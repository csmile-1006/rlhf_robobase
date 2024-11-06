import logging
import shutil
import signal
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from robobase import utils
from robobase.envs.isaaclab import IsaacLabEnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.rlhf_module.iter import get_rlhf_iter_fn
from robobase.rlhf_module.query import get_query_fn

from robobase.workspace import (
    _worker_init_fn,
    relabel_with_predictor,
    _create_default_replay_buffer,
    _create_default_query_replay_buffer,
    _create_default_feedback_replay_buffer,
)

torch.backends.cudnn.benchmark = True


class IsaacLabWorkspace:
    def __init__(
        self,
        cfg: DictConfig,
        env: gym.Env,
        env_factory: IsaacLabEnvFactory = None,
        create_replay_fn: Optional[Callable[[DictConfig], ReplayBuffer]] = None,
        work_dir: Optional[str] = None,
    ):
        if env_factory is None:
            env_factory = IsaacLabEnvFactory()
        if create_replay_fn is None:
            create_replay_fn = _create_default_replay_buffer

        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

        # Sanity checks
        if (
            cfg.replay_size_before_train * cfg.action_repeat * cfg.action_sequence
            < env.unwrapped.max_episode_length
            and cfg.replay_size_before_train > 0
        ):
            raise ValueError(
                "replay_size_before_train * action_repeat "
                f"({cfg.replay_size_before_train} * {cfg.action_repeat}) "
                f"must be >= episode_length ({env.unwrapped.max_episode_length})."
            )

        if cfg.method.is_rl and cfg.action_sequence != 1:
            raise ValueError("Action sequence > 1 is not supported for RL methods")
        if cfg.method.is_rl and cfg.execution_length != 1:
            raise ValueError("execution_length > 1 is not supported for RL methods")
        if not cfg.method.is_rl and cfg.replay.nstep != 1:
            raise ValueError("replay.nstep != 1 is not supported for IL methods")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        dev = "cpu"
        if cfg.num_gpus > 0:
            if sys.platform == "darwin":
                dev = "mps"
            else:
                dev = 0
                job_num = False
                try:
                    job_num = HydraConfig.get().job.get("num", False)
                except ValueError:
                    pass
                if job_num:
                    dev = job_num % cfg.num_gpus
        self.device = torch.device(dev)

        # create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        if (num_demos := cfg.demos) > 0:
            # Collect demos or fetch saved demos before making environments
            # to consider demo-based action space (e.g., standardization)
            self.env_factory.collect_or_fetch_demos(cfg, num_demos)

        # Make training environment
        self.train_envs = self.env_factory.make_train_env(env, cfg)

        # Create evaluation environment
        self.eval_env = None
        # self.eval_env = self.env_factory.make_eval_env(cfg)

        if num_demos > 0:
            # Post-process demos using the information from environments
            self.env_factory.post_collect_or_fetch_demos(cfg)

        # Create the RL Agent
        # observation_space = self.eval_env.observation_space
        # action_space = self.eval_env.action_space

        # observation_space = self.train_envs.unwrapped.single_observation_space
        observation_space = {}
        for key in self.train_envs.observation_space.spaces.keys():
            observation_space[key] = gym.spaces.Box(
                low=self.train_envs.observation_space.spaces[key].low[0],
                high=self.train_envs.observation_space.spaces[key].high[0],
                shape=self.train_envs.observation_space.spaces[key].shape[1:],
            )
        observation_space = gym.spaces.Dict(observation_space)

        action_space = gym.spaces.Box(
            low=np.full((1, self.train_envs.action_space.shape[-1]), -1.0),
            high=np.full((1, self.train_envs.action_space.shape[-1]), 1.0),
            shape=(1, self.train_envs.action_space.shape[-1]),
        )

        reward_space = self.train_envs.unwrapped.single_reward_space

        intrinsic_reward_module = None
        if cfg.get("intrinsic_reward_module", None):
            intrinsic_reward_module = hydra.utils.instantiate(
                cfg.intrinsic_reward_module,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
            )

        self.agent = hydra.utils.instantiate(
            cfg.method,
            device=self.device,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=cfg.num_train_envs,
            replay_alpha=cfg.replay.alpha,
            replay_beta=cfg.replay.beta,
            frame_stack_on_channel=cfg.frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
        )
        self.agent.train(False)

        self.replay_buffer = create_replay_fn(
            cfg, observation_space, action_space, extra_replay_elements=reward_space
        )
        self.prioritized_replay = cfg.replay.prioritization
        self.extra_replay_elements = self.replay_buffer.extra_replay_elements

        self.replay_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.batch_size,
            num_workers=cfg.replay.num_workers,
            pin_memory=cfg.replay.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        self._replay_iter = None

        self.use_rlhf = cfg.rlhf.use_rlhf
        if self.use_rlhf:
            self.reward_model = hydra.utils.instantiate(
                cfg.reward_method,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
                reward_space=reward_space,
            )
            self.reward_model.train(False)
            self.replay_buffer.set_reward_model(self.reward_model)

            self.query_replay_buffer = _create_default_query_replay_buffer(
                cfg,
                observation_space,
                action_space,
                save_dir=self.work_dir,
                extra_replay_elements=reward_space,
            )

            self.feedback_replay_buffer = _create_default_feedback_replay_buffer(
                cfg,
                observation_space,
                action_space,
                save_dir=self.work_dir,
                extra_replay_elements=reward_space,
            )

            self.query_replay_loader = DataLoader(
                self.query_replay_buffer,
                batch_size=self.query_replay_buffer.batch_size,
                num_workers=0,
            )
            self.feedback_replay_loader = DataLoader(
                self.feedback_replay_buffer,
                batch_size=self.feedback_replay_buffer.batch_size,
                num_workers=0,
            )
            self._query_replay_iter, self._feedback_replay_iter = None, None

            # RLHF settings
            self._reward_pretrain_step = 0
            self._total_feedback = 0

            if cfg.rlhf.feedback_type == "gemini":
                cfg.rlhf.gemini.task_description = (
                    self.env_factory.get_task_description(cfg)
                )
                cfg.rlhf.gemini.output_path = self.work_dir / "gemini"

            self._comparison_fn = get_rlhf_iter_fn(cfg)
            self._query_fn = get_query_fn(cfg.rlhf.query_type)

        # Create a separate demo replay that contains successful episodes.
        # This is designed for RL. IL algorithms don't have to use this!
        # TODO: Change the name to `self_imitation_buffer` or other names
        # Note that original buffer also contains demos, but they are not protected
        # TODO: Support demo protection in a buffer
        self.use_demo_replay = cfg.demo_batch_size is not None
        if self.use_demo_replay:
            self.demo_replay_buffer = create_replay_fn(
                cfg,
                observation_space,
                action_space,
                demo_replay=True,
                extra_replay_elements=reward_space,
            )
            self.demo_replay_loader = DataLoader(
                self.demo_replay_buffer,
                batch_size=self.demo_replay_buffer.batch_size,
                num_workers=cfg.replay.num_workers,
                pin_memory=cfg.replay.pin_memory,
                worker_init_fn=partial(_worker_init_fn, offset=3407),
            )
            if self.use_rlhf:
                self.demo_query_replay_buffer = _create_default_query_replay_buffer(
                    cfg,
                    observation_space,
                    action_space,
                    save_dir=self.work_dir,
                    use_demo=True,
                    extra_replay_elements=reward_space,
                )
                self.demo_query_replay_loader = DataLoader(
                    self.demo_query_replay_buffer,
                    batch_size=self.demo_query_replay_buffer.batch_size,
                    num_workers=0,
                )

        if self.prioritized_replay:
            if self.use_demo_replay:
                raise NotImplementedError(
                    "Demo replay is not compatible with prioritized replay"
                )

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        # from robobase.video import VideoRecorder

        # self.eval_video_recorder = VideoRecorder((self.work_dir / "eval_videos") if self.cfg.log_eval_video else None)

        self._timer = utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        if self.train_envs:
            self._episode_rollouts = [[] for _ in range(cfg.num_train_envs)]
        else:
            self._episode_rollouts = []

        # if cfg.num_eval_episodes == 0:
        #     # We no longer need the eval env
        #     self.eval_env.close()
        #     self.eval_env = None

        self._shutting_down = False

    @property
    def pretrain_steps(self):
        return self._pretrain_step

    @property
    def reward_pretrain_steps(self):
        return self._reward_pretrain_step

    @property
    def total_feedback(self):
        return self._total_feedback

    @property
    def main_loop_iterations(self):
        return self._main_loop_iterations

    @property
    def global_env_episodes(self):
        return self._global_env_episode

    @property
    def global_env_steps(self):
        """Total number of environment steps taken."""
        if not self.train_envs:
            # If train envs is not enabled, we are in pure evaluation mode.
            # Return 0 as there is no global frame.
            return 0

        # TODO: Pretrain_steps should not be included in env_steps, because it's
        # training steps but not environment steps. We need another PR to address this
        return (
            self._main_loop_iterations
            * self.cfg.action_repeat
            * self.cfg.num_train_envs
            * self.cfg.action_sequence
            + self.pretrain_steps
        )

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            _replay_iter = iter(self.replay_loader)
            if self.use_demo_replay:
                _demo_replay_iter = iter(self.demo_replay_loader)
                _replay_iter = utils.merge_replay_demo_iter(
                    _replay_iter, _demo_replay_iter
                )
            self._replay_iter = _replay_iter
        return self._replay_iter

    @property
    def query_replay_iter(self):
        if not self.use_rlhf:
            raise ValueError("reward replay is not enabled")
        if self._query_replay_iter is None:
            _query_replay_iter = iter(self.query_replay_loader)
            if self.use_demo_replay:
                _demo_query_replay_iter = iter(self.demo_query_replay_loader)
                _query_replay_iter = utils.merge_replay_demo_iter(
                    _query_replay_iter, _demo_query_replay_iter
                )
            self._query_replay_iter = _query_replay_iter
        return self._query_replay_iter

    @property
    def feedback_replay_iter(self):
        if not self.use_rlhf:
            raise ValueError("reward replay is not enabled")
        if self._feedback_replay_iter is None:
            _feedback_replay_iter = iter(self.feedback_replay_loader)
            self._feedback_replay_iter = _feedback_replay_iter
        return self._feedback_replay_iter

    def train(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        if not self.train_envs:
            raise Exception("Train envs not created! Train can't be called!")
        try:
            self._train()
        except Exception as e:
            self.shutdown()
            raise e

    def _train(self):
        # Load Demo
        self._load_demos()

        # Perform pretraining. This is suitable for behaviour cloning or Offline RL
        self._pretrain_on_demos()

        if self.use_rlhf:
            self._pretrain_reward_model_on_demos()

        # Perform online rl with exploration.
        self._online_rl() if not self.use_rlhf else self._online_rlhf()

        if self.cfg.save_snapshot:
            self.save_snapshot()
            if self.use_rlhf:
                self.save_reward_model_snapshot()

        self.shutdown()

    # Deprecated
    def eval(self) -> dict[str, Any]:
        raise RuntimeError("This method is deprecated. Please do not use.")
        return self._eval(eval_record_all_episode=True)

    # Deprecated
    def _eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        # TODO: In future, this func could do with a further refactor
        self.agent.set_eval_env_running(True)
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}
        while eval_until_episode(episode):
            observation, info = self.eval_env.reset()
            # eval agent always has last id (ids start from 0)
            self.agent.reset(self.main_loop_iterations, [self.cfg.num_train_envs])
            enabled = eval_record_all_episode or episode == 0
            self.eval_video_recorder.init(self.eval_env, enabled=enabled)
            termination, truncation = False, False
            while not (termination or truncation):
                (
                    action,
                    (next_observation, reward, termination, truncation, next_info),
                    env_metrics,
                ) = self._perform_env_steps(observation, self.eval_env, True)
                observation = next_observation
                info = next_info
                metrics.update(env_metrics)
                # Below is testing a feature wich can be enforced in v6.
                # The ability will allow agent info to be passed to envirionments.
                # This will be habdy for rednering any auxiliary outputs.
                if "agent_act_info" in env_metrics:
                    if hasattr(self.eval_env, "give_agent_info"):
                        self.eval_env.give_agent_info(env_metrics["agent_act_info"])
                self.eval_video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1
            if episode == 0:
                first_rollout = np.array(self.eval_video_recorder.frames)
            self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            success = info.get("task_success")
            if success is not None:
                successes += np.array(success).astype(int).item()
            else:
                successes = None
            episode += 1
        metrics.update(
            {
                "episode_reward": total_reward / episode,
                "episode_length": step * self.cfg.action_repeat / episode,
            }
        )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
        self.agent.set_eval_env_running(False)
        return metrics

    def _add_to_replay(
        self,
        actions,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
        next_infos,
        use_reward_model=False,
    ):
        # TODO: In future, this func could do with a further refactor
        # TODO: Add transitions into replay buffer in sliding window fashion??
        #      Currently, as train env has action sequence wrapper which only gives
        #      total reward and final obs for the full sequence, we can't perform
        #      sliding window.

        # Convert observation to list of observations ordered by train_env index
        list_of_obs_dicts = [
            dict(zip(observations, t)) for t in zip(*observations.values())
        ]
        agents_reset = []
        for i in range(self.cfg.num_train_envs):
            # Add transitions to episode rollout
            self._episode_rollouts[i].append(
                [
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    {k: infos[k][i] for k in infos.keys() if k != "log"},
                    {k: next_infos[k][i] for k in next_infos.keys() if k != "log"},
                ]
            )

            # If episode finishes, add to replay buffer.
            if terminations[i] or truncations[i]:
                agents_reset.append(i)
                ep = self._episode_rollouts[i]
                last_next_info = ep[-1][-1]
                # assert last_next_info["_final_observation"]
                # `next_info` containing `final_info` is the first info of next episode
                # we need to extract `final_info` and use it as true next_info
                # final_obs = last_next_info["final_observation"]
                # final_info = last_next_info["final_info"]

                final_obs = list_of_obs_dicts[i]
                task_success = int(last_next_info.get("task_success", 0) > 0.0)

                # Re-labeling demonstrations with reward model
                if self.use_rlhf:
                    ep = self.reward_model.compute_reward(
                        ep, use_reward_model=use_reward_model
                    )

                # Re-labeling successful demonstrations as success, following CQN
                relabeling_as_demo = (
                    task_success
                    and self.use_demo_replay
                    and self.cfg.use_self_imitation
                )
                ep_index = 0
                for act, obs, rew, term, trunc, info, next_info in ep:
                    # Only keep the last frames regardless of frame stacks because
                    # replay buffer always store single-step transitions
                    obs = {k: v[-1] for k, v in obs.items()}

                    # Strip out temporal dimension as action_sequence = 1
                    act = act[0]

                    if relabeling_as_demo:
                        info["demo"] = 1
                    else:
                        info["demo"] = 0

                    # Filter out unwanted keys in info
                    extra_replay_elements = {
                        k: v
                        for k, v in info.items()
                        if k in self.extra_replay_elements.keys()
                    }

                    self.replay_buffer.add(
                        obs, act, rew, term, trunc, **extra_replay_elements
                    )
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            obs, act, rew, term, trunc, **extra_replay_elements
                        )
                    if self.use_rlhf:
                        self.query_replay_buffer.add(
                            obs,
                            act,
                            rew,
                            term,
                            trunc,
                            ep_index,
                            **extra_replay_elements,
                        )
                    ep_index += 1

                # Add final obs
                # Only keep the last frames regardless of frame stacks because
                # replay buffer always store single-step transitions
                final_obs = {k: v[-1] for k, v in final_obs.items()}
                self.replay_buffer.add_final(final_obs)
                if relabeling_as_demo:
                    self.demo_replay_buffer.add_final(final_obs)
                if self.use_rlhf:
                    self.query_replay_buffer.add_final(final_obs)

                # clean up
                self._global_env_episode += 1
                self._episode_rollouts[i].clear()

        self.agent.reset(self.main_loop_iterations, agents_reset)  # clear hidden dim

    def _signal_handler(self, sig, frame):
        print("\nCtrl+C detected. Preparing to shutdown...")
        self._shutting_down = True

    def _load_demos(self):
        if (num_demos := self.cfg.demos) > 0:
            # NOTE: Currently we do not protect demos from being evicted from replay
            self.env_factory.load_demos_into_replay(self.cfg, self.replay_buffer)
            if self.use_demo_replay:
                # Load demos to the dedicated demo_replay_buffer
                self.env_factory.load_demos_into_replay(
                    self.cfg, self.demo_replay_buffer
                )
            if self.use_rlhf:
                # Load demos to the dedicated query_replay_buffer
                self.env_factory.load_demos_into_replay(
                    self.cfg, self.query_replay_buffer
                )
                if self.use_demo_replay:
                    self.env_factory.load_demos_into_replay(
                        self.cfg, self.demo_query_replay_buffer
                    )

        if self.cfg.replay_size_before_train > 0:
            diff = self.cfg.replay_size_before_train - len(self.replay_buffer)
            if num_demos > 0 and diff > 0:
                logging.warning(
                    f"Collecting additional {diff} random samples even though there "
                    f"are {len(self.replay_buffer)} demo samples inside the buffer. "
                    "Please make sure that this is an intended behavior."
                )

    def _perform_updates(self, unsup_train: bool = False) -> dict[str, Any]:
        if self.agent.logging:
            start_time = time.time()
        metrics = {}
        self.agent.train(True)
        for i in range(self.train_envs.num_envs):
            if (self.main_loop_iterations + i) % self.cfg.update_every_steps != 0:
                # Skip update
                continue
            for _ in range(self.cfg.num_update_steps):
                if unsup_train:
                    metrics.update(
                        self.agent.update_state_entropy(
                            self.replay_iter,
                            self.main_loop_iterations + i,
                            self.replay_buffer,
                        )
                    )
                else:
                    metrics.update(
                        self.agent.update(
                            self.replay_iter,
                            self.main_loop_iterations + i,
                            self.replay_buffer,
                        )
                    )
        self.agent.train(False)
        if self.agent.logging:
            execution_time_for_update = time.time() - start_time
            metrics["agent_batched_updates_per_second"] = (
                self.train_envs.num_envs / execution_time_for_update
            )
            metrics["agent_updates_per_second"] = (
                self.train_envs.num_envs * self.cfg.batch_size
            ) / execution_time_for_update
        return metrics

    def collect_feedback(self):
        query_batch = self._query_fn(next(self.query_replay_iter))
        feedbacks = self._comparison_fn(segments=query_batch)
        for feedback in feedbacks:
            self.feedback_replay_buffer.add_feedback(
                feedback["segment_0"], feedback["segment_1"], feedback["label"]
            )
        self._total_feedback += len(feedbacks)

    def _perform_reward_model_updates(self) -> dict[str, Any]:
        if self.reward_model.logging:
            start_time = time.time()
        metrics = {}
        self.reward_model.train(True)
        metrics.update(
            self.reward_model.update(
                self.feedback_replay_iter,
                self.main_loop_iterations,
                self.feedback_replay_buffer,
            )
        )
        self.reward_model.train(False)
        if self.reward_model.logging:
            execution_time_for_update = time.time() - start_time
            metrics["reward_model_batched_updates_per_second"] = (
                1 / execution_time_for_update
            )
            metrics["agent_updates_per_second"] = (
                1 * self.cfg.reward_replay.feedback_batch_size
            ) / execution_time_for_update

        return metrics

    # TODO: change this part according to isaaclab
    def _perform_env_steps(
        self, observations: dict[str, np.ndarray], env: gym.Env, eval_mode: bool
    ) -> tuple[np.ndarray, tuple, dict[str, Any]]:
        if self.agent.logging:
            start_time = time.time()
        with torch.no_grad(), utils.eval_mode(self.agent):
            torch_observations = observations
            if eval_mode:
                torch_observations = {
                    k: v.unsqueeze(0) for k, v in torch_observations.items()
                }
            action = self.agent.act(
                torch_observations, self.main_loop_iterations, eval_mode=eval_mode
            )
            metrics = {}
            # Below is testing a feature which can be enforced in v6.
            # The ability will allow agent info to be passed to environments.
            # This will be handy for rendering any auxiliary outputs.
            if isinstance(action, tuple):
                action, act_info = action
                metrics["agent_act_info"] = act_info
            # action = action.cpu().detach().numpy()
            if action.ndim != 3:
                raise ValueError(
                    "Expected actions from `agent.act` to have shape "
                    "(Batch, Timesteps, Action Dim)."
                )
            if eval_mode:
                action = action[0]  # we expect batch of 1 for eval

        if self.agent.logging:
            execution_time_for_act = time.time() - start_time
            metrics["agent_act_steps_per_second"] = (
                self.cfg.num_train_envs / execution_time_for_act
            )
            start_time = time.time()

        # hard-coded part for removing temporal axis in robobase stuffs.
        *env_step_tuple, next_info = env.step(action.squeeze(1))

        if self.agent.logging:
            execution_time_for_env_step = time.time() - start_time
            metrics["env_steps_per_second"] = (
                self.cfg.num_train_envs / execution_time_for_env_step
            )
            for k, v in next_info["log"].items():
                # if train env, then will be vectorised, so get first elem
                metrics[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

        return action, (*env_step_tuple, next_info), metrics

    def _pretrain_on_demos(self):
        if self.cfg.num_pretrain_steps > 0:
            pre_train_until_step = utils.Until(self.cfg.num_pretrain_steps)
            should_pretrain_log = utils.Every(self.cfg.log_pretrain_every)
            should_pretrain_eval = utils.Every(self.cfg.eval_every_steps)
            if self.cfg.log_pretrain_every > 0:
                assert self.cfg.num_pretrain_steps % self.cfg.log_pretrain_every == 0
            if len(self.replay_buffer) <= 0:
                raise ValueError(
                    "there is no sample to pre-train with in the replay buffer "
                    f"but num_pretrain_steps ({self.cfg.num_pretrain_steps}) is > 0"
                )

            while pre_train_until_step(self.pretrain_steps):
                self.agent.logging = False

                if should_pretrain_log(self.pretrain_steps):
                    self.agent.logging = True
                pretrain_metrics = self._perform_updates()

                if should_pretrain_log(self.pretrain_steps):
                    pretrain_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        pretrain_metrics, self.pretrain_steps, prefix="pretrain"
                    )

                if should_pretrain_eval(self.pretrain_steps):
                    eval_metrics = self._eval()
                    eval_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        eval_metrics, self.pretrain_steps, prefix="pretrain_eval"
                    )

                self._pretrain_step += 1

    def _pretrain_reward_model_on_demos(self):
        if self.cfg.rlhf.num_pretrain_steps > 0:
            pre_train_until_step = utils.Until(self.cfg.rlhf.num_pretrain_steps)
            should_pretrain_log = utils.Every(self.cfg.log_pretrain_every)
            if self.cfg.log_pretrain_every > 0:
                assert (
                    self.cfg.rlhf.num_pretrain_steps % self.cfg.log_pretrain_every == 0
                )
            self.collect_feedback()
            if len(self.feedback_replay_buffer) <= 0:
                raise ValueError(
                    "there is no sample to pre-train with in the replay buffer "
                    f"but num_pretrain_steps ({self.cfg.num_pretrain_steps}) is > 0"
                )

            while pre_train_until_step(self.reward_pretrain_steps):
                self.reward_model.logging = False

                if should_pretrain_log(self.reward_pretrain_steps):
                    self.reward_model.logging = True
                pretrain_metrics = self._perform_reward_model_updates()

                if should_pretrain_log(self.reward_pretrain_steps):
                    pretrain_metrics.update(self._get_common_metrics())
                    pretrain_metrics["iteration"] = self.reward_pretrain_steps
                    self.logger.log_metrics(
                        pretrain_metrics,
                        self.reward_pretrain_steps,
                        prefix="pretrain_reward",
                    )

                self._reward_pretrain_step += 1

            relabel_with_predictor(self.reward_model, self.replay_buffer)
            if self.use_demo_replay:
                relabel_with_predictor(self.reward_model, self.demo_replay_buffer)

    def _online_rl(self):
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        seed_until_size = utils.Until(self.cfg.replay_size_before_train)
        should_log = utils.Every(self.cfg.log_every)
        # eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        # should_eval = utils.Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = utils.Every(snapshot_every_n)

        observations, info = self.train_envs.reset()
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = None

        while train_until_frame(self.global_env_steps):
            metrics = {}

            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True
            if not seed_until_size(len(self.replay_buffer)):
                update_metrics = self._perform_updates()
                metrics.update(update_metrics)

            (
                action,
                (next_observations, rewards, terminations, truncations, next_info),
                env_metrics,
            ) = self._perform_env_steps(observations, self.train_envs, False)

            agent_0_reward += rewards[0]
            agent_0_ep_len += 1
            if terminations[0] or truncations[0]:
                agent_0_prev_ep_len = agent_0_ep_len
                agent_0_prev_reward = agent_0_reward
                agent_0_ep_len = agent_0_reward = 0

            metrics.update(env_metrics)
            transitions = (
                action,
                observations,
                rewards,
                terminations,
                truncations,
                info,
                next_info,
            )
            numpy_transitions = list(map(utils.convert_torch_to_numpy, transitions))
            self._add_to_replay(
                *numpy_transitions,
                use_reward_model=seed_until_size(len(self.replay_buffer)),
            )
            observations = next_observations
            info = next_info
            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_length": agent_0_prev_ep_len
                            * self.cfg.action_repeat,
                        }
                    )

                eureka_metrics = {key: value for key, value in metrics.items() if "Eureka" in key}
                non_eureka_metrics = {key: value for key, value in metrics.items() if "Eureka" not in key}
                self.logger.log_metrics(non_eureka_metrics, self.global_env_steps, prefix="train")
                # Hack code for logging Eureka metrics
                for key, value in eureka_metrics.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 1:
                        self.logger._log(key, np.mean(value), self.global_env_steps)  # type: ignore
                    else:
                        self.logger._log(key, value, self.global_env_steps)  # type: ignore
                self.logger._dump(self.global_env_steps, "train")  # type: ignore

            # temporarily disable evaluation in IsaacLab.
            if False:
                eval_metrics = self._eval()
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

    def _online_rlhf(self):
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        seed_until_size = utils.Until(self.cfg.replay_size_before_train)
        unsup_train_until_frame = utils.Until(
            self.cfg.rlhf.num_unsup_train_frames + self.cfg.replay_size_before_train
        )
        should_log = utils.Every(self.cfg.log_every)
        # eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        # should_eval = utils.Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = utils.Every(snapshot_every_n)

        should_update_reward_model = utils.Every(self.cfg.rlhf.update_every_steps)
        should_log_reward_model = utils.Every(self.cfg.rlhf.log_every)
        snapshot_reward_model_every_n = (
            self.cfg.rlhf.snapshot_every_n if self.cfg.save_snapshot else 0
        )
        should_save_reward_model_snapshot = utils.Every(snapshot_reward_model_every_n)

        observations, info = self.train_envs.reset()
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = None

        while train_until_frame(self.global_env_steps):
            metrics = {}

            num_reward_model_updates = 0
            self.reward_model.logging = False
            if (
                not seed_until_size(len(self.replay_buffer))
                and not unsup_train_until_frame(len(self.replay_buffer))
                and self.total_feedback < self.cfg.rlhf.max_feedback
                and should_update_reward_model(self.global_env_steps)
            ):
                self.reward_model.logging = True
                logging.info(
                    f"[Feedback {self.total_feedback} / {self.cfg.rlhf.max_feedback}] Collecting feedback for {self.cfg.reward_replay.num_queries} queries"  # noqa
                )
                self.collect_feedback()
                for it in range(self.cfg.rlhf.num_train_frames):
                    reward_update_metrics = self._perform_reward_model_updates()
                    metrics.update(reward_update_metrics)
                    metrics.update(self._get_common_metrics())
                    metrics["iteration"] = self.global_env_steps + it
                    if should_log_reward_model(it):
                        self.logger.log_metrics(
                            metrics, metrics["iteration"], prefix="train_reward"
                        )
                    metrics = {}
                relabel_with_predictor(self.reward_model, self.replay_buffer)
                if self.use_demo_replay:
                    relabel_with_predictor(self.reward_model, self.demo_replay_buffer)

                if num_reward_model_updates == 0:
                    (
                        self.agent.critic,
                        self.agent.critic_target,
                        self.agent.critic_opt,
                    ) = self.agent.build_critic()
                    self.agent.train(True)
                    for j in range(self.cfg.rlhf.num_reset_update_steps):
                        _ = self.agent.update(
                            self.replay_iter,
                            self.main_loop_iterations + j,
                            self.replay_buffer,
                        )
                    self.agent.train(False)
                num_reward_model_updates += 1

            if (
                self.total_feedback < self.cfg.rlhf.max_feedback
                and should_save_reward_model_snapshot(self.main_loop_iterations)
            ):
                self.save_reward_model_snapshot()

            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True

            if not seed_until_size(len(self.replay_buffer)):
                update_metrics = self._perform_updates(
                    unsup_train=unsup_train_until_frame(len(self.replay_buffer))
                )
                metrics.update(update_metrics)

            (
                action,
                (next_observations, rewards, terminations, truncations, next_info),
                env_metrics,
            ) = self._perform_env_steps(observations, self.train_envs, False)

            agent_0_reward += rewards[0]
            agent_0_ep_len += 1
            if terminations[0] or truncations[0]:
                agent_0_prev_ep_len = agent_0_ep_len
                agent_0_prev_reward = agent_0_reward
                agent_0_ep_len = agent_0_reward = 0

            metrics.update(env_metrics)
            transitions = (
                action,
                observations,
                rewards,
                terminations,
                truncations,
                info,
                next_info,
            )
            numpy_transitions = list(map(utils.convert_torch_to_numpy, transitions))
            self._add_to_replay(
                *numpy_transitions,
                use_reward_model=seed_until_size(len(self.replay_buffer)),
            )
            observations = next_observations
            info = next_info
            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_length": agent_0_prev_ep_len
                            * self.cfg.action_repeat,
                        }
                    )
                self.logger.log_metrics(
                    metrics,
                    self.global_env_steps,
                    prefix="train"
                    if not unsup_train_until_frame(len(self.replay_buffer))
                    else "unsup_train",
                )

            # temporarily disable evaluation in IsaacLab.
            if False:
                eval_metrics = self._eval()
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

    def _get_common_metrics(self) -> dict[str, Any]:
        _, total_time = self._timer.reset()
        metrics = {
            "total_time": total_time,
            "iteration": self.main_loop_iterations,
            "env_steps": self.global_env_steps,
            "env_episodes": self.global_env_episodes,
            "buffer_size": len(self.replay_buffer),
        }
        if self.use_demo_replay:
            metrics["demo_buffer_size"] = len(self.demo_replay_buffer)
        return metrics

    def shutdown(self):
        if self.eval_env:
            self.eval_env.close()

        # self.train_envs.close()
        self.replay_buffer.shutdown()
        if self.use_demo_replay:
            self.demo_replay_buffer.shutdown()

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshots" / f"{self.global_env_steps}_snapshot.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        keys_to_save = [
            "_pretrain_step",
            "_main_loop_iterations",
            "_global_env_episode",
            "cfg",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["agent"] = self.agent.state_dict()
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        latest_snapshot = self.work_dir / "snapshots" / "latest_snapshot.pt"
        shutil.copy(snapshot, latest_snapshot)

    def load_snapshot(self, path_to_snapshot_to_load=None):
        if path_to_snapshot_to_load is None:
            path_to_snapshot_to_load = (
                self.work_dir / "snapshots" / "latest_snapshot.pt"
            )
        else:
            path_to_snapshot_to_load = Path(path_to_snapshot_to_load)
        if not path_to_snapshot_to_load.is_file():
            raise ValueError(
                f"Provided file '{str(path_to_snapshot_to_load)}' is not a snapshot."
            )
        with path_to_snapshot_to_load.open("rb") as f:
            payload = torch.load(f, map_location="cpu")
        self.agent.load_state_dict(payload.pop("agent"))
        for k, v in payload.items():
            self.__dict__[k] = v

    def save_reward_model_snapshot(self):
        snapshot = (
            self.work_dir
            / "reward_model_snapshots"
            / f"{self.global_env_steps}_snapshot.pt"
        )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        keys_to_save = [
            "_pretrain_step",
            "_main_loop_iterations",
            "_global_env_episode",
            "_total_feedback",
            "cfg",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["reward_model"] = self.reward_model.state_dict()
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        latest_snapshot = (
            self.work_dir / "reward_model_snapshots" / "latest_snapshot.pt"
        )
        shutil.copy(snapshot, latest_snapshot)

    def load_reward_model_snapshot(self, path_to_snapshot_to_load=None):
        if path_to_snapshot_to_load is None:
            path_to_snapshot_to_load = (
                self.work_dir / "reward_model_snapshots" / "latest_snapshot.pt"
            )
        else:
            path_to_snapshot_to_load = Path(path_to_snapshot_to_load)
        if not path_to_snapshot_to_load.is_file():
            raise ValueError(
                f"Provided file '{str(path_to_snapshot_to_load)}' is not a snapshot."
            )
        with path_to_snapshot_to_load.open("rb") as f:
            payload = torch.load(f, map_location="cpu")
        self.reward_model.load_state_dict(payload.pop("reward_model"))
        for k, v in payload.items():
            self.__dict__[k] = v
