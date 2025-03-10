import asyncio
import logging
import os
import random
import shutil
import signal
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium import spaces
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tensordict.nn import CudaGraphModule
from torch.utils.data import DataLoader
from tqdm import tqdm

import robobase
from robobase import utils
from robobase.envs.env import EnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.rlhf.feedback_replay_buffer import FeedbackReplayBuffer
from robobase.replay_buffer.rlhf.query_replay_buffer import QueryReplayBuffer
from robobase.rlhf_module.iter import get_rlhf_iter_fn
from robobase.rlhf_module.third_party.gemini import configure_gemini

from robobase.method.ppo import ActorCritic, PPO

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"


def _worker_init_fn(worker_id, offset=0):
    seed = np.random.get_state()[1][0] + worker_id + offset
    np.random.seed(seed)
    random.seed(int(seed))


def _create_default_query_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    save_dir: Path = None,
    use_demo: bool = False,
    extra_replay_elements: dict[str, gym.Space] = None,
) -> QueryReplayBuffer:
    if extra_replay_elements is None:
        extra_replay_elements = spaces.Dict({})
    if cfg.demos > 0:
        extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)

    if cfg.demos > 0:
        batch_size = (
            (
                cfg.rlhf_replay.num_queries // 2 + 1
                if not use_demo
                else cfg.rlhf_replay.num_queries // 2
            )
            if "pairwise" in cfg.rlhf.comparison_type
            else cfg.rlhf_replay.num_queries * cfg.rlhf_replay.larger_batch_ratio
        )
    else:
        if "pairwise" in cfg.rlhf.comparison_type:
            batch_size = cfg.rlhf_replay.num_queries + 1
        else:
            batch_size = (
                cfg.rlhf_replay.num_queries * cfg.rlhf_replay.larger_batch_ratio
            )

    return QueryReplayBuffer(
        save_dir=save_dir / "queries" if not use_demo else save_dir / "demo_queries",
        batch_size=batch_size,
        replay_capacity=cfg.rlhf_replay.size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        observation_elements=observation_space,
        extra_replay_elements=extra_replay_elements,
        num_workers=0,
        sequential=True,
        transition_seq_len=cfg.rlhf_replay.seq_len,
        max_episode_number=cfg.rlhf_replay.max_episode_number if not use_demo else 0,
        upload_gemini=cfg.rlhf.feedback_type == "gemini",
        purge_replay_on_shutdown=True,
        save_snapshot=True,
    )


def _create_default_feedback_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    save_dir: Path = None,
    extra_replay_elements: dict[str, gym.Space] = None,
) -> FeedbackReplayBuffer:
    return FeedbackReplayBuffer(
        save_dir=save_dir / "feedbacks",
        batch_size=cfg.rlhf_replay.feedback_batch_size,
        replay_capacity=cfg.rlhf_replay.size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        observation_elements=observation_space,
        extra_replay_elements=extra_replay_elements,
        num_workers=cfg.rlhf_replay.num_workers,
        sequential=False,
        transition_seq_len=cfg.rlhf_replay.seq_len,
        num_labels=cfg.rlhf_replay.num_labels,
        purge_replay_on_shutdown=True,
    )


def _create_default_envs(cfg: DictConfig) -> EnvFactory:
    factory = None
    if cfg.env.env_name == "rlbench":
        from robobase.envs.rlbench import RLBenchEnvFactory

        factory = RLBenchEnvFactory()
    elif cfg.env.env_name == "dmc":
        from robobase.envs.dmc import DMCEnvFactory

        factory = DMCEnvFactory()
    elif cfg.env.env_name == "bigym":
        from robobase.envs.bigym import BiGymEnvFactory

        factory = BiGymEnvFactory()
    elif cfg.env.env_name == "d4rl":
        from robobase.envs.d4rl import D4RLEnvFactory

        factory = D4RLEnvFactory()
    elif cfg.env.env_name == "agym":
        from robobase.envs.agym import AGymEnvFactory

        factory = AGymEnvFactory()
    elif cfg.env.env_name == "humanoidbench":
        from robobase.envs.humanoidbench import HumanoidBenchEnvFactory

        factory = HumanoidBenchEnvFactory()
    elif cfg.env.env_name == "locomujoco":
        from robobase.envs.locomujoco import LocoMujocoEnvFactory

        factory = LocoMujocoEnvFactory()
    else:
        ValueError()
    return factory


class OnPolicyWorkspace:
    def __init__(
        self,
        cfg: DictConfig,
        env_factory: EnvFactory = None,
        work_dir: str = None,
    ):
        if env_factory is None:
            env_factory = _create_default_envs(cfg)

        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

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

        self.eval_env = self.env_factory.make_eval_env(cfg)

        # Create the RL Agent
        full_observation_space = self.eval_env.observation_space
        clean_observation_space = spaces.Dict(
            {
                k: v
                for k, v in full_observation_space.items()
                if "query_pixels_" not in k
            }
        )
        action_space = self.eval_env.action_space

        num_obs = clean_observation_space["low_dim_state"].shape[-1]
        num_critic_obs = num_obs

        actor_critic = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=action_space.shape[-1],
            actor_hidden_dims=cfg.method.actor_hidden_dims,
            critic_hidden_dims=cfg.method.critic_hidden_dims,
            activation=cfg.method.activation,
        )
        self.agent = PPO(
            actor_critic=actor_critic,
            num_learning_epochs=cfg.method.num_learning_epochs,
            num_mini_batches=cfg.method.num_mini_batches,
            clip_param=cfg.method.clip_param,
            gamma=cfg.method.gamma,
            lam=cfg.method.lam,
            value_loss_coef=cfg.method.value_loss_coef,
            entropy_coef=cfg.method.entropy_coef,
            learning_rate=cfg.method.learning_rate,
            max_grad_norm=cfg.method.max_grad_norm,
            use_clipped_value_loss=cfg.method.use_clipped_value_loss,
            schedule=cfg.method.schedule,
            desired_kl=cfg.method.desired_kl,
            device=self.device,
        )
        self.agent.init_storage(
            self.cfg.num_train_envs,
            self.cfg.method.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [action_space.shape[-1]],
        )

        self.agent.actor_critic.eval()

        # Make training environment
        if cfg.num_train_envs > 0:
            self.train_envs = self.env_factory.make_train_env(cfg)
        else:
            self.train_envs = None
            logging.warning("Train env is not created. Training will not be supported ")

        self.use_rlhf = cfg.rlhf.use_rlhf
        if self.cfg.env.env_name in ["humanoidbench", "dmc", "agym"]:
            reward_space = self.eval_env.unwrapped.reward_space
            extra_replay_elements = reward_space
        else:
            extra_replay_elements = None

        if self.use_rlhf:
            assert (
                self.cfg.rlhf.num_pretrain_frames == 0
                or self.cfg.rlhf.num_unsup_train_frames == 0
            ), "Either num_pretrain_frames or num_unsup_train_frames must be 0."
            self.rlhf_reset_flag = False
            self.rlhf_replay_reset_flag = False

            self.reward_model = hydra.utils.instantiate(
                cfg.reward_method,
                device=self.device,
                observation_space=clean_observation_space,
                action_space=action_space,
                reward_space=reward_space,
                reward_operator=cfg.env.get("reward_operator", "sum"),
            )
            self.reward_model.train(False)
            self.query_replay_buffer = _create_default_query_replay_buffer(
                cfg,
                observation_space=full_observation_space,
                action_space=action_space,
                save_dir=self.work_dir,
                extra_replay_elements=extra_replay_elements,
            )

            self.feedback_replay_buffer = _create_default_feedback_replay_buffer(
                cfg,
                observation_space=clean_observation_space,
                action_space=action_space,
                save_dir=self.work_dir,
                extra_replay_elements=extra_replay_elements,
            )

            self.query_replay_loader = DataLoader(
                self.query_replay_buffer,
                batch_size=self.query_replay_buffer.batch_size,
                worker_init_fn=partial(_worker_init_fn, offset=1234),
            )
            self.feedback_replay_loader = DataLoader(
                self.feedback_replay_buffer,
                batch_size=self.feedback_replay_buffer.batch_size,
                num_workers=cfg.rlhf_replay.num_workers,
                pin_memory=cfg.rlhf_replay.pin_memory,
                worker_init_fn=partial(_worker_init_fn, offset=4567),
                persistent_workers=False,
            )
            self._query_replay_iter, self._feedback_replay_iter = None, None

            # RLHF settings
            self._reward_pretrain_step = 0
            self._total_feedback = 0
            self._feedback_iter = 0

            self._unsup_update_step = 0
            self._gemini_client = None

            if cfg.rlhf.feedback_type == "gemini":
                self._gemini_client = configure_gemini()
                import asyncio

                self._loop = asyncio.get_event_loop()
                asyncio.set_event_loop(self._loop)

            self._rlhf_iter_fn = get_rlhf_iter_fn(
                self.work_dir, cfg, env_factory, self.reward_model, self._gemini_client
            )

        self.extra_replay_elements = (
            extra_replay_elements
            if extra_replay_elements is not None
            else spaces.Dict({})
        )

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        from robobase.video import VideoRecorder

        self.eval_video_recorder = VideoRecorder(
            (self.work_dir / "eval_videos") if self.cfg.log_eval_video else None
        )

        self._timer = utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._update_step = 0
        self._act_dim = self.eval_env.action_space.shape[0]
        if self.train_envs:
            self._episode_rollouts = [[] for _ in range(self.train_envs.num_envs)]
        else:
            self._episode_rollouts = []

        if cfg.num_eval_episodes == 0:
            # We no longer need the eval env
            self.eval_env.close()
            self.eval_env = None

        self._shutting_down = False
        # store code state for checking diffs
        utils.store_code_state(
            self.work_dir, [robobase.__file__, self.env_factory.env_class.__file__]
        )

    @property
    def pretrain_steps(self):
        return self._pretrain_step

    @property
    def reward_pretrain_steps(self):
        return self._reward_pretrain_step

    @property
    def update_steps(self):
        return self._update_step

    @property
    def unsup_update_steps(self):
        return self._unsup_update_step

    @property
    def total_feedback(self):
        return self._total_feedback

    @property
    def feedback_iter(self):
        return self._feedback_iter

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
            * self.cfg.method.num_steps_per_env
            * self.cfg.action_repeat
            * self.train_envs.num_envs
            * (
                self.cfg.action_sequence
                if not self.cfg.temporal_ensemble
                else self.cfg.execution_length
            )
            + self.pretrain_steps
        )

    @property
    def query_replay_iter(self):
        if not self.use_rlhf:
            raise ValueError("reward replay is not enabled")
        if self._query_replay_iter is None:
            _query_replay_iter = iter(self.query_replay_loader)
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
            logging.critical(e)
            self.shutdown()
            raise e

    def _setup_training_functions(self):
        if self.cfg.use_compile:
            self._update_fn = torch.compile(self.agent.update)
            self._act_fn = torch.compile(self.agent.act)
            torch.set_float32_matmul_precision("high")
        else:
            self._update_fn = self.agent.update
            self._act_fn = self.agent.act

        if self.cfg.rlhf.use_rlhf:
            self._reward_update_fn = self.reward_model.update

        if self.cfg.use_cuda_graph:
            self._update_fn = CudaGraphModule(self._update_fn, in_keys=[], out_keys=[])

    def _train(self):
        self._setup_training_functions()

        # Perform online rl with exploration.
        self._online_rl()

        if self.cfg.save_snapshot:
            self.save_snapshot()
            if self.use_rlhf:
                self.save_reward_model_snapshot()

        if hasattr(self, "_loop"):
            self._loop.close()

        self.shutdown()

    def eval(self) -> dict[str, Any]:
        return self._eval(eval_record_all_episode=True)

    def _eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        # TODO: In future, this func could do with a further refactor
        step, episode, total_learned_reward, total_reward, successes = 0, 0, 0, 0, 0
        if len(self.extra_replay_elements) > 0:
            reward_term_dict = {key: 0 for key in self.extra_replay_elements}
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}
        pbar = tqdm(
            total=self.cfg.num_eval_episodes, desc="Evaluating", leave=False, position=0
        )
        while eval_until_episode(episode):
            observation, info = self.eval_env.reset()
            # eval agent always has last id (ids start from 0)
            enabled = eval_record_all_episode or episode == 0
            self.eval_video_recorder.init(self.eval_env, enabled=enabled)
            termination, truncation = False, False
            episode_pbar = tqdm(
                total=self.cfg.env.episode_length,
                desc="Episode",
                leave=False,
                position=1,
            )
            critic_observation = observation
            while not (termination or truncation):
                (
                    _,
                    (next_observation, reward, termination, truncation, next_info),
                    env_metrics,
                ) = self._perform_env_steps(
                    observation, critic_observation, self.eval_env, True
                )
                observation = next_observation
                critic_observation = next_observation
                info = next_info
                metrics.update(env_metrics)
                # Below is testing a feature wich can be enforced in v6.
                # The ability will allow agent info to be passed to envirionments.
                # This will be habdy for rednering any auxiliary outputs.
                if "agent_act_info" in env_metrics:
                    if hasattr(self.eval_env, "give_agent_info"):
                        self.eval_env.give_agent_info(env_metrics["agent_act_info"])
                self.eval_video_recorder.record(self.eval_env)
                total_reward += info.get("task_reward", reward)
                total_learned_reward += reward
                if len(self.extra_replay_elements) > 0:
                    for key in info.keys():
                        if key.startswith("Reward/"):
                            reward_term_dict[key] += info[key]
                step += 1
                episode_pbar.update(1)
            if episode == 0:
                first_rollout = np.array(self.eval_video_recorder.frames)
            if self.cfg.env.task_name != "humanoidbench":
                self.eval_video_recorder.save(f"{self.global_env_steps}_{episode}.mp4")
            else:
                if episode == 0:
                    self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            success = info.get("task_success")
            if success is not None:
                successes += np.array(success).astype(int).item()
            else:
                successes = None
            episode += 1
            pbar.update(1)
        metrics.update(
            {
                "episode_reward": total_reward / episode,
                "episode_learned_reward": total_learned_reward / episode,
                "episode_length": step * self.cfg.action_repeat / episode,
            }
        )
        if len(self.extra_replay_elements) > 0:
            metrics.update(
                {
                    f"return_{key.split('/')[-1]}": val / episode
                    for key, val in reward_term_dict.items()
                }
            )
            metrics.update(
                {
                    f"real_return_{key.split('/')[-1]}": val
                    / episode
                    * self.eval_env.initial_reward_scale[key.split("/")[-1]]
                    for key, val in reward_term_dict.items()
                }
            )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
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
        for i in range(self.train_envs.num_envs):
            # Add transitions to episode rollout
            self._episode_rollouts[i].append(
                [
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    {k: infos[k][i] for k in infos.keys()},
                    {k: next_infos[k][i] for k in next_infos.keys()},
                ]
            )

            # If episode finishes, add to replay buffer.
            if terminations[i] or truncations[i]:
                agents_reset.append(i)
                ep = self._episode_rollouts[i]
                last_next_info = ep[-1][-1]
                assert last_next_info["_final_observation"]
                # `next_info` containing `final_info` is the first info of next episode
                # we need to extract `final_info` and use it as true next_info
                final_obs = last_next_info["final_observation"]
                final_info = last_next_info["final_info"]
                task_success = int(final_info.get("task_success", 0) > 0.0)

                # Re-labeling successful demonstrations as success, following CQN
                relabeling_as_demo = task_success and self.cfg.use_self_imitation
                ep_index = 0
                for act, obs, rew, term, trunc, info, next_info in ep:
                    # Only keep the last frames regardless of frame stacks because
                    # replay buffer always store single-step transitions
                    obs = {k: v[-1] for k, v in obs.items()}
                    clean_obs = {
                        k: v for k, v in obs.items() if "query_pixels_" not in k
                    }

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
                        if k in list(self.extra_replay_elements.keys())
                    }
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            clean_obs, act, rew, term, trunc, **extra_replay_elements
                        )
                    if (
                        self.use_rlhf
                        and self.total_feedback < self.cfg.rlhf.max_feedback
                    ):
                        task_rew = info["task_reward"]
                        self.query_replay_buffer.add(
                            obs,
                            act,
                            task_rew,
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
                final_clean_obs = {
                    k: v for k, v in final_obs.items() if "query_pixels_" not in k
                }
                if relabeling_as_demo:
                    self.demo_replay_buffer.add_final(final_clean_obs)
                if self.use_rlhf and self.total_feedback < self.cfg.rlhf.max_feedback:
                    self.query_replay_buffer.add_final(final_obs)

                # clean up
                self._global_env_episode += 1
                self._episode_rollouts[i].clear()

    def _signal_handler(self, sig, frame):
        print("\nCtrl+C detected. Preparing to shutdown...")
        self._shutting_down = True

    def _perform_updates(self, unsup_train: bool = False) -> dict[str, Any]:
        def choose_update_fn():
            return self._update_fn

        update_fn = choose_update_fn()
        if self.agent.logging:
            start_time = time.time()
        metrics = {}
        self.agent.actor_critic.train()
        metrics.update(update_fn())
        self.agent.actor_critic.eval()
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
        query_batch = next(self.query_replay_iter)
        if self.cfg.rlhf.feedback_type == "gemini":
            if not hasattr(self, "_loop"):
                self._loop = asyncio.get_event_loop()
                asyncio.set_event_loop(self._loop)
            feedbacks, metadata = self._loop.run_until_complete(
                self._rlhf_iter_fn(
                    segments=query_batch, feedback_iter=self.feedback_iter
                )
            )
        else:
            feedbacks, metadata = self._rlhf_iter_fn(
                segments=query_batch, feedback_iter=self.feedback_iter
            )
        if metadata:
            for feedback, metadatum in zip(feedbacks, metadata):
                self.feedback_replay_buffer.add_feedback(
                    feedback["segment_0"],
                    feedback["segment_1"],
                    feedback["label"],
                    metadatum,
                )
        else:
            for feedback in feedbacks:
                self.feedback_replay_buffer.add_feedback(
                    feedback["segment_0"],
                    feedback["segment_1"],
                    feedback["label"],
                )
        self._total_feedback += len(feedbacks)
        self._feedback_iter += 1

    def _perform_reward_model_updates(self) -> dict[str, Any]:
        if self.reward_model.logging:
            start_time = time.time()
        metrics = {}
        self.reward_model.train(True)
        feedback_one_epoch = utils.Until(
            max(self.total_feedback / self.cfg.rlhf_replay.feedback_batch_size, 1)
        )
        it = 0
        while feedback_one_epoch(it):
            batches = self.reward_model.extract_batch(self.feedback_replay_iter)
            metrics.update(self._reward_update_fn(batches))
            it += 1
        self.reward_model.train(False)
        if self.reward_model.logging:
            execution_time_for_update = time.time() - start_time
            metrics["reward_model_batched_updates_per_second"] = (
                1 / execution_time_for_update
            )
            metrics["agent_updates_per_second"] = (
                1 * self.cfg.rlhf_replay.feedback_batch_size
            ) / execution_time_for_update

        return metrics

    def _perform_env_steps(
        self,
        observations: dict[str, np.ndarray],
        critic_observations: dict[str, np.ndarray],
        env: gym.Env,
        eval_mode: bool,
    ) -> tuple[np.ndarray, tuple, dict[str, Any]]:
        if self.agent.logging:
            start_time = time.time()
        with torch.no_grad():
            torch_observations = torch.as_tensor(
                observations["low_dim_state"], dtype=torch.float32, device=self.device
            ).squeeze(-2)
            torch_critic_observations = torch.as_tensor(
                critic_observations["low_dim_state"],
                dtype=torch.float32,
                device=self.device,
            ).squeeze(-2)
            if eval_mode:
                torch_observations = torch_observations.unsqueeze(0)
                torch_critic_observations = torch_critic_observations.unsqueeze(0)
            action = self._act_fn(
                torch_observations,
                torch_critic_observations,
                eval_mode=eval_mode,
            )
            metrics = {}
            # Below is testing a feature which can be enforced in v6.
            # The ability will allow agent info to be passed to environments.
            # This will be handy for rendering any auxiliary outputs.
            if isinstance(action, tuple):
                action, act_info = action
                metrics["agent_act_info"] = act_info
            action = action.cpu().detach().numpy()[:, None, :]
            if action.ndim != 3:
                raise ValueError(
                    f"Expected actions from `agent.act` to have shape (Batch, Timesteps, Action Dim) != {action.shape}."
                )
            if eval_mode:
                action = action[0]  # we expect batch of 1 for eval

        if self.agent.logging:
            execution_time_for_act = time.time() - start_time
            metrics["agent_act_steps_per_second"] = (
                self.train_envs.num_envs / execution_time_for_act
            )
            start_time = time.time()

        next_observations, rewards, terminations, truncations, next_info = env.step(
            action
        )
        # TODO: debug details
        if self.use_rlhf and not eval_mode:
            rewards = self.reward_model.compute_reward(
                {
                    "action": action,
                    **{k: v for k, v in observations.items()},
                    **{k: v for k, v in next_info.items()},
                    "reward": rewards,
                },
                episodic=False,
            )["reward"]

        if self.agent.logging:
            execution_time_for_env_step = time.time() - start_time
            metrics["env_steps_per_second"] = (
                self.train_envs.num_envs / execution_time_for_env_step
            )
            for k, v in next_info.items():
                # if train env, then will be vectorised, so get first elem
                metrics[f"env_info/{k}"] = v if eval_mode else v[0]

        if not eval_mode:
            self.agent.process_env_step(rewards, terminations, next_info)

        if eval_mode:
            next_info.update(env.last_reward)
        else:
            _rewards = env.get_attr("last_reward")
            next_info.update(
                {
                    k: np.stack([elem[k] for elem in _rewards], axis=0)
                    for k in _rewards[0].keys()
                }
            )

        return (
            action,
            (next_observations, rewards, terminations, truncations, next_info),
            metrics,
        )

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
        if self.cfg.rlhf.num_pretrain_frames > 0:
            pre_train_until_step = utils.Until(self.cfg.rlhf.num_pretrain_frames)
            should_pretrain_log = utils.Every(self.cfg.log_pretrain_every)
            if self.cfg.log_pretrain_every > 0:
                assert (
                    self.cfg.rlhf.num_pretrain_frames % self.cfg.log_pretrain_every == 0
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

    def _online_rl(self):
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        should_log = utils.Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = utils.Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = utils.Every(snapshot_every_n)
        if self.use_rlhf:
            should_reward_log = utils.Every(self.cfg.rlhf.log_every)
            should_update_reward_model = utils.Every(self.cfg.rlhf.update_every_steps)
            snapshot_reward_model_every_n = (
                self.cfg.rlhf.snapshot_every_n if self.cfg.save_snapshot else 0
            )
            should_save_reward_model_snapshot = utils.Every(
                snapshot_reward_model_every_n
            )

        observations, info = self.train_envs.reset()
        critic_observations = observations
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = agent_0_learned_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = agent_0_prev_learned_reward = None
        while train_until_frame(self.global_env_steps):
            if self.use_rlhf and self.total_feedback >= self.cfg.rlhf.max_feedback:
                if self.rlhf_reset_flag is False and self.cfg.rlhf.reset_after_rlhf:
                    observations, info = self.reset_after_rlhf()
                    self.rlhf_reset_flag = True

            metrics = {}

            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True

            for _ in range(self.cfg.method.num_steps_per_env):
                (
                    action,
                    (next_observations, rewards, terminations, truncations, next_info),
                    env_metrics,
                ) = self._perform_env_steps(
                    observations,
                    critic_observations,
                    self.train_envs,
                    False,
                )

                agent_0_learned_reward += rewards[0]
                agent_0_reward += next_info.get("task_reward", rewards)[0]
                agent_0_ep_len += 1
                if terminations[0] or truncations[0]:
                    agent_0_prev_ep_len = agent_0_ep_len
                    agent_0_prev_reward = agent_0_reward
                    agent_0_prev_learned_reward = agent_0_learned_reward
                    agent_0_ep_len = agent_0_reward = agent_0_learned_reward = 0

                metrics.update(env_metrics)
                self._add_to_replay(
                    action,
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    info,
                    next_info,
                )
                observations = next_observations
                critic_observations = next_observations
                info = next_info

            with torch.no_grad():
                torch_critic_observations = torch.as_tensor(
                    critic_observations["low_dim_state"],
                    dtype=torch.float32,
                    device=self.device,
                ).squeeze(-2)
                self.agent.compute_returns(torch_critic_observations)

            metrics.update(self._perform_updates())

            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_learned_reward": agent_0_prev_learned_reward,
                            "episode_length": agent_0_prev_ep_len
                            * self.cfg.action_repeat,
                        }
                    )
                self.logger.log_metrics(
                    metrics,
                    self.global_env_steps,
                    prefix="train"
                    if not (
                        self.use_rlhf
                        and self.unsup_update_steps
                        < self.cfg.rlhf.num_unsup_train_frames
                    )
                    else "unsup_train",
                )

            if should_eval(self.main_loop_iterations):
                eval_metrics = self._eval(eval_record_all_episode=True)
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            if self.use_rlhf:
                if (
                    self.cfg.rlhf.num_unsup_train_frames > 0
                    and self.global_env_steps - self.cfg.rlhf.num_pretrain_frames == 0
                    and not self.reward_model.activated
                ):
                    if hasattr(self.agent, "reset_critic"):
                        logging.info("Resetting critic after unsup train")
                        self.agent.reset_critic()
                    self._setup_training_functions()

                if (
                    self.total_feedback < self.cfg.rlhf.max_feedback
                    and should_update_reward_model(
                        self.main_loop_iterations
                        - max(
                            self.cfg.rlhf.num_pretrain_frames,
                            self.cfg.rlhf.num_unsup_train_frames,
                        )
                    )
                ):
                    self.reward_model.logging = True
                    logging.info(
                        f"[Feedback {self.total_feedback} / {self.cfg.rlhf.max_feedback}] Collecting feedback for {self.cfg.rlhf_replay.num_queries} queries"  # noqa
                    )
                    self.collect_feedback()

                    # reward model reset must be after feedback collection,
                    # as reward model is used for disagreement-based query selection
                    if self.cfg.rlhf.initialize_reward_model_per_session:
                        logging.info(
                            f"Resetting reward model in feedback session {self.feedback_iter}"
                        )
                        self.reward_model.build_reward_model()

                    for it in range(self.cfg.rlhf.num_train_frames):
                        reward_update_metrics = self._perform_reward_model_updates()
                        reward_update_metrics.update(
                            {
                                "iteration": self.global_env_steps + it,
                            }
                        )
                        _, total_time = self._timer.reset()
                        reward_update_metrics.update(
                            {
                                "total_time": total_time,
                                "iteration": self.global_env_steps + it,
                                "buffer_size": len(self.feedback_replay_buffer),
                            }
                        )
                        if should_reward_log(it):
                            self.logger.log_metrics(
                                reward_update_metrics,
                                self.global_env_steps + it,
                                prefix="train_reward",
                            )
                        if self.reward_model.early_stopping_criteria(
                            reward_update_metrics
                        ):
                            logging.info(
                                f"Reward model training finished after {it} steps with accuracy {reward_update_metrics['pref_acc_label_0'] * 100:.2f} %."  # noqa
                            )
                            break

                    # agent reset can be occurred in two cases.
                    # 1. initialize_agent_per_session is True
                    if self.cfg.rlhf.initialize_agent_per_session:
                        if hasattr(self.agent, "reset_critic"):
                            logging.info(
                                f"Resetting critic after feedback session {self.feedback_iter}"
                            )
                            self.agent.reset_critic()
                        if hasattr(self.agent, "reset_actor"):
                            logging.info(
                                f"Resetting actor after feedback session {self.feedback_iter}"
                            )
                            self.agent.reset_actor()
                        if hasattr(self.agent, "reset_temperature"):
                            logging.info(
                                f"Resetting temperature after feedback session {self.feedback_iter}"
                            )
                            self.agent.reset_temperature()
                        self._setup_training_functions()

                    if not self.reward_model.activated:
                        self.reward_model.set_activated(True)

                if (
                    self.total_feedback <= self.cfg.rlhf.max_feedback
                    and should_save_reward_model_snapshot(self.global_env_steps)
                ):
                    self.save_reward_model_snapshot()

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
        }
        return metrics

    def shutdown(self):
        logging.warning(f"Shutting down workspace at {self.global_env_steps} env steps")

        if hasattr(self, "_loop"):
            self._loop.close()

        if self.eval_env:
            self.eval_env.close()

        self.train_envs.close()

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
        payload["model_state_dict"] = self.agent.actor_critic.state_dict()
        payload["optimizer_state_dict"] = self.agent.optimizer.state_dict()
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        latest_snapshot = self.work_dir / "snapshots" / "latest_snapshot.pt"
        shutil.copy(snapshot, latest_snapshot)

    def load_snapshot(
        self, path_to_snapshot_to_load=None, load_optimizer=True, override_cfg=False
    ):
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
        self.agent.actor_critic.load_state_dict(payload.pop("model_state_dict"))
        if load_optimizer:
            self.agent.optimizer.load_state_dict(payload.pop("optimizer_state_dict"))
        for k, v in payload.items():
            if (
                k != "cfg" or not override_cfg
            ):  # cfg must not be loaded for efficient post-training.
                self.__dict__[k] = v

        logging.info(f"Loaded snapshot from env_step {self.global_env_steps}")

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
