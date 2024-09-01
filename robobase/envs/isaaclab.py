from robobase.utils import (
    DemoEnv,
    add_demo_to_replay_buffer,
    add_demo_to_query_replay_buffer,
    convert_demo_to_episode_rollouts,
)
import gymnasium as gym
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    # RescaleFromTanhWithMinMax,
    # RescaleFromTanh,
    OnehotTime,
    # ActionSequence,
    AppendDemoInfo,
    FrameStack,
    ConcatDim,
    # RecedingHorizonControl,
)
from omegaconf import DictConfig


from robobase.replay_buffer.rlhf.query_replay_buffer import QueryReplayBuffer

from typing import List
import copy

UNIT_TEST = False


class IsaacLabEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        # assert cfg.demos > 0
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        obs_stats = None
        # if cfg.demos > 0:
        #     env = RescaleFromTanhWithMinMax(
        #         env=env,
        #         action_stats=self._action_stats,
        #         min_max_margin=cfg.min_max_margin,
        #     )
        #     if cfg.norm_obs:
        #         obs_stats = self._obs_stats
        # else:
        #     assert cfg.norm_obs is False, "Need to provide demos to normalize obs"
        #     env = RescaleFromTanh(env=env)

        # We normalize the low dimensional observations in the ConcatDim wrapper.
        # This is to be consistent with the original ACT implementation.
        env = ConcatDim(
            env,
            shape_length=1,
            dim=-1,
            new_name="low_dim_state",
            norm_obs=cfg.norm_obs,
            obs_stats=obs_stats,
            keys_to_ignore=["proprioception_floating_base_actions"],
            lib="torch",
        )
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, env.max_episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack, lib="torch")

        # if not demo_env:
        #     if not train:
        #         env = RecedingHorizonControl(
        #             env,
        #             cfg.action_sequence,
        #             env.max_episode_length // (cfg.env.demo_down_sample_rate),
        #             cfg.execution_length,
        #             temporal_ensemble=cfg.temporal_ensemble,
        #             gain=cfg.temporal_ensemble_gain,
        #         )
        #     else:
        #         env = ActionSequence(
        #             env,
        #             cfg.action_sequence,
        #         )

        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, action_space, observation_space
        else:
            return env

    def make_train_env(self, env: gym.Env, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(env=env, cfg=cfg, demo_env=False, train=True)

    def make_eval_env(self, env: gym.Env, cfg: DictConfig) -> gym.Env:
        raise NotImplementedError
        env, self._action_space, self._observation_space = self._wrap_env(
            env=env,
            cfg=cfg,
            demo_env=False,
            train=False,
            return_raw_spaces=True,
        )
        return env

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int, mp_list: List) -> None:
        raise NotImplementedError

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        raise NotImplementedError
        # manager = mp.Manager()
        # mp_list = manager.list()

        # p = mp.Process(
        #     target=self._get_demo_fn,
        #     args=(cfg, num_demos, mp_list),
        # )
        # p.start()
        # p.join()

        # demos = mp_list[0]
        demos = self._get_demo_fn(cfg, num_demos, None)

        self._raw_demos = demos
        self._action_stats = self._compute_action_stats(cfg, demos)
        self._obs_stats = self._compute_obs_stats(cfg, demos)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        raise NotImplementedError

    def load_demos_into_replay(
        self, cfg: DictConfig, buffer, target_indices: list[int] = None
    ):
        raise NotImplementedError
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        if target_indices:
            demos = [self._demos[i] for i in target_indices]
        else:
            demos = self._demos
        demo_env = self._wrap_env(
            DemoEnv(copy.deepcopy(demos), self._action_space, self._observation_space),
            cfg,
            demo_env=True,
            train=False,
        )
        add_demo_fn = (
            add_demo_to_query_replay_buffer
            if isinstance(buffer, QueryReplayBuffer)
            else add_demo_to_replay_buffer
        )
        for _ in range(len(demos)):
            add_demo_fn(demo_env, buffer)

    def load_demos_into_rollouts(self, cfg: DictConfig):
        raise NotImplementedError

        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos), self._action_space, self._observation_space
            ),
            cfg,
            demo_env=True,
            train=False,
        )
        demos = []
        for _ in range(len(self._demos)):
            demos.append(convert_demo_to_episode_rollouts(demo_env))
        return demos
