from typing import Any, SupportsFloat, List

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
    # RecedingHorizonControl,
)
from omegaconf import DictConfig


from robobase.replay_buffer.rlhf.query_replay_buffer import QueryReplayBuffer

import copy

UNIT_TEST = False


class IsaacLab(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, task_name, num_envs, seed, render_mode=None):
        """Create the environment for the task."""
        from omni.isaac.lab.app import AppLauncher

        device = "cuda:0"
        app_launcher = AppLauncher(headless=True, device=device, enable_cameras=True)
        simulation_app = app_launcher.app  # noqa

        import omni.isaac.lab_tasks  # noqa: F401
        from omni.isaac.lab.envs import ManagerBasedEnvCfg
        from omni.isaac.lab_tasks.utils import parse_env_cfg

        env_cfg: ManagerBasedEnvCfg = parse_env_cfg(task_name)
        env_cfg.sim.device = device
        env_cfg.seed = seed
        env_cfg.scene.num_envs = num_envs
        self._env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
        self._num_envs = num_envs
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

    @property
    def max_episode_length(self):
        return self._env.unwrapped.max_episode_length

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def observation_space(self):
        return gym.spaces.Dict({"low_dim_state": self._env.observation_space["policy"]})

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def is_vector_env(self):
        return True

    @property
    def num_envs(self):
        return self._num_envs

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        new_obs, info = self._env.reset(**kwargs)
        return {"low_dim_state": new_obs["policy"]}, info

    @property
    def device(self):
        return self._env.device

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        new_obs, reward, terminated, truncated, info = self._env.step(action)
        return {"low_dim_state": new_obs["policy"]}, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()


class IsaacLabEnvFactory(EnvFactory):
    def _wrap_env(self, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        # assert cfg.demos > 0
        env = IsaacLab(
            task_name=cfg.env.task_name,
            num_envs=cfg.num_train_envs,
            seed=cfg.seed,
            render_mode=cfg.env.render_mode,
        )
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, env.max_episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack, lib="torch")

        env = AppendDemoInfo(env)
        if return_raw_spaces:
            return env, action_space, observation_space
        else:
            return env

    def make_train_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(cfg=cfg, demo_env=False, train=True)

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        raise NotImplementedError
        env, self._action_space, self._observation_space = self._wrap_env(
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
