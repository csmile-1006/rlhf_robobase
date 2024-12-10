import numpy as np

import gymnasium as gym
from gymnasium import spaces

from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

import loco_mujoco  # noqa

from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
)
from robobase.envs.utils.locomujoco_utils import (
    TASK_DESCRIPTION,
)

import logging


def _task_name_to_description(task_name: str) -> str:
    return TASK_DESCRIPTION.get(task_name, None)


class LocoMujoco(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        task_name,
        from_pixels: bool = False,
        action_repeat: int = 1,
        visual_observation_shape: tuple[int, int] = (84, 84),
        render_mode: str = "rgb_array",
        use_rlhf: bool = False,
        query_keys: list[str] = ["right"],
        reward_mode: str = "dense",
        reward_term_type: str = "all",
        initial_terms: list[float] = [],
    ):
        self._task_name = task_name
        self._from_pixels = from_pixels
        self._action_repeat = action_repeat
        self._viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert reward_mode in ["initial", "dense"]
        self._i = 0
        self._render_mode = render_mode
        self._reward_mode = reward_mode
        self._reward_term_type = reward_term_type
        self._initial_terms = initial_terms
        self._pixels_key = "pixels"
        self._query_keys = query_keys
        assert all(
            key == self._pixels_key for key in query_keys
        ), f"Only {self._pixels_key} view is supported"

        self._locomujoco_env = None

        # Set up rendering parameters
        height, width = visual_observation_shape
        logging.info(
            f"Creating LocoMujoco environment with task name: {self._task_name}"
        )

        self._locomujoco_env = gym.make(
            "LocoMujoco",
            env_name=self._task_name,
            width=width,
            height=height,
            render_mode=render_mode,
        )

        # Set up observation space
        _obs_space = {}
        if from_pixels:
            image_shape = [3, height, width]  # (channels, height, width)
            # For pixel observations, use RGB image space
            rgb_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            _obs_space["rgb"] = rgb_space
        else:
            _obs_space["low_dim_state"] = spaces.Box(
                low=self._locomujoco_env.observation_space.low,
                high=self._locomujoco_env.observation_space.high,
                dtype=np.float32,
            )

        self._use_rlhf = use_rlhf
        if use_rlhf:
            for key in self._query_keys:
                _obs_space[f"query_video_{key}"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                )
        self.observation_space = spaces.Dict(_obs_space)
        self.action_space = self._locomujoco_env.action_space
        self.reward_space = gym.spaces.Dict(
            {
                "target_velocity": gym.spaces.Box(
                    low=1e-10, high=1e1, shape=(), dtype=np.float32
                ),
            }
        )

        if len(self._initial_terms) == 0:
            self._initial_terms = [key for key in self.reward_space.keys()]
        else:
            self._initial_terms = [f"Reward/{key}" for key in self._initial_terms]

        if self._reward_term_type == "all":
            self._reward_terms = [key for key in self.reward_space.keys()]
        elif self._reward_term_type == "initial":
            self._reward_terms = self._initial_terms
        else:
            raise ValueError(
                f"reward_term_type must be one of ['all', 'initial'], got {self._reward_term_type}"
            )

        self.reward_space = spaces.Dict(
            {
                k: gym.spaces.Box(
                    low=self.reward_space[k].low,
                    high=self.reward_space[k].high,
                    shape=self.reward_space[k].shape,
                )
                for k in self._reward_terms
            }
        )

        self.initial_reward_scale = {
            k: self.reward_space[k].high for k in self._initial_terms
        }

    def _get_obs(self, observation):
        ret_obs = {}
        if self._from_pixels:
            ret_obs["rgb"] = self._locomujoco_env.render().transpose([2, 0, 1]).copy()
        else:
            ret_obs["low_dim_state"] = observation.astype(np.float32)

        if self._use_rlhf:
            ret_obs[f"query_video_{self._query_keys[0]}"] = self.render().copy()
        return ret_obs

    def step(self, action):
        reward = 0
        info = {"task_reward": 0.0, **{f"Reward/{k}": 0.0 for k in self._reward_terms}}
        for _ in range(self._action_repeat):
            (
                next_obs,
                task_reward,
                terminated,
                truncated,
                _info,
            ) = self._locomujoco_env.step(action)
            # target velocity is the reward
            _info["target_velocity"] = task_reward
            info["task_reward"] += task_reward
            if self._reward_mode == "initial":
                _reward = np.sum(
                    [
                        self.initial_reward_scale[key] * _info[key]
                        for key in self._initial_terms
                    ]
                )
            else:
                _reward = task_reward

            reward += _reward
            for key in self._reward_terms:
                info[f"Reward/{key}"] += _info[key]
            if terminated or truncated:
                break
        # See https://github.com/google-deepmind/dm_control/blob/f2f0e2333d8bd82c0b6ba83628fe44c2bcc94ef5/dm_control/rl/control.py#L115C18-L115C29
        truncated = truncated and not terminated
        assert not np.any(
            terminated and truncated
        ), "Can't be both terminal and truncated."
        self._i += 1
        return self._get_obs(next_obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        agym_obs, info = self._locomujoco_env.reset(seed=seed)
        info.update({key: 0.0 for key in self.reward_space.keys()})
        info.update({"task_reward": 0.0})
        return self._get_obs(agym_obs), info

    def render(self) -> None:
        image = self._locomujoco_env.render()
        return image


class LocoMujocoEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.env.episode_length != 1000:
            # Used in unit tests.
            env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(
                env, cfg.env.episode_length // cfg.action_repeat
            )  # Time limits are handles by DMC
        env = ActionSequence(env, cfg.action_sequence)
        env = FrameStack(env, cfg.frame_stack)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context=None)
        # vec_env_class = gym.vector.SyncVectorEnv
        # kwargs = dict()
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    LocoMujoco(
                        task_name=cfg.env.task_name,
                        from_pixels=cfg.pixels,
                        action_repeat=cfg.action_repeat,
                        visual_observation_shape=cfg.visual_observation_shape,
                        render_mode="rgb_array",
                        use_rlhf=cfg.rlhf.use_rlhf,
                        query_keys=cfg.env.query_keys,
                        reward_mode=cfg.env.reward_mode,
                        reward_term_type=cfg.env.reward_term_type,
                        initial_terms=cfg.env.initial_terms,
                    ),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            **kwargs,
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(
            LocoMujoco(
                task_name=cfg.env.task_name,
                from_pixels=cfg.pixels,
                action_repeat=cfg.action_repeat,
                visual_observation_shape=cfg.visual_observation_shape,
                render_mode="rgb_array",
                use_rlhf=cfg.rlhf.use_rlhf,
                query_keys=cfg.env.query_keys,
                reward_mode=cfg.env.reward_mode,
                reward_term_type=cfg.env.reward_term_type,
                initial_terms=cfg.env.initial_terms,
            ),
            cfg,
        )

    def get_task_description(self, cfg: DictConfig) -> str:
        return _task_name_to_description(cfg.env.task_name)
