import numpy as np

import gym as gym_old
import gymnasium as gym
from gymnasium import spaces

from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

import assistive_gym  # noqa

from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
)


class AGym(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        task_name,
        action_repeat: int = 1,
        render_mode: str = None,
    ):
        self._action_repeat = action_repeat
        self._viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        self._agym_env = gym_old.make(task_name)
        self._agym_env = gym.wrappers.EnvCompatibility(self._agym_env)

        self.observation_space = spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    low=self._agym_env.observation_space.low,
                    high=self._agym_env.observation_space.high,
                    dtype=np.float32,
                )
            }
        )
        self.action_space = self._agym_env.action_space

    def _get_obs(self, observation):
        return {"low_dim_state": observation.astype(np.float32)}

    def _flatten_obs(self, observation):
        obs_pieces = []
        for v in observation.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=0).astype(np.float32)
        return obs

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            agym_obs, reward, terminated, truncated, info = self._agym_env.step(action)
            reward += reward
            if terminated or truncated:
                break
        return self._get_obs(agym_obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        agym_obs, info = self._agym_env.reset(seed=seed, options=options)
        return self._get_obs(agym_obs), info

    def render(self):
        img, depth = self._agym_env.env.get_camera_image_depth()

        if self._render_mode == "rgb_array":
            return img.astype(np.uint8)
        elif self._render_mode == "human":
            from PIL import Image

            return Image.fromarray(img)
        else:
            raise NotImplementedError(f"`{self._render_mode}` mode is not implemented")

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return self._agym_env.close()


class AGymEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.env.episode_length != 200:
            # Used in unit tests.
            env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length // cfg.action_repeat)  # Time limits are handles by DMC
        env = ActionSequence(env, cfg.action_sequence)
        env = FrameStack(env, cfg.frame_stack)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context="spawn")
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    AGym(
                        cfg.env.task_name,
                        cfg.action_repeat,
                        "rgb_array",
                    ),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            **kwargs,
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(
            AGym(
                cfg.env.task_name,
                cfg.action_repeat,
                "rgb_array",
            ),
            cfg,
        )
