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
from robobase.envs.utils.agym_utils import (
    TASK_DESCRIPTION,
    SUBTASK_LIST,
    GENERAL_CRITERIA,
)

import logging

logging.getLogger("pybullet").setLevel(logging.ERROR)


def _task_name_to_description(task_name: str) -> str:
    return TASK_DESCRIPTION.get(task_name, None)


def _task_name_to_subtask_list(task_name: str) -> str:
    return SUBTASK_LIST.get(task_name, None)


def _get_general_criteria() -> str:
    return GENERAL_CRITERIA


class AGym(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        task_name,
        action_repeat: int = 1,
        frame_skip: int = 2,
        render_mode: str = "rgb_array",
        query_keys: list[str] = ["right"],
        reward_mode: str = "dense",
    ):
        self._action_repeat = action_repeat
        self._viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert reward_mode in ["initial", "dense"]
        self._i = 0
        self._frame_skip = frame_skip
        self._prev_image = None
        self._render_mode = render_mode
        self._reward_mode = reward_mode

        print(f"Creating AGym environment with task name: {task_name}")
        self.__agym_env = gym_old.make(task_name)
        self._agym_env = gym.wrappers.EnvCompatibility(self.__agym_env, render_mode)

        self._query_keys = query_keys
        obs_dict = {}
        obs_dict["low_dim_state"] = spaces.Box(
            low=self._agym_env.observation_space.low,
            high=self._agym_env.observation_space.high,
            dtype=np.float32,
        )
        for key in self._query_keys:
            obs_dict[f"query_video_{key}"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.__agym_env.height, self.__agym_env.width, 3),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(obs_dict)

        self.action_space = self._agym_env.action_space
        self.reward_space = self._agym_env.env.reward_space
        self.initial_reward_scale = 1.0

    def agym_env(self):
        return self.__agym_env

    def _get_obs(self, observation, image):
        return {
            "low_dim_state": observation.astype(np.float32),
            **{f"query_video_{key}": image[key] for key in self._query_keys},
        }

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
            agym_obs, task_reward, terminated, truncated, info = self._agym_env.step(
                action
            )
            info["task_reward"] = task_reward
            if self._reward_mode == "initial":
                _reward = np.sum(
                    [
                        self.initial_reward_scale * info[key]
                        for key in self.reward_space.keys()
                    ]
                )
            else:
                _reward = task_reward
            if self._render_mode is None:
                images = {
                    key: np.zeros_like(
                        self.observation_space.sample()[f"query_video_{key}"],
                        dtype=np.uint8,
                    )
                    for key in self._query_keys
                }
            else:
                images = {key: self._render(key) for key in self._query_keys}
            reward += _reward
            if terminated or truncated:
                break
        self._i += 1
        return self._get_obs(agym_obs, images), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        agym_obs, info = self._agym_env.reset(seed=seed, options=options)
        if self._render_mode is None:
            images = {
                key: np.zeros_like(
                    self.observation_space.sample()[f"query_video_{key}"],
                    dtype=np.uint8,
                )
                for key in self._query_keys
            }
        else:
            images = {key: self._render(key) for key in self._query_keys}
        info.update({key: 0.0 for key in self.reward_space.keys()})
        info.update({"task_reward": 0.0})
        return self._get_obs(agym_obs, images), info

    def render(self, view: str = "right") -> None:
        return self._render(self._query_keys[0])

    def _render(self, view: str = "right") -> None:
        """Render the environment.

        Args:
            view (str, optional): Camera view to render from.
        """
        if view not in ["front", "right", "top"]:
            raise ValueError(
                f'view must be one of ["front", "right", "top"], got {view}'
            )
        if self._i % self._frame_skip == 0:
            img, depth = self._agym_env.env.get_camera_image_depth(view=view)
            img = img[:, :, :3]
            self._prev_image = img

        if self._render_mode == "rgb_array":
            return self._prev_image.astype(np.uint8)
        elif self._render_mode == "human":
            from PIL import Image

            return Image.fromarray(self._prev_image)
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
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    AGym(
                        task_name=cfg.env.task_name,
                        action_repeat=cfg.action_repeat,
                        frame_skip=cfg.env.frame_skip,
                        query_keys=cfg.env.query_keys,
                        render_mode="rgb_array" if cfg.rlhf.use_rlhf else None,
                        reward_mode=cfg.env.reward_mode,
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
                task_name=cfg.env.task_name,
                action_repeat=cfg.action_repeat,
                frame_skip=cfg.env.frame_skip,
                query_keys=cfg.env.query_keys,
                render_mode="rgb_array",  # always render for evaluation
                reward_mode=cfg.env.reward_mode,
            ),
            cfg,
        )

    def get_task_description(self, cfg: DictConfig) -> str:
        return _task_name_to_description(cfg.env.task_name)

    def get_subtask_list(self, cfg: DictConfig) -> str:
        return _task_name_to_subtask_list(cfg.env.task_name)

    def get_general_criteria(self, cfg: DictConfig) -> str:
        return _get_general_criteria()
