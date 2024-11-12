from abc import ABC, abstractmethod
from typing import Iterator, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.method.utils import extract_from_spec


Metrics: TypeAlias = dict[str, np.ndarray]


class RewardMethod(nn.Module, ABC):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.num_eval_envs = 1  # Fixed to 1 for now
        self._eval_env_running = False
        self.logging = False

    @abstractmethod
    def compute_reward(
        self, observations: dict[str, torch.Tensor], actions: torch.Tensor, step: int
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        pass

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        pass

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value

    @property
    def time_obs_size(self) -> int:
        time_obs_spec = extract_from_spec(
            self.observation_space, "time", missing_ok=True
        )
        time_obs_size = 0
        if time_obs_spec is not None:
            time_obs_size = time_obs_spec.shape[1]
        return time_obs_size

    @property
    def low_dim_size(self) -> int:
        low_dim_state_spec = extract_from_spec(
            self.observation_space, "low_dim_state", missing_ok=True
        )
        low_dim_in_size = 0
        if low_dim_state_spec is not None:
            low_dim_in_size = low_dim_state_spec.shape[1] * low_dim_state_spec.shape[0]
        return low_dim_in_size

    def get_fully_connected_inputs(self):
        """Get input_sizes for FullyConnectedModules"""
        input_sizes = {}
        if self.rgb_latent_size > 0:
            input_sizes["fused_view_feats"] = (self.rgb_latent_size,)
        if self.low_dim_size > 0:
            input_sizes["low_dim_obs"] = (self.low_dim_size,)
        if self.time_obs_size > 0:
            input_sizes["time_obs"] = (self.time_obs_size,)
        if self.time_dim > 0:
            for k, v in input_sizes.items():
                input_sizes[k] = (self.time_dim,) + v
        return input_sizes
