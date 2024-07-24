from abc import ABC, abstractmethod
from typing import Iterator, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from robobase.replay_buffer.replay_buffer import ReplayBuffer


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
    def compute_reward(self, observations: dict[str, torch.Tensor], actions: torch.Tensor, step: int) -> torch.Tensor:
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
