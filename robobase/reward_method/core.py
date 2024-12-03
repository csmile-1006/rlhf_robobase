from abc import ABC, abstractmethod
from typing import Dict, Iterator, Sequence, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F

from robobase.method.utils import extract_from_spec
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
        self._activated = False

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

    @property
    def activated(self) -> bool:
        return self._activated

    def set_activated(self, value: bool):
        self._activated = value

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

    def initialize_reward_model(self):
        self.build_reward_model()

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.compute_reward(x_1, member=member, return_reward=True).sum(
                axis=1
            )
            r_hat2 = self.compute_reward(x_2, member=member, return_reward=True).sum(
                axis=1
            )
            r_hat = torch.stack([r_hat1, r_hat2], axis=-1)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def get_rank_probability(
        self,
        x_1: Sequence[Dict[str, torch.Tensor]],
        x_2: Sequence[Dict[str, torch.Tensor]],
    ) -> tuple[np.ndarray, np.ndarray]:
        probs = []
        for member in range(self.num_reward_models):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.asarray(probs)
        return probs.mean(axis=0), probs.std(axis=0)
