from copy import deepcopy
from typing import Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from robobase.method.value_based import ValueBased

from robobase import utils

from robobase.method.utils import (
    random_action_if_within_delta,
    zoom_in,
    loss_weights,
    encode_action,
    decode_action,
)


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        low_dim: int,
        action_shape: Tuple,
        hidden_dim: int,
        levels: int,
        bins: int,
    ):
        super().__init__()
        self._levels = levels
        self._actor_dim = action_shape[0]
        self._bins = bins

        self.net = nn.Sequential(
            nn.Linear(low_dim + self._actor_dim + levels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.head = nn.Linear(hidden_dim, self._actor_dim * bins)
        self.output_shape = (self._actor_dim, bins)

        self.apply(utils.weight_init)
        self.head.weight.data.fill_(0.0)
        self.head.bias.data.fill_(0.0)

    def forward_each_level(
        self, level: int, low_dim_obs: torch.Tensor, prev_action: torch.Tensor
    ):
        """
        Implementation to compute Q-values at each level.

        Inputs:
        - level: level index (integer, not one-hot)
        - low_dim_obs: low-dimensional observations
        - prev_actions: actions from *all* previous levels

        Outputs:
        - q_values: (batch_size, actor_dim, bins)
        """
        level_id = (
            torch.eye(self._levels, device=low_dim_obs.device, dtype=low_dim_obs.dtype)[
                level
            ]
            .unsqueeze(0)
            .repeat_interleave(low_dim_obs.shape[0], 0)
        )

        adv_x = torch.cat([low_dim_obs, prev_action, level_id], -1)
        q_values = self.head(self.net(adv_x)).view(-1, *self.output_shape)
        return q_values

    def forward(self, obs: torch.Tensor, prev_actions: torch.Tensor):
        """
        Optimized implementation to compute Q-values at all levels in parallel.
        See `forward_each_level` for implementation that processes each level.

        Inputs:
        - low_dim_obs: low-dimensional observations
        - prev_actions: actions from *all* previous levels

        Outputs:
        - q_values: (batch_size, level, actor_dim, bins)
        """
        device, dtype = obs.device, obs.dtype
        B, L = prev_actions.shape[:2]
        # Reshape previous actions
        prev_actions = prev_actions.view(B, L, self._actor_dim)  # [B, L, T, D]

        # level id - [L, L] -> [B, L, L]
        level_id = torch.eye(L, device=device, dtype=dtype)[None, :, :].repeat(B, 1, 1)

        obs = obs[:, None, :].repeat(1, L, 1)
        adv_x = torch.cat([obs, prev_actions, level_id], -1)
        q_values = self.head(self.net(adv_x)).view(B, L, *self.output_shape)
        return q_values


class C2FCriticSimple(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        low_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        actor_dim = action_shape[0]
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.network = C2FCriticNetwork(low_dim, action_shape, hidden_dim, levels, bins)

    def get_action(self, obs: torch.Tensor):
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            qs = self.network.forward_each_level(level, obs, (low + high) / 2)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)
        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        qs_per_level = []
        qs_a_per_level = []

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        # Pre-compute previous actions for all the levels
        prev_actions = []
        for level in range(self.levels):
            prev_actions.append((low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]
            low, high = zoom_in(low, high, argmax_q, self.bins)

        qs_all = self.network(obs, torch.stack(prev_actions, 1))
        for level in range(self.levels):
            qs = qs_all[:, level]
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # qs: [B, D, bins]
            # qs_a: [B, D]
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[..., 0]

            qs_per_level.append(qs)
            qs_a_per_level.append(qs_a)

        qs = torch.stack(qs_per_level, -3)
        qs_a = torch.stack(qs_a_per_level, -2)
        return qs, qs_a

    def encode_decode_action(self, continuous_action: torch.Tensor):
        """Encode and decode actions"""
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        return continuous_action


class CQNSimple(ValueBased):
    def __init__(
        self,
        levels: int,
        critic_lambda: float,
        centralized_critic: bool,
        critic_target_interval: int,
        *args,
        **kwargs,
    ):
        self.levels = levels
        self.critic_lambda = critic_lambda
        self.centralized_critic = centralized_critic
        self.critic_target_interval = critic_target_interval
        super().__init__(*args, **kwargs)

    def build_critic(self):
        critic_cls = C2FCriticSimple
        actor_dim = np.prod(self.action_space.shape)
        input_shapes = self.get_fully_connected_inputs()
        input_shapes["level"] = (self.levels,)
        input_shapes["low_high"] = (actor_dim,)

        critic = critic_cls(
            action_shape=(actor_dim,),
            low_dim=input_shapes["low_dim_obs"][0],
            hidden_dim=512,
            levels=self.levels,
            bins=self.bins,
        ).to(self.device)
        critic_target = deepcopy(critic)
        critic_target.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.AdamW(
            critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay
        )
        critic_target.eval()
        return critic, critic_target, critic_opt

    def extract_batch(
        self, replay_iter: Iterator[dict[str, torch.Tensor]]
    ) -> tuple[dict, TensorDict]:
        batch = next(replay_iter)
        batch = TensorDict(
            {
                k: torch.as_tensor(v, dtype=v.dtype, device=self.device)
                for k, v in batch.items()
            },
            batch_size=batch["action"].shape[0],
        )
        batch["reward"] = batch["reward"].unsqueeze(1)
        batch["discount"] = batch["discount"].to(batch["reward"].dtype).unsqueeze(1)
        batch["terminal"] = batch["terminal"].to(batch["reward"].dtype)
        batch["truncated"] = batch["truncated"].to(batch["reward"].dtype)
        # 1. If not terminal and not truncated, we bootstrap
        # 2. If not terminal and truncated, we bootstrap
        # 3. If terminal and not truncated, we don't bootstrap
        # 4. If terminal and truncated,(e.g., success in last timestep)
        #    we don't bootstrap as terminal has a priortiy over truncated
        # In summary, we do not bootstrap when terminal; otherwise we do bootstrap
        batch["bootstrap"] = (1.0 - batch["terminal"]).unsqueeze(1)
        if self.always_bootstrap:
            # Override bootstrap to be 1
            batch["bootstrap"] = torch.ones_like(batch["bootstrap"])
        batch["loss_coeff"] = loss_weights(batch, self.replay_beta)
        batch["action"] = batch["action"].flatten(-2)
        return batch

    def update_critic(
        self,
        low_dim_obs,
        action,
        reward,
        discount,
        bootstrap,
        next_low_dim_obs,
        next_action,
        loss_coeff,
    ):
        critic, critic_opt = (
            self.critic,
            self.critic_opt,
        )

        with torch.no_grad():
            target_v = self.critic_target(
                next_low_dim_obs,
                next_action,
            )[1]
            target_q = (
                reward.unsqueeze(-1)
                + bootstrap.unsqueeze(-1) * discount.unsqueeze(-1) * target_v
            )

        qs_a = critic(
            low_dim_obs,
            action,
        )[1]

        q_critic_loss = F.mse_loss(qs_a, target_q)
        critic_loss = self.critic_lambda * (q_critic_loss * loss_coeff).mean()
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        return TensorDict(critic_loss=critic_loss.detach())

    def update(
        self,
        batch: TensorDict,
    ) -> dict[str, np.ndarray]:
        low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)

        with torch.no_grad():
            # NOTE: Pre-compute next_action here, outside update_critic to support
            # using the same next_action for both critic/intr_critic updates
            next_action = self.critic.get_action(next_low_dim_obs)

        metrics = self.update_critic(
            low_dim_obs,
            batch["action"],
            batch["reward"],
            batch["discount"],
            batch["bootstrap"],
            next_low_dim_obs,
            next_action,
            batch["loss_coeff"],
        )

        if self.logging:
            metrics["batch_reward"] = batch["reward"].detach().mean()

        return metrics

    def update_target_critic(self, step: int):
        # update critic target
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )

    def act(
        self,
        observations: dict[str, torch.Tensor],
        step: int,
        eval_mode: bool,
    ):
        low_dim_obs = self._act_extract_low_dim_state(observations)
        critic = self.critic
        action = critic.get_action(low_dim_obs)
        std = torch.ones_like(action) * self.get_std(step)
        dist = utils.TruncatedNormal(action, std)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_explore_steps:
                action.uniform_(-1, 1)
        action = self.critic.encode_decode_action(action)
        action = action.view(*action.shape[:-1], *self.action_space.shape)
        return action
