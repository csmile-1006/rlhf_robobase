from copy import deepcopy
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from robobase.method.value_based import ValueBased

from robobase import utils

from robobase.method.utils import (
    loss_weights,
)
from robobase.method.cqn_simple import C2FCriticSimple


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
        (
            self.extr_critic,
            self.extr_critic_target,
            self.extr_critic_opt,
        ) = self.build_critic()

    def reset_critic(self):
        self.critic, self.critic_target, self.critic_opt = self.build_critic()
        (
            self.extr_critic,
            self.extr_critic_target,
            self.extr_critic_opt,
        ) = self.build_critic()

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
        loss_coeff,
    ):
        # first update critic
        critic, critic_target, critic_opt = (
            self.critic,
            self.critic_target,
            self.critic_opt,
        )
        with torch.no_grad():
            next_action = critic.get_action(next_low_dim_obs)
            target_v = critic_target(
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

    def update_extr_critic(
        self,
        low_dim_obs,
        action,
        reward,
        discount,
        bootstrap,
        next_low_dim_obs,
        loss_coeff,
    ):
        critic, critic_target, critic_opt = (
            self.extr_critic,
            self.extr_critic_target,
            self.extr_critic_opt,
        )
        with torch.no_grad():
            next_action = critic.get_action(next_low_dim_obs)
            target_v = critic_target(
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
        extr_critic_loss = self.critic_lambda * (q_critic_loss * loss_coeff).mean()
        critic_opt.zero_grad(set_to_none=True)
        extr_critic_loss.backward()
        critic_opt.step()

        return TensorDict(extr_critic_loss=extr_critic_loss.detach())

    def update(
        self,
        batch: TensorDict,
    ) -> dict[str, np.ndarray]:
        low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)
        assert self.intrinsic_reward_module is not None
        self.intrinsic_reward_module.update(batch)

        with torch.no_grad():
            # NOTE: Pre-compute next_action here, outside update_critic to support
            # using the same next_action for both critic/intr_critic updates
            intr_action = self.extr_critic.get_action(low_dim_obs)
            Q = (
                self.extr_critic(low_dim_obs, intr_action)[1]
                .mean(dim=[-2, -1])
                .reshape(-1, 1)
            )
            Q = F.layer_norm(Q, normalized_shape=(1,))
            intrinsic_rewards = self.intrinsic_reward_module.compute_irs(batch, Q)

        metrics = self.update_critic(
            low_dim_obs,
            batch["action"],
            batch["reward"] + intrinsic_rewards,
            batch["discount"],
            batch["bootstrap"],
            next_low_dim_obs,
            batch["loss_coeff"],
        )
        metrics.update(
            self.update_extr_critic(
                low_dim_obs,
                batch["action"],
                batch["reward"],
                batch["discount"],
                batch["bootstrap"],
                next_low_dim_obs,
                batch["loss_coeff"],
            )
        )
        metrics["batch_reward"] = batch["reward"].mean().detach()
        metrics["batch_intrinsic_rewards"] = intrinsic_rewards.mean().detach()
        return metrics

    def update_unsupervised(self, batch: TensorDict):
        low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)
        assert self.intrinsic_reward_module is not None
        self.intrinsic_reward_module.update(batch)

        with torch.no_grad():
            # NOTE: Pre-compute next_action here, outside update_critic to support
            # using the same next_action for both critic/intr_critic updates
            next_action = self.critic.get_action(next_low_dim_obs)
            intrinsic_rewards = self.intrinsic_reward_module.compute_unsup_irs(batch)

        metrics = self.update_critic(
            low_dim_obs,
            batch["action"],
            intrinsic_rewards,
            batch["discount"],
            batch["bootstrap"],
            next_low_dim_obs,
            next_action,
            batch["loss_coeff"],
        )
        metrics["unsup_critic_loss"] = metrics["critic_loss"]
        metrics["batch_intrinsic_rewards"] = intrinsic_rewards.mean().detach()
        return metrics

    def update_target_critic(self, step: int):
        # update critic target
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
            utils.soft_update_params(
                self.extr_critic, self.extr_critic_target, self.critic_target_tau
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
