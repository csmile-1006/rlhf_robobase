from copy import deepcopy
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robobase.method.value_based import ValueBased

from robobase import utils
from robobase.models.fully_connected import FullyConnectedModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.method.utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
)


class C2FCriticSimple(nn.Module):
    def __init__(
        self,
        actor_dim: int,
        levels: int,
        bins: int,
        advantage_model: FullyConnectedModule,
        value_model: Optional[FullyConnectedModule] = None,
    ):
        super().__init__()
        self.adv = advantage_model
        self.value = value_model
        self.use_dueling = value_model is not None
        self.levels = levels
        self.bins = bins

        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )

        self.apply(utils.weight_init)
        self.adv.initialize_output_layer(utils.uniform_weight_init(0.0))
        if self.use_dueling:
            self.value.initialize_output_layer(utils.uniform_weight_init(0.0))

    def forward(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        time_obs,
    ):
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
            bs = low_dim_obs.shape[0]
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
            bs = fused_view_feats.shape[0]
        if time_obs is not None:
            net_ins["time_obs"] = time_obs
            bs = time_obs.shape[0]

        low = self.initial_low.repeat(bs, 1).detach()
        high = self.initial_high.repeat(bs, 1).detach()
        discrete_action = self.encode_action(action)

        qs_per_level = []
        qs_a_per_level = []
        for level in range(self.levels):
            net_ins["level"] = (
                torch.eye(self.levels, device=low.device, dtype=low.dtype)[level]
                .unsqueeze(0)
                .repeat_interleave(bs, 0)
            )
            net_ins["low_high"] = (low + high) / 2.0
            if self.use_dueling:
                advs = self.adv(net_ins)
                values = self.value(net_ins)
                qs = values + advs - advs.mean(-2, keepdim=True)
            else:
                qs = self.adv(net_ins)

            argmax_q = discrete_action[..., level, :].long()
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[..., 0]

            qs_per_level.append(qs)
            qs_a_per_level.append(qs_a)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        qs = torch.stack(qs_per_level, -3)  # [B, L, D, bins]
        qs_a = torch.stack(qs_a_per_level, -2)  # [B, L, D]
        return qs, qs_a

    def get_action(
        self,
        low_dim_obs,
        fused_view_feats,
        time_obs,
        intr_critic=None,
        return_metrics=False,
        logging=False,
    ):
        metrics = dict()
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
            bs = low_dim_obs.shape[0]
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
            bs = fused_view_feats.shape[0]
        if time_obs is not None:
            net_ins["time_obs"] = time_obs
            bs = time_obs.shape[0]

        low = self.initial_low.repeat(bs, 1).detach()
        high = self.initial_high.repeat(bs, 1).detach()

        for level in range(self.levels):
            net_ins["level"] = (
                torch.eye(self.levels, device=low.device, dtype=low.dtype)[level]
                .unsqueeze(0)
                .repeat_interleave(bs, 0)
            )
            net_ins["low_high"] = (low + high) / 2.0
            if self.use_dueling:
                advs = self.adv(net_ins)
                values = self.value(net_ins)
                qs = values + advs - advs.mean(-2, keepdim=True)
            else:
                qs = self.adv(net_ins)

            if intr_critic is not None:
                if self.use_dueling:
                    intr_advs = intr_critic.adv(net_ins)
                    intr_values = intr_critic.value(net_ins)
                    intr_qs = intr_values + intr_advs - intr_advs.mean(-2, keepdim=True)
                else:
                    intr_qs = intr_critic.adv(net_ins)
                qs += intr_qs

            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]

            if return_metrics:
                # For logging
                qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[
                    ..., 0
                ]  # [..., D]
                if logging:
                    metrics[f"critic_target_q_level{level}"] = qs_a.mean().item()

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)
        continuous_action = (high + low) / 2.0
        if return_metrics:
            return continuous_action, metrics
        else:
            return continuous_action

    def encode_action(self, continuous_action):
        return encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

    def decode_action(self, discrete_action):
        return decode_action(
            discrete_action, self.initial_low, self.initial_high, self.levels, self.bins
        )

    def encode_decode_action(self, action: torch.Tensor):
        return self.decode_action(self.encode_action(action))


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

        advantage_model = self.advantage_model(
            input_shapes=input_shapes,
            output_shape=(actor_dim, self.bins),
            num_envs=self.num_train_envs + 1,
        )
        if self.use_dueling:
            value_model = self.value_model(
                input_shapes=input_shapes,
                output_shape=(actor_dim, 1),
                num_envs=self.num_train_envs + 1,
            )
        else:
            value_model = None
        critic = critic_cls(
            actor_dim,
            self.levels,
            self.bins,
            advantage_model,
            value_model,
        ).to(self.device)
        critic_target = deepcopy(critic)
        critic_target.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.AdamW(
            critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay
        )
        critic_target.eval()
        if self.use_torch_compile:
            critic = torch.compile(critic)
            critic_target = torch.compile(critic_target)
        return critic, critic_target, critic_opt

    def update_critic(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        reward,
        discount,
        bootstrap,
        next_low_dim_obs,
        next_fused_view_feats,
        next_action,
        time_obs,
        next_time_obs,
        loss_coeff,
        demos,
        updating_intrinsic_critic,
        logging,
    ):
        lp = "intrinsic_" if updating_intrinsic_critic else ""
        if updating_intrinsic_critic:
            critic, critic_opt = (
                self.intr_critic,
                self.intr_critic_opt,
            )
        else:
            critic, critic_opt = (
                self.critic,
                self.critic_opt,
            )

        metrics = dict()
        with torch.no_grad():
            _, target_v = self.critic_target(
                next_low_dim_obs,
                next_fused_view_feats,
                next_action,
                next_time_obs,
            )
            target_q = (
                reward.unsqueeze(-1)
                + bootstrap.unsqueeze(-1) * discount.unsqueeze(-1) * target_v
            )

        qs, qs_a = critic(
            low_dim_obs,
            fused_view_feats,
            action,
            time_obs,
        )

        # q_critic_loss = F.mse_loss(qs_a, target_q, reduction="none").mean([1, 2])
        q_critic_loss = F.mse_loss(qs_a, target_q)
        critic_loss = self.critic_lambda * (q_critic_loss * loss_coeff).mean()
        if logging:
            metrics["q_critic_loss"] = q_critic_loss.mean().item()
            metrics["loss_coeff"] = loss_coeff.mean().item()

        if self.bc_lambda > 0.0 and demos is not None:
            qs = None
            demos = demos.float()
            if logging:
                metrics["ratio_of_demos"] = demos.mean().item()
            if torch.sum(demos) > 0:
                # qs: [B, L, D, bins], qs_a: [B, L, D]
                qs_cdf = torch.cumsum(qs, -1)
                qs_a_cdf = torch.cumsum(qs_a, -1)
                # qs_{a_{i}} is stochastically dominant over qs_{a_{-i}}
                bc_fosd_loss = (
                    (qs_a_cdf.unsqueeze(-2) - qs_cdf)
                    .clamp(min=0)
                    .sum(-1)
                    .mean([-1, -2, -3])
                )
                bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
                critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss
                if logging:
                    metrics["bc_fosd_loss"] = bc_fosd_loss.item()

                if self.bc_margin > 0:
                    qs = (qs * self.critic.support.expand_as(qs)).sum(-1)
                    qs_a = (qs_a * self.critic.support.expand_as(qs_a)).sum(-1)
                    margin_loss = torch.clamp(
                        self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
                    ).mean([-1, -2, -3])
                    margin_loss = (margin_loss * demos).sum() / demos.sum()
                    critic_loss = critic_loss + self.bc_lambda * margin_loss
                    if logging:
                        metrics["bc_margin_loss"] = margin_loss.item()

        # Compute priority
        new_pri = torch.sqrt(q_critic_loss + 1e-10)
        self._td_error = (new_pri / torch.max(new_pri)).cpu().detach().numpy()
        critic_loss = torch.mean(critic_loss)

        if logging:
            metrics[f"{lp}critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_grad_clip:
            critic_norm = nn.utils.clip_grad_norm_(
                critic.parameters(), self.critic_grad_clip
            )
            if logging:
                metrics[f"{lp}critic_norm"] = critic_norm.item()
        critic_opt.step()
        if self.use_pixels and self.encoder is not None:
            if self.critic_grad_clip:
                encoder_norm = nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.critic_grad_clip
                )
                if logging:
                    metrics[f"{lp}encoder_norm"] = encoder_norm.item()
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()
        return metrics

    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        if step != 0:
            num_update_steps = self.num_update_steps
        else:
            num_update_steps = 1  # pre-training when step == 0
        for _ in range(num_update_steps):
            (
                metrics,
                batch,
                action,
                reward,
                discount,
                terminal,
                truncated,
                bootstrap,
                demos,
                time_obs,
                next_time_obs,
                loss_coeff,
            ) = self.extract_batch(replay_iter)

            low_dim_obs = next_low_dim_obs = None
            fused_view_feats = next_fused_view_feats = None
            if self.low_dim_size > 0:
                low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)

            if self.use_pixels:
                rgb_obs, next_rgb_obs, ext_metrics = self.extract_pixels(batch)
                metrics.update(ext_metrics)
                enc_metrics, rgb_feats, next_rgb_feats = self.encode(
                    rgb_obs, next_rgb_obs
                )
                metrics.update(enc_metrics)
                (
                    fusion_metrics,
                    fused_view_feats,
                    next_fused_view_feats,
                ) = self.multi_view_fusion(rgb_obs, rgb_feats, next_rgb_feats)
                metrics.update(fusion_metrics)
                if not self.frame_stack_on_channel:
                    fused_view_feats = fused_view_feats.view(
                        -1, self.time_dim, *fused_view_feats.shape[1:]
                    )
                    next_fused_view_feats = next_fused_view_feats.view(
                        -1, self.time_dim, *next_fused_view_feats.shape[1:]
                    )

            with torch.no_grad():
                # NOTE: Pre-compute next_action here, outside update_critic to support
                # using the same next_action for both critic/intr_critic updates
                next_action, mets = self.critic.get_action(
                    next_low_dim_obs,
                    next_fused_view_feats,
                    next_time_obs,
                    self.intr_critic,
                    return_metrics=True,
                    logging=self.logging,
                )
                metrics.update(**mets)

            metrics.update(
                self.update_critic(
                    low_dim_obs,
                    fused_view_feats,
                    action,
                    reward,
                    discount,
                    bootstrap,
                    next_low_dim_obs,
                    next_fused_view_feats,
                    next_action,
                    time_obs,
                    next_time_obs,
                    loss_coeff,
                    demos,
                    False,
                    logging=self.logging,
                )
            )

            if isinstance(replay_buffer, PrioritizedReplayBuffer):
                replay_buffer.set_priority(
                    indices=batch["indices"].cpu().detach().numpy(),
                    priorities=self._td_error**self.replay_alpha,
                )

            if self.intrinsic_reward_module is not None:
                intrinsic_rewards = self.intrinsic_reward_module.compute_irs(
                    batch, step
                )
                self.intrinsic_reward_module.update(batch)
                metrics.update(
                    self.update_critic(
                        low_dim_obs,
                        fused_view_feats.detach()
                        if fused_view_feats is not None
                        else None,
                        action,
                        intrinsic_rewards,
                        discount,
                        bootstrap,
                        next_low_dim_obs,
                        next_fused_view_feats.detach()
                        if next_fused_view_feats is not None
                        else None,
                        next_action,
                        time_obs,
                        next_time_obs,
                        loss_coeff,
                        None,
                        True,
                        logging=self.logging,
                    )
                )
                if step % self.critic_target_interval == 0:
                    utils.soft_update_params(
                        self.intr_critic,
                        self.intr_critic_target,
                        self.critic_target_tau,
                    )

            # update critic target
            if step % self.critic_target_interval == 0:
                utils.soft_update_params(
                    self.critic, self.critic_target, self.critic_target_tau
                )

        return metrics
