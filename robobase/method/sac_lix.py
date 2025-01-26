from typing import Iterator
from distutils.dist import Distribution

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from robobase.method.utils import extract_many_from_spec

from robobase.models.lix_utils import analysis_optimizers

from robobase import utils
from robobase.method.utils import loss_weights
from robobase.method.actor_critic import ActorCritic, Actor
from robobase.models.lix_utils.analysis_modules import LIXModule

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACActor(Actor):
    def forward(self, low_dim_obs, fused_view_feats) -> Distribution:
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        mu, log_std = self.actor_model(net_ins).chunk(2, -1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        # dist = utils.SquashedNormal(mu, std)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def logprob(self, dist):
        tanh_a = dist.rsample()
        log_prob = dist.log_prob(tanh_a)
        # Enforcing Action Bound
        log_prob = log_prob.sum(-1)
        return tanh_a, log_prob


class SACLix(ActorCritic):
    """Implementation of Soft Actor-Critic (SAC) with Local sIgnal miXing (LIX) layer.

    Haarnoja, Tuomas, et al. "Soft actor-critic:
    Off-policy maximum entropy deep reinforcement learning with a stochastic actor."
    """

    def __init__(self, alpha_lr: float, init_temperature: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_space.shape).to(self.device)
        ).item()
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(init_temperature), device=self.device)
        )
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def reset_temperature(self):
        self.init_temperature = self.init_temperature
        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_space.shape).to(self.device)
        ).item()
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(self.init_temperature), device=self.device)
        )
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def build_encoder(self):
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            rgb_shapes = [s.shape for s in rgb_spaces.values()]
            assert np.all(
                [sh == rgb_shapes[0] for sh in rgb_shapes]
            ), "Expected all RGB obs to be same shape."

            num_views = len(rgb_shapes)
            if self.frame_stack_on_channel:
                obs_shape = (np.prod(rgb_shapes[0][:2]), *rgb_shapes[0][2:])
            else:
                # T is folded into batch
                obs_shape = rgb_shapes[0][1:]
            self.encoder = self.encoder_model(input_shape=(num_views, *obs_shape))
            if not isinstance(self.encoder, LIXModule):
                raise ValueError("Encoder must be of type ALIXModule.")
            self.encoder.to(self.device)
            self.encoder_opt = (
                analysis_optimizers.custom_parameterized_aug_optimizer_builder(
                    encoder_lr=self.encoder_lr, lr=2e-3, betas=[0.5, 0.999]
                )(self.encoder)
            )

    def build_actor(self):
        actor_model_obj = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1] * 2,
            num_envs=self.num_train_envs + self.num_eval_envs,
        )
        self.actor = SACActor(actor_model_obj).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def extract_batch(
        self, replay_iter: Iterator[dict[str, torch.Tensor]]
    ) -> TensorDict:
        batch = next(replay_iter)
        batch = TensorDict(
            {
                k: torch.as_tensor(v, dtype=v.dtype, device=self.device)
                for k, v in batch.items()
            },
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

        return batch

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
        time_obs,
        next_time_obs,
        loss_coeff,
        updating_intrinsic_critic,
        updating_unsup_critic,
    ):
        assert not (
            updating_intrinsic_critic and updating_unsup_critic
        ), "Cannot update both intrinsic and unsup critic."
        lp = ""

        critic, critic_opt = (
            self.critic,
            self.critic_opt,
        )

        metrics = TensorDict({})
        target_qs = self.calculate_target_q(
            next_low_dim_obs,
            next_fused_view_feats,
            next_time_obs,
            reward,
            discount,
            bootstrap,
            updating_intrinsic_critic,
        )

        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)

        target_qs = target_qs.repeat(1, self.num_critics)
        q_critic_loss = F.mse_loss(qs, target_qs, reduction="none").mean(
            -1, keepdim=True
        )
        critic_loss = q_critic_loss * loss_coeff.unsqueeze(1)

        # Compute priority
        new_pri = torch.sqrt(q_critic_loss + 1e-10)
        self._td_error = (new_pri / torch.max(new_pri)).cpu().detach().numpy()
        critic_loss = torch.mean(critic_loss)

        if self.logging:
            metrics[f"{lp}critic_target_q"] = target_qs.mean().detach()
            for i in range(1, self.num_critics):
                metrics[f"{lp}critic_q{i + 1}"] = qs[..., i].mean().detach()
            metrics[f"{lp}critic_loss"] = critic_loss.detach()

        # optimize encoder and critic
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_grad_clip:
            nn.utils.clip_grad_norm_(critic.parameters(), self.critic_grad_clip)
        critic_opt.step()
        return metrics

    def update_target_critic(self, step: int):
        # update critic target
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )

    def calculate_target_q(
        self,
        next_low_dim_obs,
        next_fused_view_feats,
        next_time_obs,
        reward,
        discount,
        bootstrap,
        updating_intrinsic_critic,
    ):
        critic_target = (
            self.intr_critic_target if updating_intrinsic_critic else self.critic_target
        )
        with torch.no_grad():
            dist = self.actor(next_low_dim_obs, next_fused_view_feats)
            next_action, log_prob = self.actor.logprob(dist)
            log_prob = log_prob.mean(-1, keepdim=True)  # account for action sequence
            target_qs = critic_target(
                next_low_dim_obs, next_fused_view_feats, next_action, next_time_obs
            )
            if self.distributional_critic:
                target_qs = critic_target.from_dist(target_qs)
            min_q = (
                target_qs.min(-1, keepdim=True)[0]
                - self.log_alpha.exp().detach() * log_prob
            )
            target_q = reward + bootstrap * discount * min_q
            if self.distributional_critic:
                return critic_target.to_dist(target_q)
            return target_q

    def _compute_actor_loss(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        time_obs,
        loss_coeff,
        critic,
        log_prob,
    ):
        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        if self.distributional_critic:
            qs = critic.from_dist(qs)
        min_q = qs.min(-1, keepdim=True)[0]
        return (
            ((self.log_alpha.exp().detach() * log_prob) - min_q)
            * loss_coeff.unsqueeze(1)
        ).mean()

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, time_obs, demos, loss_coeff
    ):
        metrics = TensorDict({})

        dist = self.actor(low_dim_obs, fused_view_feats)
        action, log_prob = self.actor.logprob(dist)

        base_actor_loss = self._compute_actor_loss(
            low_dim_obs,
            fused_view_feats,
            action,
            time_obs,
            loss_coeff,
            self.critic,
            log_prob,
        )
        intr_actor_loss = 0
        bc_metrics, bc_loss = self.get_bc_loss(dist.mean, act, demos)
        metrics.update(bc_metrics)
        actor_loss = base_actor_loss + intr_actor_loss + bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()

        self.a_optimizer.zero_grad()
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.a_optimizer.step()
        alpha = self.log_alpha.exp().detach()
        if self.logging:
            metrics["alpha"] = alpha
            metrics["alpha_loss"] = alpha_loss.detach()
            metrics["mean_act"] = dist.mean.mean().detach()
            metrics["actor_loss"] = actor_loss.detach()
            metrics["actor_logprob"] = log_prob.mean().detach()
        return metrics

    def update(
        self,
        batch: TensorDict,
    ) -> dict[str, np.ndarray]:
        low_dim_obs = next_low_dim_obs = None
        fused_view_feats = next_fused_view_feats = None
        low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)

        metrics = self.update_critic(
            low_dim_obs,
            fused_view_feats,
            batch["action"],
            batch["reward"],
            batch["discount"],
            batch["bootstrap"],
            next_low_dim_obs,
            next_fused_view_feats,
            None,
            None,
            batch["loss_coeff"],
            False,
            False,
        )

        metrics.update(
            self.update_actor(
                low_dim_obs,
                fused_view_feats,
                batch["action"],
                None,
                None,
                batch["loss_coeff"],
            )
        )

        return metrics
