import logging
from copy import deepcopy
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from torch import nn
from tqdm import trange
from typing_extensions import override

from robobase import utils
from robobase.method.utils import (
    TimeConsistentRandomShiftsAug,
    extract_from_batch,
    extract_from_spec,
    extract_many_from_batch,
    extract_many_from_spec,
    stack_tensor_dictionary,
)
from robobase.models import RoboBaseModule
from robobase.models.encoder import EncoderModule
from robobase.models.fully_connected import FullyConnectedModule
from robobase.models.fusion import FusionModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.reward_method.core import RewardMethod


class WeightRewardModel(nn.Module):
    MIN = -1.0
    MAX = 1.0

    def __init__(
        self,
        reward_model: FullyConnectedModule,
        num_reward_models: int = 1,
        num_reward_terms: int = 1,
        reward_lows: Optional[np.ndarray] = None,
        reward_highs: Optional[np.ndarray] = None,
        reg_weight: float = 0.0,
    ):
        super().__init__()
        self.ws = nn.ModuleList(
            [deepcopy(reward_model) for _ in range(num_reward_models)]
        )
        self.apply(utils.weight_init)
        if reward_lows is None:
            reward_lows = torch.full(num_reward_terms, -np.inf)
        if reward_highs is None:
            reward_highs = torch.full(num_reward_terms, np.inf)

        self.reward_lows = reward_lows
        self.reward_highs = reward_highs
        self.reg_weight = reg_weight
        self.label_margin = 0.0
        self.label_target = 1.0 - 2 * self.label_margin

    def forward(self, low_dim_obs, fused_view_feats, action, time_obs, member: int = 0):
        net_ins = {"action": action.view(action.shape[0], -1)}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs

        weights = self.ws[member](net_ins)
        weights = torch.tanh(weights)
        return weights

    def transform_to_tanh(self, weights):
        orig_min, orig_max = self.reward_lows, self.reward_highs
        scale = (orig_max - orig_min) / (self.MAX - self.MIN)
        new_weights = orig_min + scale * (weights - self.MIN)
        return new_weights.to(weights.dtype)

    def reset(self, env_index: int):
        for w in self.ws:
            w.reset(env_index)

    def set_eval_env_running(self, value: bool):
        for w in self.ws:
            w.eval_env_running = value

    def calculate_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        r_hat_weights: torch.Tensor,
    ) -> Optional[dict]:
        """
        Calculate the loss for the WeightRewardModel model.

        Args:
            logits (Tuple[torch.Tensor, torch.Tensor]): Tuple of two tensors, each of shape (bs, seq).
            labels (torch.Tensor): Tensor containing ground truth preference labels.
            r_hat_weights (torch.Tensor): Tensor containing the weights of the reward terms.

        Returns:
            Optional[dict]: Dictionary containing training metrics.
        """
        logits = torch.stack(logits, dim=-1)
        r_hat_weights = torch.stack(r_hat_weights, dim=-1)

        loss_dict = {"loss": 0.0}
        reward_loss = 0.0
        for idx, (logit, label) in enumerate(zip(logits.unbind(1), labels.unbind(1))):
            # reward_loss = F.cross_entropy(logit, label)
            uniform_index = labels == -1
            labels[uniform_index] = 0
            target_onehot = torch.zeros_like(logit).scatter(
                1, labels, self.label_target
            )
            target_onehot += self.label_margin
            if sum(uniform_index) > 0:
                target_onehot[uniform_index] = 0.5
            reward_loss += utils.softXEnt_loss(logit, target_onehot)

            loss_dict[f"pref_acc_label_{idx}"] = utils.pref_accuracy(logit, label)
            loss_dict[f"pref_loss_{idx}"] = reward_loss
            loss_dict["loss"] += reward_loss

        logit_reg_loss = torch.mean(torch.square(r_hat_weights))  # L2 regularization
        loss_dict["logit_reg_loss"] = logit_reg_loss.item()
        loss_dict["loss"] += self.reg_weight * logit_reg_loss
        return loss_dict


class WeightTunerReward(RewardMethod):
    def __init__(
        self,
        reward_space: gym.spaces.Dict,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        encoder_model: Optional[EncoderModule] = None,
        view_fusion_model: Optional[FusionModule] = None,
        reward_model: Optional[RoboBaseModule] = None,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        use_lang_cond: bool = False,
        num_label: int = 1,
        num_reward_models: int = 1,
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        reg_weight: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Weight Tuner Reward Model Agent.

        Args:
            lr (float): Learning rate for the reward model.
            lr_backbone (float): Learning rate for the backbone.
            weight_decay (float): Weight decay for optimization.
        """
        super().__init__(*args, **kwargs)

        if isinstance(reward_space, gym.spaces.Dict):
            self.reward_space = reward_space
        else:
            self.reward_space = gym.spaces.Dict(sorted(reward_space.items()))
        self.num_reward_models = num_reward_models
        self.num_reward_terms = len(self.reward_space.spaces)
        self.reward_lows = torch.from_numpy(
            np.stack([space.low for space in self.reward_space.spaces.values()])
        ).to(self.device)
        self.reward_highs = torch.from_numpy(
            np.stack([space.high for space in self.reward_space.spaces.values()])
        ).to(self.device)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.num_label = num_label
        self.seq_len = seq_len
        self.compute_batch_size = compute_batch_size
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.reward_model = reward_model
        self.reg_weight = reg_weight
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.aug = (
            TimeConsistentRandomShiftsAug(pad=4) if use_augmentation else lambda x: x
        )

        # T should be same across all obs
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1
        self.time_dim = list(self.observation_space.values())[0].shape[0]

        self.encoder = self.view_fusion = None
        self.build_encoder()
        self.build_view_fusion()
        self.build_reward_model()

        self.lr_scheduler = None
        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.actor_opt,
                num_warmup_steps=100,
                num_training_steps=num_train_steps,
            )

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
            # Multiply first two dimensions to consider frame stacking
            obs_shape = (np.prod(rgb_shapes[0][:2]), *rgb_shapes[0][2:])
            self.encoder = self.encoder_model(input_shape=(num_views, *obs_shape))
            self.encoder.to(self.device)
            self.encoder_opt = None

    def build_view_fusion(self):
        self.rgb_latent_size = 0
        if not self.use_pixels:
            return
        if self.use_multicam_fusion:
            if self.view_fusion_model is None:
                logging.warn(
                    "Multicam fusion is enabled but view_fusion_model is not set!"
                )
                self.view_fusion_opt = None
                return

            self.view_fusion = self.view_fusion_model(
                input_shape=self.encoder.output_shape
            )
            self.view_fusion.to(self.device)
            self.view_fusion_opt = None
            if len([_ for _ in self.view_fusion.parameters()]) != 0:
                # Introduce optimizer when view_fusion_model is parametrized
                self.view_fusion_opt = torch.optim.Adam(
                    self.view_fusion.parameters(), lr=self.lr
                )
            self.rgb_latent_size = self.view_fusion.output_shape[-1]
        else:
            self.view_fusion = lambda x: x[:, 0]
            self.rgb_latent_size = self.encoder.output_shape[-1]

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

    def build_reward_model(self):
        input_shapes = self.get_fully_connected_inputs()
        input_shapes["actions"] = (np.prod(self.action_space.shape),)
        reward_model = self.reward_model(
            input_shapes=input_shapes, output_shape=self.num_reward_terms
        )
        self.reward = WeightRewardModel(
            reward_model=reward_model,
            num_reward_models=self.num_reward_models,
            num_reward_terms=self.num_reward_terms,
            reward_lows=self.reward_lows,
            reward_highs=self.reward_highs,
            reg_weight=self.reg_weight,
        )
        self.reward.to(self.device)
        self.reward_opt = torch.optim.Adam(self.reward.parameters(), lr=self.lr)

    def encode_rgb_feats(self, rgb, train=False):
        # (bs * seq *v, ch, h , w)
        bs, seq, v, c, h, w = rgb.shape
        image = rgb.transpose(1, 2).reshape(bs * v, seq, c, h, w)
        if train:
            image = self.aug(image.float())
        image = (
            image.reshape(bs, v, seq, c, h, w)
            .transpose(1, 2)
            .reshape(bs * seq, v, c, h, w)
        )
        # (bs * seq, v, ch, h , w)
        image = image.float().detach()

        with torch.no_grad():
            # (bs*seq, v, c, h, w) -> (bs*seq, v, h)
            multi_view_rgb_feats = self.encoder(image)
            # (bs*seq, v*h)
            fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
            # (bs, seq, v*h)
            fused_rgb_feats = fused_rgb_feats.view(*rgb.shape[:2], -1)
        return fused_rgb_feats

    @override
    def compute_reward(
        self,
        seq: Sequence,
        member: int = -1,
        return_reward: bool = False,
    ) -> torch.Tensor:
        """
        Compute the reward from sequences.

        Args:
            seq (Sequence): same with _episode_rollouts in workspace.py.
            observations (dict): Dictionary containing observations.
            actions (torch.Tensor): The actions taken.

        Returns:
            torch.Tensor: The reward tensor.

        """

        if not self.activated:
            return seq

        start_idx = 0
        T = len(seq) - start_idx

        if isinstance(seq, list):
            # seq: list of (action, obs, reward, term, trunc, info, next_info)
            actions = utils.convert_numpy_to_torch(
                np.stack([elem[0] for elem in seq]), self.device
            )
            if seq[0][1]["low_dim_state"].ndim > 1:
                list_of_obs_dicts = [
                    {k: v[-1] for k, v in elem[1].items()} for elem in seq
                ]
            else:
                list_of_obs_dicts = [elem[1] for elem in seq]

            obs = {key: [] for key in list_of_obs_dicts[0].keys()}
            for obs_dict in list_of_obs_dicts:
                for key, val in obs_dict.items():
                    obs[key].append(val)
            obs = utils.convert_numpy_to_torch(
                {key: np.stack(val) for key, val in obs.items()}, self.device
            )
            # obs: (T, elem_shape) for elem in obs
            # actions: (T, action_shape)

            list_of_info_dicts = [elem[-2] for elem in seq]
            reward_terms = {key: [] for key in self.reward_space.keys()}
            for idx, info_dict in enumerate(list_of_info_dicts):
                if idx == 0 and not info_dict:
                    for key in reward_terms:
                        reward_terms[key].append(0.0)
                else:
                    for key in reward_terms:
                        reward_terms[key].append(info_dict[key])
            reward_terms = {
                key: utils.convert_numpy_to_torch(np.stack(val), self.device)
                for key, val in reward_terms.items()
            }
            # reward_terms: (T, num_reward_terms)
        elif isinstance(seq, dict):
            T = len(seq["action"]) - start_idx
            actions = utils.convert_numpy_to_torch(seq["action"], self.device)
            if actions.ndim > 2:
                actions = actions[..., -1, :]
            if seq["low_dim_state"].ndim > 2:
                obs = {
                    key: utils.convert_numpy_to_torch(val[start_idx:, -1], self.device)
                    for key, val in seq.items()
                    if key in self.observation_space.spaces
                }
            else:
                obs = {
                    key: utils.convert_numpy_to_torch(val[start_idx:], self.device)
                    for key, val in seq.items()
                    if key in self.observation_space.spaces
                }
            reward_terms = {
                key: utils.convert_numpy_to_torch(seq[key], self.device)
                for key in self.reward_space.keys()
            }

        if self.use_pixels:
            rgbs = (
                stack_tensor_dictionary(
                    extract_many_from_batch(obs, r"rgb(?!.*?tp1)"), 1
                )
                .unsqueeze(1)
                .to(self.device)
            )
            fused_rgb_feats = self.encode_rgb_feats(rgbs, train=False).squeeze(1)
        else:
            fused_rgb_feats = None
        qpos = (
            extract_from_batch(obs, "low_dim_state").to(self.device)
            if self.low_dim_size > 0
            else None
        )
        time_obs = (
            extract_from_batch(obs, "time", missing_ok=True).to(self.device)
            if self.time_obs_size > 0
            else None
        )
        reward_terms = stack_tensor_dictionary(reward_terms, dim=-1).to(self.device)

        rewards = []
        for i in trange(
            0,
            T,
            self.compute_batch_size,
            leave=False,
            ncols=0,
            desc="reward compute per batch",
        ):
            _range = list(range(i, min(i + self.compute_batch_size, T)))
            with torch.no_grad():
                if member == -1:
                    _weighted_rewards = []
                    for mem in range(self.num_reward_models):
                        _reward_weights = self.reward(
                            qpos[_range] if qpos is not None else None,
                            fused_rgb_feats[_range]
                            if fused_rgb_feats is not None
                            else None,
                            actions[_range],
                            time_obs[_range] if time_obs is not None else None,
                            member=mem,
                        )
                        scaled_reward_weights = self.reward.transform_to_tanh(
                            _reward_weights
                        )
                        weighted_reward = (
                            scaled_reward_weights * reward_terms[_range]
                        ).sum(dim=-1, keepdim=True)
                        _weighted_rewards.append(weighted_reward)
                    weighted_reward = torch.cat(_weighted_rewards, dim=1).mean(dim=1)
                    rewards.append(weighted_reward)
                else:
                    _reward_weights = self.reward(
                        qpos[_range] if qpos is not None else None,
                        fused_rgb_feats[_range]
                        if fused_rgb_feats is not None
                        else None,
                        actions[_range],
                        time_obs[_range] if time_obs is not None else None,
                        member=member,
                    )
                    _scaled_reward_weights = self.reward.transform_to_tanh(
                        _reward_weights
                    )
                    _reward = (_scaled_reward_weights * reward_terms[_range]).sum(
                        dim=-1
                    )
                    rewards.append(_reward)
        total_rewards = torch.cat(rewards, dim=0)
        assert len(rewards) == T, f"Expected {T} rewards, got {len(rewards)}"

        if return_reward:
            return total_rewards

        total_rewards = total_rewards.cpu().numpy()
        if isinstance(seq, list):
            for idx in range(len(seq)):
                seq[idx][2] = total_rewards[idx]

        elif isinstance(seq, dict):
            seq["reward"] = total_rewards

        return seq

    @override
    def update(
        self, replay_iter, step: int, replay_buffer: ReplayBuffer = None
    ) -> dict:
        """
        Compute the loss from binary preferences.

        Args:
            replay_iter (iterable): An iterator over a replay buffer.
            step (int): The current step.
            replay_buffer (ReplayBuffer): The replay buffer.

        Returns:
            dict: Dictionary containing training metrics.

        """

        metrics = dict()
        loss_dict = defaultdict(float)
        for member in range(self.num_reward_models):
            batch = next(replay_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            r_hats = []
            r_hat_weights = []
            for i in range(2):
                actions = batch[f"seg{i}_action"]
                if self.low_dim_size > 0:
                    # (bs, seq, low_dim)
                    qpos = extract_from_batch(batch, f"seg{i}_low_dim_state").detach()

                if self.use_pixels:
                    # (bs, seq, v, ch, h, w)
                    rgb = stack_tensor_dictionary(
                        extract_many_from_batch(batch, rf"seg{i}_rgb(?!.*?tp1)"), 2
                    )
                    fused_rgb_feats = self.encode_rgb_feats(rgb, train=True)
                else:
                    fused_rgb_feats = None

                time_obs = extract_from_batch(batch, "time", missing_ok=True)

                # extract reward terms: (bs, seq, num_reward_terms)
                reward_terms = stack_tensor_dictionary(
                    {key: batch[f"seg{i}_{key}"] for key in self.reward_space.keys()},
                    dim=-1,
                )
                # r_hat_weight: (bs * seq, num_reward_terms) -> (bs, seq, num_reward_terms)
                r_hat_weight = self.reward(
                    qpos.reshape(-1, *qpos.shape[2:]),
                    fused_rgb_feats.reshape(-1, *fused_rgb_feats.shape[2:])
                    if fused_rgb_feats is not None
                    else None,
                    actions.reshape(-1, *actions.shape[2:]),
                    time_obs.reshape(-1, *time_obs.shape[2:])
                    if time_obs is not None
                    else None,
                ).view(*actions.shape[:-2], -1, reward_terms.shape[-1])
                # r_hat: (bs, seq, num_reward_terms) -> (bs, seq, 1) -> (bs, 1)
                scaled_r_hat_weight = self.reward.transform_to_tanh(r_hat_weight)
                r_hat = (
                    (scaled_r_hat_weight * reward_terms)
                    .sum(dim=-1, keepdim=True)
                    .sum(dim=-2)
                )
                r_hats.append(r_hat)
                r_hat_weights.append(r_hat_weight)

            _loss_dict = self.reward.calculate_loss(
                r_hats, batch["label"], r_hat_weights
            )
            for k, v in _loss_dict.items():
                loss_dict[k] += v

        for i in range(self.num_label):
            loss_dict[f"pref_acc_label_{i}"] /= self.num_reward_models

        # calculate gradient
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        self.reward_opt.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()

        # step optimizer
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()
        self.reward_opt.step()

        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.logging:
            metrics["reward_loss"] = loss_dict["loss"].item()
            r_hat_weights = torch.cat(r_hat_weights, dim=0)
            for idx, term in enumerate(self.reward_space):
                metrics[f"r_hat_weights_{term.split('/')[-1]}"] = (
                    r_hat_weights[..., idx].mean().item()
                )
            for label in range(self.num_label):
                metrics[f"pref_acc_label_{label}"] = loss_dict[
                    f"pref_acc_label_{label}"
                ].item()

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
