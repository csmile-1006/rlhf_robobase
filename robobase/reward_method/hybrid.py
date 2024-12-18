import logging
from collections import defaultdict
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from tqdm import trange
from typing_extensions import override

from robobase import utils
from robobase.method.utils import (
    TimeConsistentRandomShiftsAug,
    extract_from_batch,
    extract_many_from_batch,
    extract_many_from_spec,
    stack_tensor_dictionary,
)
from robobase.models import RoboBaseModule
from robobase.models.encoder import EncoderModule
from robobase.models.fusion import FusionModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.reward_method.core import RewardMethod
from robobase.reward_method.markovian import MarkovianRewardModel
from robobase.reward_method.weight_tuner import WeightRewardModel


class HybridReward(RewardMethod):
    def __init__(
        self,
        reward_space: gym.spaces.Dict,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        encoder_model: Optional[EncoderModule] = None,
        view_fusion_model: Optional[FusionModule] = None,
        reward_model: Optional[RoboBaseModule] = None,
        weight_model: Optional[RoboBaseModule] = None,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        use_lang_cond: bool = False,
        num_labels: int = 1,
        num_reward_models: int = 1,
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        lambda_weight: float = 1.0,
        reg_weight: float = 0.0,
        apply_final_layer_tanh: bool = False,
        data_aug_ratio: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Hybrid Reward Model Agent.

        Args:
            lr (float): Learning rate for the reward model.
            lr_backbone (float): Learning rate for the backbone.
            weight_decay (float): Weight decay for optimization.
        """
        super().__init__(*args, **kwargs)

        self.reward_space = gym.spaces.Dict(sorted(reward_space.items()))
        self.num_reward_terms = len(self.reward_space.spaces)
        self.reward_lows = utils.convert_numpy_to_torch(
            np.stack([space.low for space in self.reward_space.spaces.values()]),
            self.device,
        )
        self.reward_highs = utils.convert_numpy_to_torch(
            np.stack([space.high for space in self.reward_space.spaces.values()]),
            self.device,
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.lambda_weight = lambda_weight
        self.reg_weight = reg_weight
        self.apply_final_layer_tanh = apply_final_layer_tanh

        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.num_reward_models = num_reward_models
        self.seq_len = seq_len
        self.compute_batch_size = compute_batch_size
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.reward_model = reward_model
        self.weight_model = weight_model
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.aug = (
            TimeConsistentRandomShiftsAug(pad=4) if use_augmentation else lambda x: x
        )
        self.data_aug_ratio = data_aug_ratio

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

    def build_reward_model(self):
        input_shapes = self.get_fully_connected_inputs()
        input_shapes["actions"] = (np.prod(self.action_space.shape),)
        reward_model = self.reward_model(input_shapes=input_shapes)
        weight_model = self.weight_model(
            input_shapes=input_shapes, output_shape=self.num_reward_terms
        )
        self.weight_tuner = WeightRewardModel(
            reward_model=weight_model,
            num_reward_models=self.num_reward_models,
            num_reward_terms=self.num_reward_terms,
            reward_lows=self.reward_lows,
            reward_highs=self.reward_highs,
            reg_weight=self.reg_weight,
        )
        self.weight_tuner.to(self.device)
        self.weight_tuner_opt = torch.optim.Adam(
            self.weight_tuner.parameters(), lr=self.lr
        )

        self.markovian = MarkovianRewardModel(
            reward_model=reward_model,
            num_reward_models=self.num_reward_models,
            apply_final_layer_tanh=self.apply_final_layer_tanh,
        )
        self.markovian.to(self.device)
        self.markovian_opt = torch.optim.Adam(self.markovian.parameters(), lr=self.lr)

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

    def initialize_reward_model(self):
        input_shapes = self.get_fully_connected_inputs()
        input_shapes["actions"] = (np.prod(self.action_space.shape),)
        reward_model = self.reward_model(input_shapes=input_shapes)
        self.markovian = MarkovianRewardModel(
            reward_model=reward_model,
            num_reward_models=self.num_reward_models,
            apply_final_layer_tanh=self.apply_final_layer_tanh,
        )
        self.markovian.to(self.device)
        self.markovian_opt = torch.optim.Adam(self.markovian.parameters(), lr=self.lr)

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
            for info_dict in list_of_info_dicts:
                for key in reward_terms:
                    reward_terms[key].append(info_dict[key])
            reward_terms = {
                key: utils.convert_numpy_to_torch(np.stack(val), self.device)
                for key, val in reward_terms.items()
            }
            # reward_terms: (T, num_reward_terms)
            if actions.ndim > 2 and actions.shape[-2] == 1:
                actions = actions[..., -1, :]

        elif isinstance(seq, dict):
            actions = utils.convert_numpy_to_torch(seq["action"], self.device)
            obs = utils.convert_numpy_to_torch(
                {
                    key: val[start_idx:]
                    for key, val in seq.items()
                    if key in self.observation_space.spaces
                },
                self.device,
            )
            # obs = {
            #     key: utils.convert_numpy_to_torch(val[start_idx:, -1], self.device)
            #     for key, val in seq.items()
            #     if key in self.observation_space.spaces
            # }
            if actions.ndim > 2 and actions.shape[-2] == 1:
                actions = actions[..., -1, :]
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

        # change components to be (bs * seq, -1)
        seq_len = None
        if qpos is not None and qpos.ndim > 2:
            qpos = qpos.reshape(-1, *qpos.shape[-1:])
        if fused_rgb_feats is not None:
            fused_rgb_feats = fused_rgb_feats.reshape(-1, *fused_rgb_feats.shape[-1:])
        if time_obs is not None:
            time_obs = time_obs.reshape(-1, *time_obs.shape[-1:])
        if actions.ndim > 2:
            actions = actions.reshape(-1, *actions.shape[-1:])
        if reward_terms.ndim > 2:
            seq_len = reward_terms.shape[1]
            reward_terms = reward_terms.reshape(-1, *reward_terms.shape[-1:])

        T = actions.shape[0] - start_idx
        weighted_rewards = []
        computed_rewards = []
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
                    _computed_rewards = []
                    for mem in range(self.num_reward_models):
                        _reward_weights = self.weight_tuner(
                            qpos[_range] if qpos is not None else None,
                            fused_rgb_feats[_range]
                            if fused_rgb_feats is not None
                            else None,
                            actions[_range],
                            time_obs[_range] if time_obs is not None else None,
                            member=mem,
                        )
                        _scaled_reward_weights = self.weight_tuner.transform_to_tanh(
                            _reward_weights
                        )
                        _weighted_reward = (
                            _scaled_reward_weights * reward_terms[_range]
                        ).sum(dim=-1, keepdim=True)
                        _computed_reward = self.markovian(
                            qpos[_range] if qpos is not None else None,
                            fused_rgb_feats[_range]
                            if fused_rgb_feats is not None
                            else None,
                            actions[_range],
                            time_obs[_range] if time_obs is not None else None,
                            member=mem,
                        )
                        _weighted_rewards.append(_weighted_reward)
                        _computed_rewards.append(_computed_reward)

                    weighted_reward = torch.cat(_weighted_rewards, dim=1).mean(dim=1)
                    computed_reward = torch.cat(_computed_rewards, dim=1).mean(dim=1)
                else:
                    _reward_weights = self.weight_tuner(
                        qpos[_range] if qpos is not None else None,
                        fused_rgb_feats[_range]
                        if fused_rgb_feats is not None
                        else None,
                        actions[_range],
                        time_obs[_range] if time_obs is not None else None,
                        member=member,
                    )
                    scaled_reward_weights = self.weight_tuner.transform_to_tanh(
                        _reward_weights
                    )
                    weighted_reward = (
                        scaled_reward_weights * reward_terms[_range]
                    ).sum(dim=-1)
                    computed_reward = self.markovian(
                        qpos[_range] if qpos is not None else None,
                        fused_rgb_feats[_range]
                        if fused_rgb_feats is not None
                        else None,
                        actions[_range],
                        time_obs[_range] if time_obs is not None else None,
                        member=member,
                    ).squeeze(-1)

                computed_rewards.append(computed_reward)
                weighted_rewards.append(weighted_reward)

        weighted_rewards = torch.cat(weighted_rewards, dim=0)
        computed_rewards = torch.cat(computed_rewards, dim=0)

        assert (
            len(weighted_rewards) == T
        ), f"Expected {T} weighted rewards, got {len(weighted_rewards)}"
        assert (
            len(computed_rewards) == T
        ), f"Expected {T} computed rewards, got {len(computed_rewards)}"
        total_rewards = weighted_rewards + self.lambda_weight * computed_rewards
        if seq_len is not None:
            total_rewards = total_rewards.view(-1, seq_len)

        if return_reward:
            return total_rewards

        total_rewards = total_rewards.cpu().numpy()
        if isinstance(seq, list):
            for idx in range(len(seq)):
                seq[idx][2] = total_rewards[idx]

        elif isinstance(seq, dict):
            seq["reward"] = total_rewards

        return seq

    def get_cropping_mask(self, r_hat, w):
        mask_ = []
        for i in range(w):
            B, S, _ = r_hat.shape
            length = np.random.randint(int(0.7 * S), int(0.9 * S) + 1, size=B)
            start_index = np.random.randint(0, S + 1 - length)
            mask = torch.zeros((B, S, 1)).to(self.device)
            for b in range(B):
                mask[b, start_index[b] : start_index[b] + length[b]] = 1
            mask_.append(mask)

        return torch.cat(mask_)

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
        weighted_loss_dict = defaultdict(float)
        computed_loss_dict = defaultdict(float)
        for mem in range(self.num_reward_models):
            batch = next(replay_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            weighted_rewards = []
            raw_weights = []
            normalized_weights = []
            markovian_rewards = []
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

                # Compute weighted reward
                # extract reward terms: (bs, seq, num_reward_terms)
                reward_terms = stack_tensor_dictionary(
                    {key: batch[f"seg{i}_{key}"] for key in self.reward_space.keys()},
                    dim=-1,
                )
                # raw_weight: (bs * seq, num_reward_terms) -> (bs, seq, num_reward_terms)
                args = (
                    qpos.reshape(-1, *qpos.shape[2:]),
                    fused_rgb_feats.reshape(-1, *fused_rgb_feats.shape[2:])
                    if fused_rgb_feats is not None
                    else None,
                    actions.reshape(-1, *actions.shape[2:]),
                    time_obs.reshape(-1, *time_obs.shape[2:])
                    if time_obs is not None
                    else None,
                )
                raw_weight = self.weight_tuner(*args, member=mem).view(
                    *actions.shape[:-2], -1, reward_terms.shape[-1]
                )
                normalized_weight = self.weight_tuner.transform_to_tanh(raw_weight)
                # weighted_reward: (bs, seq, num_reward_terms) -> (bs, seq, 1) -> (bs, 1)
                weighted_reward = (raw_weight * reward_terms).sum(dim=-1, keepdim=True)
                if self.data_aug_ratio > 0.0:
                    mask = self.get_cropping_mask(weighted_reward, self.data_aug_ratio)
                    weighted_reward = weighted_reward.repeat(self.data_aug_ratio, 1, 1)
                    weighted_reward = (mask * weighted_reward).sum(axis=-2)
                else:
                    weighted_reward = weighted_reward.sum(axis=-2)
                weighted_rewards.append(weighted_reward)
                raw_weights.append(raw_weight)
                normalized_weights.append(normalized_weight)

                # Compute markovian reward
                markovian_reward = self.markovian(*args, member=mem)
                # markovian_reward: (bs, seq, 1) -> (bs, 1)
                markovian_reward = markovian_reward.view(
                    *actions.shape[:-2], -1, markovian_reward.shape[-1]
                )
                if self.data_aug_ratio > 0.0:
                    mask = self.get_cropping_mask(markovian_reward, self.data_aug_ratio)
                    markovian_reward = markovian_reward.repeat(
                        self.data_aug_ratio, 1, 1
                    )
                    markovian_reward = (mask * markovian_reward).sum(axis=-2)
                else:
                    markovian_reward = markovian_reward.sum(axis=-2)
                markovian_rewards.append(markovian_reward)

            labels = batch["label"]
            if self.data_aug_ratio > 0:
                labels = labels.repeat(self.data_aug_ratio, 1)

            _weighted_loss_dict = self.weight_tuner.calculate_loss(
                weighted_rewards, labels, raw_weights
            )
            for k, v in _weighted_loss_dict.items():
                weighted_loss_dict[k] += v

            _computed_loss_dict = self.markovian.calculate_loss(
                markovian_rewards, labels
            )
            for k, v in _computed_loss_dict.items():
                computed_loss_dict[k] += v

        for i in range(self.num_labels):
            weighted_loss_dict[f"pref_acc_label_{i}"] /= self.num_reward_models
            computed_loss_dict[f"pref_acc_label_{i}"] /= self.num_reward_models

        # calculate gradient
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)

        self.weight_tuner_opt.zero_grad(set_to_none=True)
        self.markovian_opt.zero_grad(set_to_none=True)
        weighted_loss_dict["loss"].backward()
        computed_loss_dict["loss"].backward()

        # step optimizer
        self.weight_tuner_opt.step()
        self.markovian_opt.step()

        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.logging:
            metrics["weighted_reward_loss"] = weighted_loss_dict["loss"].item()
            metrics["computed_reward_loss"] = computed_loss_dict["loss"].item()
            metrics["reward_loss"] = (
                weighted_loss_dict["loss"] + computed_loss_dict["loss"]
            ).item()
            raw_weights = torch.cat(raw_weights, dim=0)
            normalized_weights = torch.cat(normalized_weights, dim=0)
            for idx, term in enumerate(self.reward_space):
                metrics[f"r_hat_weights_{term.split('/')[-1]}"] = (
                    normalized_weights[..., idx].mean().item()
                )
            for label in range(self.num_labels):
                metrics[f"weighted_pref_acc_label_{label}"] = weighted_loss_dict[
                    f"pref_acc_label_{label}"
                ].item()
                metrics[f"computed_pref_acc_label_{label}"] = computed_loss_dict[
                    f"pref_acc_label_{label}"
                ].item()
                metrics[f"pref_acc_label_{label}"] = (
                    metrics[f"weighted_pref_acc_label_{label}"]
                    + metrics[f"computed_pref_acc_label_{label}"]
                ) / 2

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
