from typing import Sequence
from typing_extensions import override
import logging

import numpy as np
import torch

from robobase.models import RoboBaseModule
from robobase.models.fusion import FusionModule
from robobase.models.encoder import EncoderModule

from robobase.reward_method.core import RewardMethod
from diffusers.optimization import get_scheduler

from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.method.utils import (
    extract_many_from_spec,
    extract_from_batch,
    stack_tensor_dictionary,
    extract_many_from_batch,
    TimeConsistentRandomShiftsAug,
)

from typing import Optional


class InvalidSequenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


class PreferenceTransformer(RewardMethod):
    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        encoder_model: Optional[EncoderModule] = None,
        view_fusion_model: Optional[FusionModule] = None,
        reward_model: Optional[RoboBaseModule] = None,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        use_lang_cond: bool = False,
        num_labels: int = 1,
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        *args,
        **kwargs,
    ):
        """
        Preference Transformer Reward Model Agent.

        Args:
            lr (float): Learning rate for the policy.
            lr_backbone (float): Learning rate for the backbone.
            weight_decay (float): Weight decay for optimization.
        """
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.compute_batch_size = compute_batch_size
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.reward_model = reward_model
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.aug = (
            TimeConsistentRandomShiftsAug(pad=4) if use_augmentation else lambda x: x
        )

        # T should be same across all obs
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1

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

        self._is_relabel = True

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
            ## Multiply first two dimensions to consider frame stacking
            # This module must ignore frame stacking (b/c input would be 1 frame)
            obs_shape = (1 * rgb_shapes[0][1], *rgb_shapes[0][2:])
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
        self.reward = self.reward_model(
            input_shape=self.encoder.output_shape,
            state_dim=np.prod(self.observation_space["low_dim_state"].shape[1:]),
            action_dim=self.action_space.shape[-1],
        )
        self.reward.to(self.device)
        self.reward_opt = torch.optim.AdamW(
            self.reward.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

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
        self, seq: Sequence, _obs_signature: Sequence = None, member: int = -1
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

        start_idx = 0
        seq_len = self.seq_len
        if isinstance(seq, list):
            T = len(seq) - start_idx

            if len(seq) < self.seq_len:
                raise InvalidSequenceError(
                    f"Input sequence must be at least {self.seq_len} steps long. Seq len is {len(seq)}"
                )

            # seq: list of (action, obs, reward, term, trunc, info, next_info)
            actions = torch.from_numpy(np.stack([elem[0] for elem in seq]))
            if actions.ndim > 2:
                actions = actions[..., -1, :]
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
            obs = {key: torch.from_numpy(np.stack(val)) for key, val in obs.items()}
            # obs: (T, elem_shape) for elem in obs
            # actions: (T, action_shape)
        elif isinstance(seq, dict):
            assert _obs_signature is not None, "Need obs_signature for dict input."
            # print("action length", len(seq["action"]))
            T = len(seq["action"]) - start_idx

            if T < self.seq_len:
                raise InvalidSequenceError(
                    f"Input sequence must be at least {self.seq_len} steps long. Seq len is {T}"
                )

            actions = torch.from_numpy(seq["action"])
            if actions.ndim > 2:
                actions = actions[..., -1, :]
            if seq["low_dim_state"].ndim > 2:
                obs = {
                    key: torch.from_numpy(val[start_idx:, -1])
                    for key, val in seq.items()
                    if key in _obs_signature
                }
            else:
                obs = {
                    key: torch.from_numpy(val[start_idx:])
                    for key, val in seq.items()
                    if key in _obs_signature
                }
            # obs: (T, elem_shape) for elem in obs
            # actions: (T, action_shape)

        idxs = list(range(T - self.seq_len + 1))
        rgbs = (
            stack_tensor_dictionary(extract_many_from_batch(obs, r"rgb(?!.*?tp1)"), 1)
            .unsqueeze(1)
            .to(self.device)
        )
        fused_rgb_feats = self.encode_rgb_feats(rgbs, train=False).squeeze(1)
        batch_fused_rgb_feats = torch.stack(
            [fused_rgb_feats[idx : idx + seq_len] for idx in idxs]
        )
        qpos = extract_from_batch(obs, "low_dim_state")

        batch_qposes = torch.stack([qpos[idx : idx + seq_len] for idx in idxs]).to(
            self.device
        )
        batch_actions = torch.stack([actions[idx : idx + seq_len] for idx in idxs]).to(
            self.device
        )

        rewards = []
        for i in range(
            0,
            len(idxs),
            self.compute_batch_size,
        ):
            _range = list(range(i, min(i + self.compute_batch_size, len(idxs))))
            with torch.no_grad():
                _reward = self.reward(
                    batch_fused_rgb_feats[_range],
                    batch_qposes[_range],
                    batch_actions[_range],
                    member=member,
                )
            rewards.append(_reward)

        rewards = torch.cat(rewards, dim=0)
        assert (
            len(rewards) == T - seq_len + 1
        ), f"Expected {T - seq_len + 1} rewards, got {len(rewards)}"

        # compute the part before the full sequence.
        preset_indices = np.concatenate(
            [np.zeros((seq_len - 1,)), np.arange(seq_len)], axis=0
        )
        preset_indices = np.lib.stride_tricks.sliding_window_view(
            preset_indices, seq_len
        ).astype(np.int32)
        first_fused_rgb_feats = torch.stack(
            [fused_rgb_feats[idx] for idx in preset_indices]
        ).to(self.device)
        first_qposes = torch.stack([qpos[idx] for idx in preset_indices]).to(
            self.device
        )
        first_actions = torch.stack([actions[idx] for idx in preset_indices]).to(
            self.device
        )
        first_attn_masks = torch.fliplr(
            torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32), 1)
        ).to(self.device)

        with torch.no_grad():
            first_rewards = self.reward(
                first_fused_rgb_feats,
                first_qposes,
                first_actions,
                attn_mask=first_attn_masks,
                member=member,
            )[:-1]

        assert (
            len(first_rewards) == seq_len - 1
        ), f"Expected {seq_len - 1} rewards, got {len(first_rewards)}"

        total_rewards = (
            torch.cat([first_rewards, rewards], dim=0).mean(dim=1).cpu().numpy()
        )

        if isinstance(seq, list):
            for idx in range(len(seq)):
                seq[idx] = list(seq[idx])
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
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # actions = batch["action"]
        # reward = batch["reward"]

        loss, loss_dict = 0.0, dict()
        r_hats = [[] for _ in range(self.reward.num_ensembles)]
        for i in range(2):
            actions = batch[f"seg{i}_action"]
            if self.low_dim_size > 0:
                # (bs, seq, low_dim)
                obs = extract_from_batch(batch, f"seg{i}_low_dim_state")
                qpos = obs.detach()

            if self.use_pixels:
                # (bs, seq, v, ch, h, w)
                rgb = stack_tensor_dictionary(
                    extract_many_from_batch(batch, rf"seg{i}_rgb(?!.*?tp1)"), 2
                )
                fused_rgb_feats = self.encode_rgb_feats(rgb, train=True)

            for member in range(self.reward.num_ensembles):
                r_hat = self.reward(fused_rgb_feats, qpos, actions, member=member)
                r_hats[member].append(r_hat)

        for member in range(self.reward.num_ensembles):
            _loss, _loss_dict = self.reward.calculate_loss(
                r_hats[member], batch["label"]
            )
            loss += _loss
            for key in _loss_dict:
                loss_dict[f"{key}_member_{member}"] = _loss_dict[key]
        loss_dict["loss"] = loss / self.reward.num_ensembles

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
            for member in range(self.reward.num_ensembles):
                for label in range(self.num_labels):
                    metrics[f"pref_acc_label_{label}_member_{member}"] = loss_dict[
                        f"pref_acc_label_{label}_member_{member}"
                    ].item()

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
