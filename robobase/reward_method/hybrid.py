import logging
from typing import Optional, Sequence, Dict

import gymnasium as gym
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from tqdm import trange
from typing_extensions import override

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
from robobase.reward_method.weight_tuner import WeightRewardModel
from robobase.reward_method.markovian import MarkovianRewardModel


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
        num_label: int = 1,
        num_reward_models: int = 1,
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        lambda_weight: float = 1.0,
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

        self.reward_space = gym.spaces.Dict(sorted(reward_space.items()))
        self.num_reward_terms = len(self.reward_space.spaces)
        self.reward_lows = torch.from_numpy(
            np.stack([space.low for space in self.reward_space.spaces.values()])
        ).to(self.device)
        self.reward_highs = torch.from_numpy(
            np.stack([space.high for space in self.reward_space.spaces.values()])
        ).to(self.device)

        self._i = 0
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.lambda_weight = lambda_weight

        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.num_label = num_label
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
        self.weight_tuner = WeightRewardModel(
            reward_model=reward_model,
            num_reward_terms=self.num_reward_terms,
            reward_lows=self.reward_lows,
            reward_highs=self.reward_highs,
        )
        self.weight_tuner.to(self.device)
        self.weight_tuner_opt = torch.optim.Adam(
            self.weight_tuner.parameters(), lr=self.lr
        )

        self.markovian = MarkovianRewardModel(
            reward_model=reward_model, num_reward_models=self.num_reward_models
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
            reward_model=reward_model, num_reward_models=self.num_reward_models
        )
        self.markovian.to(self.device)
        self.markovian_opt = torch.optim.Adam(self.markovian.parameters(), lr=self.lr)

    @override
    def compute_reward(
        self,
        seq: Sequence,
        _obs_signature: Dict[str, gym.Space] = None,
        activate_reward_model: bool = True,
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

        if not activate_reward_model:
            return seq

        start_idx = 0
        T = len(seq) - start_idx

        if isinstance(seq, list):
            # seq: list of (action, obs, reward, term, trunc, info, next_info)
            actions = torch.from_numpy(np.stack([elem[0] for elem in seq])).to(
                self.device
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
            obs = {key: torch.from_numpy(np.stack(val)) for key, val in obs.items()}
            # obs: (T, elem_shape) for elem in obs
            # actions: (T, action_shape)

            list_of_info_dicts = [elem[-2] for elem in seq]
            reward_terms = {key: [] for key in self.reward_space.keys()}
            for info_dict in list_of_info_dicts:
                for key in reward_terms:
                    reward_terms[key].append(info_dict[key])
            reward_terms = {
                key: torch.from_numpy(np.stack(val))
                for key, val in reward_terms.items()
            }
            # reward_terms: (T, num_reward_terms)
        elif isinstance(seq, dict):
            assert _obs_signature is not None, "Need obs_signature for dict input."
            # print("action length", len(seq["action"]))
            T = len(seq["action"]) - start_idx
            actions = torch.from_numpy(seq["action"]).to(self.device)
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
            reward_terms = {
                key: torch.from_numpy(seq[key]) for key in self.reward_space.keys()
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
                _reward_weights = self.weight_tuner(
                    qpos[_range] if qpos is not None else None,
                    fused_rgb_feats[_range] if fused_rgb_feats is not None else None,
                    actions[_range],
                    time_obs[_range] if time_obs is not None else None,
                )
                weighted_reward = (_reward_weights * reward_terms[_range]).sum(
                    dim=-1, keepdim=True
                )

                _reward = self.markovian(
                    qpos[_range] if qpos is not None else None,
                    fused_rgb_feats[_range] if fused_rgb_feats is not None else None,
                    actions[_range],
                    time_obs[_range] if time_obs is not None else None,
                )

            weighted_rewards.append(weighted_reward)
            computed_rewards.append(_reward)
        weighted_rewards = torch.cat(weighted_rewards, dim=0)
        computed_rewards = torch.cat(computed_rewards, dim=0)

        assert (
            len(weighted_rewards) == T
        ), f"Expected {T} weighted rewards, got {len(weighted_rewards)}"
        assert (
            len(computed_rewards) == T
        ), f"Expected {T} computed rewards, got {len(computed_rewards)}"

        weighted_rewards = weighted_rewards.mean(dim=1).cpu().numpy()
        computed_rewards = computed_rewards.mean(dim=1).cpu().numpy()
        total_rewards = weighted_rewards + self.lambda_weight * computed_rewards
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
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        r_hats_weighted = []
        r_hat_weights = []
        r_hats_computed = []
        fused_rgb_feats = None
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

            time_obs = extract_from_batch(batch, "time", missing_ok=True)

            # Compute weighted reward
            # extract reward terms: (bs, seq, num_reward_terms)
            reward_terms = stack_tensor_dictionary(
                {key: batch[f"seg{i}_{key}"] for key in self.reward_space.keys()},
                dim=-1,
            )
            # r_hat_weight: (bs * seq, num_reward_terms) -> (bs, seq, num_reward_terms)
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
            r_hat_weight = self.weight_tuner(*args).view(
                *actions.shape[:-2], -1, reward_terms.shape[-1]
            )
            # r_hat: (bs, seq, num_reward_terms) -> (bs, seq, 1) -> (bs, 1)
            r_hat = (r_hat_weight * reward_terms).sum(dim=-1, keepdim=True).mean(dim=-2)
            r_hats_weighted.append(r_hat)
            r_hat_weights.append(r_hat_weight)

            # Compute computed reward
            r_hat_computed = self.markovian(*args)
            # r_hat_computed: (bs, seq, 1) -> (bs, 1)
            r_hat_computed = r_hat_computed.view(
                *actions.shape[:-2], -1, r_hat_computed.shape[-1]
            ).mean(dim=-2)
            r_hats_computed.append(r_hat_computed)

        weighted_loss_dict = self.weight_tuner.calculate_loss(
            r_hats_weighted, batch["label"], r_hat_weights
        )
        computed_loss_dict = self.markovian.calculate_loss(
            r_hats_computed, batch["label"]
        )

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
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()
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
            r_hat_weights = torch.cat(r_hat_weights, dim=0)
            for idx, term in enumerate(self.reward_space):
                metrics[f"r_hat_weights_{term.split('/')[-1]}"] = (
                    r_hat_weights[..., idx].mean().item()
                )
            for label in range(self.num_label):
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

        self._i += 1
        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
