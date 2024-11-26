import logging
from copy import deepcopy
from typing import Optional, Sequence, Tuple, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
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
    def __init__(
        self,
        reward_model: FullyConnectedModule,
        num_reward_terms: int = 1,
        reward_lows: Optional[np.ndarray] = None,
        reward_highs: Optional[np.ndarray] = None,
        reg_weight: float = 0.0,
    ):
        super().__init__()
        self.ws = nn.ModuleList(
            [deepcopy(reward_model) for _ in range(num_reward_terms)]
        )
        self.apply(utils.weight_init)
        if reward_lows is None:
            reward_lows = torch.full(num_reward_terms, -np.inf)
        if reward_highs is None:
            reward_highs = torch.full(num_reward_terms, np.inf)

        self.reward_lows = reward_lows
        self.reward_highs = reward_highs
        self.reg_weight = reg_weight

    def forward(self, low_dim_obs, fused_view_feats, action, time_obs):
        net_ins = {"action": action.view(action.shape[0], -1)}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs

        weights = torch.cat(
            [weight_model(net_ins) for weight_model in self.ws],
            -1,
        )
        return torch.clamp(weights, self.reward_lows, self.reward_highs)

    def reset(self, env_index: int):
        self.reward_model.reset(env_index)

    def set_eval_env_running(self, value: bool):
        self.reward_model.eval_env_running = value

    def calculate_loss(
        self,
        input_feats: Tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        r_hat_weights: torch.Tensor,
    ) -> Optional[dict]:
        """
        Calculate the loss for the WeightRewardModel model.

        Args:
            input_feats (Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]):
                    Tuple containing action predictions, padding predictions,
                    and a list of latent variables [mu, logvar].
            labels (torch.Tensor): Tensor containing ground truth preference labels.
            is_pad (torch.Tensor): Tensor indicating padding positions.

        Returns:
            Optional[Tuple[torch.Tensor, dict]]:
                    Tuple containing the loss tensor and a dictionary of loss
                    components.
        """
        logits = torch.stack(input_feats, dim=-1)
        r_hat_weights = torch.stack(r_hat_weights, dim=-1)

        loss_dict = {"loss": 0.0}
        reward_loss = 0.0
        for idx, (logit, label) in enumerate(zip(logits.unbind(1), labels.unbind(1))):
            label_stack = torch.stack([1 - label, label], dim=-1)
            reward_loss += F.cross_entropy(logit, label_stack)
            loss_dict[f"pref_acc_label_{idx}"] = utils.pref_accuracy(
                logit, torch.argmax(label_stack, dim=-1)
            )
            loss_dict[f"pref_loss_{idx}"] = reward_loss
            loss_dict["loss"] += reward_loss

        logit_reg_loss = torch.mean(torch.square(r_hat_weights))  # L2 regularization
        loss_dict[f"logit_reg_loss_{idx}"] = logit_reg_loss.item()
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
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        reg_weight: float = 0.0,
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

        if isinstance(reward_space, gym.spaces.Dict):
            self.reward_space = reward_space
        else:
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
        reward_model = self.reward_model(input_shapes=input_shapes)
        self.reward = WeightRewardModel(
            reward_model=reward_model,
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
        _obs_signature: Dict[str, gym.Space] = None,
        activate_reward_model: bool = False,
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
            for idx, info_dict in enumerate(list_of_info_dicts):
                if idx == 0 and not info_dict:
                    for key in reward_terms:
                        reward_terms[key].append(0.0)
                else:
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
                _reward_weights = self.reward(
                    qpos[_range] if qpos is not None else None,
                    fused_rgb_feats[_range] if fused_rgb_feats is not None else None,
                    actions[_range],
                    time_obs[_range] if time_obs is not None else None,
                )
                _reward = (_reward_weights * reward_terms[_range]).sum(
                    dim=-1, keepdim=True
                )
            rewards.append(_reward)
        rewards = torch.cat(rewards, dim=0)

        assert len(rewards) == T, f"Expected {T} rewards, got {len(rewards)}"

        total_rewards = rewards.mean(dim=1).cpu().numpy()
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
        # elem = {key: val.shape for key, val in batch.items()}
        # print(elem)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # actions = batch["action"]
        # reward = batch["reward"]

        r_hats = []
        r_hat_weights = []
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
            r_hat = (r_hat_weight * reward_terms).sum(dim=-1, keepdim=True).mean(dim=-2)
            r_hats.append(r_hat)
            r_hat_weights.append(r_hat_weight)

        loss_dict = self.reward.calculate_loss(r_hats, batch["label"], r_hat_weights)

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

        self._i += 1
        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
