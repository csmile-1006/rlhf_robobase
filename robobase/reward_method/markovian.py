import logging
from copy import deepcopy
from typing import Optional, Sequence, Tuple

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


class InvalidSequenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MarkovianRewardModel(nn.Module):
    def __init__(
        self,
        reward_model: FullyConnectedModule,
        num_reward_models: int = 1,
        apply_final_layer_tanh: bool = False,
    ):
        super().__init__()
        self.rs = nn.ModuleList(
            [deepcopy(reward_model) for _ in range(num_reward_models)]
        )
        self.apply(utils.weight_init)
        self.apply_final_layer_tanh = apply_final_layer_tanh

    def forward(self, low_dim_obs, fused_view_feats, action, time_obs):
        net_ins = {"action": action.view(action.shape[0], -1)}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs

        reward_outs = torch.cat(
            [reward(net_ins) for reward in self.rs],
            -1,
        )
        if self.apply_final_layer_tanh:
            reward_outs = torch.tanh(reward_outs)
        return reward_outs

    def reset(self, env_index: int):
        self.reward_model.reset(env_index)

    def set_eval_env_running(self, value: bool):
        self.reward_model.eval_env_running = value

    def calculate_loss(
        self,
        input_feats: Tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """
        Calculate the loss for the MultiViewTransformerEncoderDecoderACT model.

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
        reward_hat_1, reward_hat_2 = input_feats
        logits = torch.stack([reward_hat_1, reward_hat_2], dim=-1)

        loss_dict = dict(loss=0.0)
        reward_loss = 0.0
        for idx, (logit, label) in enumerate(zip(logits.unbind(1), labels.unbind(1))):
            label_stack = torch.stack([1 - label, label], dim=-1)
            reward_loss += F.cross_entropy(logit, label_stack)
            loss_dict[f"pref_acc_label_{idx}"] = utils.pref_accuracy(
                logit, torch.argmax(label_stack, dim=-1)
            )
            loss_dict[f"pref_loss_{idx}"] = reward_loss
            loss_dict["loss"] += reward_loss

        loss_dict["loss"] /= len(labels)
        return loss_dict


class MarkovianReward(RewardMethod):
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
        num_label: int = 1,
        num_reward_models: int = 1,
        seq_len: int = 50,
        compute_batch_size: int = 32,
        use_augmentation: bool = False,
        reward_space: gym.spaces.Dict = None,
        apply_final_layer_tanh: bool = False,
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
        self.reward_space = gym.spaces.Dict(sorted(reward_space.items()))

        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.num_label = num_label
        self.num_reward_models = num_reward_models
        self.seq_len = seq_len
        self.apply_final_layer_tanh = apply_final_layer_tanh
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
        self.reward = MarkovianRewardModel(
            reward_model=reward_model,
            num_reward_models=self.num_reward_models,
            apply_final_layer_tanh=self.apply_final_layer_tanh,
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
        self, seq: Sequence, activate_reward_model: bool = True
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

        if len(seq) < self.seq_len:
            raise InvalidSequenceError(
                f"Input sequence must be at least {self.seq_len_steps} steps long.\
                Seq len is {len(seq)}"
            )

        # seq: list of (action, obs, reward, term, trunc, info, next_info)
        actions = torch.from_numpy(np.stack([elem[0] for elem in seq])).to(self.device)
        if seq[0][1]["low_dim_state"].ndim > 1:
            list_of_obs_dicts = [{k: v[-1] for k, v in elem[1].items()} for elem in seq]
        else:
            list_of_obs_dicts = [elem[1] for elem in seq]

        obs = {key: [] for key in list_of_obs_dicts[0].keys()}
        for obs_dict in list_of_obs_dicts:
            for key, val in obs_dict.items():
                obs[key].append(val)
        obs = {key: torch.from_numpy(np.stack(val)) for key, val in obs.items()}
        # obs: (T, elem_shape) for elem in obs
        # actions: (T, action_shape)

        rgbs = (
            stack_tensor_dictionary(extract_many_from_batch(obs, r"rgb(?!.*?tp1)"), 1)
            .unsqueeze(1)
            .to(self.device)
        )
        fused_rgb_feats = self.encode_rgb_feats(rgbs, train=False).squeeze(1)
        qpos = extract_from_batch(obs, "low_dim_state").to(self.device)
        time_obs = extract_from_batch(obs, "time", missing_ok=True)

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
                _reward = self.reward(
                    qpos[_range],
                    fused_rgb_feats[_range],
                    actions[_range],
                    time_obs[_range] if time_obs is not None else None,
                )
            rewards.append(_reward)

        rewards = torch.cat(rewards, dim=0)
        assert len(rewards) == T, f"Expected {T} rewards, got {len(rewards)}"

        total_rewards = rewards.mean(dim=1).cpu().numpy()
        for idx in range(len(seq)):
            seq[idx][2] = total_rewards[idx]

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

        r_hats = []
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

            time_obs = extract_from_batch(batch, "time", missing_ok=True)
            r_hat_segment = self.reward(
                qpos.reshape(-1, *qpos.shape[2:]),
                fused_rgb_feats.reshape(-1, *fused_rgb_feats.shape[2:])
                if fused_rgb_feats is not None
                else None,
                actions.reshape(-1, *actions.shape[2:]),
                time_obs.reshape(-1, *time_obs.shape[2:])
                if time_obs is not None
                else None,
            )
            r_hat = r_hat_segment.view(
                *actions.shape[:-2], -1, r_hat_segment.shape[-1]
            ).mean(dim=-2)
            r_hats.append(r_hat)

        loss_dict = self.reward.calculate_loss(r_hats, batch["label"])

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
            for label in range(self.num_label):
                metrics[f"pref_acc_label_{label}"] = loss_dict[
                    f"pref_acc_label_{label}"
                ].item()

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.
