import re
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dm_env import specs


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class TimeConsistentRandomShiftsAug(nn.Module):
    def __init__(self, pad: int):
        """
        Applies random shift augmentation to videos of shape [B, T, C, H, W].
        Augmentations are differently applied to different videos,
        but same augmentations are applied to the frames within the video.

        Args:
            pad (int): Size of padding for augmentation
        """
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): [B, T, C, H, W]-shaped torch tensor

        Returns:
            x (torch.Tensor): [B, T, C, H, W]-shaped torch tensor
        """
        B, T, C, H, W = x.size()
        assert H == W
        padding = tuple([self.pad] * 4)
        # NOTE: Padding behaves differently from RandomShiftsAug!
        x = F.pad(x, padding, mode="constant", value=0)
        eps = 1.0 / (H + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, H + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:H]
        arange = arange.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(B, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (H + 2 * self.pad)  # [B, 1, 1, 2]

        grid = base_grid + shift
        # Repeats the same grid for T times for time-consistent augmentation
        grid = grid.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, H, W, 2)

        x = x.view(B * T, C, H + 2 * self.pad, W + 2 * self.pad)
        out = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        out = out.view(B, T, C, H, W)
        return out


def flatten_time_dim_into_channel_dim(
    tensor: torch.Tensor, has_view_axis: bool = False
):
    if has_view_axis:
        bs, v, t, ch = tensor.shape[:4]
        return tensor.view(bs, v, t * ch, *tensor.shape[4:])
    bs, t, ch = tensor.shape[:3]
    return tensor.view(bs, t * ch, *tensor.shape[3:])


def merge_tensor_dictionary(tensor_dict: Dict[str, torch.Tensor], dim: int):
    return torch.cat(list(tensor_dict.values()), dim)


def stack_tensor_dictionary(tensor_dict: Dict[str, torch.Tensor], dim: int):
    return torch.stack(list(tensor_dict.values()), dim)


def extract_from_batch(
    batch: Dict[str, torch.Tensor], key: str, missing_ok: bool = False
):
    if key not in batch:
        if missing_ok:
            return None
        raise ValueError(
            f"Couldn't find '{key}' in the batch. "
            f"Available keys are: {list(batch.keys())}"
        )
    return batch[key]


def extract_from_spec(spec: gym.spaces.Dict, key, missing_ok: bool = False):
    if key not in list(spec.keys()):
        if missing_ok:
            return None
        raise ValueError(
            f"Couldn't find '{key}' in the space. "
            f"Available keys are: {list(spec.keys())}"
        )
    return spec[key]


def extract_many_from_spec(
    spec: gym.spaces.Dict, pattern: str, missing_ok: bool = False
):
    filtered_dict = {}
    regex = re.compile(pattern)
    for key, value in spec.items():
        if regex.search(key):
            filtered_dict[key] = value
    if len(filtered_dict) == 0 and not missing_ok:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the space. "
            f"'Available keys are: {list(spec.keys())}"
        )
    return filtered_dict


def extract_many_from_batch(batch, pattern: str):
    filtered_dict = {}
    regex = re.compile(pattern)
    for key, value in batch.items():
        if regex.search(key):
            filtered_dict[key] = value
    if len(filtered_dict) == 0:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the batch. "
            f"'Available keys are: {list(batch.keys())}"
        )
    return filtered_dict


def loss_weights(replay_sample, beta=1.0):
    if "sampling_probabilities" in replay_sample:
        probs = replay_sample["sampling_probabilities"]
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
        return loss_weights
    else:
        return torch.ones(replay_sample["action"].shape[0]).to(
            replay_sample["action"].device
        )


def random_action_if_within_delta(qs, delta=0.0001):
    q_diff = qs.max(-1).values - qs.min(-1).values
    random_action_mask = q_diff < delta
    argmax_q = qs.max(-1)[1]
    random_actions = torch.randint(
        0, qs.size(-1), random_action_mask.shape, device=qs.device
    )
    argmax_q = torch.where(random_action_mask, random_actions, argmax_q)
    return argmax_q


def encode_action(
    continuous_action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    levels: int,
    bins: int,
):
    """Encode continuous action to discrete action

    Args:
        continuous_action (torch.Tensor): [..., D] shape tensor
    Returns:
        torch.Tensor: [..., L, D] shape tensor where L is the level
    """
    low = low.repeat(*continuous_action.shape[:-1], 1)
    high = high.repeat(*continuous_action.shape[:-1], 1)

    idxs = []
    for _ in range(levels):
        # Put continuous values into bin
        slice_range = (high - low) / bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, bins - 1)
        idxs.append(idx)

        # Re-compute low/high for each bin (i.e., Zoom-in)
        recalculated_action = low + slice_range * idx
        recalculated_action = torch.clip(recalculated_action, -1.0, 1.0)
        low = recalculated_action
        high = recalculated_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    discrete_action = torch.stack(idxs, -2)
    return discrete_action


def decode_action(
    discrete_action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    levels: int,
    bins: int,
):
    """Decode discrete action to continuous action

    Args:
        discrete_action (torch.Tensor): [..., L, D] shape tensor
    Returns:
        torch.Tensor: [..., D] shape continuous action tensor
    """
    low = low.repeat(*discrete_action.shape[:-2], 1)
    high = high.repeat(*discrete_action.shape[:-2], 1)
    for i in range(levels):
        slice_range = (high - low) / bins
        continuous_action = low + slice_range * discrete_action[..., i, :]
        low = continuous_action
        high = continuous_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    continuous_action = (high + low) / 2.0
    return continuous_action


def zoom_in(low: torch.Tensor, high: torch.Tensor, argmax_q: torch.Tensor, bins: int):
    """Zoom-in to the selected interval

    Args:
        low: [D] shape tensor that denotes minimum of the current interval
        high: [D] shape tensor that denotes maximum of the current interval
    Returns:
        low: [D] shape tensor that denotes minimum of the *next* interval
        high: [D] shape tensor that denotes maximum of the *next* interval
    """
    slice_range = (high - low) / bins
    continuous_action = low + slice_range * argmax_q
    low = continuous_action
    high = continuous_action + slice_range
    low = torch.maximum(-torch.ones_like(low), low)
    high = torch.minimum(torch.ones_like(high), high)
    return low, high


class TemporalEnsembleControl:
    def __init__(
        self,
        episode_length: int,
        action_spec: specs.BoundedArray,
        action_sequence: int,
        gain: float = 0.01,
    ):
        self._episode_length = episode_length
        self._action_dtype = action_spec.dtype
        self._action_dim = action_spec.shape[0]
        self._action_sequence = action_sequence
        self._gain = gain
        self._init_action_history()

    def _init_action_history(self):
        self._action_history = np.zeros(
            [
                self._episode_length,
                self._episode_length + self._action_sequence,
                self._action_dim,
            ],
            self._action_dtype,
        )
        self._cur_step = 0

    def register_action_sequence(self, action_sequence):
        self._action_history[self._cur_step, self._cur_step : self._cur_step + self._action_sequence] = action_sequence

    def get_action(self):
        cur_actions = self._action_history[:, self._cur_step]
        indices = np.all(cur_actions != 0, axis=1)
        cur_actions = cur_actions[indices]

        # earlier predicted actions will have smaller weights.
        exp_weights = np.exp(-self._gain * np.arange(len(cur_actions), dtype=self._action_dtype))
        exp_weights = (exp_weights / exp_weights.sum())[:, None]
        action = (cur_actions * exp_weights).sum(axis=0)
        self._cur_step += 1
        return action

    def reset(self):
        self._init_action_history()


class TrainTemporalEnsembleControl(TemporalEnsembleControl):
    def __init__(
        self,
        num_train_envs: int,
        episode_length: int,
        action_spec: specs.BoundedArray,
        action_sequence: int,
        gain: float = 0.01,
    ):
        super().__init__(episode_length, action_spec, action_sequence, gain)
        self._num_train_envs = num_train_envs
        self._init_action_history()

    def _init_action_history(self):
        self._action_history = np.zeros(
            [
                self._num_train_envs,
                self._episode_length,
                self._episode_length + self._action_sequence,
                self._action_dim,
            ],
            self._action_dtype,
        )
        self._cur_step = 0

    def register_action_sequence(self, action_sequence):
        self._action_history[self._cur_step, self._cur_step : self._cur_step + self._action_sequence] = action_sequence

    def get_action(self):
        cur_actions = self._action_history[:, self._cur_step]
        indices = np.all(cur_actions != 0, axis=1)
        cur_actions = cur_actions[indices]

        # earlier predicted actions will have smaller weights.
        exp_weights = np.exp(-self._gain * np.arange(len(cur_actions), dtype=self._action_dtype))
        exp_weights = (exp_weights / exp_weights.sum())[:, None]
        action = (cur_actions * exp_weights).sum(axis=0)
        self._cur_step += 1
        return action

    def reset(self):
        self._init_action_history()
