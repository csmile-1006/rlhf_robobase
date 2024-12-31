import numpy as np
import torch
from torch import nn

from robobase import utils
from robobase.intrinsic_reward_module.core import IntrinsicRewardModule


class PBE(object):
    """particle-based entropy based on knn normalized by running mean"""

    def __init__(self, knn_k):
        self.knn_k = knn_k

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(
            source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
            dim=-1,
            p=2,
        )
        reward, _ = sim_matrix.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward


class Encoder(nn.Module):
    """Encoder for encoding observations."""

    def __init__(self, obs_shape: tuple, latent_dim: int) -> None:
        """Init.

        Args:
            obs_shape: The data shape of observations.
            latent_dim: The dimension of encoding vectors.

        Returns:
            Encoder instance.
        """
        super().__init__()
        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            # no need to introduce random encoder for low-dim state
            self.trunk = nn.Identity()
            self.linear = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input tensors.

        Args:
            obs: Observations.

        Returns:
            Encoding tensors.
        """
        if len(obs.shape) == 4:
            # RGB image
            obs = obs.float() / 255.0 - 0.5
        else:
            obs /= obs.shape[-1]
        return self.linear(self.trunk(obs))


class RE3(IntrinsicRewardModule):
    """.

    Younggyo, Seo, et al. "State Entropy Maximization with Random Encoders for Efficient Exploration"
    https://arxiv.org/pdf/2102.09430

    If pixels are used, then any low-dimensional input will not be passed into RND.
    All pixel observations are concatenated on channel axis, and assumed same shape.
    """

    def __init__(
        self,
        latent_dim: int = 50,
        knn_k: int = 3,
        beta=0.1,
        kappa=0.000025,
        *args,
        **kwargs
    ) -> None:
        """Init.

        Args:
            latent_dim: The dimension of encoding vectors.
            lr: The learning rate.
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.kappa = kappa
        if self.use_pixels:
            obs_shapes = [v.shape for v in self.rgb_spaces.values()]
            # Fuse num views and time into channel axis
            obs_shape = (len(obs_shapes) * np.prod(obs_shapes[0][:2]),) + obs_shapes[0][
                2:
            ]
        else:
            obs_shape = self.low_dim_space.shape[-1:]
        self.target = Encoder(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
        ).to(self.device)
        self.se = PBE(knn_k=knn_k)
        self.state_ent_stats = utils.TorchRunningMeanStd(shape=(1,), device=self.device)
        self._step = 0

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_irs(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """See Base."""
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * (1.0 - self.kappa) ** self._step
        obs = self._extract_obs(batch, r"rgb.*" if self.use_pixels else "low_dim_state")
        with torch.no_grad():
            feats = self.target(obs)
        intrinsic_rewards = self.se(feats).reshape(-1, 1)
        self.state_ent_stats.update(intrinsic_rewards)
        intrinsic_rewards = intrinsic_rewards / self.state_ent_stats.mean
        return intrinsic_rewards * beta_t

    def compute_unsup_irs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """See Base."""
        obs = self._extract_obs(batch, r"rgb.*" if self.use_pixels else "low_dim_state")
        with torch.no_grad():
            feats = self.target(obs)
        intrinsic_rewards = self.se(feats).reshape(-1, 1)
        self.state_ent_stats.update(intrinsic_rewards)
        intrinsic_rewards = intrinsic_rewards / self.state_ent_stats.mean
        return intrinsic_rewards

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        # there's no need for update, since the target is frozen.
        self._step += 1
