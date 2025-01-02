import torch

from robobase import utils
from robobase.intrinsic_reward_module.core import IntrinsicRewardModule
from robobase.intrinsic_reward_module.re3 import PBE


class VCSEModel(object):
    def __init__(self, knn_k):
        self.knn_k = knn_k

    def __call__(self, state, value):
        # value => [b1 , 1]
        # state => [b1 , c]
        # z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(
            source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
            dim=-1,
            p=2,
        )

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(
            source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
            dim=-1,
            p=2,
        )

        sim_matrix = torch.max(
            torch.cat((sim_matrix_s.unsqueeze(-1), sim_matrix_v.unsqueeze(-1)), dim=-1),
            dim=-1,
        )[0]
        eps, index = sim_matrix.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)

        state_norm, index = sim_matrix_s.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)

        value_norm, index = sim_matrix_v.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)

        eps = eps[:, -1]  # k-th nearest distance
        eps = eps.reshape(-1, 1)  # (b1, 1)

        state_norm = state_norm[:, -1]  # k-th nearest distance
        state_norm = state_norm.reshape(-1, 1)  # (b1, 1)

        value_norm = value_norm[:, -1]  # k-th nearest distance
        value_norm = value_norm.reshape(-1, 1)  # (b1, 1)

        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v, dim=1, keepdim=True)  # (b1,1)

        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s, dim=1, keepdim=True)  # (b1,1)

        reward = torch.special.digamma(n_v + 1) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v, n_s, eps, state_norm, value_norm


class VCSE(IntrinsicRewardModule):
    """.

    Younggyo, Seo, et al. "State Entropy Maximization with Random Encoders for Efficient Exploration"
    https://arxiv.org/pdf/2102.09430

    If pixels are used, then any low-dimensional input will not be passed into RND.
    All pixel observations are concatenated on channel axis, and assumed same shape.
    """

    def __init__(
        self, latent_dim: int = 50, knn_k: int = 3, beta=0.1, *args, **kwargs
    ) -> None:
        """Init.

        Args:
            latent_dim: The dimension of encoding vectors.
            lr: The learning rate.
        """
        super().__init__(*args, **kwargs)
        assert not self.use_pixels, "VCSE does not support pixels"
        self.se = PBE(knn_k=knn_k)
        self.state_ent_stats = utils.TorchRunningMeanStd(shape=(1,), device=self.device)
        self.vcse = VCSEModel(knn_k=knn_k)
        self._step = 0
        self.beta = beta

    def compute_irs(
        self, batch: dict[str, torch.Tensor], value: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """See Base."""
        # compute the weighting coefficient of timestep t
        beta_t = self.beta
        obs = self._extract_obs(batch, r"rgb.*" if self.use_pixels else "low_dim_state")
        feats = obs
        intrinsic_rewards = self.vcse(feats, value)[0].reshape(-1, 1)
        self.state_ent_stats.update(intrinsic_rewards)
        return intrinsic_rewards * beta_t

    def compute_unsup_irs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """In unsup training, we don't have value, so we use the state entropy directly."""
        obs = self._extract_obs(batch, r"rgb.*" if self.use_pixels else "low_dim_state")
        feats = obs
        intrinsic_rewards = self.se(feats).reshape(-1, 1)
        self.state_ent_stats.update(intrinsic_rewards)
        intrinsic_rewards = intrinsic_rewards / self.state_ent_stats.mean
        return intrinsic_rewards

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        # there's no need for update, since the target is frozen.
        self._step += 1
