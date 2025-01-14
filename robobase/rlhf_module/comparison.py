from abc import ABC, abstractmethod
import logging

import numpy as np
from omegaconf import DictConfig
from robobase.reward_method.core import RewardMethod

"""
How to compare pairwise preferences (sequential, sequential_pairwise, root_pairwise)
"""


class ComparisonFn(ABC):
    def initialize(self, segments):
        self._i = 0
        large_batch_size = len(segments[list(segments.keys())[0]])
        half_size = large_batch_size // 2
        self.half_size = half_size
        self.indices = [(i, i + half_size) for i in range(half_size)]

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def increment(self):
        if self._i >= self.half_size:
            logging.warning(
                "Resetting index for comparison function. This case must not be happened."
            )
            self._i = 0
        self._i += 1

    def update(self, pair, label):
        pass


class SequentialComparisonFn(ComparisonFn):
    def __call__(self):
        return self.indices[self._i]


class SequentialPairwiseComparisonFn(ComparisonFn):
    def initialize(self, segments):
        self._i = 0
        large_batch_size = len(segments[list(segments.keys())[0]])
        self.indices = [(i, i + 1) for i in range(large_batch_size - 1)]

    def __call__(self):
        return self.indices[self._i]


class RootPairwiseComparisonFn(ComparisonFn):
    def initialize(self, segments):
        super(RootPairwiseComparisonFn, self).initialize(segments)
        self.best_choice = 0

    def __call__(self):
        return np.random.permutation([self._i, self.best_choice])

    def update(self, pair, label):
        if self.best_choice != pair[label]:
            self.best_choice = pair[label]


class DisagreementComparisonFn(ComparisonFn):
    def __init__(self, reward_model: RewardMethod):
        self.reward_model = reward_model

    def initialize(self, segments):
        super(DisagreementComparisonFn, self).initialize(segments)
        if not self.reward_model.activated:
            # If reward model is not activated, we use the default sequential comparison
            self.top_k_index = np.arange(self.half_size)

        else:
            # If reward model is activated, we use the disagreement comparison
            # Split segments into two batches
            x_1 = {k: v[: self.half_size] for k, v in segments.items()}
            x_2 = {k: v[self.half_size :] for k, v in segments.items()}
            self.indices = [(i, i + self.half_size) for i in range(self.half_size)]

            _, disagree = self.reward_model.get_rank_probability(x_1, x_2)
            self.top_k_index = (-disagree).argsort()

    def __call__(self):
        return self.indices[self.top_k_index[self._i]]


class MajorColumnComparisonFn(ComparisonFn):
    def __init__(self, major_column: str):
        self.major_column = major_column

    def initialize(self, segments):
        super(MajorColumnComparisonFn, self).initialize(segments)
        major_column_returns = segments[f"Reward/{self.major_column}"].sum(axis=-1)
        # Sort indices by major_column_returns
        sorted_indices = np.argsort(major_column_returns)
        # remove bottom 30% with particulary smaller major_column_returns
        sorted_indices = sorted_indices[: int(len(sorted_indices) * 0.7)]

        # make pairs with similar major_column_returns by randomly selecting from 10 nearest neighbors
        self.indices = []
        window_size = 10  # Compare with one of 10 nearest neighbors
        for i in range(len(sorted_indices) - 1):
            # Determine valid window range (don't go past array bounds)
            max_window = min(window_size, len(sorted_indices) - i - 1)
            # Randomly select one index from the window to pair with
            j = i + 1 + np.random.randint(max_window)
            self.indices.append((sorted_indices[i], sorted_indices[j]))
        # shuffle self.indices not to be biased to larger major_column_returns
        np.random.shuffle(self.indices)

    def __call__(self):
        return self.indices[self._i]


def get_comparison_fn(cfg: DictConfig, reward_model: RewardMethod):
    comparison_type = cfg.rlhf.comparison_type
    match comparison_type:
        case "sequential":
            return SequentialComparisonFn()
        case "sequential_pairwise":
            return SequentialPairwiseComparisonFn()
        case "root_pairwise":
            return RootPairwiseComparisonFn()
        case "disagreement":
            return DisagreementComparisonFn(reward_model)
        case "major_column":
            major_column = cfg.rlhf.major_column
            return MajorColumnComparisonFn(major_column)
        case _:
            raise ValueError(
                f"Unknown comparison type: {comparison_type}, please choose between 'sequential',"
                "'sequential_pairwise', 'root_pairwise', 'disagreement', 'major_column'."
            )
