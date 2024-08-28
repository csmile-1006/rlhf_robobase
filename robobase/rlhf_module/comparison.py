from abc import ABC, abstractmethod

import numpy as np

"""
How to compare pairwise preferences (sequential, sequential_pairwise, root_pairwise)
"""


class ComparisonFn(ABC):
    @abstractmethod
    def __call__(self, i):
        pass

    def __str__(self):
        return self.__class__.__name__

    def update(self, pair, label):
        return pair


class SequentialComparisonFn(ComparisonFn):
    def __call__(self, i):
        return [i, 2 * i]


class SequentialPairwiseComparisonFn(ComparisonFn):
    def __call__(self, i):
        # return np.random.permutation([i, i - 1])
        return [i - 1, i]


class RootPairwiseComparisonFn(ComparisonFn):
    def __init__(self):
        self.best_choice = 0

    def __call__(self, i):
        return np.random.permutation([i, self.best_choice])

    def update(self, pair, label):
        if self.best_choice != pair[label]:
            self.best_choice = pair[label]


def get_comparison_fn(comparison_type):
    match comparison_type:
        case "sequential":
            return SequentialComparisonFn()
        case "sequential_pairwise":
            return SequentialPairwiseComparisonFn()
        case "root_pairwise":
            return RootPairwiseComparisonFn()
        case _:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
