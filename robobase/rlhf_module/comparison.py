import logging
from typing import Callable, Sequence

import numpy as np


"""
How to compare pairwise preferences (sequential, sequential_pairwise, root_pairwise)
"""


def collect_sequential_preferences(segments: Sequence, feedback_fn: Callable):
    n_queries = len(segments["action"])
    assert n_queries % 2 == 0, "The number of queries must be even."
    tot_queries = range(n_queries // 2)
    logging.info("START!")

    feedbacks = []
    for i in tot_queries:
        pair = [i, 2 * i]
        label = feedback_fn(segments, pair, i, tot_queries)

        pref_dict = {
            "segment_0": {
                key: np.asarray(segments[key][pair[0]]) for key in segments.keys()
            },
            "segment_1": {
                key: np.asarray(segments[key][pair[1]]) for key in segments.keys()
            },
            "label": np.asarray(label)[np.newaxis],
        }
        feedbacks.append(pref_dict)
    logging.info("FINISH!")

    return feedbacks


def collect_sequential_pairwise_preferences(segments: Sequence, feedback_fn: Callable):
    n_queries = len(segments["action"])
    tot_queries = range(1, n_queries)
    logging.info("START!")

    feedbacks = []
    for i in tot_queries:
        # Required to shuffle the previous best choice for removing position bias.
        shuffled_pair = np.random.permutation([i, i - 1])
        label = feedback_fn(segments, shuffled_pair, i, tot_queries)

        pref_dict = {
            "segment_0": {
                key: np.asarray(segments[key][shuffled_pair[0]])
                for key in segments.keys()
            },
            "segment_1": {
                key: np.asarray(segments[key][shuffled_pair[1]])
                for key in segments.keys()
            },
            "label": np.asarray(label)[np.newaxis],
        }
        feedbacks.append(pref_dict)
    logging.info("FINISH!")

    return feedbacks


def collect_root_pairwise_preferences(segments: Sequence, feedback_fn: Callable):
    n_queries = len(segments["action"])
    tot_queries = range(1, n_queries)
    logging.info("START!")

    best_choice = 0
    feedbacks = []
    for i in tot_queries:
        # Required to shuffle the previous best choice for removing position bias.
        shuffled_pair = np.random.permutation([i, best_choice])
        index_to_shuffled_pair = {0: shuffled_pair[0], 1: shuffled_pair[1]}

        label = feedback_fn(segments, shuffled_pair, i, tot_queries)

        if best_choice != index_to_shuffled_pair[label]:
            best_choice = index_to_shuffled_pair[label]
        pref_dict = {
            "segment_0": {
                key: np.asarray(segments[key][shuffled_pair[0]])
                for key in segments.keys()
            },
            "segment_1": {
                key: np.asarray(segments[key][shuffled_pair[1]])
                for key in segments.keys()
            },
            "label": np.asarray(label)[np.newaxis],
        }
        feedbacks.append(pref_dict)
    logging.info("FINISH!")

    return feedbacks


def get_comparison_fn(comparison_type):
    if comparison_type == "sequential":
        return collect_sequential_preferences
    elif comparison_type == "sequential_pairwise":
        return collect_sequential_pairwise_preferences
    elif comparison_type == "root_pairwise":
        return collect_root_pairwise_preferences
    else:
        raise ValueError(
            "Invalid preference type. Please choose between\
            'sequential' or 'sequential_pairwise' or 'root_pairwise'."
        )
