import logging
from functools import partial
from typing import Callable, Sequence

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm, trange

from robobase.rlhf_module.comparison import get_comparison_fn, SequentialComparisonFn
from robobase.rlhf_module.feedback import get_feedback_fn
from robobase.rlhf_module.prompt import subtask_identification_prompt
from robobase.rlhf_module.third_party.gemini import (
    get_general_criteria,
    get_subtask_lists,
    load_model,
    preprocess_video_gemini,
)
from robobase.rlhf_module.utils import retry_on_error

"""
General function to collect preferences (LLM vs non-LLM)
"""


def collect_preferences(
    segments: Sequence, comparison_fn: object, feedback_fn: Callable
):
    n_queries = len(segments["action"])
    if isinstance(comparison_fn, SequentialComparisonFn):
        assert n_queries % 2 == 0, "The number of queries must be even."
        tot_queries = range(n_queries // 2)
    tot_queries = range(1, n_queries)
    logging.info("START!")

    feedbacks = []
    for i in tot_queries:
        pair = comparison_fn(i)
        label = feedback_fn(segments, pair, index=i, len_tot_queries=len(tot_queries))
        comparison_fn.update(pair, label)

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


def collect_gemini_preferences(
    segments: Sequence,
    comparison_fn: object,
    feedback_fn: Callable,
    gemini_model: Callable,
    cfg: DictConfig,
    general_criteria: str,
    task_description: str,
    subtasks: str,
):
    (
        output_path,
        target_viewpoints,
    ) = (cfg.rlhf.gemini.output_path, cfg.rlhf.gemini.target_viewpoints)
    if gemini_model is None:
        raise ValueError("Gemini model is not provided.")

    idx2link = preprocess_video_gemini(
        segments, output_path, target_viewpoints=target_viewpoints
    )

    n_queries = len(segments["action"])
    if isinstance(comparison_fn, SequentialComparisonFn):
        assert n_queries % 2 == 0, "The number of queries must be even."
        tot_queries = range(n_queries // 2)
    tot_queries = range(1, n_queries)
    logging.info("START!")
    # 1. Identify subtasks for each video.

    @retry_on_error(10)
    def identify_subtasks(idx):
        return gemini_model.generate_content(
            [
                subtask_identification_prompt.format(
                    task_description=task_description, subtasks=subtasks
                ),
                *idx2link[idx],
            ]
        )

    identified_subtasks = {}
    for idx in trange(
        len(idx2link), desc="Identifying subtasks", position=0, leave=False
    ):
        identified_subtasks[idx] = identify_subtasks(idx)

    feedbacks = []
    for i in tqdm(tot_queries, desc="Collecting preferences", position=0, leave=False):
        pair = comparison_fn(i)
        label = feedback_fn(
            idx2link,
            pair,
            gemini_model=gemini_model,
            general_criteria=general_criteria,
            task_description=task_description,
            target_viewpoints=target_viewpoints,
            subtasks=subtasks,
            identified_subtasks=identified_subtasks,
        )
        comparison_fn.update(pair, label)

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


def get_rlhf_iter_fn(cfg: DictConfig):
    comparison_fn = get_comparison_fn(cfg.rlhf.comparison_type)
    feedback_fn = get_feedback_fn(cfg.rlhf.feedback_type)

    match cfg.rlhf.feedback_type:
        case "gemini":
            task_description = cfg.rlhf.gemini.task_description
            assert task_description is not None, "Task description is not provided."
            gemini_model = load_model(cfg.rlhf.gemini)
            general_criteria = get_general_criteria(gemini_model)
            subtasks = get_subtask_lists(
                gemini_model, task_description=task_description
            )
            return partial(
                collect_gemini_preferences,
                comparison_fn=comparison_fn,
                feedback_fn=feedback_fn,
                gemini_model=gemini_model,
                cfg=cfg,
                task_description=task_description,
                general_criteria=general_criteria,
                subtasks=subtasks,
            )
        case "human" | "random" | "script":
            return partial(
                collect_preferences,
                comparison_fn=comparison_fn,
                feedback_fn=feedback_fn,
            )
        case _:
            raise ValueError(
                "Invalid feedback type. Please choose between 'random' or 'script' or 'human' or 'gemini'."
            )
