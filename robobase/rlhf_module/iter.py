import logging
from functools import partial
from typing import Callable, Sequence

import time
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm, trange

from robobase.envs.env import EnvFactory
from robobase.rlhf_module.comparison import get_comparison_fn, SequentialComparisonFn
from robobase.rlhf_module.feedback import get_feedback_fn
from robobase.rlhf_module.prompt import get_zeroshot_video_evaluation_prompt
from robobase.rlhf_module.third_party.gemini import (
    load_gemini_model,
    get_gemini_video_ids,
)
from robobase.rlhf_module.utils import retry_on_error

"""
General function to collect preferences (LLM vs non-LLM)
"""


def collect_basic_preferences(
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
    gemini_model_config: DictConfig,
    general_criteria: str,
    task_description: str,
    subtasks: str,
):
    target_viewpoints = gemini_model_config.target_viewpoints
    n_queries = len(segments["action"])
    if isinstance(comparison_fn, SequentialComparisonFn):
        assert n_queries % 2 == 0, "The number of queries must be even."
        tot_queries = range(n_queries // 2)
    tot_queries = range(1, n_queries)
    logging.info("START!")
    # 1. Identify subtasks for each video.

    @retry_on_error(
        10,
        sleep_time=1,
        callback_fn=lambda *_: "[Failure from Gemini API] Subtask identification failed.",
    )
    def identify_subtasks(idx):
        gemini_model = load_gemini_model(gemini_model_config)
        quest = get_zeroshot_video_evaluation_prompt(
            task_description=task_description,
            subtasks=subtasks,
            videos=get_gemini_video_ids(segments, idx, target_viewpoints),
            viewpoints=target_viewpoints,
        )
        return gemini_model.generate_content(quest).text

    identified_subtasks = {}
    for idx in trange(n_queries, desc="Identifying subtasks", position=0, leave=False):
        # Ensure we don't exceed 2 requests per second to Gemini
        if idx > 0 and idx % 2 == 0:
            # Sleep for 1 second after every 2 requests
            time.sleep(1)
        identified_subtasks[idx] = identify_subtasks(idx)

    feedbacks = []
    for i in tqdm(tot_queries, desc="Collecting preferences", position=0, leave=False):
        if i > 0 and i % 2 == 0:
            # Sleep for 1 second after every 2 requests
            time.sleep(1)
        pair = comparison_fn(i)
        label = feedback_fn(
            segments,
            pair,
            gemini_model_config=gemini_model_config,
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


def get_rlhf_iter_fn(cfg: DictConfig, env_factory: EnvFactory):
    comparison_fn = get_comparison_fn(cfg.rlhf.comparison_type)
    feedback_fn = get_feedback_fn(cfg.rlhf.feedback_type)

    match cfg.rlhf.feedback_type:
        case "gemini":
            task_description = env_factory.get_task_description(cfg)
            assert task_description is not None, "Task description is not provided."
            gemini_model_config = cfg.rlhf.gemini
            general_criteria = env_factory.get_general_criteria(cfg)
            subtasks = env_factory.get_subtask_list(cfg)
            return partial(
                collect_gemini_preferences,
                comparison_fn=comparison_fn,
                feedback_fn=feedback_fn,
                gemini_model_config=gemini_model_config,
                task_description=task_description,
                general_criteria=general_criteria,
                subtasks=subtasks,
            )
        case "human" | "random" | "script":
            return partial(
                collect_basic_preferences,
                comparison_fn=comparison_fn,
                feedback_fn=feedback_fn,
            )
        case _:
            raise ValueError(
                "Invalid feedback type. Please choose between 'random' or 'script' or 'human' or 'gemini'."
            )
