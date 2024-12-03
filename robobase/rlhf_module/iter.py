import logging
from functools import partial
from typing import Callable, Sequence

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm, trange

from robobase.envs.env import EnvFactory
from robobase.reward_method.core import RewardMethod
from robobase.rlhf_module.comparison import get_comparison_fn
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
    segments: Sequence, num_queries: int, comparison_fn: object, feedback_fn: Callable
):
    tot_queries = range(num_queries)
    logging.info("START!")
    comparison_fn.initialize(segments)

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

    return feedbacks, None


def collect_gemini_preferences(
    segments: Sequence,
    num_queries: int,
    comparison_fn: object,
    feedback_fn: Callable,
    gemini_model_config: DictConfig,
    general_criteria: str,
    task_description: str,
    subtasks: str,
):
    target_viewpoints = gemini_model_config.target_viewpoints
    tot_queries = range(num_queries)
    logging.info("START!")
    comparison_fn.initialize(segments)
    # 1. Identify subtasks for each video.

    @retry_on_error(
        10,
        sleep_time=0,
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
    for idx in trange(
        num_queries, desc="Identifying subtasks", position=0, leave=False
    ):
        # Ensure we don't exceed 2 requests per second to Gemini
        identified_subtasks[idx] = identify_subtasks(idx)

    feedbacks = []
    total_metadata = []
    for i in tqdm(tot_queries, desc="Collecting preferences", position=0, leave=False):
        pair = comparison_fn(i)
        label, metadata = feedback_fn(
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
        total_metadata.append(metadata)
    logging.info("FINISH!")

    return feedbacks, total_metadata


def get_rlhf_iter_fn(
    cfg: DictConfig, env_factory: EnvFactory, reward_model: RewardMethod
):
    comparison_fn = get_comparison_fn(cfg.rlhf.comparison_type, reward_model)
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
                num_queries=cfg.rlhf_replay.num_queries,
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
                num_queries=cfg.rlhf_replay.num_queries,
                comparison_fn=comparison_fn,
                feedback_fn=feedback_fn,
            )
        case _:
            raise ValueError(
                "Invalid feedback type. Please choose between 'random' or 'script' or 'human' or 'gemini'."
            )
