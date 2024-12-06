import logging
from functools import partial
from typing import Callable, Sequence

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

from robobase.envs.env import EnvFactory
from robobase.reward_method.core import RewardMethod
from robobase.rlhf_module.comparison import get_comparison_fn
from robobase.rlhf_module.feedback import get_feedback_fn
from robobase.rlhf_module.prompt import (
    get_zeroshot_video_evaluation_prompt,
    get_zeroshot_subtask_identification_prompt,
)
from robobase.rlhf_module.third_party.gemini import (
    load_gemini_model,
    get_gemini_video_ids,
)
from robobase.rlhf_module.utils import retry_on_error

"""
General function to collect preferences (LLM vs non-LLM)
"""


def collect_basic_preferences(
    segments: Sequence,
    num_queries: int,
    comparison_fn: object,
    feedback_fn: Callable,
    feedback_iter: int,
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


def collect_gemini_manipulation_preferences(
    segments: Sequence,
    num_queries: int,
    comparison_fn: object,
    feedback_fn: Callable,
    gemini_model_config: DictConfig,
    general_criteria: str,
    task_description: str,
    subtasks: str,
    video_path: Path,
    feedback_iter: int,
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
    def identify_subtasks(videos):
        gemini_model = load_gemini_model(gemini_model_config)
        quest = get_zeroshot_subtask_identification_prompt(
            task_description=task_description,
            subtasks=subtasks,
            videos=videos,
            viewpoints=target_viewpoints,
        )
        return gemini_model.generate_content(quest).text

    identified_subtasks = {}
    feedbacks = []
    total_metadata = []

    for i in tqdm(tot_queries, desc="Collecting preferences", position=0, leave=False):
        pair = comparison_fn(i)

        video1 = get_gemini_video_ids(
            segments, pair[0], target_viewpoints, video_path, feedback_iter, i, 0
        )
        video2 = get_gemini_video_ids(
            segments, pair[1], target_viewpoints, video_path, feedback_iter, i, 1
        )

        video_evaluation1 = identify_subtasks(video1)
        video_evaluation2 = identify_subtasks(video2)

        label, metadata = feedback_fn(
            video1=video1,
            video2=video2,
            video_evaluation1=video_evaluation1,
            video_evaluation2=video_evaluation2,
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


def collect_gemini_locomotion_preferences(
    segments: Sequence,
    num_queries: int,
    comparison_fn: object,
    feedback_fn: Callable,
    gemini_model_config: DictConfig,
    task_description: str,
    video_path: Path,
    feedback_iter: int,
):
    target_viewpoints = gemini_model_config.target_viewpoints
    tot_queries = range(num_queries)
    logging.info("START!")
    comparison_fn.initialize(segments)
    # 1. Evaluate videos.

    @retry_on_error(
        10,
        sleep_time=0,
        callback_fn=lambda *_: "[Failure from Gemini API] Video evaluation failed.",
    )
    def evaluate_videos(videos):
        gemini_model = load_gemini_model(gemini_model_config)
        quest = get_zeroshot_video_evaluation_prompt(
            task_description=task_description,
            videos=videos,
        )
        return gemini_model.generate_content(quest).text

    feedbacks = []
    total_metadata = []

    for i in tqdm(tot_queries, desc="Collecting preferences", position=0, leave=False):
        pair = comparison_fn(i)

        video1 = get_gemini_video_ids(
            segments, pair[0], target_viewpoints, video_path, feedback_iter, i, 0
        )
        video2 = get_gemini_video_ids(
            segments, pair[1], target_viewpoints, video_path, feedback_iter, i, 1
        )

        video_evaluation1 = evaluate_videos(video1)
        video_evaluation2 = evaluate_videos(video2)
        label, metadata = feedback_fn(
            video1=video1,
            video2=video2,
            video_evaluation1=video_evaluation1,
            video_evaluation2=video_evaluation2,
            gemini_model_config=gemini_model_config,
            task_description=task_description,
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
    work_dir: Path, cfg: DictConfig, env_factory: EnvFactory, reward_model: RewardMethod
):
    comparison_fn = get_comparison_fn(cfg.rlhf.comparison_type, reward_model)
    feedback_fn = get_feedback_fn(cfg.env.env_name, cfg.rlhf.feedback_type)

    match cfg.rlhf.feedback_type:
        case "gemini":
            task_description = env_factory.get_task_description(cfg)
            assert task_description is not None, "Task description is not provided."
            gemini_model_config = cfg.rlhf.gemini
            video_path = work_dir / "videos"
            video_path.mkdir(parents=True, exist_ok=True)
            if cfg.env.env_name == "agym":
                general_criteria = env_factory.get_general_criteria(cfg)
                subtasks = env_factory.get_subtask_list(cfg)
                return partial(
                    collect_gemini_manipulation_preferences,
                    num_queries=cfg.rlhf_replay.num_queries,
                    comparison_fn=comparison_fn,
                    feedback_fn=feedback_fn,
                    gemini_model_config=gemini_model_config,
                    task_description=task_description,
                    general_criteria=general_criteria,
                    subtasks=subtasks,
                    video_path=video_path,
                )
            elif cfg.env.env_name == "dmc":
                return partial(
                    collect_gemini_locomotion_preferences,
                    num_queries=cfg.rlhf_replay.num_queries,
                    comparison_fn=comparison_fn,
                    feedback_fn=feedback_fn,
                    gemini_model_config=gemini_model_config,
                    task_description=task_description,
                    video_path=video_path,
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
