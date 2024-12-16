import asyncio
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from robobase.envs.env import EnvFactory
from robobase.reward_method.core import RewardMethod
from robobase.rlhf_module.comparison import get_comparison_fn
from robobase.rlhf_module.feedback import get_feedback_fn
from robobase.rlhf_module.prompt import (
    get_zeroshot_locomotion_pairwise_comparison_prompt,
    get_zeroshot_manipulation_pairwise_comparison_prompt,
    get_zeroshot_subtask_identification_prompt,
    get_zeroshot_video_evaluation_prompt,
)
from robobase.rlhf_module.third_party.gemini import (
    get_gemini_video_ids,
    load_gemini_model,
    postprocess_gemini_response,
)

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


# 1. evaluate videos.
async def _identify_subtask_manipulation_videos(
    videos,
    gemini_model_config,
    task_description,
    subtasks,
    viewpoints,
    general_criteria,
):
    gemini_model = load_gemini_model(gemini_model_config)
    quest = get_zeroshot_subtask_identification_prompt(
        task_description=task_description,
        videos=videos,
        subtasks=subtasks,
        viewpoints=viewpoints,
        general_criteria=general_criteria,
    )
    response = await gemini_model.generate_content_async(quest)
    return response.text


# 2. get feedback.
async def _get_manipulation_feedback(
    video1,
    video2,
    gemini_model_config,
    task_description,
    subtasks,
    viewpoints,
    general_criteria,
):
    gemini_model = load_gemini_model(gemini_model_config)
    video_evaluation1 = await _identify_subtask_manipulation_videos(
        video1,
        gemini_model_config,
        task_description,
        subtasks,
        viewpoints,
        general_criteria,
    )
    video_evaluation2 = await _identify_subtask_manipulation_videos(
        video2,
        gemini_model_config,
        task_description,
        subtasks,
        viewpoints,
        general_criteria,
    )
    quest = get_zeroshot_manipulation_pairwise_comparison_prompt(
        subtasks=subtasks,
        viewpoints=viewpoints,
        general_criteria=general_criteria,
        task_description=task_description,
        video1=video1,
        video1_evaluations=video_evaluation1,
        video2=video2,
        video2_evaluations=video_evaluation2,
    )
    response = await gemini_model.generate_content_async(quest)
    return response, quest, video_evaluation1, video_evaluation2


# 3. collect feedback using gemini.
async def _collect_manipulation_feedback(
    videos,
    gemini_model_config,
    task_description,
    subtasks,
    viewpoints,
    general_criteria,
):
    responses = await asyncio.gather(
        *[
            _get_manipulation_feedback(
                video1,
                video2,
                gemini_model_config,
                task_description,
                subtasks,
                viewpoints,
                general_criteria,
            )
            for video1, video2 in videos
        ]
    )
    return responses


async def collect_gemini_manipulation_preferences(
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

    feedbacks = []
    total_metadata = []

    # upload videos in linear way
    pair_indices = []
    videos = []
    for i in tqdm(tot_queries, desc="Uploading videos", position=0, leave=False):
        pair = comparison_fn(i)
        video1 = get_gemini_video_ids(
            segments, pair[0], target_viewpoints, video_path, feedback_iter, i, 0
        )
        video2 = get_gemini_video_ids(
            segments, pair[1], target_viewpoints, video_path, feedback_iter, i, 1
        )
        pair_indices.append(pair)
        videos.append(video1)
        videos.append(video2)
        comparison_fn.update(pair, i)

    videos = [(videos[i], videos[i + 1]) for i in range(0, len(videos), 2)]
    responses = await _collect_manipulation_feedback(
        videos,
        gemini_model_config,
        task_description,
        subtasks,
        target_viewpoints,
        general_criteria,
    )
    results = []
    for pair, (response, quest, video_evaluation1, video_evaluation2) in zip(
        pair_indices, responses
    ):
        label = postprocess_gemini_response(response)
        pref_dict = {
            "segment_0": {
                key: np.asarray(segments[key][pair[0]]) for key in segments.keys()
            },
            "segment_1": {
                key: np.asarray(segments[key][pair[1]]) for key in segments.keys()
            },
            "label": np.asarray(label)[np.newaxis],
        }
        results.append(
            (response, label, pref_dict, quest, video_evaluation1, video_evaluation2)
        )

    for (video1, video2), (
        response,
        label,
        pref_dict,
        quest,
        video_evaluation1,
        video_evaluation2,
    ) in zip(videos, results):
        metadata = {
            "response": response,
            "video1": video1,
            "video2": video2,
            "video_evaluation1": video_evaluation1,
            "video_evaluation2": video_evaluation2,
        }
        feedbacks.append(pref_dict)
        total_metadata.append(metadata)
    logging.info("FINISH!")

    return feedbacks, total_metadata


# 1. evaluate videos.
async def _evaluate_locomotion_videos(videos, gemini_model_config, task_description):
    gemini_model = load_gemini_model(gemini_model_config)
    quest = get_zeroshot_video_evaluation_prompt(
        task_description=task_description,
        videos=videos,
    )
    response = await gemini_model.generate_content_async(quest)
    return response.text


# 2. get feedback.
async def _get_locomotion_feedback(
    video1, video2, gemini_model_config, task_description
):
    gemini_model = load_gemini_model(gemini_model_config)
    video_evaluation1 = await _evaluate_locomotion_videos(
        video1, gemini_model_config, task_description
    )
    video_evaluation2 = await _evaluate_locomotion_videos(
        video2, gemini_model_config, task_description
    )
    quest = get_zeroshot_locomotion_pairwise_comparison_prompt(
        task_description=task_description,
        video1=video1,
        video1_evaluations=video_evaluation1,
        video2=video2,
        video2_evaluations=video_evaluation2,
    )
    response = await gemini_model.generate_content_async(quest)
    return response, quest, video_evaluation1, video_evaluation2


# 3. collect feedback using gemini.
async def _collect_locomotion_feedback(videos, gemini_model_config, task_description):
    responses = await asyncio.gather(
        *[
            _get_locomotion_feedback(
                video1, video2, gemini_model_config, task_description
            )
            for video1, video2 in videos
        ]
    )
    return responses


async def collect_gemini_locomotion_preferences(
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

    feedbacks = []
    total_metadata = []

    # upload videos in linear way
    pair_indices = []
    videos = []
    for i in tqdm(tot_queries, desc="Uploading videos", position=0, leave=False):
        pair = comparison_fn(i)
        video1 = get_gemini_video_ids(
            segments, pair[0], target_viewpoints, video_path, feedback_iter, i, 0
        )
        video2 = get_gemini_video_ids(
            segments, pair[1], target_viewpoints, video_path, feedback_iter, i, 1
        )
        pair_indices.append(pair)
        videos.append(video1)
        videos.append(video2)
        comparison_fn.update(pair, i)

    videos = [(videos[i], videos[i + 1]) for i in range(0, len(videos), 2)]
    responses = await _collect_locomotion_feedback(
        videos, gemini_model_config, task_description
    )
    results = []
    for pair, (response, quest, video_evaluation1, video_evaluation2) in zip(
        pair_indices, responses
    ):
        label = postprocess_gemini_response(response)
        pref_dict = {
            "segment_0": {
                key: np.asarray(segments[key][pair[0]]) for key in segments.keys()
            },
            "segment_1": {
                key: np.asarray(segments[key][pair[1]]) for key in segments.keys()
            },
            "label": np.asarray(label)[np.newaxis],
        }
        results.append(
            (response, label, pref_dict, quest, video_evaluation1, video_evaluation2)
        )

    for (video1, video2), (
        response,
        label,
        pref_dict,
        quest,
        video_evaluation1,
        video_evaluation2,
    ) in zip(videos, results):
        metadata = {
            "response": response.text,
            # "quest": quest,
            "video_evaluation1": video_evaluation1,
            "video_evaluation2": video_evaluation2,
            "label": label,
        }
        metadata.update(
            {f"video1_{key}": val.display_name for key, val in video1.items()}
        )
        metadata.update(
            {f"video2_{key}": val.display_name for key, val in video2.items()}
        )
        feedbacks.append(pref_dict)
        total_metadata.append(metadata)
    logging.info("FINISH!")

    if gemini_model_config.compute_self_consistency:
        # compute self consistency with different temperatures
        sc_gemini_model_config = deepcopy(gemini_model_config)
        sc_gemini_model_config.temperature = (
            gemini_model_config.self_consistency_temperature
        )
        # Create multiple copies of videos for self-consistency evaluation
        num_original_videos = len(videos)
        num_samples = gemini_model_config.n_self_consistency_samples

        # Duplicate each video pair num_samples times
        self_consistency_videos = videos * num_samples

        # Get feedback for all duplicated videos
        self_consistency_responses = await _collect_locomotion_feedback(
            self_consistency_videos, sc_gemini_model_config, task_description
        )

        # Group responses by original video pair
        # e.g. if we have 2 video pairs and 3 samples:
        # [v1_s1, v1_s2, v1_s3, v2_s1, v2_s2, v2_s3] -> [[v1_s1, v1_s2, v1_s3], [v2_s1, v2_s2, v2_s3]]
        sc_responses = [
            self_consistency_responses[i::num_original_videos]
            for i in range(num_original_videos)
        ]

        sc_metadata = []
        for i in tot_queries:
            target_elem = total_metadata[i]
            for j in range(gemini_model_config.n_self_consistency_samples):
                response, quest, video_evaluation1, video_evaluation2 = sc_responses[i][
                    j
                ]
                label = postprocess_gemini_response(response)
                target_elem.update(
                    {
                        f"sc_{j}_response": response.text,
                        # f"sc_{j}_quest": quest,
                        f"sc_{j}_video_evaluation1": video_evaluation1,
                        f"sc_{j}_video_evaluation2": video_evaluation2,
                        f"sc_{j}_label": label,
                    }
                )
            sc_metadata.append(target_elem)
        total_metadata = sc_metadata

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
            elif cfg.env.env_name in ["dmc", "locomujoco"]:
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
