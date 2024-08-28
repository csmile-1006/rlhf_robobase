import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import google.generativeai as genai
import numpy as np
from tqdm import tqdm

from robobase.rlhf_module.prompt import (
    humanoid_criteria_generation_prompt,
    subtask_generation_prompt,
)

from robobase.rlhf_module.utils import retry_on_error


def load_model(cfg):
    api_key = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_output_tokens": cfg.max_output_tokens,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel(
        model_name=cfg.model_type,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    return model


@retry_on_error(10)
def load_video_to_genai(video_path):
    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        logging.info("Waiting for video to be processed.")
        time.sleep(1)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    logging.info("Video processing complete: " + video_file.uri)
    return video_file


def save_segments_local(batch, video_path, fps=30):
    """
    output: {idx: ["{ts}_head_query_{idx}.mp4", "{ts}_right_wrist_query_{idx}.mp4", ...], ...}
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    indices = len(list(batch.values())[0])
    idx2filenames = defaultdict(list)

    for idx in range(indices):
        image_seq = {
            key: batch[key][idx]
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .astype(np.uint8)
            for key in batch
            if "rgb" in key
        }
        for key in image_seq:
            save_key = key.replace("rgb_", "")
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            filename = f"{ts}_{save_key}_query_{idx}.mp4"
            out = cv2.VideoWriter(video_path / filename, fourcc, fps, (84, 84))
            for i in range(image_seq[key].shape[0]):
                out.write(cv2.cvtColor(image_seq[key][i], cv2.COLOR_RGB2BGR))
            out.release()
            idx2filenames[idx].append(filename)
    return idx2filenames


def preprocess_video_gemini(segments, video_path, target_viewpoints):
    """
    output: {
        idx: [
            genai.File(name="{ts}_head_query_{idx}.mp4", ...),
            ...
            ],
        ...}
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    idx2link = defaultdict(list)
    idx2filenames = save_segments_local(segments, video_path)
    for idx, filenames in tqdm(
        idx2filenames.items(),
        total=len(idx2filenames),
        desc="Uploading videos to GenAI",
        position=0,
        leave=False,
    ):
        for filename in filenames:
            if any([viewpoint in filename for viewpoint in target_viewpoints]):
                link = load_video_to_genai(video_path=video_path / filename)
                idx2link[idx].append(link)
    return idx2link


def get_general_criteria(model):
    return model.generate_content([humanoid_criteria_generation_prompt])


def get_subtask_lists(model, task_description):
    return model.generate_content(
        [subtask_generation_prompt.format(task_description=task_description)]
    )


def postprocess_gemini_response(response):
    """
    Response format:
    <Answer>: Video 1
    """
    text = response.text
    try:
        stripped_text = text[:17]
        postprocessed_text = int(stripped_text.split(":")[1].strip().split(" ")[-1])
        return postprocessed_text
    except Exception as e:
        print(f"Error in postprocessing: {text} / {e}")
        raise
