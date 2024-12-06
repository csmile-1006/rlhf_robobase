import logging
import os
import time

import google.generativeai as genai
import imageio

from robobase.rlhf_module.utils import retry_on_error


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)


def load_gemini_model(cfg):
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


@retry_on_error(
    10, callback_fn=lambda *_: ValueError("Failed to upload video to Gemini")
)
def upload_video_to_genai(video_path, verbose=False):
    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        if verbose:
            logging.info("Waiting for video to be processed.")
        time.sleep(1)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    if verbose:
        logging.info("Video processing complete: " + video_file.uri)
    return video_file


def postprocess_gemini_response(response):
    """
    Response format:
    <Answer>: Video 1
    """
    text = response.text
    try:
        stripped_text = text[:17]
        postprocessed_index = int(stripped_text.split(":")[1].strip().split(" ")[-1])
        return postprocessed_index - 1
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return -1


def get_gemini_video_ids(
    segments, idx, target_viewpoints, video_path, feedback_iter, i, j
):
    output = {}
    for viewpoint in target_viewpoints:
        assert (
            f"query_video_{viewpoint}" in segments
        ), "query_video_{viewpoint} not found in segments"
        index = segments["indices"][idx]
        video_file_path = (
            video_path
            / f"query_video-{viewpoint}-feedback_iter{feedback_iter}-pair{i}_{j}-timestep_{index}_{index + segments['action'].shape[1]}.mp4"  # noqa
        )
        imageio.mimsave(
            video_file_path, segments[f"query_video_{viewpoint}"][idx], fps=20
        )
        gemini_video_file_path = upload_video_to_genai(video_file_path, verbose=False)
        output[viewpoint] = gemini_video_file_path
    return output
