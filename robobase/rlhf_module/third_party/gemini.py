import os
import time

import imageio
from google import genai


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def load_gemini_model(cfg, system_instruction=None):
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
        system_instruction=system_instruction,
    )
    return model


def upload_video_to_genai(client, video_path, verbose=False):
    video_file = client.files.upload(file=video_path)
    while video_file.state == "PROCESSING":
        print("Waiting for video to be processed.")
        time.sleep(1.0)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    print("Video processing complete: " + video_file.uri)
    return video_file
    # video_file = genai.upload_file(path=video_path)
    # while video_file.state.name == "PROCESSING":
    #     if verbose:
    #         logging.info("Waiting for video to be processed.")
    # if video_file.state.name == "FAILED":
    #     raise ValueError(video_file.state.name)
    # if verbose:
    #     logging.info("Video processing complete: " + video_file.uri)
    # return video_file


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
    client, segments, idx, target_viewpoints, video_path, feedback_iter, i, j
):
    output = {}
    for viewpoint in target_viewpoints:
        assert (
            f"query_pixels_{viewpoint}" in segments
        ), "query_pixels_{viewpoint} not found in segments"
        index = segments["indices"][idx]
        video_file_path = (
            video_path
            / f"query_pixels-{viewpoint}-feedback_iter{feedback_iter}-pair{i}_{j}-idx{segments['global_steps'][idx]}-ep{segments['episode_number'][idx]}-timestep_{index}_{index + segments['action'].shape[1]}.mp4"  # noqa
        )
        imageio.mimsave(
            video_file_path, segments[f"query_pixels_{viewpoint}"][idx], fps=20
        )
        gemini_video_file_path = upload_video_to_genai(
            client, video_file_path, verbose=False
        )
        output[viewpoint] = gemini_video_file_path
    return output
