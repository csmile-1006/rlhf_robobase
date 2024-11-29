import logging

import numpy as np
from IPython.display import HTML, clear_output, display

from robobase.rlhf_module.prompt import get_zeroshot_pairwise_comparison_prompt
from robobase.rlhf_module.third_party.gemini import (
    load_gemini_model,
    postprocess_gemini_response,
    get_gemini_video_ids,
)
from robobase.rlhf_module.utils import (
    get_label,
    get_video_embed,
    preprocess_video,
    retry_on_error,
    return_random_label,
)

"""
How to collect feedbacks
(real human feedback from jupyter notebook, random feedback, script feedback)
"""


def human_feedback_fn(segments, indices, **kwargs):
    index, len_tot_queries = kwargs["index"], kwargs["len_tot_queries"]
    camera_keys = [key for key in segments.keys() if "rgb" in key]
    anim = get_video_embed(
        preprocess_video(segments, idxs=indices, camera_keys=camera_keys),
        num_pairs=len(indices),
        num_cameras=len(camera_keys),
    )
    html_video = anim.to_html5_video()
    html_video_with_autoplay = html_video.replace("<video ", "<video autoplay muted ")

    while True:
        clear_output(True)
        display(HTML(html_video_with_autoplay))
        choice = input(
            f"[{index}/{len_tot_queries}] Put Preference (a (left), d (right), quit(quit)):  "
        ).strip()
        label = get_label(choice)
        if label is not None:
            logging.info(f"Is this your answer?: {label}")
            out = input("next=1 / again=2 ")
            try:
                out = int(out)
            except Exception:
                continue
            if out == 1:
                return label
        else:
            print("Invalid input. Please try again.")
    clear_output(True)


def random_feedback_fn(*_, **kwargs):
    random_label = np.random.choice([0, 1])
    return random_label


def scripted_feedback_fn(segments, indices, **kwargs):
    segment_return_1 = segments["reward"][indices[0]].sum(dim=-1)
    segment_return_2 = segments["reward"][indices[1]].sum(dim=-1)

    if segment_return_1 > segment_return_2:
        script_label = 0
    elif segment_return_1 == segment_return_2:
        script_label = np.random.choice([0, 1])
    else:
        script_label = 1

    return script_label


@retry_on_error(10, callback_fn=return_random_label)
def gemini_feedback_fn(
    segments,
    indices,
    gemini_model_config,
    general_criteria,
    task_description,
    target_viewpoints,
    subtasks,
    identified_subtasks,
):
    # Collect feedbacks for pair of videos.
    video1 = get_gemini_video_ids(segments, indices[0], target_viewpoints)
    video_evaluation1 = identified_subtasks[indices[0]]
    video2 = get_gemini_video_ids(segments, indices[1], target_viewpoints)
    video_evaluation2 = identified_subtasks[indices[1]]
    quest = get_zeroshot_pairwise_comparison_prompt(
        general_criteria=general_criteria,
        task_description=task_description,
        subtasks=subtasks,
        viewpoints=target_viewpoints,
        video1=video1,
        video2=video2,
        video1_evaluations=video_evaluation1,
        video2_evaluations=video_evaluation2,
    )
    gemini_model = load_gemini_model(gemini_model_config)
    response = gemini_model.generate_content(quest)
    label = postprocess_gemini_response(response)
    metadata = {
        "video1": video1,
        "video2": video2,
        "video_evaluation1": video_evaluation1,
        "video_evaluation2": video_evaluation2,
        "quest": quest,
        "response": response,
    }
    return label, metadata


def get_feedback_fn(feedback_type):
    if feedback_type == "random":
        return random_feedback_fn
    elif feedback_type == "script":
        return scripted_feedback_fn
    elif feedback_type == "human":
        return human_feedback_fn
    elif feedback_type == "gemini":
        return gemini_feedback_fn
    else:
        raise ValueError(
            "Invalid feedback type. Please choose between 'random' or 'script' or 'human'."
        )
