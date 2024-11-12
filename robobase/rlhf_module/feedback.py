import logging

import numpy as np
from IPython.display import HTML, clear_output, display

from robobase.rlhf_module.prompt import pairwise_comparison_prompt
from robobase.rlhf_module.third_party.gemini import (
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


def collect_human_feedback(segments, indices, **kwargs):
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


def collect_random_feedback(*_, **kwargs):
    random_label = np.random.choice([0, 1])
    return random_label


def collect_script_feedback(segments, indices, **kwargs):
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
def collect_gemini_feedback(
    segments,
    indices,
    gemini_model,
    general_criteria,
    task_description,
    target_viewpoints,
    subtasks,
    identified_subtasks,
):
    # Collect feedbacks for pair of videos.
    response = gemini_model.generate_content(
        [
            pairwise_comparison_prompt.format(
                task_description=task_description,
                subtasks=subtasks,
                viewpoint_order="/".join(target_viewpoints),
                general_criteria=general_criteria,
                identified_subtasks=identified_subtasks,
                video1_subtask=identified_subtasks[indices[0]],
                video2_subtask=identified_subtasks[indices[1]],
            ),
            *get_gemini_video_ids(segments, indices[0], target_viewpoints),
            *get_gemini_video_ids(segments, indices[1], target_viewpoints),
        ]
    )
    label = postprocess_gemini_response(response)
    return label


def get_feedback_fn(feedback_type):
    if feedback_type == "random":
        return collect_random_feedback
    elif feedback_type == "script":
        return collect_script_feedback
    elif feedback_type == "human":
        return collect_human_feedback
    elif feedback_type == "gemini":
        return collect_gemini_feedback
    else:
        raise ValueError(
            "Invalid feedback type. Please choose between 'random' or 'script' or 'human'."
        )
