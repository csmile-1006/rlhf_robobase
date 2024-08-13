import logging
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, clear_output, display
from matplotlib import animation


def preprocess_video(
    subtrajectories: Dict,
    idxs: Sequence,
    camera_keys: Sequence = ("rgb_head", "rgb_right_wrist", "rgb_left_wrist"),
):
    videos = []
    for camera in camera_keys:
        videos.append(
            np.concatenate([subtrajectories[camera][idx] for idx in idxs], axis=-1)
        )
    return np.concatenate(videos, axis=-2).transpose(0, 2, 3, 1)


def get_video_embed(video, num_pairs=2, num_cameras=3):
    # Video shape must be (N, H, W, C)
    fig = plt.figure(figsize=(num_pairs * 2, num_cameras * 2))
    im = plt.imshow(video[0, :, :, :])
    plt.axis("off")

    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=video.shape[0], interval=50
    )
    return anim


def get_label(ans):
    if ans not in ["a", "d"]:
        print("Invalid option.")
        return None
    if ans == "a":
        return 0
    elif ans == "d":
        return 1


"""
How to collect feedbacks
(real human feedback from jupyter notebook, random feedback, script feedback)
"""


def collect_human_feedback(segments, indices, index, tot_queries):
    anim = get_video_embed(preprocess_video(segments, idxs=indices))
    html_video = anim.to_html5_video()
    html_video_with_autoplay = html_video.replace("<video ", "<video autoplay muted ")

    while True:
        clear_output(True)
        display(HTML(html_video_with_autoplay))
        choice = input(
            f"[{index}/{len(tot_queries)}] Put Preference (a (left), d (right), quit(quit)):  "
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


def collect_random_feedback(segments, indices, index, tot_queries):
    random_label = np.random.choice([0, 1])
    print(
        f"[{index}/{len(tot_queries)}] Random Preference on (a (left), d (right), quit(quit)):  {random_label}"
    )
    return random_label


def collect_script_feedback(segments, indices, index, tot_queries):
    segment_return_1 = segments["reward"][indices[0]]
    segment_return_2 = segments["reward"][indices[1]]

    if segment_return_1 > segment_return_2:
        script_label = 0
    elif segment_return_1 == segment_return_2:
        script_label = np.random.choice([0, 1])
    else:
        script_label = 1

    return script_label


def get_feedback_fn(feedback_type):
    if feedback_type == "random":
        return collect_random_feedback
    elif feedback_type == "script":
        return collect_script_feedback
    elif feedback_type == "human":
        return collect_human_feedback
    else:
        raise ValueError(
            "Invalid feedback type. Please choose between 'random' or 'script' or 'human'."
        )
