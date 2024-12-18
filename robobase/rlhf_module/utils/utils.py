import logging
import time
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def return_random_label(*_, **kwargs):
    return np.random.choice([0, 1])


def retry_on_error(times, callback_fn=lambda *_: None, sleep_time=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Function execution failed. Retrying... {e}")
                    pass
                time.sleep(sleep_time)
            logging.warning(f"Function execution failed after {times} attempts, ")
            return callback_fn(*args, **kwargs)

        return wrapper

    return decorator


def timeout_callback(max_time=20):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import signal

            def handler(signum, frame):
                raise TimeoutError(
                    f"Function execution timed out after {max_time} seconds"
                )

            # Set signal handler
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(max_time)  # Trigger alarm in max_time seconds

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable alarm
            return result

        return wrapper

    return decorator


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


def get_label(ans):
    if ans not in ["a", "d"]:
        print("Invalid option.")
        return None
    if ans == "a":
        return 0
    elif ans == "d":
        return 1


def check_valid_pair(segments, pair):
    index_1, index_2 = segments["indices"][pair[0]], segments["indices"][pair[1]]
    ep_num_1, ep_num_2 = (
        segments["episode_number"][pair[0]],
        segments["episode_number"][pair[1]],
    )
    return index_1 != index_2 or ep_num_1 != ep_num_2
