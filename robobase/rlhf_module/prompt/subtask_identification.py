# ruff: noqa
# ZERO-SHOT VERSION
zeroshot_subtask_infer_prompt = """
You are an AI model who is responsible for determining the behavior of the robot using videos from {viewpoint} viewpoint(s).

The goal of the agent is as follows:
{task_description}

To achieve this goal, the agent should solve the subtasks below:
{subtasks}

Based on the video, identify which subtask from the list above the robot is currently executing.
Then, evaluate whether the task is successfully achieved.

As an output, use the format below:

Subtask 1: <Success/Success but not perfect/Failure/WIP>

- Reason: <Reason>

Subtask 2: <Not started/Success/Success but not perfect/Failure/WIP>

- Reason: <Reason>

...

Please consider the following instructions:

- Provide clear, specific reasons for each evaluation.
- Be consistent to assess movements and task completion.
- Pay attention to all segments of the video, especially the later part.
- Strictly follow the format and do not include any unnecessary information in your response.
- Consider all camera angles when evaluating, and ensure your evaluation takes into account the complete perspective of the scene.
"""


def get_zeroshot_video_evaluation_prompt(
    task_description, subtasks, videos, viewpoints
):
    # Generate separate prompts for each viewpoint
    prompt = [
        zeroshot_subtask_infer_prompt.format(
            task_description=task_description.strip(),
            subtasks=subtasks.strip(),
            viewpoint="/".join(viewpoints),
        ).strip()
    ]
    # videos: {viewpoint: video for viewpoint in viewpoints}
    prompt.append("<VIDEO START>")
    for viewpoint, video in videos.items():
        prompt.append(f"FROM {viewpoint.upper()} VIEWPOINT:")
        prompt.append(video)
    prompt.append("<VIDEO END>")
    return prompt
