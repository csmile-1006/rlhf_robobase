# ruff: noqa
# ZERO-SHOT VERSION
zeroshot_video_evaluation_prompt = """
You are an AI model who is responsible for determining the behavior of the robot using videos.

The goal of the agent is as follows:
{task_description}

Based on the video, evaluate whether the task is successfully achieved.

As an output, use the format below:

<Task achievement>: <Description of the video from start to end frame, be detailed and specific, whether the task is successfully achieved>
<Naturalness>: <Assessment of the robot's behavior, whether the robot is moving smoothly and naturally, be detailed and specific>

Please consider the following instructions:

- Provide clear, specific reasons for each evaluation.
- Make each evaluation to be independent of others.
- Be consistent to assess movements and task completion.
- Pay attention to all segments of the video, especially the later part.
- Strictly follow the format and do not include any unnecessary information in your response.
- Don't be too harsh or too lenient, assume that the robot has innate differences with human.
- Consider all camera angles when evaluating, and ensure your evaluation takes into account the complete perspective of the scene.
"""


def get_zeroshot_video_evaluation_prompt(task_description, videos):
    # Generate separate prompts for each viewpoint
    prompt = [
        zeroshot_video_evaluation_prompt.format(
            task_description=task_description.strip(),
        ).strip()
    ]
    # videos: {viewpoint: video for viewpoint in viewpoints}
    prompt.append("<VIDEO START>")
    for viewpoint, video in videos.items():
        prompt.append(f"FROM {viewpoint.upper()} VIEWPOINT:")
        prompt.append(video)
    prompt.append("<VIDEO END>")
    return prompt
