# ruff: noqa
# ZERO_SHOT VERSION FOR MANIPULATION TASK
zeroshot_manipulation_pairwise_comparison_prompt_header = """
You are an AI model who is responsible for determining which agent is better in terms of achieving the goal and friendly to human.

The goal of the agent is as follows:
{task_description}

To achieve this goal, the agent should solve subtasks below:
{subtasks}

As a robot interacting with human subject, the agent should move smoothly and human-friendly by following criteria below:

<CRITERIA START>
{general_criteria}
<CRITERIA END>

We will show you two agents from {viewpoint} viewpoint.
Moreover, we will show you the evaluation of each agent from {viewpoint} viewpoint.
"""

zeroshot_pairwise_comparison_prompt_footer = """
Using all information, determine which agent is better in terms of solving the task and being human-friendly.

As an output, use the format below:

<Answer>: <chosen one between Agent 1/Agent 2/equally preferred>
Reason: <Reason for the choice>

Please follow the instructions below:
- Please explain detailed and specific reasons as possible.
- Do not include any unnecessary information in your response.
- Consider all viewpoints when making a decision and ensure your evaluation takes into account the complete perspective of the scene.
- Compare the behaviors from different angles to get a comprehensive understanding.
- If different viewpoints provide conflicting information, explain how you weighted and reconciled the different perspectives.
"""

# Additionally, you are provided with the evaluation of each agent from different viewpoints.
# - Note that 'Failure' is better than 'Not started', as 'Failure' means the agent has made an attempt to complete the task.


def get_zeroshot_manipulation_pairwise_comparison_prompt(
    general_criteria,
    task_description,
    subtasks,
    viewpoints,
    video1,
    video1_evaluations,
    video2,
    video2_evaluations,
):
    prompt = [
        zeroshot_manipulation_pairwise_comparison_prompt_header.format(
            task_description=task_description,
            viewpoint="/".join(viewpoints),
            subtasks=subtasks,
            general_criteria=general_criteria,
        )
    ]

    prompt.append("<AGENT 1 START>")
    for viewpoint, video in video1.items():
        prompt.append(f"VIDEO FROM {viewpoint.upper()} VIEWPOINT:")
        prompt.append(video)
    prompt.append("<AGENT 1 END>")

    prompt.append("<AGENT 2 START>")
    for viewpoint, video in video2.items():
        prompt.append(f"VIDEO FROM {viewpoint.upper()} VIEWPOINT:")
        prompt.append(video)
    prompt.append("<AGENT 2 END>")

    video1_eval_text = ""
    # Add headers for each viewpoint's evaluation for video 1
    video1_eval_text += "\n\n"
    video1_eval_text += (
        f"<EVALUATION FOR AGENT 1 FROM {'/'.join(viewpoints).upper()} VIEWPOINT>\n"
    )
    video1_eval_text += video1_evaluations.replace("\n", "\n    ")

    video2_eval_text = ""
    video2_eval_text += "\n\n"
    video2_eval_text += (
        f"<EVALUATION FOR AGENT 2 FROM {'/'.join(viewpoints).upper()} VIEWPOINT>\n"
    )
    video2_eval_text += video2_evaluations.replace("\n", "\n    ")
    prompt.append(
        zeroshot_pairwise_comparison_prompt_footer.format(
            viewpoints="/".join(viewpoints),
            video1_evaluations=video1_eval_text,
            video2_evaluations=video2_eval_text,
        )
    )

    return prompt


# ZERO_SHOT VERSION FOR LOCOMOTION TASK
zeroshot_locomotion_pairwise_comparison_prompt_header = """
You are an AI model who is responsible for determining which agent is better in terms of achieving the goal and friendly to human.

The goal of the agent is as follows:
{task_description}

We will show you videos of two different agents along with their evaluations.
"""


def get_zeroshot_locomotion_pairwise_comparison_prompt(
    task_description,
    video1,
    video1_evaluations,
    video2,
    video2_evaluations,
):
    prompt = [
        zeroshot_locomotion_pairwise_comparison_prompt_header.format(
            task_description=task_description
        )
    ]

    prompt.append("<AGENT 1 START>")
    for idx, video in enumerate(video1.values()):
        prompt.append(f"VIDEO from viewpoint {idx + 1}:")
        prompt.append(video)
    prompt.append("<AGENT 1 END>")

    prompt.append("<AGENT 2 START>")
    for idx, video in enumerate(video2.values()):
        prompt.append(f"VIDEO from viewpoint {idx + 1}:")
        prompt.append(video)
    prompt.append("<AGENT 2 END>")

    video1_eval_text = ""
    video1_eval_text += "\n\n"
    video1_eval_text += f"<EVALUATION FOR AGENT 1>\n"
    video1_eval_text += video1_evaluations.replace("\n", "\n    ")

    video2_eval_text = ""
    video2_eval_text += "\n\n"
    video2_eval_text += f"<EVALUATION FOR AGENT 2>\n"
    video2_eval_text += video2_evaluations.replace("\n", "\n    ")
    prompt.append(zeroshot_pairwise_comparison_prompt_footer)

    return prompt
