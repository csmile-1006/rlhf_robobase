# ruff: noqa
pairwise_comparison_prompt = """
You are an AI model who is responsible for determining which video is better by evaluating the behavior of the humanoid robot.

The goal of the agent is as follows:
{task_description}

To achieve this goal, the agent should solve subtasks below:
{subtasks}

As a humanoid robot, the agent should move like a human by following criteria below:

<CRITERIA START>
{general_criteria}
<CRITERIA END>

We will show you two videos from {viewpoint_order} viewpoints.
By referring to this, determine which video is better in terms of achieving the goal.

For your help, we provide the list of subtasks that each humanoid has been performing.
<Video 0>
{video1_subtask}

<Video 1>
{video2_subtask}

Using all information, determine which video is better in terms of achieving each criterion in the <CRITERIA START> section.

As an output, use the format below:

<Answer>: Video <number>

Do not include any other information in your response.
Please consider all viewpoints and explain detailed and specific reasons as possible.
Let's think step by step and be concise.

"""

pairwise_comparison_prompt_detailed = """
You are an AI model who is responsible for determining which video is better by evaluating the behavior of the humanoid robot.

The goal of the agent is as follows:
{task_description}

To achieve this goal, the agent should solve subtasks below:
{subtasks}

As a humanoid robot, the agent should move like a human by following criteria below:

<CRITERIA START>
{general_criteria}
<CRITERIA END>

We will show you two videos from {viewpoint_order} viewpoints.
By referring to this, determine which video is better in terms of achieving the goal.

For your help, we provide the list of subtasks that each humanoid has been performing.
<Video 0>
{video1_subtask}

<Video 1>
{video2_subtask}

Using all information, determine which video is better in terms of achieving each criterion in the <CRITERIA START> section.

As an output, use the format below:

<Comparison>
<Criterion 1: <summary of criterion>>: Video <number>
Reason: <Reason for the choice>

<Criterion 2: <summary of criterion>>: Video <number>
Reason: <Reason for the choice>

...

<Total>: Video <number>
Reason: <Reason for the choice>


Do not include any other information in your response.
Please consider all viewpoints and explain detailed and specific reasons as possible.
Let's think step by step and be concise.

"""
