# ruff: noqa
subtask_generation_prompt = """
You are an AI model responsible for analyzing the objective of the humanoid robot.

The goal of the agent is as follows:
{task_description}

Given this, please list the subtasks the agent should solve in the order in which they must be solved.

As an output, use the format below:
Subtask 1: <Title>
- <Description>

Subtask 2: <Title>
- <Description>

...

Make sure that each subtask is atomic, essential, and measurable.
Refrain from including any other information in your response.
Specify the subtask for each instance of manipulating the object.
Let's think step by step and be precise.
"""


subtask_identification_prompt = """
You are an AI model who is responsible for determining the behavior of the humanoid robot using videos from different viewpoints.

The goal of the agent is as follows:
{task_description}

To achieve this goal, the agent should solve subtasks below:
{subtasks}

By referring to this, first, determine which sub-criterion the agent is currently performing.
Then, evaluate whether the task is successfully achieved.

As an output, use the format below:

With in progress: Subtask <number>
Evaluation:
Subtask <number>: <Success/Failure/WIP>
- <Reason>

Subtask <number>: <Success/Failure/WIP>
- <Reason>

...

Do not include any other information in your response.
Please consider all viewpoints and explain detailed and specific reasons as possible.
Let's think step by step and be concise.

"""
