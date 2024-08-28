# ruff: noqa
humanoid_criteria_generation_prompt = """
You are an AI model who is responsible for analyzing the objective of the humanoid robot.
Consider the difference between humanoid robots and arm robots.
Please suggest a list of top 5 criteria that the humanoid robot should additionally keep in mind compared to robot arms, especially in terms of control and movement.

As an output, use the format below:
Criterion 1: <Title>
- Humanoid agent must be <Description>

Criterion 2: <Title>
- Humanoid agent must be <Description>

...

Each criterion should be specific, atomic, and actionable.
Do not add markdown syntax or any other characters.
Let's think step by step and be precise.
"""
