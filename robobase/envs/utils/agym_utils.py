# ruff: noqa
GENERAL_CRITERIA = """
1. Did the robot complete all the steps?
- The robot needs to finish each step in order, like putting on clothes or feeding someone
- Each step must be done correctly before moving to the next one

2. Does the robot move smoothly?
- The robot's movements should be steady and fluid
- No jerky or shaky movements
- The robot's movements must not threaten the person

3. Is the robot gentle enough?
- The robot must be careful when touching people's skin/body or their clothes.
- It shouldn't pull or push too hard or move too fast
- The person should feel comfortable and safe

4. Does the robot avoid bumping into things?
- The robot arm must not hit the person
- It must stay clear of wheelchairs and other objects nearby
"""

TASK_DESCRIPTION = {
    "DressingPR2-v0": "Carefully moving the sleeve up a person's stationary left arm. Start from the forearm near the wrist, move past the elbow, and continue up to the shoulder. Note that the robot is already holding the sleeve.",  # noqa
    "FeedingBaxter-v0": "Carefully moving the food on the spoon to the person's mouth without spilling.",
}

SUBTASK_LIST = {
    "DressingPR2-v0": """
    Subtask 1: Pull Sleeve Opening Toward Hand
    - Description: Success is when: 1) The magenta-colored sphere hand is covered inside the sleeve opening, 2) After a moment, the hand becomes visible again after passing through the sleeve. Must succeed before proceeding.

    Subtask 2: Pull Sleeve Up to Elbow
    - Description: After the hand is through the sleeve, pull the sleeve up along the arm until it reaches the bright green marker at the elbow.

    Subtask 3: Pull Sleeve Up to Shoulder
    - Description: After reaching the elbow, keep pulling the sleeve up along the arm until it reaches the cyan marker at the shoulder.
    """,
    "FeedingBaxter-v0": """
    Subtask 1: Move Spoon to Mouth
    - Description: Moving the spoon in a straight line to feed the food to the person's mouth, which is marked by a bright green sphere, without spilling.

    Subtask 2: Feed Food to Mouth
    - Description: Feed the food on the spoon to the person's mouth without spilling.
    """,
}
