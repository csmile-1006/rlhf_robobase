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
    "FeedingBaxter-v0": """Task: Robot-Assisted Feeding
Goal: Use a robot to feed a person with a spoon holding pink-colored food (simulated as particles).
Environment Setup:
- The person's mouth is indicated by a bright green sphere
- The robot holds and maneuvers the spoon
- The robot must deliver the pink-colored food from the spoon to the person's mouth without spilling.
- If the food particle is within 0.02 meters of the person's mouth, it is considered fed, then the particle is removed from the spoon and the robot arm earns a reward.

Success Criteria:
- Ensure all pink-colored food successfully reaches the person's mouth and disappears
- Prevent any spilling of the pink-colored food
- After feeding all food, **THE ROBOT ARM MUST RETURN TO THE STARTING POSITION**

General Instructions:
- The robot's movements should be smooth and predictable
- Avoid uncomfortable contact between the spoon and the person, and prevent spoon vibration near the person's mouth
- The robot arm maintains a safe distance from the person
""",
    "DrinkingPR2-v0": """Task: Robot-Assisted Drinking
Goal: Use a robot to feed a person with a cup containing cyan-colored water (simulated as particles).
Environment Setup:
- The person's mouth is indicated by a bright green sphere
- The robot holds and maneuvers the spoon
- The robot must deliver the cyan-colored water from the cup to the person's mouth without spilling.
- If the water particle is within 0.03 meters of the person's mouth, it is considered fed, then the particle is removed from the cup and the robot arm earns a reward.

Success Criteria:
- Ensure all cyan-colored water successfully reaches the person's mouth and disappears
- Prevent any spilling of the blue-colored water
- After feeding all water, **THE ROBOT ARM MUST RETURN TO THE STARTING POSITION**

General Instructions:
- The robot's movements should be smooth and predictable
- Avoid uncomfortable contact between the spoon and the person, and prevent spoon vibration near the person's mouth
- The robot arm maintains a safe distance from the person
""",
    "DrinkingSaywer-v0": """Task: Robot-Assisted Drinking
Goal: Use a robot to make a person drink a cup containing cyan-colored water (simulated as particles).
Environment Setup:
- The person's mouth is indicated by a bright green sphere
- The robot holds and maneuvers the spoon
- The robot must deliver the cyan-colored water from the cup to the person's mouth without spilling.
- If the water particle is within 0.03 meters of the person's mouth, it is considered fed, then the particle is removed from the cup and the robot arm earns a reward.

Success Criteria:
- Ensure all cyan-colored water successfully reaches the person's mouth and disappears
- Prevent any spilling of the cyan-colored water
- After feeding all water, **THE ROBOT ARM MUST RETURN TO THE STARTING POSITION**

General Instructions:
- The robot's movements should be smooth and predictable
- Avoid uncomfortable contact between the spoon and the person, and prevent spoon vibration near the person's mouth
- The robot arm maintains a safe distance from the person
""",
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
