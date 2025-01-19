# ruff: noqa
TASK_DESCRIPTION = {
    "h1-walk-v0": """Task: Make the H1 humanoid robot walk naturally at 2m/s

Requirements:
1. Speed: Maintain 2 meter per second walking speed
2. Direction: Walk in a straight line without body rotation
3. Posture: Keep torso and head upright, minimal forward/backward tilt
4. Gait:
   - Use natural stride length, neither too short nor too long
   - Alternate legs in a cyclical walking pattern
   - Plant one foot while swinging the other forward
   - No foot dragging during ground contact
5. Arm Motion:
   - Keep arms and hands close to body
   - Natural arm swing coordinated with leg motion
   - Right arm forward when left leg steps forward and vice versa
6. Balance: Maintain stability and avoid falling while walking
""",
    "h1-run-v0": """Task: Make the H1 humanoid robot run naturally at 5m/s

Requirements:
1. Speed: Maintain 5 meter per second running speed
2. Direction: Run in a straight line without body rotation
3. Posture: Keep torso and head upright, minimal forward/backward tilt
4. Gait:
   - Use natural stride length, neither too short nor too long
   - Alternate legs in a cyclical running pattern
   - Plant one foot while swinging the other forward
   - No foot dragging during ground contact
5. Arm Motion:
   - Keep arms and hands close to body
   - Natural arm swing coordinated with leg motion
   - Right arm forward when left leg steps forward and vice versa
6. Balance: Maintain stability and avoid falling while running
""",
    "h1hand-walk-v0": "The H1 humanoid robot should walk forward at 1m/s speed without falling. The walking motion should be stable and closely resemble natural human walking.",
    "g1-walk-v0": "The G1 humanoid robot should walk forward at 1m/s speed. The walking motion should be stable, energy-efficient, and closely resemble natural human walking.",
    "gr1-walk-v0": "The GR-1 humanoid robot should walk forward at 1m/s speed without falling. The walking motion should be stable and closely resemble natural human walking.",
}
