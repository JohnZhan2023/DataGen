def HierarchicalPlanning():
    prompt = """
    Please analyze the bird's-eye view (BEV) image, paying special attention to the plotted trajectory, and provide a structured output for meta-actions and decision description:

    # Meta-actions ($A$):
    - Identify and describe the short-term driving decisions based on the trajectory in the image. These actions include:
    - Speed-control actions (e.g., speed up, slow down, stop, wait)
    - Turning actions (e.g., turn left, turn right, turn around)
    - Lane-control actions (e.g., change lane, shift slightly to the left or right)

    # Decision Description ($D$):
    - For each meta-action, provide a decision description including:
    1. **Action ($A$)**: The specific meta-action, such as 'turn', 'accelerate', or 'stop'.
    2. **Subject ($S$)**: The interacting object or lane involved, such as a pedestrian, vehicle, or lane.
    3. **Duration ($D$)**: Specify the time frame for the action, including how long it should last or when it should begin.

    your output format should be a json object with the following structure:
    {
        "Meta_Actions": [
            "speed up",
            "turn right"
        ],
        "Decision_Description": [
            {
                "Action": "speed up",
                "Subject": "ego vehicle",
                "Duration": "5 seconds"
            },
            {
                "Action": "turn right",
                "Subject": "left lane",
                "Duration": "2 seconds"
            }
        ]
    }
    """
    return prompt
