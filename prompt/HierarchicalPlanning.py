def HierarchicalPlanning(description, analysis, historical_trajectory, future_trajecotry):
    prompt = f'''
    Here is a description of the scene:
    {description}

    Here is a scene analysis:
    {analysis}

    Please analyze the bird's-eye view (BEV) image, paying special attention to the plotted trajectory,(red points denote the past position and blue ones denote future trajectory) and provide a structured output for meta-actions and decision description:

    To help you give precise description of the meta action, we provide the exact historical trajectory and future trajectory of the ego vehicle in the image. The historical trajectory is represented as a sequence of waypoints in the form of (x, y, z, yaw, velocity, velocity_yaw, acceleration, ) coordinates. The future trajectory is represented as a sequence of waypoints in the form of (x, y, z, yaw) coordinates. You can use this information to infer the driving decisions and actions.
    Historical Trajectory: {historical_trajectory} (-2s, -1s, -0.5s, 0s)
    Future Trajectory: {future_trajecotry} (0s-8s, frequency: 10Hz)
    
    # Meta-actions ($A$):
    - Firstly, you should locate the blue line (futuWre trajectory) on the picture. Identify and describe the short-term driving decisions based on the trajectory in the image. These actions include:
    - Speed-control actions: speed up, slow down, stop, wait
    - Turning actions: turn left, turn right, turn around
    - Lane-control actions: change lane, shift slightly to the left or right

    # Decision Description ($D$):
    - For each meta-action, provide a combined decision description from 3 aspects:
    1. **Action ($A$)**: The specific meta-action, such as 'turn', 'accelerate', or 'stop'.
    2. **Subject ($S$)**: The interacting object or lane involved, such as a pedestrian, vehicle, or lane.
    3. **Duration ($D$)**: Specify the time frame for the action, including how long it should last or when it should begin.

    your output format should be a json object with the following structure:
    {{
        "Meta_Actions": [
            $$chosen from meta actions and can be two or more$$
        ],
        "Decision_Description": $$the instructions and reasons for the car to follow in a sentence$$
    }}
    '''
    return prompt
