def SceneDescription():
    prompt = """
        Please analyze the image content and provide a formatted output for the environment description and critical object identification:

        ### Environment Description:
        - **E_weather**: Describe the weather condition (e.g., sunny, rainy, snowy, etc.), and explain its impact on visibility and vehicle traction.
        - **E_time**: Describe the time of day (e.g., daytime or nighttime), and explain how it affects driving strategies due to visibility changes.
        - **E_road**: Describe the type of road (e.g., urban road or highway), and explain the challenges associated with it for driving.
        - **E_lane**: Describe the current lane positioning and possible maneuvers, particularly focusing on lane selection and control decisions.

        ### Critical Object Identification:
        - Identify each critical object in the scene. For each object, provide:
        1. Object category (e.g., pedestrian, vehicle, traffic signal).
        2. Approximate bounding box coordinates in the format (x1, y1, x2, y2).
        3. Explain the significance of the object to the current driving scenario.
        
        your output format should be a json object with the following structure:
        {
            "E_weather": "sunny",
            "E_time": "daytime",
            "E_road": "urban road",
            "E_lane": "middle lane",
            "Critical_Objects": [
                {
                    "Category": "pedestrian",
                    "BoundingBox": [x1, y1, x2, y2],
                    "Description": "Pedestrian crossing the road."
                },
                {
                    "Category": "vehicle",
                    "BoundingBox": [x1, y1, x2, y2],
                    "Description": "Vehicle making a U-turn."
                }
            ]
        }
        """
    return prompt