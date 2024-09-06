def SceneAnalysis():
    prompt = """
        Please analyze the image and provide a comprehensive scene analysis, focusing on both the driving environment and critical objects:

        # Scene Analysis:
        - Provide a detailed description of the driving environment, including weather, road type, time of day, and lane conditions.

        # Critical Object Analysis:
        For each critical object identified in the scene, analyze it from the following three perspectives:
        1. **Static attributes ($C_s$)**: Describe inherent properties of the object, such as visual cues (e.g., roadside billboards or oversized cargo on a truck).
        2. **Motion states ($C_m$)**: Describe the object's dynamics, including its position, direction, and current action.
        3. **Particular behaviors ($C_b$)**: Highlight any special actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian's gesture or sudden movement).
        
        your output format should be a json object with the following structure:
        {
            "Scene_Summary": "The ego vehicle changes lanes from the wrong-way lane to the right-way lane, with a bicycle fallen in front.",
                
            "Critical_Objects": [
                {
                    "Category": "pedestrian",
                    "Static_Attributes": "Carrying a backpack",
                    "Motion_States": "Crossing the road",
                    "Particular_Behaviors": "Looking at the oncoming traffic"
                },
                {
                    "Category": "vehicle",
                    "Static_Attributes": "Truck with oversized cargo",
                    "Motion_States": "Turning right",
                    "Particular_Behaviors": "Sudden lane change"
                }
            ]
        }
        """
    return prompt