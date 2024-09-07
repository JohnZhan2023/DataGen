# Design of Dataset
## Example Pic
![sampled fpv](pic/0a04b286e2dd5602.jpg)


## Scene Description (input: sensor)
### Environment Description
The scene description module identifies the driving environment and critical objects. The driving environment can be described by several key conditions that impact driving difficulty, collectively represented as:
$$
\{E_{weather}, E_{time}, E_{road}, E_{lane}\}
$$

- **$E_{weather}$**: Spans conditions from sunny to snowy, affecting visibility and traction.
- **$E_{time}$**: Indicates daytime or nighttime, influencing driving strategies due to visibility changes.
- **$E_{road}$**: Represents road types, such as urban or highway, introducing different challenges to driving.
- **$E_{lane}$**: Focuses on the current lane positioning and possible maneuvers, crucial for safe driving decisions.

### Critical Object Identification
In addition to environmental conditions, identifying critical objects in the driving scenario is essential for safe driving decisions. Each critical object, denoted as $O_c$, includes:
- Object category $c$
- Approximate bounding box coordinates $b(x_1, y_1, x_2, y_2)$ on the image

These attributes are mapped to language token IDs, enabling integration with subsequent modules.

```json
{
    "E_weather": "overcast",
    "E_time": "daytime",
    "E_road": "urban road",
    "E_lane": "middle lane",
    "Critical_Objects": [
        {
            "Category": "vehicle",
            "BoundingBox": [350, 480, 410, 580],
            "Description": "Truck driving towards the camera."
        },
        {
            "Category": "vehicle",
            "BoundingBox": [670, 510, 770, 610],
            "Description": "Van parked on the right side of the road."
        },
        {
            "Category": "vehicle",
            "BoundingBox": [530, 500, 580, 560],
            "Description": "Truck parked on the right side of the road."
        },
        {
            "Category": "pedestrian",
            "BoundingBox": [580, 520, 600, 580],
            "Description": "Person standing near the parked vehicles."
        }
    ]
}
```

## Scene Analysis (input: sensor)
The scene analysis module provides a comprehensive understanding of the driving environment and critical objects. 

### Critical Object Analysis
Critical objects are analyzed from three perspectives:
- **Static attributes ($C_s$)**: Properties like the visual cues of roadside billboards or the oversized cargo of a truck.
- **Motion states ($C_m$)**: Describes the object's dynamics, such as position, direction, and action.
- **Particular behaviors ($C_b$)**: Refers to special actions, like a pedestrian's gesture, that could influence the ego vehicle's next decision.
```json
{
    "Scene_Summary": "The ego vehicle is driving on a city street with construction activity on the right side. The road is clear, and the weather appears to be overcast. There are several vehicles, including a truck with oversized cargo, and construction workers present.",
    
    "Critical_Objects": [
        {
            "Category": "vehicle",
            "Static_Attributes": "Truck with oversized cargo",
            "Motion_States": "Moving forward",
            "Particular_Behaviors": "No special actions observed"
        },
        {
            "Category": "construction_worker",
            "Static_Attributes": "Wearing high-visibility clothing",
            "Motion_States": "Standing near the construction site",
            "Particular_Behaviors": "No special actions observed"
        }
    ]
}
```


## Hierarchical Planning (input: bev and trajectory)
Once the scene-level summary is generated, it is combined with the route, ego vehicle's pose, and velocity to inform planning. Planning occurs in three stages:

### Meta-actions ($A$)
Meta-actions represent short-term driving decisions, such as accelerating, decelerating, turning, or changing lanes.

| **Category**          | **Meta-actions**                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| **Speed-control actions** | Speed up, Slow down, Slow down rapidly, Go straight slowly, Go straight at a constant speed, Stop, Wait, Reverse |
| **Turning actions**    | Turn left, Turn right, Turn around                                                                  |
| **Lane-control actions** | Change lane to the left, Change lane to the right, Shift slightly to the left, Shift slightly to the right      |


### Decision Description ($D$)
The decision description provides a fine-grained driving strategy, including:
- **Action ($A$)**: The specific meta-action, such as 'turn' or 'accelerate.'
- **Subject ($S$)**: The object or lane involved in the interaction, such as a pedestrian or traffic signal.
- **Duration ($D$)**: Specifies when the action should occur and how long it should last.

### Trajectory Waypoints ($W$)
Trajectory waypoints, denoted as:
$$
W = \{w_1, w_2, ..., w_n\}, \quad w_i = (x_i, y_i)
$$
These waypoints define the vehicle's path over a certain future period with predetermined intervals $\Delta t$.
