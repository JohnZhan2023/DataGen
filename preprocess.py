import os
from datasets.arrow_dataset import Dataset
import time
import json
import re
from pipeline.utils import pic_path 

# load the indexes from the saved root and add a route information to the indexes
# finally save the indexes to the new root
def load_indexes(save_root):
    return Dataset.load_from_disk(save_root)

def add_route_info(index_root, new_data_root):
    dataset = load_indexes(index_root)
    new_dataset = []
    for i, data in enumerate(dataset):
        meta_action = data["hierarchical_planning"]
        route = "go straight"
        try:
            meta_action = json.loads(meta_action[8:-4])
        except:
            try:
                meta_action = json.loads(meta_action)
            except:
                try:
                    meta_action = re.search(r'```json(.*?)```', meta_action, re.DOTALL).group(1)
                    meta_action = json.loads(meta_action)
                except:
                    continue
        for action in meta_action["Meta_Actions"]:
            if "turn left" in action["Action"] :
                route = "left"
                break
            if "turn right" in action["Action"]:
                route = "right"
                break
            if "wait" in action["Action"]:
                route = "wait"
                break
        data["route"] = route
        new_dataset.append(data)
        if "Waiting for Traffic Lights" not in data["scene_description"] and "Near Multiple Vehicles" not in data["scene_description"]: 
            print("number of data processed: ", i)
            print(data["scene_description"])
            path = pic_path(data["images_path"][0])
            print(path)
        
        # time.sleep(10)
    # new_dataset = Dataset.from_list(new_dataset)
    # new_dataset.save_to_disk(new_data_root)

if __name__ == "__main__":
    index_root = "/public/MARS/datasets/dataGen/boston_train copy"
    new_data_root = index_root[:-5]+"_with_route_info"
    add_route_info(index_root,  new_data_root)