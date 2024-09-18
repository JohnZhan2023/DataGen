import os
import base64
from datasets.arrow_dataset import Dataset
import sys
sys.path.append("/home/zhanjh/workspace/DataGen/")
from api.qwen import qwen_api
from utils import load_dataset
import pickle
from functools import partial
from utils import save_raster

from prompt.HierarchicalPlanning import HierarchicalPlanning
from prompt.SceneAnalysis import SceneAnalysis
from prompt.SceneDescription import SceneDescription
from utils import pic_path

def data_annotation(index_root, split, data_scale, data_root, new_data_root):
    dataset = load_dataset(index_root, split, dataset_scale=data_scale, select=True)

    from map.map import return_map_dic
    all_maps_dic = return_map_dic()

    indexes = []
    
    for i, data in enumerate(dataset):
        from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
        sample = nuplan_rasterize_collate_func(
            [data], all_maps_dic=all_maps_dic, dic_path=data_root
        )   
        index = sample["index"]
        img_path = pic_path(data["images_path"])
        description = qwen_api(img_path, SceneDescription())
        index["description"] = description
        print(description)

        prediction = qwen_api(img_path, SceneAnalysis(description))
        index["prediction"] = prediction
        print(prediction)

        save_raster(sample, 0, file_index=i, path_to_save="./raster")
        high_res_image_path = os.path.join("./raster", f"test_{i}_0_high_res_raster.png")
        low_res_image_path = os.path.join("./raster", f"test_{i}_0_low_res_raster.png")
        planning = qwen_api(high_res_image_path, HierarchicalPlanning(description, prediction, sample["context_actions"], sample["trajectory_label"]))
        index["planning"] = planning
        print(planning)
        
        indexes.append(index)
    new_dataset = Dataset.from_list(indexes)
    new_dataset.save_to_disk(os.path.join(new_data_root, split))
    
if __name__ == "__main__":
    data_annotation(index_root="/home/zhanjh/data/index",
                    split="val",
                    data_scale=0.0002,
                    data_root="/home/zhanjh/nuplan/online_s6",
                    new_data_root="/home/zhanjh/data/DataGen",
                    )
    