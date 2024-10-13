import os
import base64
from datasets.arrow_dataset import Dataset
import sys
sys.path.append("/home/zhanjh/workspace/DataGen/")
from api.qwen import qwen_api
from api.gpt import gpt_4o, gpt_4o_text
from utils import load_dataset
import pickle
from functools import partial
from utils import save_raster
import time
import random
import json

from prompt.HierarchicalPlanning import HierarchicalPlanning
from prompt.SceneAnalysis import SceneAnalysis
from prompt.SceneDescription import SceneDescription
from prompt.ScenePrompt import ScenePrompt
from utils import pic_path, selected_scenario_type, flip_image_horizontally, delete_image, pose_v_describe, concatenate_images
# to continue the annotation we laod the existing data
from datasets.arrow_dataset import Dataset

def load_indexes(save_root):
    return Dataset.load_from_disk(save_root)
    
    

def data_annotation(index_root, split, data_scale, data_root, new_data_root):
    dataset = load_dataset(index_root, split, dataset_scale=data_scale, select=True, debug=True)
    selected_scenario_types = selected_scenario_type()

    from map.map import return_map_dic
    all_maps_dic = return_map_dic()
    try:
        indexes = load_indexes(f"/public/MARS/datasets/dataGen/new_val_{split}")
        # turn the indexes(Datasets) into a list
        indexes = [index for index in indexes]
    except:
        indexes = []
    saved_num = len(indexes)
    file_name = None
    frame_id = None
    num_data = 0
    print("the length of dataset is ", len(dataset))
    for i, data in enumerate(dataset):
        try:
            from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
            sample = nuplan_rasterize_collate_func(
                [data], all_maps_dic=all_maps_dic, dic_path=data_root
            )   
            index = sample["index"]
            if sample["scenario_type"][0] not in selected_scenario_types:
                continue
            if file_name == index["file_name"]:
                if abs(index["frame_id"]-frame_id)<50:
                    continue
                
            print(f"{sample['scenario_type']} is annotated as the {num_data}th data")
            num_data += 1
            if num_data <= saved_num:
                print(f"skip {num_data-1}th data")
                continue
            frame_id = index["frame_id"]
            file_name = index["file_name"]
            img_path = pic_path(data["images_path"])
            # if index["map"][0]=="sg-one-north":
            #     original_img_path = img_path
            #     print("flip the image")
            #     img_path = flip_image_horizontally(img_path)
            print(i, img_path)

            save_raster(sample, 0, file_index=i, path_to_save="raster")
            high_res_image_path = os.path.join("/home/sunq/home/jiahaozhan/DataGen/raster", f"test_{i}_0_high_res_raster.jpg")
            low_res_image_path = os.path.join("/home/sunq/home/jiahaozhan/DataGen/raster", f"test_{i}_0_low_res_raster.jpg")
            delete_image(low_res_image_path)

            ######################################################################################################################
            scene_description = gpt_4o(img_path, SceneDescription())
            
            description = pose_v_describe(sample["other_agent_position"], sample["other_agent_v"])
            prompt = SceneAnalysis(description, scene_description)
            scene_analysis = gpt_4o(high_res_image_path, prompt)
            
            sampled_trajectory = sample["trajectory_label"][:, ::10, :2]
            hierarchicalPrompt = HierarchicalPlanning(sample["context_actions"], sampled_trajectory)
            meta_actions = gpt_4o(high_res_image_path, hierarchicalPrompt)
            
            # if index["map"][0]=="sg-one-north":
            #     delete_image(img_path)
            #     img_path = original_img_path
            ######################################################################################################################
            index["scene_analysis"] = scene_analysis
            index["scene_description"] = scene_description
            index["hierarchical_planning"] = meta_actions
            print(meta_actions)
                    
            indexes.append(index)
            with open(f"test_data/cache_{i}.txt", "w") as file:
                file.write(scene_analysis)
                file.write("\n")
                file.write(scene_description)
                file.write("\n")
                file.write(meta_actions)
        except Exception as e:
            print(e)
            continue
        if num_data % 100==0:
            new_dataset = Dataset.from_list(indexes)
            new_dataset.save_to_disk(os.path.join(new_data_root, f"new_val_{split}"))
    
if __name__ == "__main__":
    random.seed(2024)
    data_annotation(index_root="/public/MARS/datasets/nuPlanCache/online_s6_inter10_wImages/index",
                    split="train",
                    data_scale=1,
                    data_root="/localdata_ssd", # /localdata_ssd/nuplan_speed  # /public/MARS/datasets/nuPlanCache/online_s6
                    new_data_root="/public/MARS/datasets/dataGen",
                    )
    