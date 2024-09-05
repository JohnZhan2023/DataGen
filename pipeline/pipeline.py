import os
import base64
from datasets.arrow_dataset import Dataset

from api.qwen import qwen_api
from utils import load_dataset
import pickle
from functools import partial
from utils import save_raster

def data_annotation(index_root, split, data_scale, data_root, map_root, new_data_root):
    dataset = load_dataset(index_root, split, dataset_scale=data_scale)
    all_maps_dic = {}
    map_folder = os.path.join(map_root, 'map')
    for each_map in os.listdir(map_folder):
        if each_map.endswith('.pkl'):
            map_path = os.path.join(map_folder, each_map)
            with open(map_path, 'rb') as f:
                map_dic = pickle.load(f)
            map_name = each_map.split('.')[0]
            all_maps_dic[map_name] = map_dic
    from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
    samples = nuplan_rasterize_collate_func(
        dataset, all_maps_dic, data_root
    )   
    indexes = []
    for i in range(len(samples)):
        sample = samples[i]
        save_raster(sample, 0, file_index=i, path_to_save="./raster")
        high_res_image_path = os.path.join("./raster", f"test_{i}_0_high_res_raster.png")
        low_res_image_path = os.path.join("./raster", f"test_{i}_0_low_res_raster.png")
        response = qwen_api(high_res_image_path, "describe the image")
        print(response)
        index = sample["index"]
        index["description"] = response
        indexes.append(index)
    new_dataset = Dataset.from_list(indexes)
    new_dataset.save_to_disk(os.path.join(new_data_root, split))
    
if __name__ == "__main__":
    NotImplementedError
    