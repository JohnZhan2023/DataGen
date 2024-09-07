import os
import base64
from datasets.arrow_dataset import Dataset

from api.qwen import qwen_api
from utils import load_dataset
import pickle
from functools import partial
from utils import save_raster

def description(index_root, split, data_scale, new_data_root):
    dataset = load_dataset(index_root, split, dataset_scale=data_scale)
    indexes = []
    for data in dataset:
        from utils import pic_path
        img_path = pic_path(data["images_path"])
        from prompt.SceneDescription import SceneDescription
        response = qwen_api(img_path, SceneDescription())
        data["prediction"] = response
        indexes.append(data)
    new_dataset = Dataset.from_list(indexes)
    new_dataset.save_to_disk(os.path.join(new_data_root, split))
    
if __name__ == "__main__":
    description(index_root="/cephfs/shared/nuplan/online_s6/index",
                    split="test",
                    data_scale=1,
                    data_root="/cephfs/shared/nuplan/online_s6",
                    map_root="/cephfs/shared/nuplan/online_s6/map",
                    new_data_root="/cephfs/shared/DataGen",
                    )
    