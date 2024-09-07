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

    from bev.map import return_map_dic
    all_maps_dic = return_map_dic()

    
    new_dataset = Dataset.from_list(indexes)
    new_dataset.save_to_disk(os.path.join(new_data_root, split))
    
if __name__ == "__main__":
    data_annotation(index_root="/cephfs/shared/nuplan/online_s6/index",
                    split="test",
                    data_scale=1,
                    data_root="/cephfs/shared/nuplan/online_s6",
                    map_root="/cephfs/shared/nuplan/online_s6/map",
                    new_data_root="/cephfs/shared/DataGen",
                    )
    