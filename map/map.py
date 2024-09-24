import os
import pickle
def return_map_dic():
    all_maps_dic = {}
    map_folder = os.path.join("/public/MARS/datasets/nuPlanCache/online_s6", 'map')
    for each_map in os.listdir(map_folder):
        if each_map.endswith('.pkl'):
            map_path = os.path.join(map_folder, each_map)
            with open(map_path, 'rb') as f:
                map_dic = pickle.load(f)
            map_name = each_map.split('.')[0]
            all_maps_dic[map_name] = map_dic
    return all_maps_dic