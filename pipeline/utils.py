from datasets import Dataset, Value

import logging
import os
import random
import shutil
import sys
import pickle
import copy
import torch
from tqdm import tqdm
import copy
import multiprocessing as mp
import datasets
import numpy as np
from datasets import Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets

def load_dataset(root, split='train', dataset_scale=1, agent_type="all", select=False, debug = False):
    datasets = []
    index_root_folders = os.path.join(root, split)
    indices = os.listdir(index_root_folders)
    # ['train-index_vegas3', 'train-index_singapore', 'train-index_pittsburgh', 'train-index_vegas2', 'train-index_vegas4', 'train-index_vegas5', 'train-index_boston', 'train-index_vegas1', 'train-index_vegas6']
    if debug:
        print(indices)
        indices = indices[2:3]
    for index in indices:
        index_path = os.path.join(index_root_folders, index)
        if os.path.isdir(index_path):
            # load training dataset
            logging.info("Loading dataset {}".format(index_path))
            dataset = Dataset.load_from_disk(index_path)
            if dataset is not None:
                datasets.append(dataset)
        else:
            continue
    # For nuplan dataset directory structure, each split obtains multi cities directories, so concat is required;
    # But for waymo dataset, index directory is just the datset, so load directory directly to build dataset. 
    if len(datasets) > 0: 
        dataset = _concatenate_map_style_datasets(datasets)
        for each in datasets:
            each.cleanup_cache_files()
    else: 
        dataset = Dataset.load_from_disk(index_root_folders)

    # add split column
    dataset.features.update({'split': Value('string')})
    try:
        # for some new dataset, split column is already added
        if split == 'train_alltype':
            dataset = dataset.add_column(name='split', column=['train'] * len(dataset))
        else:
            dataset = dataset.add_column(name='split', column=[split] * len(dataset))
    except:
        pass

    dataset.set_format(type='torch')

    if agent_type != "all":
        agent_type_list = agent_type.split()
        agent_type_list = [int(t) for t in agent_type_list]
        dataset = dataset.filter(lambda example: example["object_type"] in agent_type_list, num_proc=mp.cpu_count())

    if select:
        samples = int(len(dataset) * float(dataset_scale))
        dataset = dataset.select(range(samples))

    return dataset

random.seed(203044)
color_pallet = np.random.randint(0, 255, size=(21, 4)) * 0.5
def save_raster(inputs, sample_index, file_index=0,
                prediction_trajectory=None, path_to_save=None,
                high_scale=4, low_scale=0.77,
                prediction_key_point=None,
                prediction_key_point_by_gen=None,
                prediction_trajectory_by_gen=None):
    import cv2
    # save rasters
    image_shape = None
    image_to_save = {
        'high_res_raster': None,
        'low_res_raster': None
    }
    past_frames_num = inputs['context_actions'][sample_index].shape[0]
    agent_type_num = 8
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = inputs[each_key][sample_index]
        if isinstance(each_img, torch.Tensor):
            each_img = each_img.cpu().numpy()
        goal = each_img[:, :, :2]
        road = each_img[:, :, 2:22]
        traffic_lights = each_img[:, :, 22:26]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=float)
        image_shape = target_image.shape

        for i in [5, 17, 18, 19]:
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, :2][goal == 1] = 255
        for i in range(20):
            if i in [5, 17, 18, 19]:
                continue
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(4):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        if 'high' in each_key:
            scale = high_scale
        elif 'low' in each_key:
            scale = 300
            # scale = low_scale
        # draw context actions, and trajectory label
        for each_traj_key in ['context_actions', 'trajectory_label']:
            if each_traj_key not in inputs:
                continue
            if isinstance(inputs[each_traj_key], torch.Tensor):
                pts = inputs[each_traj_key][sample_index].cpu().numpy()
            else:
                pts = inputs[each_traj_key][sample_index]
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    if 'actions' in each_traj_key:
                        target_image[x, y, :] = [255, 0, 255]
                    elif 'label' in each_traj_key:
                        target_image[x-1:x+1, y-1:y+1, :] = [255, 255, 255]

        tray_point_size = max(2, int(0.75 * scale * 4 / 7 / 20))
        key_point_size = max(2, int(3 * scale * 4 / 7))
        # draw prediction trajectory
        if prediction_trajectory is not None:
            for i in range(prediction_trajectory.shape[0]):
                x = int(prediction_trajectory[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 <y < target_image.shape[1]:
                    target_image[x-tray_point_size:x+tray_point_size, y-tray_point_size:y+tray_point_size, 1:] += 200

            x = int(0 * scale) + target_image.shape[0] // 2
            y = int(0 * scale) + target_image.shape[1] // 2
            if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                target_image[x - tray_point_size:x + tray_point_size, y - tray_point_size:y + tray_point_size, 2] += 200

        # draw prediction trajectory by generation
        if prediction_trajectory_by_gen is not None:
            for i in range(prediction_trajectory_by_gen.shape[0]):
                x = int(prediction_trajectory_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 <y < target_image.shape[1]:
                    target_image[x-tray_point_size:x+tray_point_size, y-tray_point_size:y+tray_point_size, :2] += 100

        # draw key points
        if prediction_key_point is not None:
            for i in range(prediction_key_point.shape[0]):
                x = int(prediction_key_point[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 <y < target_image.shape[1]:
                    target_image[x-key_point_size:x+key_point_size, y-key_point_size:y+key_point_size, 1] += 200

        # draw prediction key points during generation
        if prediction_key_point_by_gen is not None:
            for i in range(prediction_key_point_by_gen.shape[0]):
                x = int(prediction_key_point_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 <y < target_image.shape[1]:
                    target_image[x-key_point_size:x+key_point_size, y-key_point_size:y+key_point_size, 2] += 200

        target_image = np.clip(target_image, 0, 255)
        image_to_save[each_key] = target_image
    # import wandb
    if path_to_save is not None:
        for each_key in image_to_save:
            # images = wandb.Image(
            #     image_to_save[each_key],
            #     caption=f"{file_index}-{each_key}"
            # )
            # self.log({"pred examples": images})
            cv2.imwrite(os.path.join(path_to_save, 'test' + '_' + str(file_index) + '_' + str(sample_index) + '_' + str(each_key) + '.jpg'), image_to_save[each_key])
    else:
        return image_to_save

def pic_path(images_path, num=0):
    root = os.getenv("SENSOR_BLOBS_ROOT")
    return os.path.join(root, images_path[num])


def selected_scenario_type():
    """
        high_magnitude_speed
        changing_lane
        stationary_in_traffic
        following_lane_with_lead
        traversing_pickup_dropoff
        stopping_with_lead
        low_magnitude_speed
        starting_straight_intersection_traversal
        near_multiple_vehicles
        waiting_for_pedestrian_to_cross
        high_lateral_acceleration
        behind_long_vehicle
        starting_right_turn
        starting_left_turn
    """
    filter_types = [
            "waiting_for_pedestrian_to_cross",
            "near_multiple_vehicles",
            "changing_lane",
            "starting_right_turn",
            "behind_long_vehicle",
            "high_magnitude_speed",
            "stationary_in_traffic",
            "following_lane_with_lead",
            "traversing_pickup_dropoff",
            "stopping_with_lead",
            "starting_straight_intersection_traversal",
            "high_lateral_acceleration",
            "starting_left_turn",
            "near_pedestrian_on_crosswalk",
            "on_stopline_traffic_light",
            "on_intersection",
            "on_traffic_light_intersection",
            "traversing_intersection",
            "traversing_traffic_light_intersection",
            "near_construction_zone_sign",
            "on_pickup_dropoff",
            "near_pedestrian_at_pickup_dropoff"
        ]
    return filter_types


# near_pedestrian_on_crosswalk
# on_stopline_traffic_light
# on_intersection
# on_traffic_light_intersection
# traversing_intersection
# traversing_traffic_light_intersection
# near_construction_zone_sign
# near_multiple_vehicles
# on_pickup_dropoff
# traversing_pickup_dropoff
# 
from PIL import Image
def flip_image_horizontally(image_path):
    """水平翻转图片并返回翻转后的图片路径。"""
    with Image.open(image_path) as img:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img_path = image_path.replace('.jpg', '_flipped.jpg')  # 修改文件名
        flipped_img.save(flipped_img_path)
    return flipped_img_path

def delete_image(image_path):
    """删除指定路径的图片。"""
    if os.path.exists(image_path):
        os.remove(image_path)
    else:
        print(f"Image not found: {image_path}")
        
        
        
def pose_v_describe(poses, velocity):
    description=""
    for pose in poses:
        description += f"the agent is at the relative place of {pose}(x, y, z, yaw) with a velocity of {velocity}. \n"
    return description
from PIL import Image
def concatenate_images(img_path1, img_path2, save_path):
    """
    拼接两张图片，img_path1 在左，img_path2 在右，先调整两张图片的高度使其相同，然后保存到 save_path。
    """
    # 打开两张图片
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)

    # 获取两张图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 确定新的高度为两者之中的最大值
    new_height = max(height1, height2)

    # 如果两张图片的高度不同，调整高度较小的图片
    if height1 != new_height:
        # 计算新的宽度，保持原始宽高比
        new_width1 = int(new_height * width1 / height1)
        image1 = image1.resize((new_width1, new_height), Image.Resampling.LANCZOS)
    if height2 != new_height:
        # 计算新的宽度，保持原始宽高比
        new_width2 = int(new_height * width2 / height2)
        image2 = image2.resize((new_width2, new_height), Image.Resampling.LANCZOS)

    # 获取调整大小后的宽度
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 创建一个新的画布，宽度为两张图片宽度之和，高度为调整后的统一高度
    new_width = width1 + width2
    new_image = Image.new('RGB', (new_width, new_height))

    # 将两张图片粘贴到新图片上
    new_image.paste(image1, (0, 0))  # 第一张图片粘贴在左边
    new_image.paste(image2, (width1, 0))  # 第二张图片粘贴在右边

    # 保存到指定路径
    new_image.save(save_path)