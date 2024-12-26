import torch
import os
import argparse
import json
import cv2
import copy

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

class ndarrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

class pizzaSlice():
    def __init__(self, data_path, clip_path = None, view = 0, meta_path = None):
        self.view = view
        self.data_path = data_path
        if meta_path is None:
            # self.slices = self.data_metainfo()
            self.slices, self.aligned_data, self.aligned_joints = self.complete_data_in_charts(view = view)
            self.chosen_ids = self.correct_img_indices()
            # self.action, self.status = self.get_action()
            self.prompts = self.get_prompt()
        else:
            self.read_meta(meta_path)
            
        self.action = self.get_aligned_action()
        self.action_wo_gripper = self.get_aligned_action(seventh_dim_keep=True)
        self.mean, self.std = self.get_mean_std()
        self.means_wo_gripper, self.stds_wo_gripper = self.mean, self.std
        self.joints_mean, self.joints_std = self.get_joints_mean_std()
        self.means_wo_gripper[6] = 0
        self.stds_wo_gripper[6] = 1
        if clip_path is not None:
            self.clip_path = clip_path
    
    def norm_numpylize(self, data:dict):
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v)
            elif isinstance(v, dict):
                data[k] = self.norm_numpylize(v)
        return data
    
    def aligned_data_numpylize(self, data:dict):
        for k, v in data.items():
            for kk, vv in v.items():
                ndarray_vv = []
                for vvv in vv:
                    ndarray_vv.append(np.asanyarray(vvv).astype(np.float32))
                data[k][kk] = ndarray_vv
        return data
    
    def read_meta(self, meta_path:str):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.slices = self.norm_numpylize(meta['slices'])
        self.aligned_data = self.aligned_data_numpylize(meta['aligned_data'])
        self.aligned_joints = self.aligned_data_numpylize(meta['aligned_joints'])
        self.chosen_ids = meta['chosen_ids']
        self.prompts = meta['prompts']

    def data_metainfo(self):
        # each_task = range(1, 23)
        if self.view == 0:
            view = "right_rgb"
        elif self.view == 1:
            view = "left_rgb"
        elif self.view == 2:
            view = "top_rgb"
        elif self.view == 3:
            view = "inhand_rgb"
        slices = {}
        for task in tqdm(range(1, 23)):
            if task == 3 or task == 19 or task == 20:
                continue
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            task_slice = {}
            for episode in episodes:
                # print(task, episode)
                episode_path = os.path.join(task_path,episode)
                npy_path = os.path.join(episode_path, "franka_data.npy")
                npy = np.load(npy_path, allow_pickle=True)
                time_table = []
                actual_status = []
                actual_joints = []
                for trajotry in npy:
                    timestamp = trajotry['timestamp']
                    time_table.append(timestamp)
                    actual_status.append([trajotry['eef_position'], trajotry['eef_quaternion'], trajotry['gripper_width']])
                    actual_joints.append(trajotry['joint_position'])
                start_time = time_table[0]
                end_time = time_table[-1]
                slice_start = start_time % 1
                slice_start = int(slice_start / 0.2) + 1
                slice_timer = int(start_time) + slice_start * 0.2
                slice_count = 1
                while True:
                    # print(slice_timer)
                    slice_count += 1
                    slice_timer += 0.2
                    if slice_timer > end_time:
                        break
                episode_info = np.zeros((2, slice_count))
                episode_info[0][0] = 1
                slice_timer_start = int(start_time) + slice_start * 0.2
                current_slice = 0
                for i in range(len(time_table) - 1):
                    id = i + 1
                    slice_timer = slice_timer_start + current_slice * 0.2
                    if time_table[i] >= slice_timer:
                        if time_table[i] < slice_timer + 0.2:
                            episode_info[0][current_slice + 1] += 1
                            current_slice += 1
                            continue
                        else:
                            while time_table[i] >= slice_timer + 0.2:
                                episode_info[0][current_slice + 1] = -1
                                current_slice += 1
                                slice_timer += 0.2
                            episode_info[0][current_slice + 1] += 1
                    else:
                        episode_info[0][current_slice] += 1
                        continue
                image_path = os.path.join(episode_path, "images", view)
                if os.path.exists(image_path):
                    for i in range(slice_count):
                        img_id = f"{i:03d}.jpg"
                        if os.path.exists(os.path.join(image_path, img_id)):
                            episode_info[1][i] = 1
                else:
                    for i in range(slice_count):
                        episode_info[1][i] = -1
                task_slice[episode] = episode_info
            slices[str(task)] = task_slice
        return slices
    def get_clip(self):
        # TODO
        pass
    def get_prompt(self):
        prompt = []
        prompt_path = os.path.join(self.data_path, "prompts.txt")
        with open(prompt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Task "):
                    task = line.split(": ")[1].strip()
                    prompt.append(task)
        return prompt
    def correct_img_indices(self):
        correct_indeces = {}
        for task in tqdm(range(1, 23)):
            if task == 3 or task == 19 or task == 20:
                continue
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            task_metainfo = self.slices[str(task)]
            task_correct_indeces = {}
            for episode in episodes:
                epi_correct_id = []
                epi_metainfo = task_metainfo[episode]
                for id in range(len(epi_metainfo[0])):
                    if epi_metainfo[0][id] >= 1 and epi_metainfo[1][id] == 1:
                        epi_correct_id.append(id)
                task_correct_indeces[episode] = epi_correct_id
            correct_indeces[str(task)] = task_correct_indeces
        return correct_indeces
    def complete_data_in_charts(self, view=0):
        if view == 0:
            view = "right_rgb"
        elif view == 1:
            view = "left_rgb"
        elif view == 2:
            view = "top_rgb"
        elif view == 3:
            view = "inhand_rgb"
        
        slices = {}
        datas = {}
        joints = {}
        for task in tqdm(range(1, 23)):
            if task == 3 or task == 19 or task == 20:
                continue
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            task_slice = {}
            task_data = {}
            task_joint = {}
            for episode in episodes:
                # print(task, episode)
                episode_path = os.path.join(task_path,episode)
                npy_path = os.path.join(episode_path, "franka_data.npy")
                npy = np.load(npy_path, allow_pickle=True)
                time_table = []
                actual_status = []
                actual_joints = []
                for trajotry in npy:
                    timestamp = trajotry['timestamp']
                    time_table.append(timestamp)
                    actual_joints.append(trajotry['joint_position'])
                    actual_status.append(self.status_to_7dim([trajotry['eef_position'], trajotry['eef_quaternion'], trajotry['gripper_width']]))
                start_time = time_table[0]
                end_time = time_table[-1]
                slice_start = start_time % 1
                slice_start = int(slice_start / 0.2) + 1
                slice_timer = int(start_time) + slice_start * 0.2
                slice_count = 1
                while True:
                    # print(slice_timer)
                    slice_count += 1
                    slice_timer += 0.2
                    if slice_timer > end_time:
                        break
                slice_timer_start = int(start_time) + slice_start * 0.2
                abnormal_status = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], -1.0]
                abnormal_joints = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])
                epi_joints = []
                epi_data = []
                episode_info = np.zeros((2, slice_count))
                episode_info[0][0] = 1
                epi_data.append([])
                epi_joints.append([])
                epi_joints[0].append(actual_joints[0])
                epi_data[0].append(actual_status[0])
                epi_joints.append([])
                epi_data.append([])
                current_slice = 0
                for i in range(len(time_table) - 1):
                    id = i + 1
                    slice_timer = slice_timer_start + current_slice * 0.2
                    if time_table[id] >= slice_timer:
                        if time_table[id] < slice_timer + 0.2:
                            episode_info[0][current_slice + 1] += 1
                            epi_joints[current_slice + 1].append(actual_joints[id])
                            epi_data[current_slice + 1].append(actual_status[id])
                            current_slice += 1
                            epi_joints.append([])
                            epi_data.append([])
                            continue
                        else:
                            while time_table[id] >= slice_timer + 0.2:
                                episode_info[0][current_slice + 1] = -1
                                epi_joints[current_slice + 1].append(abnormal_joints)
                                epi_data[current_slice + 1].append(self.status_to_7dim(abnormal_status))
                                current_slice += 1
                                epi_joints.append([])
                                epi_data.append([])
                                slice_timer += 0.2
                            episode_info[0][current_slice + 1] += 1
                            epi_joints[current_slice + 1].append(actual_joints[id])
                            epi_data[current_slice + 1].append(actual_status[id])
                            current_slice += 1
                            epi_joints.append([])
                            epi_data.append([])

                    else:
                        episode_info[0][current_slice] += 1
                        epi_joints[current_slice].append(actual_joints[id])
                        epi_data[current_slice].append(actual_status[id])
                        continue
                image_path = os.path.join(episode_path, "images", view)
                if os.path.exists(image_path):
                    for i in range(slice_count):
                        img_id = f"{i:03d}.jpg"
                        if os.path.exists(os.path.join(image_path, img_id)):
                            episode_info[1][i] = 1
                else:
                    for i in range(slice_count):
                        episode_info[1][i] = -1
                if epi_data[-1] == []:
                    epi_data = epi_data[:-1]
                if epi_joints[-1] == []:
                    epi_joints = epi_joints[:-1]
                task_joint[episode] = epi_joints
                task_slice[episode] = episode_info
                task_data[episode] = epi_data
            slices[str(task)] = task_slice
            datas[str(task)] = task_data
            joints[str(task)] = task_joint
        return slices, datas, joints
        
    def get_indices_include_first(self, indice, step=7):  
        # 首先包含索引0，然后是每step个元素的最后一个索引  
        indices = [0] + [indice[i] for i in range(step - 1, len(indice), step)]  
        # 如果数组的长度恰好是step的倍数，上一步已经包含了最后一个元素的索引，否则添加最后一个元素的索引  
        if len(indice) % step != 0 or len(indice) == 1:  # 添加len(arr) == 1来处理只有一个元素的情况  
            indices.append(indice[len(indice) - 1])  
        return indices  
    
    def get_indice_packs(self, indice, step = 7):
        # 首先包含索引0，然后是每step个元素的最后一个索引  
        indices = []
        
        frame_pack = np.zeros((step)).astype(np.int32)
        frame_id = 0
        for i in range(0, len(indice)):
            frame_pack[frame_id] = indice[i]
            frame_id += 1
            if frame_id == step:
                indices.append(frame_pack)
                frame_pack = np.zeros((step)).astype(np.int32)
                frame_id = 0
        # 如果数组的长度恰好是step的倍数，上一步已经包含了最后一个元素的索引，否则添加最后一个元素的索引  
        if frame_id != 0:
            for i in range(frame_id, step):
                frame_pack[i] = indice[len(indice) - 1]
            indices.append(frame_pack)
        return indices
    
    def get_indice_packs_include_first(self, indice, step=7):  
        # 首先包含索引0，然后是每step个元素的最后一个索引  
        indices = []
        indices.append(np.zeros((step)).astype(np.int32))
        frame_pack = np.zeros((step)).astype(np.int32)
        frame_id = 0
        for i in range(0, len(indice)):
            frame_pack[frame_id] = indice[i]
            frame_id += 1
            if frame_id == step:
                indices.append(frame_pack)
                frame_pack = np.zeros((step)).astype(np.int32)
                frame_id = 0
        # 如果数组的长度恰好是step的倍数，上一步已经包含了最后一个元素的索引，否则添加最后一个元素的索引  
        if frame_id != 0:
            for i in range(frame_id, step):
                frame_pack[i] = indice[len(indice) - 1]
            indices.append(frame_pack)
        return indices
    
    def get_aligned_action(self, seventh_dim_keep=False):
        action = {}
        abnormal_status = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], -1.0]
        abnormal_status = self.status_to_7dim(abnormal_status)
        for task in tqdm(range(1,23)):
            task_id = str(task)
            if task == 3 or task == 19 or task == 20:
                continue
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            task_act = {}
            for episode in episodes:

                action_vector = np.zeros((len(self.aligned_data[task_id][episode]) - 1, 7)).astype(np.float32)
                
                for i in range(0, len(self.aligned_data[task_id][episode]) - 1):
                    current_status = self.aligned_data[task_id][episode][i][-1]
                    if (current_status == abnormal_status).all():
                        action_vector[i] = abnormal_status
                        continue
                    append_status = 0
                    for j in range(i + 1, len(self.aligned_data[task_id][episode])):
                        next_status = self.aligned_data[task_id][episode][j][-1]
                        if (next_status == abnormal_status).all():
                            continue
                        else:
                            action_vector[i] = self._7dof_status_to_action(current_status, next_status, seventh_dim_keep)
                            append_status = 1
                            break
                    if not append_status:
                        action_vector[i] = abnormal_status
                task_act[episode] = action_vector
            action[task_id] = task_act
        return action
    
    def _7dof_status_to_action(self, statusA, statusB, seventh_dim_keep = True):
        
        action = statusB - statusA
        statusA_euler = statusA[3:6]
        statusB_euler = statusB[3:6]
        statusA_rot_mat = R.from_euler('xyz', statusA_euler, degrees=False).as_matrix()
        statusB_rot_mat = R.from_euler('xyz', statusB_euler, degrees=False).as_matrix()
        rot_mat = statusB_rot_mat @ np.linalg.inv(statusA_rot_mat)
        action[3:6] = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
        
        if seventh_dim_keep:
            action[6] = statusA[6]

        for i in range(3, 6):
            if action[i] > np.pi:
                action[i] -= 2 * np.pi
            elif action[i] < -np.pi:
                action[i] += 2 * np.pi
        return action
        
    def get_action(self):
        action_mat = {}
        status_mat = {}
        for task in tqdm(range(1, 23)):
            if task == 3 or task == 19 or task == 20:
                continue
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            task_action = {}
            task_status = {}
            for episode in episodes:
                # print(task, episode)
                epi_status = []
                episode_path = os.path.join(task_path,episode)
                npy_path = os.path.join(episode_path, "franka_data.npy")
                npy = np.load(npy_path, allow_pickle=True)
                current_id = 0
                for idx in range(len(self.slices[str(task)][episode][0])):
                    if idx == 0:
                        epi_status.append(self.npy_to_7dim_status(npy, current_id))
                        current_id += self.slices[str(task)][episode][0][idx]
                        continue
                    if self.slices[str(task)][episode][0][idx] == -1 or self.slices[str(task)][episode][1][idx] != 1:
                        epi_status.append([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
                        continue
                    else:
                        # print(current_id + self.slices[str(task)][episode][0][idx] - 1)
                        epi_status.append(self.npy_to_7dim_status(npy, int(current_id + self.slices[str(task)][episode][0][idx] - 1)))
                        current_id += self.slices[str(task)][episode][0][idx]
                        continue
                # epi_action.append(np.array([.0, .0, .0, .0, .0, .0, .0]))
                current_id = 0
                epi_action = np.zeros([len(epi_status) - 1, 7])
                for idx in range(len(epi_status) - 1, 0, -1):
                    if epi_status[idx][0] == -1.0 and epi_status[idx][1] == -1.0:
                        if idx != len(epi_status) - 1:
                            epi_action[idx] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                        continue
                    search_range = idx - 1
                    for i in range(search_range, -1, -1):
                        if epi_status[i][0] == -1.0 and epi_status[i][1] == -1.0:
                            epi_action[i] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                            continue
                        else:
                            epi_action[i] = epi_status[idx] - epi_status[i]
                            idx = i
                            break
                
                task_action[episode] = epi_action
                task_status[episode] = epi_status
            action_mat[str(task)] = task_action
            status_mat[str(task)] = task_status
        return action_mat, status_mat
                
    def status_to_7dim(self, status):
        status_vector = np.zeros(7).astype(np.float32)
        status_vector[0:3] = status[0]
        status_vector[3:6] = R.from_quat(status[1]).as_euler('xyz', degrees=False)
        status_vector[6] = status[2]

        status_vector[1] = -status_vector[1]
        status_vector[2] = -status_vector[2]
        return status_vector
    
    def npy_to_7dim_status(self, npy, idx):
        status = npy[idx]
        status_vector = np.zeros(7).astype(np.float32)
        status_vector[0:3] = status['eef_position']
        status_vector[3:6] = R.from_quat(status['eef_quaternion']).as_euler('xyz', degrees=False)
        status_vector[6] = status['gripper_width']
        
        status_vector[1] = -status_vector[1]
        status_vector[2] = -status_vector[2]
        return status_vector
    
    def get_joints_mean_std(self):
        joints_mat = []
        for k, v in self.chosen_ids.items():
            for kk, vv in v.items():
                for idx in vv:
                    joint = self.aligned_joints[k][kk][idx][-1]
                    joints_mat.append(joint)
        joints_mat = np.array(joints_mat)
        return np.concatenate([np.mean(joints_mat, axis=0), [0.0]], axis=0), np.concatenate([np.std(joints_mat, axis=0), [1.00]], axis=0)
    
    def get_mean_std(self):
        action_mat = []
        for k,v in self.action.items():
            for kk, vv in v.items():
                for idx in range(len(vv)):
                    if vv[idx][0] == -1.0 and vv[idx][1] == -1.0:
                        continue
                    action_mat.append(vv[idx])
        action_mat = np.array(action_mat)
        return np.mean(action_mat, axis=0), np.std(action_mat, axis=0)
    
    def get_gripper_movement(self, threshold = 0.01):
        gripper_frame = []
        for k,v in self.action.items():
            for kk, vv in v.items():
                for idx in range(len(vv)):
                    if vv[idx][0] == -1.0 and vv[idx][1] == -1.0 and vv[idx][6] == -1.0:
                        continue
                    if abs(vv[idx][6]) > threshold:
                        gripper_info = {
                            'task': k,
                            'episode': kk,
                            'frame': idx,
                            'action': vv[idx]
                        }
                        gripper_frame.append(gripper_info)
        return gripper_frame
    
    def get_gripper_movement_ids(self, threshold = 0.01, forward_search_range = 20, backward_search_range = 10):
        abnormal_status = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], -1.0]
        abnormal_status = self.status_to_7dim(abnormal_status)


        gripper_frame = []
        total_info = {}
        for k,v in tqdm(self.action.items()):
            task_info = {}
            for kk, vv in v.items():
                epi_gripper_packs = []
                for idx in range(len(vv)):
                    chosed_ids = []
                    gripper_frame = []
                    if (vv[idx] == abnormal_status).all():
                        continue
                    if abs(vv[idx][6]) >= threshold:
                        if idx not in chosed_ids:
                            chosed_ids.append(idx)
                            gripper_info = {
                                'frame': idx,
                                'action': vv[idx].tolist(),
                                'action_gripper': self.action_wo_gripper[k][kk][idx].tolist()
                            }
                            gripper_frame.append(gripper_info)
                        forward_id = 1
                        while forward_id <= forward_search_range:
                            if idx + forward_id >= len(vv):
                                break
                            if idx + forward_id in chosed_ids:
                                forward_id += 1
                                continue
                            elif (vv[idx + forward_id] == abnormal_status).all():
                                forward_id += 1
                                continue
                            else:
                                chosed_ids.append(idx + forward_id)
                                gripper_info = {
                                'frame': idx + forward_id,
                                'action': vv[idx + forward_id].tolist(),
                                'action_gripper': self.action_wo_gripper[k][kk][idx + forward_id].tolist()
                                }
                                gripper_frame.append(gripper_info)
                                forward_id += 1
                        backward_id = 1
                        while backward_id <= backward_search_range:
                            if idx - backward_id < 0:
                                break
                            if idx - backward_id in chosed_ids:
                                backward_id += 1
                                continue
                            elif (vv[idx - backward_id] == abnormal_status).all():
                                backward_id += 1
                                continue
                            else:
                                chosed_ids.append(idx - backward_id)
                                gripper_info = {
                                'frame': idx - backward_id,
                                'action': vv[idx - backward_id].tolist(),
                                'action_gripper': self.action_wo_gripper[k][kk][idx - backward_id].tolist()
                                }
                                gripper_frame.append(gripper_info)
                                backward_id += 1
                        epi_gripper_packs.append(sorted(gripper_frame, key = lambda x: x['frame']))
                task_info[kk] = epi_gripper_packs
                gripper_frame = []
            total_info[k] = task_info
        return total_info
    
    def get_action_frame_pack(self, task, epi, id):
        if self.view == 0:
            view = "right_rgb"
        elif self.view == 1:
            view = "left_rgb"
        elif self.view == 2:
            view = "top_rgb"
        elif self.view == 3:
            view = "inhand_rgb"
        pack = {}
        task_id = str(task)
        action_search = self.action[task_id][epi]
        if id == len(action_search):
            action = np.zeros(7).astype(np.float32)
        else:
            action = action_search[id]
        pack['action'] = action
        image_path = self.data_path
        image_path = os.path.join(image_path, task_id, epi, 'images', view, f"{id:03d}.jpg")
        pack['frame'] = image_path
        pack['id'] = id
        return pack
    
    def get_six_frame_pack(self):
        if self.view == 0:
            view = "right_rgb"
        elif self.view == 1:
            view = "left_rgb"
        elif self.view == 2:
            view = "top_rgb"
        elif self.view == 3:
            view = "inhand_rgb"
        six_frame_pack = []
        for task in tqdm(range(1, 23)):
            # # single task test
            # if task != 22:
            #     continue
            if task == 3 or task == 19 or task == 20:
                continue
            task_id = str(task)
            task_path = os.path.join(self.data_path,str(task))
            episodes = os.listdir(task_path)
            for episode in episodes:
                epi_pack = {}
                epi_pack['task'] = task_id
                epi_pack['episode'] = episode
                epi_pack['prompt'] = self.prompts[task-1]
                img_indice_list = self.chosen_ids[task_id][episode]
                img_paths = os.path.join(task_path, episode, "images", view)
                epi_indices_pack = self.get_indice_packs_include_first(img_indice_list, step=6)
                epi_img = []
                for epi_indices in epi_indices_pack:
                    single_img_pack = []
                    single_act_pack = []
                    for id in range(len(epi_indices)):
                        img_id = f"{epi_indices[id]:03d}.jpg"
                        img_path = os.path.join(img_paths, img_id)
                        img = cv2.imread(img_path)
                        single_img_pack.append(img_path)
                    epi_img.append(single_img_pack)
                epi_pack['frame'] = epi_img
                six_frame_pack.append(epi_pack)
        return six_frame_pack
    
    def action_thershold_detection(self, action, threshold=[0.05, 5, 0.01], upper_thershold = [0.015, 0.78, 0.015]):
        for i in range(len(threshold)):
            assert threshold[i] <= upper_thershold[i], "threshold should be smaller than upper threshold"
        translation = np.sqrt(action[0]**2 + action[1]**2 + action[2]**2)
        rotation = np.abs(action[3]) + np.abs(action[4]) + np.abs(action[5]) 
        gripper = np.abs(action[6])
        action_state = [0,0,0]
        if translation > threshold[0]:
            action_state[0] = 1
            if translation > upper_thershold[0]:
                action_state[0] = -1
        if rotation > threshold[1]:
            action_state[1] = 1
            if rotation > upper_thershold[1]:
                action_state[1] = -1
        if gripper > threshold[2]:  
            action_state[2] = 1
            if gripper > upper_thershold[2]:
                action_state[2] = -1
        return action_state
    
    def _action_megre(self,actionA, actionB, gripper_state = "width", gripper_thershold = 0.7):
        actionC = np.array(actionA) + np.array(actionB)
        rotA = actionA[3:6]
        rotB = actionB[3:6]
        matA = R.from_euler('xyz', rotA, degrees=False).as_matrix()
        matB = R.from_euler('xyz', rotB, degrees=False).as_matrix()
        matC = np.matmul(matA, matB)
        rotC = R.from_matrix(matC).as_euler('xyz', degrees=False)
        for i in range(3):
            if rotC[i] > np.pi:
                rotC[i] = rotC[i] - 2*np.pi
            elif rotC[i] < -np.pi:
                rotC[i] = rotC[i] + 2*np.pi
        actionC[3:6] = rotC
        if gripper_state == "speed":
            actionC[6] = actionA[6] + actionB[6]
        elif gripper_state == 'width':
            actionC[6] = actionA[6]
            if actionC[6] > gripper_thershold:
                actionC[6] = 1
            else:
                actionC[6] = 0
        actionC = actionC.tolist()
        return actionC
    
    def _backward_action_search(self, task:int, episode:str, chosen_id, action_pack, id, threshold:list, upper_thershold:list):
        action = action_pack[id]
        state_pack = {
            'action': action,
            'id': id,
            'task': task,
            'episode': episode,
            'state':self.action_thershold_detection(action, threshold, upper_thershold)
        }
        index = chosen_id.index(id)
        end_index = index+1
        
        for k in range(index+1, len(chosen_id)-1):
            if np.array(self.action_thershold_detection(action_pack[chosen_id[k]], threshold, upper_thershold)).any():
                end_index = k-1
                
                break
            else:   
                pre_merge = self._action_megre(state_pack['action'], action_pack[chosen_id[k]])
                state_pack['action'] = pre_merge
                state_pack['id'] = chosen_id[k]
                state_pack['state'] = self.action_thershold_detection(pre_merge, threshold, upper_thershold)
                if np.array(self.action_thershold_detection(pre_merge, threshold, upper_thershold)).any():
                    end_index = k
                    
                    break
                
                continue
        
        return state_pack, end_index
    
    def merge_actions(self, task=1, threshold=[0.005, 0.16, 0.005], upper_thershold = [0.015, 0.78, 0.015]):
        if task == 3 or task == 19 or task == 20:
            print(f"Task {task} cannot be merged")
            return False
        task_str = str(task)
        merged_pack = []
        
        task_path = os.path.join(self.data_path,str(task))
        episodes = os.listdir(task_path)
        episode_id = 0
        back = 0
        while episode_id < len(episodes):
            # print(f"Processing episode {episode}")
            episode = episodes[episode_id]
            chosen_id = self.chosen_ids[task_str][episode][:-1]
            action_pack = self.action_wo_gripper[task_str][episode]
            state_pack = {}
            i = 0
            # print("Another loop start")
            while i < len(chosen_id):
                
                id = chosen_id[i]
                action = action_pack[id]
                # print(episode_id, episode, i, chosen_id[i], len(chosen_id), len(merged_pack))
                
                if np.array(self.action_thershold_detection(action, threshold, upper_thershold)).any():
                    state_pack = {
                        'action': action,
                        'id': id,
                        'task': task,
                        'episode': episode,
                        'state':self.action_thershold_detection(action, threshold, upper_thershold)
                    }
                    merged_pack.append(state_pack)
                    i += 1
                    continue
                else:
                    if len(merged_pack) > 0:
                        forward_state = merged_pack[-1]
                        forward_action = forward_state['action']
                        pre_merge = self._action_megre(action, forward_action)
                        state_check = self.action_thershold_detection(pre_merge, threshold, upper_thershold)
                        if -1 not in state_check and 1 in state_check:
                            merged_pack[-1]['action'] = pre_merge
                            merged_pack[-1]['state'] = self.action_thershold_detection(pre_merge, threshold, upper_thershold)
                            i += 1
                            continue
                        elif -1 not in state_check and 1 not in state_check:
                            search_pack, end_index = self._backward_action_search(task=task, episode=episode, chosen_id=chosen_id, action_pack=action_pack, id=id, threshold=threshold, upper_thershold=upper_thershold)
                            back += 1
                            merged_pack.append(search_pack)
                            i = end_index + 1
                            continue
                        elif -1 in state_check and 1 not in state_check:
                            if -1 in merged_pack[-1]['state']:
                                search_pack, end_index = self._backward_action_search(task=task, episode=episode, chosen_id=chosen_id, action_pack=action_pack, id=id, threshold=threshold, upper_thershold=upper_thershold)
                                back += 1
                                merged_pack.append(search_pack)
                                i = max(end_index + 1, i+1)
                                continue
                            else:
                                merged_pack[-1]['action'] = pre_merge
                                merged_pack[-1]['state'] = self.action_thershold_detection(pre_merge, threshold, upper_thershold)
                                i += 1  
                                continue     
                        elif -1 in state_check and 1 in state_check:
                            if -1 in merged_pack[-1]['state']:
                                search_pack, end_index = self._backward_action_search(task=task, episode=episode, chosen_id=chosen_id, action_pack=action_pack, id=id, threshold=threshold, upper_thershold=upper_thershold)
                                back += 1
                                merged_pack.append(search_pack)
                                i = max(end_index + 1, i+1)
                                continue
                            else:
                                merged_pack[-1]['action'] = pre_merge
                                merged_pack[-1]['state'] = self.action_thershold_detection(pre_merge, threshold, upper_thershold)
                                i += 1
                                continue
                    else:
                        search_pack, end_index = self._backward_action_search(task=task, episode=episode, chosen_id=chosen_id, action_pack=action_pack, id=id, threshold=threshold, upper_thershold=upper_thershold)
                        back += 1
                        merged_pack.append(search_pack)
                        i = end_index +1
                        continue
            episode_id += 1
        # Calculate mean & std of the merged pack
        action_vec = []
        for state_pack in merged_pack:
            action_vec.append(state_pack['action'])
        action_np = np.array(action_vec)
        mean = np.mean(action_np, axis=0)
        std = np.std(action_np, axis=0)
        print("Backward search count: ", back)
        return merged_pack, mean, std

if __name__ == "__main__":
    
    meta_save = 'meta_view0.json'
    pizza1 = pizzaSlice(data_path='/datahdd_8T/sep_pizza_builder/pizza_dataset/')
    slice1 = pizza1.slices
    aligned_data1 = pizza1.aligned_data
    aligned_joints1 = pizza1.aligned_joints
    prompts1 = pizza1.prompts
    chosen_ids1 = pizza1.chosen_ids
    
    pizza2 = pizzaSlice(data_path='/datahdd_8T/sep_pizza_builder/pizza_dataset/', meta_path='meta_view0.json')
    slice2 = pizza2.slices
    aligned_data2 = pizza2.aligned_data
    aligned_joints2 = pizza2.aligned_joints
    prompts2 = pizza2.prompts
    chosen_ids2 = pizza2.chosen_ids

    # print(slice1['1']['20230913195953'])
    # print(isinstance(slice1['1']['20230913195953'][0], np.ndarray))
    slice_equal = True

    for k, v in slice1.items():
        for kk, vv in v.items():
            for i in range(len(vv)):
                if (vv[i] == slice2[k][kk][i]).all():
                    continue
                else:
                    slice_equal = False
                    print(f"Item {kk} in {k} Not Equal: ", vv[i],slice2[k][kk][i])
                    print('------------------------')
    print("Slice equal stat: ", slice_equal)

    # print((slice1 == slice2).all())
    aligned_data_equal = True
    for k, v in aligned_data1.items():
        for kk, vv in v.items():
            for i in range(len(vv)):
                # print(isinstance(vv[i][0], np.ndarray), isinstance(aligned_data2[k][kk][i][0], np.ndarray))
                # break
                for j in range(len(vv[i])):
                    if (vv[i][j] == aligned_data2[k][kk][i][j]).all() :
                        continue
                    else:
                        aligned_data_equal = False
                        print(f"Item {kk} in {k} Not Equal: ", vv[i][j],aligned_data2[k][kk][i][j])
                        print('------------------------')
    aligned_joints_equal = True
    print("Aligned data equal stat: ", aligned_data_equal)
    # print(aligned_joints1['1']['20230913195953'][0])
    # print(aligned_joints2['1']['20230913195953'][0])
    def joint_equal(joint1:np.ndarray, joint2:np.ndarray):
        delta = np.abs(joint1 - joint2)
        if delta.max() > 0.000001:
            return False
        else:
            return True
    for k, v in aligned_joints1.items():
        for kk, vv in v.items():
            for i in range(len(vv)):
                for j in range(len(vv[i])):
                    if joint_equal(vv[i][j],aligned_joints2[k][kk][i][j]):
                        continue
                    else:
                        aligned_joints_equal = False
                        print(f"Item {kk} in {k} Not Equal: ", vv[i][j],aligned_joints2[k][kk][i][j])
                        print('------------------------')
    print("Aligned joints equal stat: ", aligned_joints_equal)

    chosen_ids_equal = True

    for k, v in chosen_ids1.items():
        for kk, vv in v.items():
            if vv == chosen_ids2[k][kk]:
                continue
            else:
                chosen_ids_equal = False
                print(f"Item {kk} in {k} Not Equal: ", vv,chosen_ids2[k][kk])
                print('------------------------')
            
    print("Chosen ids equal stat: ", chosen_ids_equal)

    prompts_equal = (prompts1 == prompts2)
    print("Prompts equal stat: ", prompts_equal)
