import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

# from configs.state_vec import STATE_VEC_IDX_MAPPING
from data.pizza_robot import pizza_data_class as pdc
# from pizza_robot import pizza_data_class as pdc

STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        'arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    **{
        'right_arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    'gripper_open': 10, # alias of right_gripper_joint_0_pos
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{
        'arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    **{
        'right_arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        'gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    'gripper_open_vel': 25, # alias of right_gripper_joint_0_vel
    'right_gripper_open_vel': 25,
    # [30, 33): right end effector positions
    'eef_pos_x': 30,
    'right_eef_pos_x': 30,
    'eef_pos_y': 31,
    'right_eef_pos_y': 31,
    'eef_pos_z': 32,
    'right_eef_pos_z': 32,
    # [33, 39): right end effector 6D pose
    'eef_angle_0': 33,
    'right_eef_angle_0': 33,
    'eef_angle_1': 34,
    'right_eef_angle_1': 34,
    'eef_angle_2': 35,
    'right_eef_angle_2': 35,
    'eef_angle_3': 36,
    'right_eef_angle_3': 36,
    'eef_angle_4': 37,
    'right_eef_angle_4': 37,
    'eef_angle_5': 38,
    'right_eef_angle_5': 38,
    # [39, 42): right end effector velocities
    'eef_vel_x': 39,
    'right_eef_vel_x': 39,
    'eef_vel_y': 40,
    'right_eef_vel_y': 40,
    'eef_vel_z': 41,
    'right_eef_vel_z': 41,
    # [42, 45): right end effector angular velocities
    'eef_angular_vel_roll': 42,
    'right_eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'right_eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
    'right_eef_angular_vel_yaw': 44,
    # [45, 50): reserved 
    # [50, 60): left arm joint positions
    **{
        'left_arm_joint_{}_pos'.format(i): i + 50 for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        'left_gripper_joint_{}_pos'.format(i): i + 60 for i in range(5)
    },
    'left_gripper_open': 60, # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        'left_arm_joint_{}_vel'.format(i): i + 65 for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        'left_gripper_joint_{}_vel'.format(i): i + 75 for i in range(5)
    },
    'left_gripper_open_vel': 75, # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    'left_eef_pos_x': 80,
    'left_eef_pos_y': 81,
    'left_eef_pos_z': 82,
    # [83, 89): left end effector 6D pose
    'left_eef_angle_0': 83,
    'left_eef_angle_1': 84,
    'left_eef_angle_2': 85,
    'left_eef_angle_3': 86,
    'left_eef_angle_4': 87,
    'left_eef_angle_5': 88,
    # [89, 92): left end effector velocities
    'left_eef_vel_x': 89,
    'left_eef_vel_y': 90,
    'left_eef_vel_z': 91,
    # [92, 95): left end effector angular velocities
    'left_eef_angular_vel_roll': 92,
    'left_eef_angular_vel_pitch': 93,
    'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    'base_vel_x': 100,
    'base_vel_y': 101,
    # [102, 103): base angular velocities
    'base_angular_vel': 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128



class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "/datahdd_8T/sep_pizza_builder/pizza_dataset/"
        HDF5_META = "/home/v-wenhuitan/RDT/RoboticsDiffusionTransformer/data/pizza_robot/meta_view0.json"
        self.emb_path = '/datahdd_8T/sep_pizza_builder/pizza_embedded/'
        HDF5_DIR = "/mnt/robotdata/datasets/pizza_robot/"
        HDF5_META = "/mnt/robotdata/datasets/pizza_robot/meta_view0.json"
        self.emb_path = '/mnt/robotdata/datasets/pizza_t5_embedded/'
        self.DATASET_NAME = "pizza_robot"
        
        
        self.file_paths = []
        # for root, _, files in os.walk(HDF5_DIR):
        #     for filename in fnmatch.filter(files, '*.hdf5'):
        #         file_path = os.path.join(root, filename)
        #         self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Get pizza data class
        self.pizza_data = pdc.pizzaSlice(HDF5_DIR, meta_path=HDF5_META)
        self.chosen_ids = self.pizza_data.chosen_ids
        for k,v in self.chosen_ids.items():
            for kk, vv in v.items():
                self.file_paths.append(os.path.join(HDF5_DIR, k, kk))
    
        # Get each episode's len
        # episode_lens = []
        # for file_path in self.file_paths:
        #     valid, res = self.parse_hdf5_file(file_path)
        #     # valid, res = self.parse_hdf5_file_state_only(file_path)
        #     _len = res['state'].shape[0] if valid else 0
        #     episode_lens.append(_len)
        # self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths)
                # file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path = None):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        # [Modify] We randomly sample a episode
        if file_path is None:
            task_list = np.arange(1, 23)
            task_id = np.random.choice(task_list)
            while task_id == 3 or task_id == 19 or task_id == 20:
                task_id = np.random.choice(task_list)
            chosen_task_ids = self.chosen_ids[str(task_id)]

            episode_list = np.asanyarray(list(chosen_task_ids.keys()))
            # We randomly sample a episode
            episode_id = np.random.choice(episode_list)
        else:
            info = file_path.split("/")
            task_id = int(info[-2])
            chosen_task_ids = self.chosen_ids[str(task_id)]
            episode_id  = info[-1]

        num_steps = len(chosen_task_ids[episode_id])

        # [Optional] We drop too-short episode
        if num_steps < 24:
            return False, None

        # We randomly sample a timestep
        first_idx = chosen_task_ids[episode_id][0]
        step_id = np.random.randint(first_idx+1, num_steps)

        # Load the instruction
        # emb_path = os.path.join(self.pizza_data.data_path, "..", "pizza_embedded")
        instruction = os.path.join(self.emb_path, f"lang_embed_{task_id}.pt")
        #"path/to/lang_embed[task_id].pt"
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        # [Not Used] Rescale gripper to [0, 1]
        # print(len(self.pizza_data.aligned_joints[str(task_id)][episode_id]))
        joints = np.zeros((len(self.pizza_data.aligned_joints[str(task_id)][episode_id]), 7)).astype(np.float32)
        for i, joint in enumerate(self.pizza_data.aligned_joints[str(task_id)][episode_id]):
            joints[i] = joint[-1]
        gripper = self.pizza_data.action_wo_gripper[str(task_id)][episode_id][:, 6:]
        gripper_last = gripper[-1]
        gripper = np.vstack((gripper, gripper_last))
        # [Not Used] Rescale gripper to [0, 1]
        for i in range(len(gripper)):
            if gripper[i][0] <= 0.067:
                gripper[i][0] = 0
            else:
                gripper[i][0] = 1
        qpos = np.hstack((joints, gripper))
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        qpos_ids = chosen_task_ids[episode_id]
        # Actual Chunk Size: 16
        CHUNK_SIZE = self.CHUNK_SIZE
        ACTUAL_CHUNK_SIZE = 16
        target_ids = qpos_ids[step_id - 1 : step_id + ACTUAL_CHUNK_SIZE]
        target_qpos = np.zeros((self.CHUNK_SIZE + 1, 8)).astype(np.float32)
        ACTUAL_LENGTH = 0
        for i in range(len(target_ids)):
            target_qpos[i] = qpos[target_ids[i]]
            ACTUAL_LENGTH += 1
        ACTUAL_LENGTH -= 1
        actions = target_qpos[1:]

        # Parse the state and action
        state = np.zeros((1, 8)).astype(np.float32)
        state[0] = target_qpos[0]

        zero_action = np.zeros(8).astype(np.float32)
        for i in range(8):
            zero_action[i] = 100

        def padding_state(value, actual, expected):

            if actual < expected:
                for i in range(1, expected - actual + 1):
                    value[-i] = zero_action
            return value
        
        actions = padding_state(actions, ACTUAL_LENGTH, CHUNK_SIZE)

        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 7 joints + 1 gripper for franka arm
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
            ] + [
                STATE_VEC_IDX_MAPPING["right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        
        state_indicator = fill_in_state(np.ones_like(state[0]))
        state = fill_in_state(state)
        state_std = self.pizza_data.joints_std
        state_mean = self.pizza_data.joints_mean
        state_std = fill_in_state(state_std)
        state_norm = fill_in_state(state_norm)
        state_mean = fill_in_state(state_mean)

        actions = fill_in_state(actions)

        def unavailable_img():
            return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))

        def parse_img(task: str, epi: str, id):
            imgs= []
            if self.pizza_data.view == 0:
                view = "right_rgb"
            elif self.pizza_data.view == 1:
                view = "left_rgb"
            elif self.pizza_data.view == 2:
                view = "top_rgb"
            elif self.pizza_data.view == 3:
                view = "inhand_rgb"
            
            for i in range(max(id - self.IMG_HISORY_SIZE + 1, 0), id + 1):
                img_id = self.pizza_data.chosen_ids[task][epi][i]
                img_path = os.path.join(self.pizza_data.data_path, task, epi, 'images', view, f"{img_id:03d}.jpg")
                img = cv2.imread(img_path)
                img = img[40:720,200:880,:] 
                # img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
                imgs.append(img)
            imgs = np.stack(imgs)
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                # Pad the images using the first image
                imgs = np.concatenate([
                    np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                    imgs
                ], axis=0)
            return imgs
        
        cam_right_wrist = parse_img(str(task_id), episode_id, step_id)
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        cam_right_wrist_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
        
        cam_left_wrist = unavailable_img()
        cam_left_wrist_mask = cam_right_wrist_mask.copy()

        cam_high = unavailable_img()
        cam_high_mask = cam_right_wrist_mask.copy()

        # Return the resulting sample
        # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
        # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
        # if the left-wrist camera is unavailable on your robot
        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }
    
    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        # [Modify] We randomly sample a episode
        if file_path is None:
            task_list = np.arange(1, 23)
            task_id = np.random.choice(task_list)
            while task_id == 3 or task_id == 19 or task_id == 20:
                task_id = np.random.choice(task_list)
            chosen_task_ids = self.chosen_ids[str(task_id)]

            episode_list = np.asanyarray(list(chosen_task_ids.keys()))
            # We randomly sample a episode
            episode_id = np.random.choice(episode_list)
        else:
            info = file_path.split("/")
            task_id = int(info[-2])
            chosen_task_ids = self.chosen_ids[str(task_id)]
            episode_id  = info[-1]

        num_steps = len(chosen_task_ids[episode_id])

        # [Optional] We drop too-short episode
        if num_steps < 24:
            return False, None

        # We randomly sample a timestep
        first_idx = chosen_task_ids[episode_id][0]

        joints = np.zeros((len(self.pizza_data.aligned_joints[str(task_id)][episode_id]), 7)).astype(np.float32)
        for i, joint in enumerate(self.pizza_data.aligned_joints[str(task_id)][episode_id]):
            joints[i] = joint[-1]
        gripper = self.pizza_data.action_wo_gripper[str(task_id)][episode_id][:, 6:]
        gripper_last = gripper[-1]
        gripper = np.vstack((gripper, gripper_last))
        # [Not Used] Rescale gripper to [0, 1]
        for i in range(len(gripper)):
            if gripper[i][0] <= 0.067:
                gripper[i][0] = 0
            else:
                gripper[i][0] = 1
        qpos = np.hstack((joints, gripper))

        state = np.zeros((len(chosen_task_ids[episode_id]), 8)).astype(np.float32)

        for id in range(len(chosen_task_ids[episode_id])):
            state[id] = qpos[chosen_task_ids[episode_id][id]]
        actions = state[1:]

        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 7 joints + 1 gripper for franka arm
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
            ] + [
                STATE_VEC_IDX_MAPPING["right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        state = fill_in_state(state)
        action = fill_in_state(actions)

        return True,{
            'state': state,
            'action': action
        }
        
        # with h5py.File(file_path, 'r') as f:
        #     qpos = f['observations']['qpos'][:]
        #     num_steps = qpos.shape[0]
        #     # [Optional] We drop too-short episode
        #     if num_steps < 128:
        #         return False, None
            
        #     # [Optional] We skip the first few still steps
        #     EPS = 1e-2
        #     # Get the idx of the first qpos whose delta exceeds the threshold
        #     qpos_delta = np.abs(qpos - qpos[0:1])
        #     indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        #     if len(indices) > 0:
        #         first_idx = indices[0]
        #     else:
        #         raise ValueError("Found no qpos that exceeds the threshold.")
            
        #     # Rescale gripper to [0, 1]
        #     qpos = qpos / np.array(
        #        [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
        #     )
        #     target_qpos = f['action'][:] / np.array(
        #        [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
        #     )
            
        #     # Parse the state and action
        #     state = qpos[first_idx-1:]
        #     action = target_qpos[first_idx-1:]
            
        #     # Fill the state/action into the unified vector
        #     def fill_in_state(values):
        #         # Target indices corresponding to your state space
        #         # In this example: 6 joints + 1 gripper for each arm
        #         UNI_STATE_INDICES = [
        #             STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
        #         ] + [
        #             STATE_VEC_IDX_MAPPING["left_gripper_open"]
        #         ] + [
        #             STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
        #         ] + [
        #             STATE_VEC_IDX_MAPPING["right_gripper_open"]
        #         ]
        #         uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
        #         uni_vec[..., UNI_STATE_INDICES] = values
        #         return uni_vec
        #     state = fill_in_state(state)
        #     action = fill_in_state(action)
            
        #     # Return the resulting sample
        #     return True, {
        #         "state": state,
        #         "action": action
        #     }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    sample = ds.get_item(100)
    print(sample['actions'].shape, sample['state'].shape, sample['state_indicator'].shape)
    print(f"std: {sample['state_std'].shape}, mean: {sample['state_mean'].shape}, norm: {sample['state_norm'].shape}, actions: {sample['actions'].shape}, cam_right_wrist: {sample['cam_right_wrist'].shape}, cam_right_wrist_mask: {sample['cam_right_wrist_mask'].shape}")
    print(f"cam_high: {sample['cam_high'].shape}, cam_high_mask: {sample['cam_high_mask'].shape}, cam_left_wrist: {sample['cam_left_wrist'].shape}, cam_left_wrist_mask: {sample['cam_left_wrist_mask'].shape}")
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        sample = ds.get_item(i, state_only=False)
    #     # print(sample['cam_high'].shape, sample['cam_left_wrist'].shape, sample['cam_right_wrist'].shape)
    #     # print(sample['state_std'])
    #     print(sample['state'].shape)
    # import json
    # stat_path = "/home/v-wenhuitan/RDT/RoboticsDiffusionTransformer/configs/dataset_stat.json"
    # with open(stat_path, 'r') as f:
    #     dataset_stat = json.load(f)
    # ds_state_mean = np.array(dataset_stat['pizza_robot']['state_mean'])
    # print(ds_state_mean.shape)
    # ds_state_mean = np.tile(ds_state_mean[None], (1, 1))
    # print(ds_state_mean.shape)
    # for i in range(len(sample['actions'])):
    #     print(sample['actions'][i][:7] )