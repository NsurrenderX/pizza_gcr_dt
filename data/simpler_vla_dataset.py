import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()
    
    return quat

def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix

def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=False)
    
    return euler

def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out


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
        HDF5_DIR = "/mnt/wangxiaofa/robot_dataset/simpler_data/"
        # HDF5_DIR = "/data_16T/simpler/simpler_data/"
        self.DATASET_NAME = "simpler"
        self.emb_path = ""
        
        self.file_paths = []
        hdf5_list = os.listdir(HDF5_DIR)
        for task_dir in tqdm(hdf5_list, desc="Loading datalist from dataset"):
            task_dir_path = os.path.join(HDF5_DIR, task_dir)
            task_list = os.listdir(task_dir_path)
            for file in task_list:
                specific_path = os.path.join(task_dir_path, file)
                self.file_paths.append(specific_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in tqdm(self.file_paths, desc="Generating weights from raw data"):
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
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
                # file_path = np.random.choice(self.file_paths)
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
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
        # Load both frames.npy & action.npy
        frame_path = os.path.join(file_path, 'frames.npy')
        action_path = os.path.join(file_path, 'action.npy')
        frames = np.load(frame_path)
        actions = np.load(action_path)
        
        num_steps = len(actions)
        # [Optional] We drop too-short episode
        if num_steps < 30:
            return False, None
        
        init_coordinate = np.zeros_like(actions[0]).astype(np.float32)
        
        # We randomly sample a timestep
        first_idx = 0
        step_id = np.random.randint(0, num_steps-1)
        
        # Load the instruction
        instruction = os.path.join(file_path, "lang_embed.pt")
        
        #meta data for this trajectory
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }
        
        def get_state_from_action(id):
            current_coordinate = np.zeros((1, 10)).astype(np.float32)
            init_rotmat = convert_euler_to_rotation_matrix(np.expand_dims(init_coordinate[3:6], axis=0))[0]
            current_rotmat = init_rotmat
            for i in range(id):
                for k in range(3):
                    current_coordinate[0][k] += actions[i][k]
                rotmat = convert_euler_to_rotation_matrix(np.expand_dims(actions[i][3:6], axis=0))[0]
                current_rotmat = rotmat @ current_rotmat
            
            current_ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(current_rotmat, axis=0))[0]
            current_coordinate[0][3:9] = current_ortho6d
            current_coordinate[0][9] = 0 if actions[id][6] < 0 else 1
            return current_coordinate
        
        qpos = np.zeros((actions.shape[0], 10)).astype(np.float32)
        for i in range(actions.shape[0]):
            qpos[i][:3] = actions[i][:3]
            rotmat = convert_euler_to_rotation_matrix(np.expand_dims(actions[i][3:6], axis=0))[0]
            ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(rotmat, axis=0))[0]
            qpos[i][3:9] = ortho6d
            qpos[i][9] = 0 if actions[i][6] < 0 else 1
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        
        CHUNK_SIZE = self.CHUNK_SIZE
        ACTUAL_CHUNK_SIZE = min(CHUNK_SIZE, num_steps - step_id)
        
        target_qpos = np.zeros((self.CHUNK_SIZE, 10)).astype(np.float32)
        state = np.zeros((1, 10)).astype(np.float32)
        
        for i in range(ACTUAL_CHUNK_SIZE):
            target_qpos[i] = qpos[step_id + i]
        
        state[0] = get_state_from_action(step_id)
        
        zero_action = np.zeros(10).astype(np.float32)
        for i in range(zero_action.shape[0]):
            zero_action[i] = np.pi*2
        
        def padding_state(value, actual, expected=0):
            
            if actual < expected:
                for i in range(1, expected - actual + 1 ):
                    # value[-i] = value[actual - 1]
                    value[-i] = zero_action
            return value
        
        target_qpos = padding_state(target_qpos, ACTUAL_CHUNK_SIZE, CHUNK_SIZE)
        
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 3 translation, 6 rotation, 1 gripper
            UNI_STATE_INDICES = [ STATE_VEC_IDX_MAPPING['eef_pos_x']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_y']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
                    ] + [
                        STATE_VEC_IDX_MAPPING["right_gripper_open"]
                    ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
            
        state_indicator = fill_in_state(np.ones_like(state[0]))
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        
        actions = fill_in_state(target_qpos)
        
        def unavailable_img():
            return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
        
        def parse_img(id:int):
            imgs = []

            if id > 0:
                for i in range(max(id - self.IMG_HISORY_SIZE + 1, 0), id + 1):
                    img = frames[i]
                    # img = img[:, 65:575, :]
                    imgs.append(img)
            else:
                img = frames[id]
                imgs.append(img)
            imgs = np.stack(imgs)
            
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                # Pad the image squence using the first image
                imgs = np.concatenate([
                    np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                    imgs
                    ], axis=0)
            
            return imgs
        
        cam_right_wrist = parse_img(step_id)
        valid_len = min(step_id - first_idx + 1, self.IMG_HISORY_SIZE)
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
        # Load both action.npy
        action_path = os.path.join(file_path, 'action.npy')

        actions = np.load(action_path)
        
        num_steps = len(actions)
        # [Optional] We drop too-short episode
        if num_steps < 30:
            return False, None
        
        init_coordinate = np.zeros_like(actions[0]).astype(np.float32)
        
        def get_state_from_action(id):
            current_coordinate = np.zeros((1, 10)).astype(np.float32)
            init_rotmat = convert_euler_to_rotation_matrix(np.expand_dims(init_coordinate[3:6], axis=0))[0]
            current_rotmat = init_rotmat
            for i in range(id):
                for k in range(3):
                    current_coordinate[0][k] += actions[i][k]
                rotmat = convert_euler_to_rotation_matrix(np.expand_dims(actions[i][3:6], axis=0))[0]
                current_rotmat = rotmat @ current_rotmat
            
            current_ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(current_rotmat, axis=0))[0]
            # print(current_ortho6d.shape)
            current_coordinate[0][3:9] = current_ortho6d
            current_coordinate[0][9] = 0 if actions[id][6] < 0 else 1
            return current_coordinate
        qpos = np.zeros((actions.shape[0] + 1, 10)).astype(np.float32)
        
        for i in range(1, actions.shape[0] + 1):
            qpos[i] = get_state_from_action(i-1)
            
        target_qos = np.zeros((actions.shape[0], 10)).astype(np.float32)
        for i in range(actions.shape[0]):
            target_qos[i][:3] = actions[i][:3]
            euler = actions[i][3:6]
            rotmat = convert_euler_to_rotation_matrix(np.expand_dims(euler, axis=0))[0]
            ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(rotmat, axis=0))[0]
            target_qos[i][3:9] = ortho6d
            target_qos[i][9] = 0 if actions[i][6] < 0 else 1
            
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 3 translation, 6 rotation, 1 gripper
            UNI_STATE_INDICES = [ STATE_VEC_IDX_MAPPING['eef_pos_x']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_y']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
                    ] + [
                        STATE_VEC_IDX_MAPPING["right_gripper_open"]
                    ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        
        qpos = fill_in_state(qpos)
        target_qos = fill_in_state(target_qos)
        
        return True, {
            "state": qpos,
            "action": target_qos
        }
        

if __name__ == "__main__":
    ds = HDF5VLADataset()
    abnormal_shape =0
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        sample = ds.get_item(i)
        print(sample['cam_right_wrist'].shape[0], sample['cam_right_wrist_mask'].shape[0], abnormal_shape)
        if (sample['cam_right_wrist'].shape[0] != 2) or (sample['cam_right_wrist_mask'].shape[0] != 2):
            abnormal_shape += 1
    print(abnormal_shape)
        
