# Third Party
import torch
import time
import numpy as np
from frankapy import FrankaArm

import math

from tqdm import tqdm

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from scipy.spatial.transform import Rotation as R

class frankaRe3:
    def __init__(self, yaml_path = 'fr3.yml', with_robot = True):
        self.ik_solver = self.init_curobo_ik_solver(yaml_path)
        if with_robot:
            self.arm = FrankaArm()
            self.init_trans, self.init_rotate = self.get_init_arm_pose()
            self.target_trans, self.target_rotate = self.init_trans, self.init_rotate

    def init_curobo_ik_solver(self, yaml_path):
        tensor_args = TensorDeviceType()

        config_file = load_yaml(join_path(get_robot_configs_path(), yaml_path))
        urdf_file = config_file["robot_cfg"]["kinematics"][
            "urdf_path"
        ]  # Send global path starting with "/"
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)
        return ik_solver
    
    def reset_joint(self):
        self.arm.reset_joints()

    def reset_gripper(self):
        self.arm.open_gripper()

    def reset_arm(self):
        
        self.reset_gripper()
        self.reset_joint()

    def get_init_arm_pose(self):
        joint_state = self.arm.get_joints()
        joint_state = torch.tensor(joint_state).float().cuda()
        kin_state = self.ik_solver.fk(joint_state)
        trans = kin_state.ee_position[0].cpu().numpy().tolist()
        rotate = self.scipy_quaternion_to_euler(kin_state.ee_quaternion[0].cpu().numpy().tolist())
        rotate_mat = self.scipy_euler_to_matrix(rotate)
        # rotate = [rotate[2], rotate[1], rotate[0]]
        return trans, rotate_mat
    
    def get_current_pose(self):
        joint_state = self.arm.get_joints()
        joint_state = torch.tensor(joint_state).float().cuda()
        kin_state = self.ik_solver.fk(joint_state)
        trans = kin_state.ee_position[0].cpu().numpy().tolist()
        rotate = kin_state.ee_quaternion[0].cpu().numpy().tolist()
        return trans, rotate
    
    def get_ik_res(self, action):
        action = self.format_action_euler(action)
        for i in range(3):
            self.target_trans[i] += action[0][i]
            # self.target_rotate[i] += action[1][i]
        rotate_mat = self.scipy_euler_to_matrix(action[1])
        self.target_rotate = rotate_mat @ self.target_rotate
        eef_pos = torch.tensor(self.target_trans).float().cuda()
        eef_rot = torch.tensor(self.scipy_matrix_to_quaternion(self.target_rotate)).float().cuda()
        goal = Pose(eef_pos, eef_rot)

        result = self.ik_solver.solve_batch(goal)
        solution = result.solution[result.success][0].cpu().numpy().tolist()
        return solution
    
    def format_action_euler(self, action):
        euler = action[1]
        euler = [euler[2], euler[1], euler[0]]
        action[1] = euler

        return action
        

    def get_ik_res_markov(self, action):
        joint_state = self.arm.get_joints()
        joint_state = torch.tensor(joint_state).float().cuda()

        kin_state = self.ik_solver.fk(joint_state)
        eef_pos = kin_state.ee_position[0].cpu().numpy().tolist()
        action = self.format_action_euler(action)
        for i in range(3):
            eef_pos[i] += action[0][i]
        
        eef_euler = self.scipy_quaternion_to_euler(kin_state.ee_quaternion[0].cpu().numpy().tolist())
        eef_matrix = self.scipy_euler_to_matrix(eef_euler)
        rotate_matrix = self.scipy_euler_to_matrix(action[1])
        eef_matrix = np.dot(rotate_matrix, eef_matrix)
        

        goal_pose = torch.tensor(eef_pos).float().cuda()
        goal_quat = torch.tensor(self.scipy_matrix_to_quaternion(eef_matrix)).float().cuda()
        goal = Pose(goal_pose, goal_quat)

        result = self.ik_solver.solve_batch(goal)

        solution = result.solution[result.success][0].cpu().numpy().tolist()
        return solution
    
    def scipy_quaternion_to_euler(self, wxvz_quat):
        return R.from_quat(wxvz_quat).as_euler('xyz', degrees=False)
    
    def scipy_euler_to_matrix(self, euler):
        return R.from_euler('xyz', euler, degrees=False).as_matrix()

    def scipy_euler_to_quaternion(self, euler):
        return R.from_euler('xyz', euler, degrees=False).as_quat()
    
    def scipy_matrix_to_quaternion(self, matrix):
        return R.from_matrix(matrix).as_quat()
    
    def exec_action(self, action, duration = 5.0, buffer_time = 1.0, ignore_virtual_walls=True):
        time_start = time.time()
        solution = self.get_ik_res(action)
        time_get_result = time.time()
        self.arm.goto_joints(solution, duration, buffer_time, ignore_virtual_walls=ignore_virtual_walls)
        time_set_joints = time.time()

        gripper_width = action[2]
        current_width = self.arm.get_gripper_width()
        print("Current gripper width: ", current_width, "Gripper width: ", gripper_width)
        if current_width >= 0.015:
            current_state = 1
        else:
            current_state = -1
        if gripper_width > 0:
            gripper = -1
        else:
            gripper = 1
        if current_state * gripper == -1:
            if current_state == 1:
                print("Closing gripper")
                self.arm.goto_gripper(0.0, speed=1.5)
            else:
                print("Opening gripper")
                self.arm.goto_gripper(0.078, speed=1.5)
        
        print("get result time: ", time_get_result - time_start)
        print("set joints time: ", time_set_joints - time_get_result)
        print("total time: ", time_set_joints - time_start)
        return (time_set_joints - time_start), solution
    def exec_action_markov(self, action, duration = 5.0, buffer_time = 1.0):
        time_start = time.time()
        solution = self.get_ik_res_markov(action)
        time_get_result = time.time()
        self.arm.goto_joints(solution, duration, buffer_time)
        time_set_joints = time.time()

        gripper_width = action[2]
        current_width = self.arm.get_gripper_width()
        if current_width >= 0.01:
            current_state = 1
        else:
            current_state = -1
        if gripper_width > 0:
            gripper = -1
        else:
            gripper = 1
        if current_state * gripper == -1:
            if current_state == 1:
                self.arm.goto_gripper(0.0, speed=1.5)
            else:
                self.arm.goto_gripper(0.078, speed=1.5)
        
        print("get result time: ", time_get_result - time_start)
        print("set joints time: ", time_set_joints - time_get_result)
        print("total time: ", time_set_joints - time_start)
        return (time_set_joints - time_start), solution
    def get_batched_ik_res(self, actions):
        batch_size = actions[0].shape[0]
        batched_trans = np.zeros((batch_size, 3)).astype(np.float32)
        batched_quaternion = np.zeros((batch_size, 4)).astype(np.float32)
        
        for i in range(batch_size):
            
            single_pos = actions[0][i]
            single_rot = self.scipy_euler_to_quaternion(actions[1][i])
            batched_trans[i] =  single_pos
            batched_quaternion[i] = single_rot
            
        eef_pos = torch.tensor(batched_trans).float().cuda()
        eef_rot = torch.tensor(batched_quaternion).float().cuda()
        goal = Pose(eef_pos, eef_rot)

        result = self.ik_solver.solve_batch(goal)
        solution = result.solution[result.success][0].cpu().numpy().tolist()
        return solution
    def benchmark(self, total_solver = 1000, batch_size = 6):
        print("Initiating benchmark...")
        sample_trans = np.random.rand(batch_size, 3)
        sample_quat = np.random.rand(batch_size, 3)
        sample_action = [sample_trans, sample_quat]
        iter_time = int(total_solver / batch_size + 0.999)
        print("Start benchmarking, total iteration: ", iter_time, "batch size: ", batch_size)
        solution = self.get_batched_ik_res(sample_action)
        start_time = time.time()
        for i in tqdm(range(iter_time)):
            time_mark = time.time()
            solution = self.get_batched_ik_res(sample_action)
            print("Iteration time: ", time.time() - time_mark)
        print("Benchmark finished, total step: ", total_solver, ", total time: ", time.time() - start_time, "batch size: ", batch_size, ", single step time: ", (time.time() - start_time)/total_solver)
    