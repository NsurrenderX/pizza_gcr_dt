import torch

# action_mask = torch.randn((32, 1, 1, 128))
# action_mask = action_mask.expand(-1, 65, -1, -1)
# state_action_traj = torch.randn((32, 65, 2, 128))
# state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
# print(state_action_traj.shape)

# print(action_mask.shape)

def padding_state(value, actual, expected):

    if actual < expected:
        for i in range(1, expected - actual + 1):
            value[-i] = value[actual - 1]
    return value

import numpy as np

if __name__ == '__main__':
    npy_path = "/datahdd_8T/sep_pizza_builder/pizza_dataset/6/20230825161210/franka_data.npy"
    imag_path = "/datahdd_8T/sep_pizza_builder/pizza_dataset/6/20230825161210/images/right_rgb/001.jpg"

    robot_state = np.load(npy_path, allow_pickle=True)
    print(robot_state[0]['joint_position'])
    print(isinstance(robot_state[0]['joint_position'], np.ndarray))
    print(robot_state[0]['joint_position'].shape)