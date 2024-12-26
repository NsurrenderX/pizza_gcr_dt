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
    value = np.zeros([6, 3])
    rand = np.random.rand(3, 3)
    value[:rand.shape[0], :rand.shape[1]] = rand
    print(value)
    print(padding_state(value, 3, 6))