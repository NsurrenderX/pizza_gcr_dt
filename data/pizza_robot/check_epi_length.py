from pizza_data_class import pizzaSlice
import numpy as np

if __name__ == "__main__":
    pizza_path = "/datahdd_8T/sep_pizza_builder/pizza_dataset/"
    pizza = pizzaSlice(data_path=pizza_path)

    small_epi = 0
    tiny_epi = 0
    ex_tiny_epi = 0
    count_epi = 0

    for task in range(1, 23):
        task_id = str(task)
        if task == 3 or task == 19 or task == 20:
            continue
        choosen_indices = pizza.chosen_ids[task_id]
        for episode, indices in choosen_indices.items():
            # print(task, episode, len(indices))
            if len(indices) < 128:
                small_epi += 1
                if len(indices) < 65:
                    tiny_epi += 1
                    if len(indices) < 32:
                        ex_tiny_epi += 1
            count_epi += 1

    print("small_epi", small_epi)
    print("tiny_epi", tiny_epi)
    print("ex_tiny_epi", ex_tiny_epi)
    print("count_epi", count_epi)
    print(pizza.chosen_ids['1'].keys())
    print(len(pizza.chosen_ids['1'].keys()))
    epi_list = np.asanyarray(list(pizza.chosen_ids['1'].keys()))
    epi_id = np.random.choice(epi_list)
    id_len = len(pizza.chosen_ids['1'][epi_id])
    print(epi_id, id_len)
    # print(pizza.aligned_joints['1'][epi_id])
    print(pizza.aligned_joints['1'][epi_id][0])
    joints = np.zeros((len(pizza.aligned_joints['1'][epi_id]), 7)).astype(np.float32)
    for i, joint in enumerate(pizza.aligned_joints['1'][epi_id]):
        joints[i] = joint[-1]
    print(joints.shape)
    # print(np.asanyarray(pizza.aligned_joints['1'][epi_id]).reshape(-1, 7).shape)
    gripper = pizza.action_wo_gripper['1'][epi_id][:, 6:]
    gripper_last = gripper[-1]
    gripper = np.vstack((gripper, gripper_last))
    for i in range(len(gripper)):
        if gripper[i][0] <= 0.067:
            gripper[i][0] = 0
        else:
            gripper[i][0] = 1
    print(gripper.shape)
    action_w_joints = np.hstack((joints, gripper))
    print(action_w_joints.shape)
    # print(action_w_joints)
    print(action_w_joints[0])
    print(action_w_joints[-1])
    print(pizza.joints_std, pizza.joints_mean)