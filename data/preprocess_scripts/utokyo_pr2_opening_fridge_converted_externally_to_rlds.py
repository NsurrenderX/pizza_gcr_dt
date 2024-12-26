import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix, \
    rotation_matrix_to_ortho6d


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.reduce_all(tf.equal(terminate_act, tf.constant([1, 0, 0], dtype=tf.float32)))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    
    # Convert raw action to our action
    action = step['action']
    # Robot action, consists of [3x end effector pos, 3x robot rpy angles, 1x gripper open/close command, 1x terminal action].
    eef_delta_pos = action[:3]/1000 # change from mm to m
    eef_ang = action[3:6]
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open = tf.expand_dims(1 - action[6], axis=0)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    
    action['terminate'] = step['is_terminal']
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")


    # Convert raw state to our state
    state = step['observation']['state']
    # Robot state, consists of [3x end effector pos, 3x robot rpy angles, 1x gripper position].
    gripper_pos = state[:3]/1000 # change from mm to m
    gripper_ang = state[3:6]
    gripper_ang = euler_to_rotation_matrix(gripper_ang)
    gripper_ang = rotation_matrix_to_ortho6d(gripper_ang)
    gripper_open = state[6:7]/1000 * 11.54 # rescale to [0, 1]


    # Concatenate the state
    state = step['observation']
    state['arm_concat'] = tf.concat([gripper_pos, gripper_ang, gripper_open], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_joint_0_pos")

    # Clean the task instruction
    # Define the replacements (old, new) as a dictionary
    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
        
    }
    instr = step['language_instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'fractal20220817_data'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
