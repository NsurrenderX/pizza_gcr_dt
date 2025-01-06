from frankaRe3Curobo import frankaRe3

import requests
import os
import yaml
import base64
import numpy as np
import json
import cv2
import imageio

from PIL import Image

def decode_b64_image(b64image):
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image_cv2 = image_cv2[40:720,200:880,:]
    return image_cv2

def get_image(camera_url = "http://127.0.0.1:5000/get_full"):
    resp = requests.get(url=camera_url)
    images = resp.json()
    images = images
    print("Image received")
    return images

def exec_robot(fr3, idx, action):
    print("=======================================")
    print("Executing action: ")
    # print("Current Step: ", idx)
    print("Detailed action: ", action)
    fr3.arm.goto_joints(action[:7], duration=1.5, buffer_time=0.1, ignore_virtual_walls = True)
    current_width = fr3.arm.get_gripper_width()
    tgt_gripper = action[7]
    if current_width >= 0.015:
        current_gripper = 1
    else:
        current_gripper = -1
    if tgt_gripper > 0.001:
        gripper = -1
    else:
        gripper = 1
    if current_gripper * gripper == -1:
        if current_gripper == 1:
            print("Closing Gripper")
            fr3.arm.goto_gripper(0.0, speed=1.5)
        else:
            print("Opening Gripper")
            fr3.arm.goto_gripper(0.078, speed=1.5)
    
    joint = fr3.arm.get_joints()
    gripper = fr3.arm.get_gripper_width()
    print("Current Joint: ", joint)
    print("Current Gripper: ", gripper)
    print("=======================================\n")

    joint_queue.append(joint.tolist())
    gripper_queue.append(gripper)
    image_queue.append(get_image()['k4a_0'])

    return joint, gripper

def control_loop(fr3, task_id, exec_per_step, max_step):
    global joint_queue, gripper_queue, image_queue, config
    step = 0
    while step < max_step:
        # images = get_image()
        # image_queue.append(images['k4a_0'])
        # joint = fr3.arm.get_joints()
        # gripper = fr3.arm.get_gripper_width()
        # joint_queue.append(joint.tolist())
        # gripper_queue.append(gripper)
        instance_data = {
            'joints': joint_queue,
            'gripper': gripper_queue,
            'image': [image_queue[-2], image_queue[-1]] if len(image_queue) > 1 else [image_queue[-1]],
            'task_id': task_id
        }
        action_json = requests.post(url = config['url'], json=instance_data)
        actions = action_json.json()['actions']

        for idx in range(exec_per_step):
            print("Current Step: ", step)
            action = actions[idx]
            exec_robot(fr3, idx, action)
            step += 1



if __name__ == "__main__":
    # Load config
    yaml_path = "configs/pizza.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Robot Init
    fr3 = frankaRe3(config['franka']['config'])

    # Init State Data
    joint_queue = []
    gripper_queue = []
    first_joint = fr3.arm.get_joints()
    first_gripper = fr3.arm.get_gripper_width()

    joint_queue.append(first_joint.tolist())
    gripper_queue.append(first_gripper)

    # Init Image Data
    images = get_image()
    image_queue = []
    image_queue.append(images['k4a_0'])

    control_loop(fr3, config['task_id'], config['exec']['inference_per_step'], config['exec']['max_step'])

    img_IMAGE_queue = []
    for img in image_queue:
        img_IMAGE_queue.append(Image.fromarray(decode_b64_image(img)))

    # Save Data
    imageio.mimsave('./latest.gif', img_IMAGE_queue, "GIF", fps=10)



