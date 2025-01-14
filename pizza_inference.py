import os
import numpy as np
import torch
import requests
import cv2
import yaml
import base64
import time
import math

from flask import Flask, request, jsonify
from PIL import Image

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
# from frankaRe3Curobo import frankaRe3

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

def get_image(camera_url = "http://127.0.0.1:5000/get_full"):
    resp = requests.get(url=camera_url)
    images = resp.json()
    images = images
    print("Image received")
    return images

def decode_b64_image(b64image):
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image_cv2 = image_cv2[40:720,200:880,:]
    return image_cv2

def extract_state(action_chunk):
    action_chunk = action_chunk.to(dtype = torch.float32).detach().cpu().numpy()[0]
    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING["right_gripper_open"]
    ]
    uni_vec = action_chunk[ : , UNI_STATE_INDICES]
    return uni_vec.tolist()

def fill_in_state(values):
    # Target indices corresponding to your state space
    # In our data: 7 joints + 1 gripper for franka arm
    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING["right_gripper_open"]
    ]
    uni_vec = np.zeros(values.shape[:-1] + (STATE_VEC_LEN,))
    uni_vec[..., UNI_STATE_INDICES] = values
    return uni_vec

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def pizza_in_datastream(instance_data: dict):
    """
        instance_data: dict
            {
                'joints': list history*7
                'gripper': float history*1
                'image'
                'task_id'    
            }
    """
    lang_embed_dir = '/datahdd_8T/sep_pizza_builder/pizza_embedded/'
    lang_embed_format = "lang_embed_format.pt"
    lang_embed_path = os.path.join(lang_embed_dir, lang_embed_format.replace("format", str(instance_data['task_id'])))
    meta = {
        "dataset_name": "pizza_robot",
        "instruction": lang_embed_path
    }
    img_history_size = 2
    num_cameras = 3

    history = len(instance_data['joints'])
    history_gripper = len(instance_data['gripper'])
    assert history == history_gripper , "joint length and gripper length should be the same"
    dof = len(instance_data['joints'][0])
    joint = np.zeros((history, dof)).astype(np.float32)
    for idx in range(history):
        joint[idx] = instance_data['joints'][-(history-idx)]
    gripper = np.array(instance_data['gripper']).astype(np.float32)
    gripper = np.expand_dims(gripper, axis=1)
    print(joint.shape, gripper.shape)
    qpos = np.hstack((joint, gripper))
    state_norm = np.sqrt(np.mean(qpos**2, axis=0))
    state = qpos[-1]
    state_indicator = fill_in_state(np.ones_like(state[0]))
    state = fill_in_state(state)
    state_norm = fill_in_state(state_norm)
    actions = state.copy()
    state = np.expand_dims(state, axis=0)
    def unavailable_img():
        return np.zeros((img_history_size, 0, 0, 0))
    def parse_img(image_queue):
        imgs = []
        status = True
        for idx, img in enumerate(image_queue):
            #crop the image to target square
            img = img[40:720, 200:880, :]
            imgs.append(img)
        imgs = np.stack(imgs)
        if imgs.shape[0] < img_history_size:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (img_history_size - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)
            status = False
        return imgs, status
    cam_right_wrist, status = parse_img(instance_data['image'])
    if status:
        valid_len = 2
    else:
        valid_len = 1
    cam_right_wrist_mask = np.array(
        [False] * (img_history_size - valid_len) + [True] * valid_len
        )

    cam_left_wrist = unavailable_img()
    cam_left_wrist_mask = cam_right_wrist_mask.copy()

    cam_high = unavailable_img()
    cam_high_mask = cam_right_wrist_mask.copy()

    # print('cam_right_wrist_mask', cam_right_wrist_mask)
    # print('cam_left_wrist_mask', cam_left_wrist_mask)
    # print('cam_high_mask', cam_high_mask)
    # print('cam_high prob', math.prod(cam_high.shape))
    # print('cam_right_wrist prob', math.prod(cam_right_wrist.shape))
    # print('cam_left_wrist prob', math.prod(cam_left_wrist.shape))

    data_dict = {
        'img_history_size': img_history_size,
        'num_cameras': num_cameras,
        'state_indicator': state_indicator,
        'state': state,
        'state_norm': state_norm,
        'meta': meta,
        'actions': actions,
        'cam_high': cam_high,
        'cam_high_mask': cam_high_mask,
        'cam_right_wrist': cam_right_wrist,
        'cam_right_wrist_mask': cam_right_wrist_mask,
        'cam_left_wrist': cam_left_wrist,
        'cam_left_wrist_mask': cam_left_wrist_mask,
    }

    return data_dict

@torch.no_grad()
def pizza_preprocess(instance_data: dict, image_processor):
    """
        instance_data: dict
            {
                'img_history_size'
                'num_cameras'
                'meta'
                'state'
                'actions'
                'state_indicator'
                'image_metas'(cam_xx, cam_xx_mask)
                'ctrl_freq': 5
            }
    """

    data_dict = {}

    img_history_size = instance_data['img_history_size']
    num_cameras = instance_data['num_cameras']
    content = instance_data['meta']
    states = instance_data['state']
    actions = instance_data['actions']
    state_elem_mask = instance_data['state_indicator']
    image_metas = [
                instance_data['cam_high'], instance_data['cam_high_mask'],
                instance_data['cam_right_wrist'], instance_data['cam_right_wrist_mask'],
                instance_data['cam_left_wrist'], instance_data['cam_left_wrist_mask'],
            ]
    # state_std = instance_data['state_std']
    # state_mean = instance_data['state_mean']
    state_norm = instance_data['state_norm']

    data_dict['dataset_name'] = content['dataset_name']
    data_dict['ctrl_freq'] = 5
    data_dict["states"] = states
    data_dict["actions"] = actions
    data_dict["state_elem_mask"] = state_elem_mask
    data_dict["state_norm"] = state_norm

    background_color = np.array([
        int(x*255) for x in image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((
        image_processor.size["height"], 
        image_processor.size["width"], 3), dtype=np.uint8
                ) * background_color
    
    image_metas = list(pairwise(image_metas))
    rearranged_images = []
    for i in range(img_history_size):
        for j in range(num_cameras):
            images, image_mask = image_metas[j]
            image, valid = images[i], image_mask[i]
            # print(valid)
            if valid and (math.prod(image.shape) > 0):
                rearranged_images.append((image, True))
            else:
                rearranged_images.append((background_image.copy(), False))
    
    preprocessed_images = []
    processor = image_processor
    for image, valid in rearranged_images:
        image = Image.fromarray(image)
        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        preprocessed_images.append(image)
    data_dict["images"] = preprocessed_images
    data_dict["lang_embed"] = torch.load(content["instruction"])

    #Add batch = 1 for lang_embed
    data_dict["lang_embed"] = data_dict["lang_embed"].unsqueeze(0)

    lang_embeds = torch.nn.utils.rnn.pad_sequence(
                data_dict["lang_embed"],
                batch_first=True,
                padding_value=0)
    lang_embed_lens = []
    for i in range(lang_embeds.shape[0]):
        lang_embed_lens.append(data_dict['lang_embed'][i].shape[0])
    
    input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
    for i, l in enumerate(lang_embed_lens):
        input_lang_attn_mask[i, :l] = True
    data_dict["lang_attn_mask"] = input_lang_attn_mask
    data_dict["lang_embed"] = lang_embeds

    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v)
    return data_dict

@torch.no_grad()
def data_torchlize(instance_data, vision_encoder):
    """
        instance_data: dict
            {
                "images": [PIL.Image, ...]
                "states:
                "lang_attn_mask":
                "lang_embeds"
                "actions"
                "state_elem_mask"
                "ctrl_freqs"
                }
    """
    images = torch.stack(instance_data['images'], dim=0).unsqueeze(0)
    # images = instance_data['images'].to(dtype=torch.bfloat16).unsqueeze(0)
    states = instance_data['states'].to(dtype=torch.bfloat16).unsqueeze(0)
    states = states[:, -1:, :]
    actions = instance_data["actions"].to(dtype=torch.bfloat16)
    state_elem_mask = instance_data["state_elem_mask"].to(dtype=torch.bfloat16).unsqueeze(0)
    ctrl_freqs = torch.tensor([instance_data["ctrl_freq"]]).view(-1).to(dtype=torch.bfloat16)
    lang_attn_mask = instance_data['lang_attn_mask'].to(dtype=torch.bfloat16)
    text_embeds = instance_data['lang_embed'].to(dtype=torch.bfloat16)
    state_elem_mask = state_elem_mask.unsqueeze(1)

    batch_size, _, C, H, W = images.shape
    with torch.no_grad():
        image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
        
        # time_s = time.time()
        # for i in range(100):
        #     image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
        # time_e = time.time()
        # print(f"Time for 100 inference: {time_e - time_s}")
        image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size)).to(dtype=torch.bfloat16)
    forward_dict = {
        'lang_tokens': text_embeds,
        'lang_attn_mask': lang_attn_mask,
        'img_tokens': image_embeds,
        'state_tokens': states,
        'action_gt': actions,
        'action_mask': state_elem_mask,
        'ctrl_freqs': ctrl_freqs
    }
    return forward_dict

@torch.no_grad()
def pizza_data_process(instance_data, vision_encoder, image_processor):
    data_dict = pizza_in_datastream(instance_data)
    data_dict = pizza_preprocess(data_dict, image_processor)
    forward_dict = data_torchlize(data_dict, vision_encoder)
    return forward_dict

@torch.no_grad()
def predict_action(instance_data, vision_encoder, rdt):
    global device
    image_processor = vision_encoder.image_processor
    forward_dict = pizza_data_process(instance_data, vision_encoder, image_processor)
    trajectory = rdt.predict_action(
        lang_tokens=forward_dict['lang_tokens'].to(device),
        lang_attn_mask=forward_dict['lang_attn_mask'].to(device),
        img_tokens=forward_dict['img_tokens'].to(device),
        state_tokens=forward_dict['state_tokens'].to(device),
        action_mask = forward_dict['action_mask'].to(device),
        ctrl_freqs = forward_dict['ctrl_freqs'].to(device)
    )
    actions = extract_state(trajectory)
    return actions

@torch.no_grad()
def func_test(npy_path, image_path):
    robot_state = np.load(npy_path, allow_pickle=True)
    image = cv2.imread(image_path)
    joint = []
    gripper = []
    for i in range(1):
        joint.append(robot_state[i]['joint_position'])
        gripper.append(robot_state[i]['gripper_width'])
    joint = np.array(joint)
    gripper = np.array(gripper)
    gripper = np.expand_dims(gripper, axis=1)

    img = []
    img.append(image)
    task_id = 1
    instance_data = {
        'joints': joint,
        'gripper': gripper,
        'image': img,
        'task_id': task_id
    }
    ins_data = pizza_in_datastream(instance_data)
    for k,  v in ins_data.items():
        print(k)
    global image_processor, vision_encoder, rdt, device
    data_dict = pizza_preprocess(ins_data, image_processor)

    print("-------------------")

    forward_dict = data_torchlize(data_dict, vision_encoder=vision_encoder)
    for k, v in forward_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype)
        elif isinstance(v, list):
            print(k, len(v))
            for i in range(len(v)):
                print(k, i, v[i].shape, v.dtype)
    print("-------------------")
    trajectory = rdt.predict_action(
        lang_tokens=forward_dict['lang_tokens'].to(device),
        lang_attn_mask=forward_dict['lang_attn_mask'].to(device),
        img_tokens=forward_dict['img_tokens'].to(device),
        state_tokens=forward_dict['state_tokens'].to(device),
        action_mask = forward_dict['action_mask'].to(device),
        ctrl_freqs = forward_dict['ctrl_freqs'].to(device)
    )
    print(trajectory.shape)


    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING["right_gripper_open"]
    ]
    trajectory = trajectory.to(dtype = torch.float32).detach().cpu().numpy()[0]
    actions = trajectory[:, UNI_STATE_INDICES].tolist()
    print(actions)
    print(trajectory[-1])

def one_step(fr3, joint_tgt, gripper_tgt):
    global img, joint_queue, gripper_queue
    fr3.arm.goto_joints(joint_tgt, duration = 1.5, buffer_time = 0.2, ignore_virtual_walls = True)
    current_width = fr3.arm.get_gripper_width()
    if current_width >= 0.015:
        current_gripper = 1
    else:
        current_gripper = -1
    if gripper_tgt > 0:
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
    joint_queue.append(joint)
    gripper_queue.append(gripper)
    images= get_image()
    k4a_0 = decode_b64_image(images['k4a_0'])
    img.append(k4a_0)
    return joint, gripper

def control_loop(fr3, task_id, vision_encoder, rdt, exec_step, max_step):
    joint_queue = []
    gripper_queue = []
    img = []
    step = 0
    while step < max_step:
        images= get_image()
        k4a_0 = decode_b64_image(images['k4a_0'])
        img.append(k4a_0)
        joint = fr3.arm.get_joints()
        gripper = fr3.arm.get_gripper_width()
        joint_queue.append(joint)
        gripper_queue.append(gripper)
        instance_data = {
            'joints': joint_queue,
            'gripper': gripper_queue,
            'image': img,
            'task_id': task_id
        }
        trajectory = predict_action(instance_data, vision_encoder, rdt)
        for idx in range(exec_step):
            joint_tgt = trajectory['joint'][idx]
            gripper_tgt = trajectory['gripper'][idx]
            print(one_step(fr3, joint_tgt, gripper_tgt))
            step += 1


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Cuda realtime vla model inference service!'

@app.route('/predict', methods=['POST'])
def predict():
    global rdt, vision_encoder

    resp = request.get_json()
    images = resp['image']
    img_queue = []
    for img in images:
        img_queue.append(decode_b64_image(img))
    resp['image'] = img_queue
    instance_data = resp
    trajectory = predict_action(instance_data, vision_encoder, rdt)
    data_dict = {
        'actions': trajectory
    }
    print(data_dict)
    print(len(trajectory))
    return jsonify(data_dict)


if __name__ == "__main__":
    device = 'cuda:0'
    pretrained_vision_encoder_name_or_path = "/datahdd_8T/vla_pizza/RDT_module_params/siglip-so400m-patch14-384/"
    pretrained_model_name_or_path = "/datahdd_8T/vla_pizza/rdt_checkpoint/170M_test_16chunk/checkpoint-100/"

    print("Loading Model from: ", pretrained_model_name_or_path)
    print("Loading vision encoder from: ", pretrained_vision_encoder_name_or_path)
    vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None).to(device=device, dtype=torch.bfloat16)
    image_processor = vision_encoder.image_processor
    rdt = RDTRunner.from_pretrained(pretrained_model_name_or_path)
    rdt.reconfig_horizon(16)
    rdt = rdt.to(device=device, dtype=torch.bfloat16)
    # for param in rdt.lang_adaptor.parameters():
    #     print(param.dtype)
    # print(rdt.lang_adaptor.parameters())
    # print(rdt, image_processor)
    print("All Model loaded, starting service...")
    # fr3 = frankaRe3('franka.yml')
    # task_id = 1
    # image_path = '/datahdd_8T/sep_pizza_builder/pizza_dataset/1/20230828114129/images/right_rgb/001.jpg'
    # npy_path = '/datahdd_8T/sep_pizza_builder/pizza_dataset/1/20230828114129/franka_data.npy'
    # func_test(npy_path, image_path)
    # # control_loop(fr3, task_id, vision_encoder, rdt, 8, 100)

    # Start Flask Service
    app.run(host='0.0.0.0', port=7777)
