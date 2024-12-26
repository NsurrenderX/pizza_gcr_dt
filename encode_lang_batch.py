import os
import json

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder
from data.pizza_robot.pizza_data_class import pizzaSlice


GPU = 0
MODEL_PATH = "/datahdd_8T/vla_pizza/RDT_module_params/t5-v1_1-xxl/"
DATA_DIR = "/datahdd_8T/sep_pizza_builder/pizza_dataset/"
SAVE_DIR = '/datahdd_8T/sep_pizza_builder/pizza_embedded/'
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
TARGET_DIR = "data/datasets/agilex/tfrecords/"

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = '/datahdd_8T/vla_pizza/RDT_module_params/offload_T5/'  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    print("T5 model loaded, loading dataset ...")
    dataset = pizzaSlice(data_path=DATA_DIR)
    print("Dataset loaded, encoding ...")
    # Get all the task paths
    # task_paths = []
    # for sub_dir in os.listdir(TARGET_DIR):
    #     middle_dir = os.path.join(TARGET_DIR, sub_dir)
    #     if os.path.isdir(middle_dir):
    #         for task_dir in os.listdir(middle_dir):
    #             task_path = os.path.join(middle_dir, task_dir)
    #             if os.path.isdir(task_path):
    #                 task_paths.append(task_path)

    # For each task, encode the instructions
    # for task_path in tqdm(task_paths):
    instructions = []
    for task_id in tqdm(range(1, 23)):
        # Load the instructions corresponding to the task from the directory
        # with open(os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
        #     instruction_dict = json.load(f_instr)
        # instructions = [instruction_dict['instruction']] + instruction_dict['simplified_instruction'] + \
        #     instruction_dict['expanded_instruction']

        instructions.append(dataset.prompts[task_id - 1])
        # instructions = [dataset.prompts[task_id - 1]]
        # print(instructions)
    
        # Encode the instructions
    tokenized_res = tokenizer(
        instructions, return_tensors="pt",
        padding="longest",
        truncation=True
    )
    # print(tokenized_res)
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)
    
    with torch.no_grad():
        text_embeds = text_encoder(
            input_ids=tokens,
            attention_mask=attn_mask
        )["last_hidden_state"].detach().cpu()
    
    attn_mask = attn_mask.cpu().bool()
    # print(tokens, text_embeds, attn_mask)
    # Save the embeddings for training use
    for i in tqdm(range(len(instructions))):
        print(i, instructions[i])
        text_embed = text_embeds[i][attn_mask[i]]
        print(text_embed, text_embed.shape)
        save_path = os.path.join(SAVE_DIR, f"lang_embed_{i+1}.pt")
        torch.save(text_embed, save_path)
    # for i in range(len(instructions)):
    #     text_embed = text_embeds[i][attn_mask[i]]
    #     save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
    #     torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
