'''
TODO
1. set up initial framework for distillation - for now do it even on very small numbered examples
2. set up framework to blur objects and create entire datasets

'''

import config
import argparse
import os
import re
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText ,TrainingArguments, Trainer, TrainerCallback
import torch
from convert_utils import *
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from processing_smolvlm import SmolVLMProcessor
import numpy as np

import config
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=['uniform_blur', 'cfg', 'BB', 'BB_extended', 'optFlow_extended', 'BB_test', 'BB_w_blur', 'BB_w_blur2', 'BB_w_blur_extended', 'optical_flow','optical_flow_w_cfg'],
                                default = 'BB_extended',
                                help='')

    parser.add_argument('--model-size', type=str,
                                choices=['500M', '2.2B', '2.2B_quant'],
                                default = '2.2B',
                                help='')
    
    args = parser.parse_args()
    config.get_config(args)

    args.save_dir = f'{args.paths["distilled_models"]}/{args.mode}/{args.model_size}'

    if True:
        args.model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    else:
        args.model_id = "distilled_models/uniform_blur/2.2B/checkpoint-2445"

    os.makedirs(args.save_dir, exist_ok=True)
    return args




class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_loss.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, model=None,  logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: loss = {logs['loss']:.4f}\n")



def collate_fn(examples,image_token_id,model,processor):
    instances = []
    for example in examples:
        prompt = example["generated_prompt"]

        user_content = [{"type": "text", "text": "Caption the video."}]
        user_content.append({"type": "video", "path": example["blur_full_video_path"]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
        instances.append(instance)


    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    )

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out





def collate_fn_blur(examples,image_token_id,model,processor, withBlur, extended, num_frames):
    videoPath = "origin_video_path" if withBlur==False else "blur_full_video_path"
    instances = []
    for example in examples:
        
        dummy_input = [{"type": "text", "text": "Caption the video."}]
        dummy_input.append({"type": "video", "path": example[videoPath]})
        dummy_input = [
            {"role": "user", "content": dummy_input},
            {"role": "assistant", "content": [{"type": "text", "text": f""}]}
        ]

        _,indices =  processor.apply_chat_template(dummy_input, add_generation_prompt=False, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt", return_frame_indices=True)
        
        prompt = ast.literal_eval(example["bbox"]) 
        prompt = np.array(prompt)
        indices[-1] = min(indices[-1], len(prompt)-1) 
        prompt = prompt[indices]
        prompt = ';'.join(prompt.tolist())


        user_content = [{"type": "text", "text": "Return xyxy coordinates for the object in the video"}]
        user_content.append({"type": "video", "path": example[videoPath]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
        
        instances.append(instance)


    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    )

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out






def collate_fn_optFlow(examples,image_token_id,model,processor, withBlur, extended, num_frames):
    videoPath = "origin_video_path" if withBlur==False else "blur_full_video_path"
    instances = []
    for example in examples:
        
        dummy_input = [{"type": "text", "text": "Caption the video."}]
        dummy_input.append({"type": "video", "path": example[videoPath]})
        dummy_input = [
            {"role": "user", "content": dummy_input},
            {"role": "assistant", "content": [{"type": "text", "text": f""}]}
        ]

        dummy_instance,indices =  processor.apply_chat_template(dummy_input, add_generation_prompt=False, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt", return_frame_indices=True)
        
        
        target_H = dummy_instance["pixel_values"].shape[-2]
        target_W = dummy_instance["pixel_values"].shape[-1]
        
        prompt = ast.literal_eval(example["bbox"]) 


        prompt = np.array(prompt)
        indices[-1] = min(indices[-1], len(prompt)-1) 
        prompt = prompt[indices]
        bboxes = []
        valid = True
        for i in range(prompt.shape[0]):
            numbers = re.findall(r'<loc(\d+)>', prompt[i])
            if len(numbers)!=4:
                valid = False
                break
            y_min, x_min, y_max, x_max = map(int, numbers)
            x_min *= (target_W / 1024)
            y_min *= (target_H / 1024)
            x_max *= (target_W / 1024)
            y_max *= (target_H / 1024)

            x_min =  int(x_min)
            y_min =  int(y_min)
            x_max =  int(x_max)
            y_max =  int(y_max)
            bboxes.append([x_min, y_min, x_max, y_max])

        if valid == False:
            continue
        bboxes = np.array(bboxes).astype(np.int32)
        inner_means, outer_means = collect_difference_vectors(dummy_instance["pixel_values"], example[videoPath], bboxes, indices,target_W,target_H)
        prompt = ""
        #print(outer_means)
        for vec_idx in range(len(inner_means)):
            for time in range(2):
                lst = inner_means if time ==0 else outer_means
                pre = "obj" if time==0 else "cam"
                
                val_x, val_y = lst[vec_idx]
                val_x = round(val_x.item())
                val_y = round(val_y.item())

                sign_char_x = '+' if val_x >= 0 else '-'
                sign_char_y = '+' if val_y >= 0 else '-'

                val_x = min(abs(val_x),512)
                val_y = min(abs(val_y),512)

                prompt+=f"{pre}<{sign_char_x}{val_x:03d}><{sign_char_y}{val_y:03d}>"
        
            if vec_idx < len(inner_means)-1:
                prompt+=";"

        #print(prompt)
        #exit(1)
        #print(prompt)
        #print(len(inner_means))
        #print(inner_means[0].shape)
        #exit(1)
        #prompt = ';'.join(prompt.tolist())

        

        #user_content = [{"type": "text", "text": "Return xyxy coordinates for the object in the video"}]
        user_content = [{"type": "text", "text": "Return xy motion vectors for the object and camera in the video"}]

        #object<loc0012><loc0034>background<loc0987><loc0756>;


        user_content.append({"type": "video", "path": example[videoPath]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
        
        instances.append(instance)


    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    )

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out





def basic_distillation(args):
    
    model_id = args.model_id
    processor_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    quant    = True if "quant" in  args.model_size else False 
    USE_QLORA = False

    if quant:
        lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            device_map="auto"
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        processor = AutoProcessor.from_pretrained(processor_id) if (("BB" not in args.mode) and ("optFlow" not in args.mode)) else SmolVLMProcessor.from_pretrained(processor_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #_attn_implementation="flash_attention_2",
        ).to("cuda")
        for name, param in model.model.vision_model.named_parameters():
            if 'vision_model.encoder' not in name:
                param.requires_grad = False
    
        #for param in model.model.vision_model.parameters():
        #    param.requires_grad = False

    dataset = load_dataset("csv", data_files=args.dataset_csv)["train"]
    #print(dataset)
    #print(f"prompt:  {dataset[0]['generated_prompt']}, video: {dataset['origin_video_path']}")
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    num_train_epochs=10
    if "2" in args.mode:
        num_train_epochs = 10
    
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,                  #5
        per_device_train_batch_size=16, #16
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        optim="adamw_torch" if quant==False else "paged_adamw_8bit", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
        bf16=True,
        #resume_from_checkpoint=True,
        report_to="tensorboard",

        output_dir=f"./{args.save_dir}",
        hub_model_id=f"./{args.save_dir}",
        logging_dir=f"./{args.save_dir}/logs", 
        remove_unused_columns=False,
        gradient_checkpointing=True,
     
        dataloader_pin_memory=False
    )
    #resume_from_checkpoint=True
    if (("BB" in args.mode) or ("optFlow" in args.mode)):
        withBlur = False
        withExtended = None
        num_frames  = None
        if "BB_w_blur" in args.mode:
            withBlur = True
        if "extended" in args.mode:
            withExtended = True
            num_frames = 30 if "500M" in args.model_size else 30
        
     

        if "BB" in args.mode:
            data_collator_fn = partial(collate_fn_blur, image_token_id=image_token_id, model=model, processor=processor, withBlur = withBlur, extended=withExtended, num_frames=num_frames)  
        else:
            data_collator_fn = partial(collate_fn_optFlow, image_token_id=image_token_id, model=model, processor=processor, withBlur = withBlur, extended=withExtended, num_frames=num_frames)  

    else:
         data_collator_fn = partial(collate_fn, image_token_id=image_token_id, model=model, processor=processor)

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator= data_collator_fn,
    train_dataset=dataset,
    callbacks=[LossLoggerCallback(f"./{args.save_dir}/logs/log.txt")] 
    )

    
    trainer.train() #resume_from_checkpoint=True



if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)
    if ("BB" in args.mode) or ("optFlow" in args.mode):
        args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined_w_BB.csv'
    else:
        args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined.csv' #f"dataset/got10k/teacher/train/{args.mode}/combined.csv" 

    basic_distillation(args)