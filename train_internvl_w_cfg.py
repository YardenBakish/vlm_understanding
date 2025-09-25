'''
TODO
1. set up initial framework for distillation - for now do it even on very small numbered examples
2. set up framework to blur objects and create entire datasets

'''
from datetime import timedelta

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
import numpy as np
from modules.utils.callback import LossLoggerCallback2, CustomTrainer
import torch.distributed as dist

from internvl.model_internvl import InternVLForConditionalGeneration
from internvl.processing_internvl import InternVLProcessor


def setup_distributed():
    """Initialize distributed training if multiple GPUs are available"""

    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        
        

        # This will be automatically handled by torchrun/torch.distributed.launch
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            
            torch.cuda.set_device(local_rank)
            print(local_rank)
            print(rank)
            torch.distributed.init_process_group(backend="nccl",
            #world_size = world_size,
            #rank=rank,
            timeout=timedelta(minutes=0.2),

            device_id=local_rank)
            print(f"Initialized DDP: rank {rank}, world_size {world_size}, local_rank {local_rank}")
            return True
    
    else:
        print("HERE1")
        exit(1)

    
    return False



import config
import argparse
def parse_args():

    
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=[
    

                                    'joint_learning_extended_freeze',
                                    'joint_learning_extended_mask_freeze',
                                    'joint_learning_extended_mask_emph_freeze',

                                    
                                    'joint_learning_extended_mask_foc_freeze',
                                    'joint_learning_extended_foc_freeze',
                                    'joint_learning_extended_foc_hard_freeze',

                                    'joint_learning_extended_l2_freeze',
                                    'joint_learning_extended_mask_l2_freeze',

                                    ],
                                default = 'joint_learning_extended_mask_l2_freeze',
                                help='')
    parser.add_argument('--resume', action='store_true', help='use small model')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    
    

    args = parser.parse_args()
    

    args.model_size = 'internVL'

    config.get_config(args)

    args.save_dir = f'{args.paths["finetuned_models"]}/{args.mode}/{args.model_size}'

    if True:
        args.model_id = "OpenGVLab/InternVL3-1B-hf"

    
    
    #setup_distributed()
    os.makedirs(args.save_dir, exist_ok=True)
    return args




class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_loss.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, model=None,  logs=None, **kwargs):
        #for name, param in model.named_parameters():
        #    print(name, param.grad is not None, None if param.grad is None else param.grad.norm().item())
        if logs is not None and "loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: loss = {logs['loss']:.4f}\n")



def collate_fn(examples,image_token_id,model,processor):
    instances = []
    for example in examples:
        #print(example["origin_video_path"])
     
        prompt = example["generated_prompt"]
        
        user_content = [{"type": "text", "text": "Caption the video."}]
        user_content.append({"type": "video", "path": example["origin_video_path"]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 num_frames=30,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(model.dtype)
        

        
        #print(processor.batch_decode(instance["input_ids"],skip_special_tokens=False,)[0].count("<IMG_CONTEXT>"))
        #exit(1)
        
        
       
        
        
        instances.append(instance)

    special_tokens = ['<IMG_CONTEXT>',
                            '<img>',
                          '</img>',
                            '<quad>',
                           '</quad>',
                            '<ref>',
                            '</ref>',
                           '<box>',
                          '</box>',]

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

    special_token_ids = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens]
    for tid in special_token_ids:
        #print((labels == tid).any())
        labels[labels == tid] = -100
    
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






def collate_fn_optFlow(examples,image_token_id,model,processor, withBlur, extended, num_frames, coarse, is_joint_learning,foc,foc_hard):
    videoPath = "origin_video_path" 
    instances = []
    B_diffs   = []
    B_vis   = []
    B_inner_mask   = []


    for example in examples:


        print(f"\t PATH {example[videoPath]}")
        #example[videoPath] = "dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_000966/video_original.mp4"
        

        
        dummy_input = [{"type": "video", "path": example[videoPath]}]
        dummy_input.append({"type": "text", "text": "Caption the video."})
        dummy_input = [
            {"role": "user", "content": dummy_input},
            {"role": "assistant", "content": [{"type": "text", "text": f""}]}
        ]

        dummy_instance = processor.apply_chat_template(dummy_input, add_generation_prompt=False,
                                                 num_frames=30,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(model.dtype)
        
        
        #print(f"\t indices.shape {indices.shape}")
        #print(dummy_instance["pixel_values"].shape)
        
        
        target_H = dummy_instance["pixel_values"].shape[-2]
        target_W = dummy_instance["pixel_values"].shape[-1]
        target_T = dummy_instance["pixel_values"].shape[0]
        
        #print(dummy_instance["pixel_values"].shape)
        prompt = ast.literal_eval(example["bbox"]) 


        prompt = np.array(prompt)
       
        indices = np.linspace(0, len(prompt)-1, 30, dtype=int)
        
        indices[-1] = min(indices[-1], len(prompt)-1)
        if indices.shape[0] < target_T:
            indices = np.append(indices, indices[-1])
      
            
        prompt = prompt[indices]
        bboxes = []
        valid = True
        for i in range(prompt.shape[0]):
            numbers = re.findall(r'<loc(\d+)>', prompt[i])
            if len(numbers)!=4:
                bboxes.append([-1, -1, -1, -1])
            else:
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

        #if valid == False:
        #    continue

        #print(f"\t bboxes len {len(bboxes)}")

        bboxes = np.array(bboxes).astype(np.int32)
        inner_means, outer_means, diffs, pred_visibility, inner_mask = collect_difference_vectors(
            dummy_instance["pixel_values"], 
            example[videoPath], 
            bboxes, 
            indices,
            target_W,target_H, 
            modified=True,foc = False, foc_hard = False,
            internVL = True)
    
        #print(f"\t len(inner_means) {len(inner_means)}")
        #print(f"\t len(diffs) {diffs.shape}")


        B_diffs.append(diffs)
        B_vis.append(pred_visibility)
        B_inner_mask.append(inner_mask)


        if is_joint_learning:
            prompt = example["generated_prompt"]
            user_text_learning = "Caption the video."
            
        else:
            user_text_learning = "Return the 2D movement vectors (dx, dy) for the object and for the camera, for every frame in the video."
            prompt = ""
            for vec_idx in range(len(inner_means)):
                for time in range(2):
                    lst = inner_means if time ==0 else outer_means
                    pre = "obj" if time==0 else "cam"

                    val_x, val_y = lst[vec_idx]
                    val_x = round(val_x.item())
                    val_y = round(val_y.item())

                    sign_char_x = '+' if val_x >= 0 else '-'
                    sign_char_y = '+' if val_y >= 0 else '-'

                    if coarse == False:
                        val_x = min(abs(val_x),512)
                        val_y = min(abs(val_y),512)
                        prompt+=f"<{pre}dx{sign_char_x}{val_x:03d}><{pre}dy{sign_char_y}{val_y:03d}>"
                    else:
                        val_x = min(abs(val_x),1)
                        val_y = min(abs(val_y),1)
                        prompt+=f"<{pre}dx{sign_char_x}{val_x:01d}><{pre}dy{sign_char_y}{val_y:01d}>"


                if vec_idx < len(inner_means)-1:
                    prompt+=";"

      
        
        #user_content = [{"type": "text", "text": "Return xyxy coordinates for the object in the video"}]
        
        user_content = [{"type": "video", "path": example[videoPath]}]
        user_content.append({"type": "text", "text": user_text_learning})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 num_frames=30,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(model.dtype)
        
        #if instance["pixel_values"].shape[1] < num_frames:
        #    
        #    last_frame = instance["pixel_values"][:, -1:, ...]         # shape: [1, 1, 3, 384, 384]
        #    instance["pixel_values"] = torch.cat([instance["pixel_values"], last_frame], dim=1)  # shape: [1, N, 3, 384, 384]
        instances.append(instance)
        


    movement_vectors = torch.cat(B_diffs, dim=0)
    pred_visibility = torch.cat(B_vis, dim=0)
    inner_mask = torch.cat(B_inner_mask, dim=0)
    
    
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

    special_tokens = ['<IMG_CONTEXT>',
                            '<img>',
                          '</img>',
                            '<quad>',
                           '</quad>',
                            '<ref>',
                            '</ref>',
                           '<box>',
                          '</box>',]

    special_token_ids = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens]
    for tid in special_token_ids:
        #print((labels == tid).any())
        labels[labels == tid] = -100

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "movement_vectors": movement_vectors,
        'pred_visibility': pred_visibility,
        "inner_mask": inner_mask

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
    #torch.cuda.set_device(args.local_rank)


    
    model_id = args.model_id
    processor_id = "OpenGVLab/InternVL3-1B-hf"
    quant    = True if "quant" in  args.model_size else False 
    USE_QLORA = False

    if False:
        lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        #target_parameters=[],
        use_dora=False ,
        init_lora_weights="gaussian",
        target_modules = [
            "out_proj",
            "in_proj_weight"

           ]
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = InternVLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            device_map="auto",
            use_emph = ("emph" in args.mode),

       
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
        processor =  InternVLProcessor.from_pretrained(processor_id)

    else:
        processor =  InternVLProcessor.from_pretrained(processor_id)

        if False:
            model = model_jl.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            use_mask = "mask" in args.mode,
            use_emph = ("mask" in args.mode) and ("emph" in args.mode),
            freeze   = ("freeze" in args.mode),
            foc   = ("foc" in args.mode),
            l2    = ("l2" in args.mode)

       
            #_attn_implementation="flash_attention_2",
        )
        else:
            model = InternVLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                 use_emph = ("emph" in args.mode),
                 device_map = args.device
                #_attn_implementation="flash_attention_2",
            )

           
           

    for name, param in model.named_parameters():
        if 'movement_encoder' not in name and 'MCA_layers' not in name:
            param.requires_grad = False
            
        if args.resume == False and name == "encoder.MCA_layers.3.transformer.out_proj.weight":
            param.data.zero_()

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    
        
    dataset = load_dataset("csv", data_files=args.dataset_csv)["train"]
   
 
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<img>")
    ]
    

    num_train_epochs=10
   
    b_size = 2 if "joint_learning" in args.mode else 16

    
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,                  #5
        per_device_train_batch_size=b_size, #16
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1000,
        optim="adamw_torch" if quant==False else "paged_adamw_8bit", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
        bf16=True,
        #resume_from_checkpoint=True,
        report_to="tensorboard",
        dataloader_drop_last = True,

        output_dir=f"./{args.save_dir}",
        hub_model_id=f"./{args.save_dir}",
        logging_dir=f"./{args.save_dir}/logs", 
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        
        
        #batch_eval_metrics = True,
        #per_device_eval_batch_size=8,
        #evaluation_strategy="steps",
        #eval_steps=301,
        #eval_on_start =True,



    )
    #resume_from_checkpoint=True
    withExtended = True
    num_frames = 30 
        
     

    #if "BB" in args.mode:
    #    data_collator_fn = partial(collate_fn_blur, image_token_id=image_token_id, model=model, processor=processor, withBlur = False, extended=withExtended, num_frames=num_frames)  
    #else:

    data_collator_fn = partial(collate_fn_optFlow, image_token_id=image_token_id, model=model, processor=processor, withBlur = False, extended=withExtended, num_frames=num_frames, 
                                   coarse= "coarse" in args.mode, 
                                   is_joint_learning = "joint_learning" in args.mode,
                                   foc = "foc" in args.mode,
                                   foc_hard = "hard" in args.mode)  

    #data_collator_fn = partial(collate_fn, image_token_id=image_token_id, model=model, processor=processor)
    
    if False:
        trainer = Trainer(
        model=model,
        
        args=training_args,
        data_collator= data_collator_fn,
        train_dataset=dataset,
        callbacks=[LossLoggerCallback(f"./{args.save_dir}/logs/log.txt")] 
        )
    else:

        

        #compute_metrics_fn = partial(compute_custom_metrics,compute_result=True, tokenizer=processor)

        trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator= data_collator_fn,
        train_dataset=dataset,
        #eval_dataset=dataset,
        #compute_metrics = compute_metrics_fn,
        callbacks=[LossLoggerCallback2(f"./{args.save_dir}/logs/log.txt")] 
        )
    
    
    #if dist.is_initialized():
    #    print(dist.get_rank())
    #    print("rank", dist.get_rank(), "before barrier", flush=True)
    #    sys.stdout.flush()
    #    dist.barrier()
    #    print("rank", dist.get_rank(), "after barrier", flush=True)

    #print("PID", os.getpid(), "RANK", os.environ.get("RANK"), "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
    #  "WORLD_SIZE", os.environ.get("WORLD_SIZE"), "MASTER_ADDR", os.environ.get("MASTER_ADDR"),
    #  "MASTER_PORT", os.environ.get("MASTER_PORT"), flush=True)
    #print("----------------")
    #
    #exit(1)
    #
    #for i, batch in enumerate(trainer.get_train_dataloader()):
    #    print(i)
        
#
    #exit(1)
    #
    trainer.train(resume_from_checkpoint=args.resume) #resume_from_checkpoint=True



if __name__ == "__main__":

    args          = parse_args()
    config.get_config(args)
    args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined_w_BB.csv'
    #if ("BB" in args.mode) or ("optFlow" in args.mode):
    #    args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined_w_BB.csv'
    #else:
    #    args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined.csv' #f"dataset/got10k/teacher/train/{args.mode}/combined.csv" 

    basic_distillation(args)