

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
from model_optical_flow import SmolVLMForConditionalGeneration
from glob import glob

import config
import argparse
from typing import Optional, Union, Any
from torch import nn

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)



class CustomTrainer(Trainer):
     def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
       
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        self.log({"optical_loss": outputs.optical_loss.item()})
        self.log({"cap_loss": outputs.cap_loss.item()})

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if Trainer._is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss




def parse_args():
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=['cfg_cap', 'cfg_BB', 'cfg_optflow_cap', 'cfg_optflow_BB'],
                                default = 'cfg_cap',
                                help='')

    parser.add_argument('--model-size', type=str,
                                choices=['500M', '2.2B'],
                                default = '2.2B',
                                help='')
    
    parser.add_argument('--resume', action='store_true')
   
    
    args = parser.parse_args()
    config.get_config(args)

    args.save_dir = f'{args.paths["finetuned_models"]}/{args.mode}/{args.model_size}'

   
    args.model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  

    os.makedirs(args.save_dir, exist_ok=True)
    return args




class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_loss.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, model=None,  logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: loss = {logs['loss']:.4f}\n")

        if logs is not None and "optical_loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: optical_loss = {logs['optical_loss']:.4f}\n")

        if logs is not None and "cap_loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: cap_loss = {logs['cap_loss']:.4f}\n")



def collate_fn(examples,image_token_id,model,processor, withBlur, extended, num_frames, use_cfg, use_optFlow):
    pattern = re.compile(r'.*_(\d+)$')
    
    videoPath = "origin_video_path" if withBlur==False else "blur_full_video_path"
    instances = []
    all_flow_maps = []
    all_pred_tracks = []
    all_pred_visibility = []

    for example in examples:
        prompt = example["generated_prompt"]
       


        
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
        
        if use_optFlow:
            example_flow_maps = collect_optical_flow(examples,example, videoPath, indices, target_W, target_H)
            all_flow_maps.append(example_flow_maps)
        else:
            all_flow_maps = None

        if use_cfg:
            ex_pred_tracks, ex_pred_visibility = collect_tracks(example, videoPath, indices, target_W, target_H)
            all_pred_tracks.append(ex_pred_tracks)
            all_pred_visibility.append(ex_pred_visibility)

        else:
             pred_tracks, pred_visibility = None, None

        user_content = [{"type": "text", "text": "Caption the video."}]
        user_content.append({"type": "video", "path": example[videoPath]})
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": prompt}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
        instances.append(instance)

    optical_flow_maps = None
    if use_optFlow:
        optical_flow_maps = pad_optical_flow(all_flow_maps)


    pred_tracks = None
    pred_visibility = None
    if use_cfg:
       
        pred_tracks = torch.cat(all_pred_tracks, dim=0)
        

        pred_visibility = torch.cat(all_pred_visibility, dim=0)


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
        "labels": labels,
        'optical_flow_maps': optical_flow_maps,
         'pred_tracks': pred_tracks,
         'pred_visibility': pred_visibility,
        
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


def adjust_architecture(model, args):
    if "optflow" in args.mode :
        if args.resume != True:
            flow_component =  create_gmflow_model(load_weights=args.resume == True).to(model.device)
            model.model.vision_model.optical_flow = flow_component

        for name, param in model.named_parameters():
            #param.requires_grad = False
            if "optical_flow" in name:
                if "backbone" in name:
                    param.requires_grad = False
            #else:
            #    param.requires_grad = False

    if "cfg" in args.mode:
        for name, param in model.named_parameters():
            if 'trackTention' not in name:
                param.requires_grad = False

            if 'attentional_splatting.W_out' in name and args.resume != True:
                param.data.zero_()
    
    #for param in model.model.vision_model.parameters():
    #    param.requires_grad = False
        #param.requires_grad = False
    #for param in model.model.text_model.parameters():
    #    param.requires_grad = False
   
    


def train_w_movement(args):
    
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

        model = SmolVLMForConditionalGeneration.from_pretrained(
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
        processor = SmolVLMProcessor.from_pretrained(processor_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_cfg = "cfg" in args.mode,
            use_optFlow = "optFlow" in args.mode
            #_attn_implementation="flash_attention_2",
        ).to("cuda")
        #for name, param in model.model.vision_model.named_parameters():
        #    print(name)
    
        #for param in model.model.vision_model.parameters():
        #    param.requires_grad = False



 
    adjust_architecture(model,args)


    dataset = load_dataset("csv", data_files=args.dataset_csv)["train"]
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    num_train_epochs=10

    training_args = TrainingArguments(
        num_train_epochs= num_train_epochs,   #num_train_epochs,                  #5
        per_device_train_batch_size=16, #16
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25, #25
        save_strategy="steps",
        save_steps=200,
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
    withExtended = True
    num_frames = 10 if "500M" in args.model_size else 30
    withBlur = False
    data_collator_fn = partial(collate_fn, 
                               image_token_id=image_token_id, 
                               model=model, 
                               processor=processor, 
                               withBlur = withBlur, 
                               extended=withExtended, 
                               num_frames=num_frames,
                               use_cfg = 'cfg' in args.mode,
                               use_optFlow = 'optFlow' in args.mode)  

   


   

    if "optFlow" in args.mode:
        trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator= data_collator_fn,
        train_dataset=dataset,
        callbacks=[LossLoggerCallback(f"./{args.save_dir}/logs/log.txt")] 
        )
    else:
        trainer = Trainer(
        model=model,
        args=training_args,
        data_collator= data_collator_fn,
        train_dataset=dataset,
        callbacks=[LossLoggerCallback(f"./{args.save_dir}/logs/log.txt")] 
        )


    if args.resume:
        trainer.train(resume_from_checkpoint=True) #resume_from_checkpoint=True
    else:
        trainer.train()


if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)
    if "BB" in args.mode:
        args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined_w_BB.csv'
    else:
        args.dataset_csv =  'dataset/got10k/teacher/train/uniform_blur/combined.csv' #f"dataset/got10k/teacher/train/{args.mode}/combined.csv" 

    train_w_movement(args)