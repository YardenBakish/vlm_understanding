'''
TODO
1. set up initial framework for distillation - for now do it even on very small numbered examples
2. set up framework to blur objects and create entire datasets

'''
import os
from pathlib import Path
import torch.nn.functional as F

import config
import argparse
import matplotlib.pyplot as plt

import re
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText ,TrainingArguments, Trainer, TrainerCallback
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file
import torch
from convert_utils import *
import pandas as pd
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from model_optical_flow import SmolVLMForConditionalGeneration
from model_w_cfg import SmolVLMForConditionalGeneration as BasicModel
from model_joint_learning import SmolVLMForConditionalGeneration as model_jl
from model_w_cfg import SmolVLMForConditionalGeneration as model_cfg
from processing_smolvlm import SmolVLMProcessor
#from internvl.model_internvl import InternVLForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#from internvl.processing_internvl import InternVLProcessor


import config
import argparse

import matplotlib.font_manager as fm
from moviepy.editor import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch



SAMPLES_TO_TEST = [


#"GOT-10k_Val_000029",
#"GOT-10k_Val_000001",
#"GOT-10k_Val_000095",

"GOT-10k_Val_000053",



#"GOT-10k_Val_000030",
#"GOT-10k_Val_000029",



#"GOT-10k_Val_000107",
#"GOT-10k_Val_000015",
#"GOT-10k_Val_000085",
#"GOT-10k_Val_000107",
#"GOT-10k_Val_000017",








#"GOT-10k_Val_000022",
#"GOT-10k_Val_000003",
#"GOT-10k_Val_000007",
#"GOT-10k_Val_000027",
#"GOT-10k_Val_000132",
#"GOT-10k_Val_000014",





]

SAMPLES_EMOTIONS_TO_TEST =[
    #"vid1",
    #"vid2",
    "vid3",
    #"vid4",
    #"vid5",
]



def parse_args():
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=[
                                        'distil_BB_extended', 'distil_optFlow_extended','distil_optFlow_extended_coarse',
                                        'finetune_cfg_cap', 'finetune_cfg_BB',
                                        'finetune_cfg_optflow_cap', 'finetune_cfg_optflow_BB',

                                        'finetune_cfg_cap_simple_extended',
                                        'finetune_joint_learning_extended_freeze',
                                        'finetune_joint_learning_extended_mask_freeze',
                                        'finetune_joint_learning_extended_mask_emph_freeze',


                                        'distil_baseline',
                                        'finetune_joint_learning_extended',
                                        'finetune_joint_learning_extended_mask',
                                        'finetune_joint_learning_extended_mask_emph',
                                               
                                        'finetune_joint_learning_extended_mask_foc_freeze',
                                        'finetune_joint_learning_extended_foc_freeze',
                                        'finetune_joint_learning_extended_foc_hard_freeze',
                                        'finetune_joint_learning_extended_l2_freeze',
                                        'finetune_joint_learning_extended_mask_l2_freeze',
                                        






                                        ],
                                default = 'distil_BB_extended',
                                help='')
    
    parser.add_argument('--compare-mode', type=str,
                                choices=['vis', 'metric'],
                                default = 'vis',
                                help='')

    parser.add_argument('--epoch', type=int,
                              
                                help='')

    parser.add_argument('--eval', type=str,
                                choices=['blur_video', 'blur_video_ker','tempcomp', 
                                'tempcomp_caption_matching',
                                'tempcomp_yes_no', 
                                'tempcomp_motion_bench',
                                'tempcomp_motion_bench_sports',
                                'tempcomp_det_ker_check',
                                'tempcomp_det_check',
                                'blur_video_sorted', 'blur_video_ker_sorted',



                                'tempcomp_mvbench',

                                
                                
                                'directions', 'action_rec'],
                                help='')
    

    parser.add_argument('--model-size', type=str,
                                choices=['500M', '2.2B', 'internVL'],
                                default = '2.2B',
                                help='')

    parser.add_argument('--emotion', action='store_true', help='use small model')

   

    
    args = parser.parse_args()
    config.get_config(args)
    pref = args.mode.split("_")[0]
    args.mode        = "_".join(args.mode.split("_")[1:])
   
    work_dir         = f'{args.mode}/{args.model_size}'
    args.dir_type    = pref
    
    if args.dir_type =="finetune":
        args.dir_type = "finetuned_models"
    else:
        args.dir_type = "distilled_models"

    
    finedtuned_dir   = f'{args.paths[args.dir_type]}/{work_dir}'
    eval_dir         = f"eval/{work_dir}"

    
    os.makedirs(eval_dir,exist_ok=True)
    args.eval_dir = eval_dir

    if "BB" in args.mode:
        args.orig_dir = get_last_checkpoint_dir(f'{args.paths["distilled_models"]}/BB_extended/{args.model_size}')
    else:
        if "internVL" not in args.model_size:
            args.orig_dir = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        else:
            args.orig_dir = "OpenGVLab/InternVL3-1B-hf"

   
    if args.eval:
        if args.epoch is None and 'baseline' in args.mode:
            args.epoch = 0
        elif args.epoch is None and 'baseline' not in args.mode:
            print("MUST SPECIFY EPOCH")
            exit(1)
        args.finetuned_dir = f"{finedtuned_dir}/checkpoint-{args.epoch}"
    
    elif "internVL" in args.model_size:
        args.finetuned_dir = get_last_checkpoint_dir(finedtuned_dir,last=True) if 'baseline' not in args.mode else None
        
    else:
        args.finetuned_dir =  get_last_checkpoint_dir(finedtuned_dir,last=True) if 'baseline' not in args.mode else None
    #args.finetuned_dir = "finetuned_models/joint_learning_extended_mask/2.2B/checkpoint-2100"
    
    args.prompt_orig     = "Describe this video in detail" if "BB" not in args.mode else "Return xyxy coordinates for the object in the video"
    
    if "BB" in args.mode:
        args.prompt_finetune = "Return xyxy coordinates for the object in the video"
        #args.prompt_finetune = "Return the 2D movement vectors (dx, dy) for the object and for the camera, for every frame in the video."
    elif (args.dir_type == "distilled_models" and "optFlow" in args.mode) or (args.dir_type == "finetuned_models" and "cfg_cap" in args.mode) :
         args.prompt_finetune = "Return the 2D movement vectors (dx, dy) for the object and for the camera, for every frame in the video."
    else:
        args.prompt_finetune = "Caption the video."


   
    #torch.manual_seed(42)
    #np.random.seed(42)
    args.extended = True
    args.num_frames = 30
    #args.extend_frames = None

    return args





def get_normalized_BB(groundTruthPath, metaInfoPath,return_raw=False):
    bboxes = parse_groundtruth(groundTruthPath)
   
    if os.path.isfile(metaInfoPath):
            with open(metaInfoPath, 'r') as meta_file:
                 for line in meta_file:
                    if 'resolution' in line:
                      
                        _, resolution = line.strip().split(':')
                        resolution = resolution.strip()
                        resolution_str = resolution.strip("()")
                  
                        width, height = map(int, resolution_str.split(", "))
                        normalized_boxes = []

                        for bbox in bboxes:
                            x1, y1, dx, dy = bbox

                            # Normalize coordinates and dimensions
                            x_min = int((x1 / width) * 1024)
                            y_min = int((y1 / height) * 1024)
                            dx = (dx / width) * 1024
                            dy = (dy / height) * 1024
                            x_max = int(x_min + dx)
                            y_max = int(y_min + dy)

                            if return_raw:
                                normalized_boxes.append([x_min,y_min,x_max,y_max])
                            else:
                                coord = f"<loc{y_min:04d}><loc{x_min:04d}><loc{y_max:04d}><loc{x_max:04d}>"
                                normalized_boxes.append(coord)
                     
                        return normalized_boxes




def get_movement_vectors(video_path, indices, groundTruthPath, metaInfoPath, input_shape, modified=False):
    
    
    normalized_bboxes = get_normalized_BB(groundTruthPath, metaInfoPath)
    normalized_bboxes = np.array(normalized_bboxes)
    normalized_bboxes = normalized_bboxes[indices]
    bboxes = []
    for i in range(len(normalized_bboxes)):
        numbers = re.findall(r'<loc(\d+)>', normalized_bboxes[i])
        if len(numbers)!=4:
            valid = False
            print("CHECK ME")
            exit(1)
        y_min, x_min, y_max, x_max = map(int, numbers)
        x_min *= (input_shape / 1024)
        y_min *= (input_shape / 1024)
        x_max *= (input_shape / 1024)
        y_max *= (input_shape / 1024)
        x_min =  int(x_min)
        y_min =  int(y_min)
        x_max =  int(x_max)
        y_max =  int(y_max)
        bboxes.append([x_min, y_min, x_max, y_max])

    bboxes = np.array(bboxes).astype(np.int32)
    gt_tracks = collect_difference_vectors(None, video_path, bboxes, indices, input_shape, input_shape, modified=modified)
    
   
    return gt_tracks
    



def get_gt(mode, indices, video_path, input_shape, return_raw=False,modified = False):
    subdir = "/".join(video_path.split("/")[:-1])
    groundTruthPath =  os.path.join(subdir,"groundtruth.txt")
    metaInfoPath =  os.path.join(subdir,"meta_info.ini")
    if "BB" in mode:
        return get_normalized_BB(groundTruthPath, metaInfoPath,return_raw=return_raw)
    else:
        return get_movement_vectors(video_path, indices, groundTruthPath, metaInfoPath, input_shape, modified=modified)


def eval_performance_per_frame(d_eval, video_type, mode, model_pred_per_frame, gt_data_per_frame ):
    if "BB" in mode:
        numbers1 = re.findall(r'<loc(\d+)>', model_pred_per_frame)
        numbers2 = re.findall(r'<loc(\d+)>', gt_data_per_frame)

        if len(numbers1) != 4 or len(numbers2) != 4:
            return
 

        pred_box = tuple(map(int, numbers1))  # (ymin, xmin, ymax, xmax)
        gt_box   = tuple(map(int, numbers2))

        iou = compute_iou(pred_box, gt_box)

        d_eval[video_type]["iou"] += iou


def eval_performance(d_eval,video_type, explaination, indices, mode, video_path,input_shape ):
    gt_data =  get_gt(args.mode, indices, video_path, input_shape)
  

    if "BB" in mode:
        gt_data    = gt_data[indices]
        gt_data = np.array(gt_data)
        model_pred =  parse_loc_string(explaination)
        model_pred = np.array(model_pred)

        for j in range(min(len(gt_data), len(model_pred))):
            d_eval[video_type]["num"]+=1
            eval_performance_per_frame(d_eval,video_type,mode, model_pred[j], gt_data[j])
            if j == 1:
                break
    else:
        inner_means, outer_means = gt_data
        l1_obj, l1_cam = compute_l1(inner_means, outer_means, explaination)

        d_eval[video_type]["num"]+=len(indices)

        d_eval[video_type]["inner"] +=l1_obj
        d_eval[video_type]["outer"] +=l1_cam



        #print(inner_means)
        #print(outer_means)

        #print(explaination)
      
    







def gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,compare_mode="BB", indices_dict = None):

    model_type1 = "orig" 
    model_type2 = "finetuned" 
    
    text_row1 = "Original"
    text_row2 = "finetuned" 


    videos_types = ["original", "blur_full"]
    rev_d = {}
    for model_type in explainations_compare_dict.keys():
        for sample in explainations_compare_dict[model_type]:
            if sample not in rev_d:
                rev_d[sample] = {}
            rev_d[sample][model_type] = explainations_compare_dict[model_type][sample]

    #between models
    if compare_mode == "BB":
        for sample in rev_d:
            output_dir = f'{args.eval_dir}/{sample}'
            for i in range(len(videos_types)):
                orig_pred_images      =  visualize(inputs_dict[sample][i],pred=parse_loc_string(rev_d[sample][model_type1][i]),indices=None,vis_pred=True, save_im = False, compare_mode = compare_mode)
                distilles_pred_images =  visualize(inputs_dict[sample][i],pred=parse_loc_string(rev_d[sample][model_type2][i]),indices=None,vis_pred=True, save_im = False, compare_mode = compare_mode)

                if orig_pred_images == -1 or distilles_pred_images == -1:
                    continue

                save_image_grid(orig_pred_images, distilles_pred_images, f"{output_dir}/{videos_types[i]}_compare_models.png", text_row1=text_row1, text_row2=text_row2)
    

    #for the finetuned model
    for sample in rev_d:
        ext = "BB" if compare_mode == "BB" else "optFlow"
        pred1 = parse_loc_string(rev_d[sample][model_type2][0]) if compare_mode == "BB" else rev_d[sample][model_type2][0]
        pred2 = parse_loc_string(rev_d[sample][model_type2][1]) if compare_mode == "BB" else rev_d[sample][model_type2][1]

        output_dir = f'{args.eval_dir}/{sample}'
        print(sample)
        orig_pred_images   =  visualize(inputs_dict[sample][0],pred=pred1,indices=indices_dict[sample][0],vis_pred=True, save_im = False,compare_mode=compare_mode, video_path = sample)
        blurry_pred_images =  visualize(inputs_dict[sample][1],pred=pred2,indices=indices_dict[sample][1],vis_pred=True, save_im = False, compare_mode=compare_mode, video_path = sample)
        
        if orig_pred_images == -1 or blurry_pred_images == -1:
            continue
        
        save_image_grid(orig_pred_images, blurry_pred_images, f"{output_dir}/compare_model_blur_{ext}.png", text_row1=text_row1, text_row2=text_row2)



def create_text_image(text, width, font_size=50, bg_color=(0, 0, 0), text_color=(255, 255, 255), padding=10):
    """
    Create an image containing text with the specified width.
    
    Args:
        text (str): The text to display
        width (int): Width of the image
        font_size (int): Font size
        bg_color (tuple): Background color (R,G,B)
        text_color (tuple): Text color (R,G,B)
        padding (int): Padding around text
        
    Returns:
        PIL.Image: An image with the rendered text
    """
    # Use default font with specified size
    # Note: PIL's default font doesn't support custom sizes well
    # So we'll adjust other parameters to make the text more prominent
    font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf", font_size)

    
    # Create a temporary image to calculate text dimensions
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Adjust line height based on requested font size
    line_height = max(16, int(font_size * 0.75))  # Scale based on requested size
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width = temp_draw.textbbox((0, 0), test_line, font=font)[2]
        
        if test_width <= (width - 2 * padding):
            current_line.append(word)
        else:
            if current_line:  # If the current line has words, append it
                lines.append(' '.join(current_line))
                current_line = [word]
            else:  # If a single word is too long, we still have to include it
                lines.append(word)
                current_line = []
    
    # Don't forget to add the last line
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate the height required for the text
    line_height = font_size + 4  # Add a bit of spacing between lines
    text_height = line_height * len(lines)
    
    # Create the actual image
    img = Image.new('RGB', (width, text_height + 2 * padding), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw each line of text
    y_position = padding
    for line in lines:
        draw.text((padding, y_position), line, font=font, fill=text_color)
        y_position += line_height
    
    return img


def create_video_with_text(video_files, text_list, output_path):
    """
    Creates a single video with multiple videos arranged in a row,
    with corresponding text beneath each video.
    
    Args:
        video_files (list): List of paths to video files
        text_list (list): List of texts corresponding to each video
        output_path (str): Path where the final video will be saved
    """
    if len(video_files) != len(text_list):
        raise ValueError("Number of videos and texts must match!")
    
    # Load all video clips
    video_clips = [VideoFileClip(file) for file in video_files]
    
    # Find the smallest height and use it for all clips to ensure consistency
    min_height = min(clip.h for clip in video_clips)

    
    
    # Resize all clips to have the same height (keeping aspect ratio)
    resized_clips = []
    for clip in video_clips:
        w_new = int(clip.w * (min_height / clip.h))
        resized_clips.append(clip.resize(height=min_height))
    
    # Create text clips using PIL and convert to MoviePy clips
    text_image_clips = []
    for i, text in enumerate(text_list):
        # Create a text image with the same width as the corresponding video
        text_img = create_text_image(text, width=resized_clips[i].w)
        
        # Convert PIL image to MoviePy ImageClip
        img_clip = ImageClip(np.array(text_img))
        img_clip = img_clip.set_duration(resized_clips[i].duration)
        
        text_image_clips.append(img_clip)
    
    # Create video-text pairs (each video with its text below)
    video_text_pairs = []
    for i in range(len(resized_clips)):
        pair = CompositeVideoClip(
            [
                resized_clips[i].set_position(('center', 'top')),
                text_image_clips[i].set_position(('center', resized_clips[i].h))
            ],
            size=(resized_clips[i].w, resized_clips[i].h + text_image_clips[i].h)
        )
        video_text_pairs.append(pair)
    
    # Arrange all pairs in one row
    final_width = sum(pair.w for pair in video_text_pairs)
    final_height = max(pair.h for pair in video_text_pairs)
    
    # Position each pair horizontally
    x_position = 0
    positioned_pairs = []
    for pair in video_text_pairs:
        positioned_pair = pair.set_position((x_position, 0))
        positioned_pairs.append(positioned_pair)
        x_position += pair.w
    
    # Create the final composition
    final_composition = CompositeVideoClip(
        positioned_pairs,
        size=(final_width, final_height)
    )
    
    # Set the duration to the maximum of all video durations
    max_duration = max(clip.duration for clip in video_clips)
   
    final_composition = final_composition.set_duration(max_duration)

   
    # Write the final video
    final_composition.write_videofile(
        output_path, 
        codec='libx264', 
        audio_codec='aac',
        fps=24
    )
    
    # Close all clips
    for clip in video_clips + [final_composition]:
        clip.close()
    # Image clips don't need explicit closing
    
    return output_path




def visualize(inputs,pred=None, indices = None,vis_pred=False, save_im = True, compare_mode="BB",video_path=None):
    inputs = inputs.pixel_values.squeeze(0)
    if compare_mode == "optFlow":
        parent_dir = "dataset/GOT10KVAL_teacher"
        bbox = get_gt("BB", indices, f"{parent_dir}/{video_path}/x", inputs.shape[-1])
        bbox = np.array(bbox)
        bbox = bbox[indices]
        pred = pred.split(";")
        inputs = inputs[:-1,:,:,:]
        pred = np.array(pred)

    else:
        pred = np.array(pred)
        bbox = pred

    H, W = inputs.shape[-2], inputs.shape[-1]


    #pred_len = len(pred)
    if vis_pred == False:
        indices[-1] = min(indices[-1], len(pred)-1) 
        pred = pred[indices]
   
       
    else:
        if len(pred) < inputs.shape[0]:
            print("NOT ENOUGH BBOXES")
            return -1
   
    images_w_BB = []
    for i in range(inputs.shape[0]):

        x = inputs[i,:,:,:]
        x = x.permute(1, 2, 0)
        x = x.cpu().float().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        x = np.float32(x)
        x =  np.uint8(255 * x)
        x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        
        
        numbers = re.findall(r'<loc(\d+)>', bbox[i])
        y_min, x_min, y_max, x_max = map(int, numbers)
        x_min *= (W / 1024)
        y_min *= (H / 1024)
        x_max *= (W / 1024)
        y_max *= (H / 1024)

        x_min =  int(x_min)
        y_min =  int(y_min)
        x_max =  int(x_max)
        y_max =  int(y_max)
            
        if compare_mode=="BB":
            cv2.rectangle(x, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
        else:

            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            # image center
            icx, icy = W // 2, H // 2

            # parse flow string
            flow_str = pred[i]

            obj_match = re.search(r"obj<([-+]?\d+)><([-+]?\d+)>", flow_str)
            cam_match = re.search(r"cam<([-+]?\d+)><([-+]?\d+)>", flow_str)

            if not obj_match:
                print(flow_str)

            if obj_match:
                dx, dy = int(obj_match.group(1)), int(obj_match.group(2))
                #cv2.arrowedLine(x, (cx, cy), (cx + dx, cy + dy), (255, 0, 0), 2, tipLength=0.3)
                draw_arrow(x, (cx, cy), dx, dy, (255, 0, 0))
            if cam_match:
                dx, dy = int(cam_match.group(1)), int(cam_match.group(2))
                #cv2.arrowedLine(x, (icx, icy), (icx + dx, icy + dy), (0, 0, 255), 2, tipLength=0.3)
                draw_arrow(x, (icx, icy), dx, dy, (0, 0, 255))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        images_w_BB.append(x)
        if save_im:
            plt.imsave(f"to_del/img_{i}.png", x)
    
    idx = np.linspace(0, len(images_w_BB)-1, 10, dtype=int)
    return [images_w_BB[i] for i in idx]
   




def configure_options(args, model_type):
    d = {}
    if model_type == "orig": 
        d["model_path"] = args.orig_dir
        d["prompt"]     = args.prompt_orig
    else:
        d["model_path"] = args.finetuned_dir
        d["prompt"]     = args.prompt_finetune
    
    d["use_cfg"] = ("cfg" in args.mode and model_type != "orig")
    d["use_optflow"] = ("optflow" in args.mode and model_type != "orig")
    return d



# we want to both compare variants but also compare performance on blurry vs not blurry video
# the only compare right now - (1) evertyhing finetuned vs not (2) distilled but only after optical flow

# we want to ouput videos per-model, only if we do caption
# if we it is BB/ optical flow we do something else
#also applies for between models


def vis(args):
    model_types           = ["orig" ,"finetuned"  ,            ]
    if "internVL" not in args.model_size:
        processor_path        = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    else:
        processor_path        = "OpenGVLab/InternVL3-1B-hf"

    
    explainations_compare = []
    prompt                = ""


    explainations_compare_dict   = {"finetuned": {}, "orig": {}}
    inputs_dict                  = {}
    indices_dict                  = {}


    #ext = "labels_w_BB.csv" if "BB" in args.mode else "labels.csv"
    for model_type in model_types:
        ops = configure_options(args, model_type)
        model_path  = ops["model_path"]
        prompt      = ops["prompt"]
        use_cfg     = ops["use_cfg"]
        use_optflow = ops["use_optflow"]

        print(prompt)
        print(model_path)
       

       

       
        if "internVL" in args.model_size:  

            if True:
                model = InternVLForConditionalGeneration.from_pretrained(
                "finetuned_models/joint_learning_extended_mask_emph_freeze/internVL2/checkpoint-1500",# model_path, #model_path,  #, #  model_path, 
                dtype=torch.bfloat16,
                use_emph = ("emph" in args.mode),
                #is_pooling = True,
                #_attn_implementation="flash_attention_2",
                    ).to("cuda")
                processor = InternVLProcessor.from_pretrained(processor_path)
                #processor.video_processor.size= { "height": 384,"width": 384}
            else:


            #    bnb_config = BitsAndBytesConfig(
            #    load_in_4bit=True,
            #    bnb_4bit_use_double_quant=True,
            #    bnb_4bit_quant_type="nf4",
            #    bnb_4bit_compute_dtype=torch.bfloat16
            #)


                model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
               # quantization_config=bnb_config

            
                    ).to("cuda")
                processor = InternVLProcessor.from_pretrained(processor_path)
                #processor.video_processor.size= { "height": 384,"width": 384}
            
           
            

        else:
            processor = SmolVLMProcessor.from_pretrained(processor_path)

            if "joint_learning" in args.mode and model_type=="finetuned":
                model = model_jl.from_pretrained(
               "finetuned_models/joint_learning_extended_mask/2.2B/checkpoint-1800",
                torch_dtype=torch.bfloat16,
                use_mask = "mask" in args.mode,
                use_emph = ("mask" in args.mode) and ("emph" in args.mode),
                foc   = ("foc" in args.mode),
                l2    = ("l2" in args.mode)


                #_attn_implementation="flash_attention_2",
                ).to("cuda")

            else:
                if model_type=="finetuned" and "cfg" in args.mode:
                    
                    model = model_cfg.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            #_attn_implementation="flash_attention_2",
                            ).to("cuda")
                else:
                    model = SmolVLMForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        use_cfg = False,
                        use_optflow = use_optflow,
                     #   config = config
                        #_attn_implementation="flash_attention_2"
                    ).to("cuda")
        

        
        #if False:
        #    for name, param in model.model.vision_model.named_parameters():
        #        #print(name)
        #        if 'attentional_splatting.W_out' in name:
        #            param.data.zero_()

        #if False:
        #    flow_component =  create_gmflow_model(load_weights=True)#create_optical_flow_model()
        #    model.model.vision_model.optical_flow = flow_component
     
        
        #config = AutoConfig.from_pretrained(model_path)
        #model = SmolVLMForConditionalGeneration(config).to("cuda").to(torch.bfloat16)
        #exit(1)

        arr_to_test = SAMPLES_EMOTIONS_TO_TEST if args.emotion else SAMPLES_TO_TEST
        pref_dir = "emotion"  if args.emotion else "dataset/GOT10KVAL_teacher"

      
      
        for sample in arr_to_test:

            videos        = ["video_original.mp4",  "video_blur_full.mp4",        ] # , 
            explainations = [None for i in range(len(videos))]
            paths         = [f"{pref_dir}/{sample}/{videos[i]}" for i in range(len(videos))]
           
            #paths         = [f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_003748/{videos[i]}" for i in range(len(videos))]
            #paths         = [f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_008365/{videos[i]}" for i in range(len(videos))]

            #normalized_bbox = extract_bbox(f"dataset/GOT10KVAL_teacher/{sample}/{ext}") if "BB" in args.mode else None
            output_dir = f'{args.eval_dir}/{sample}' 
          
            os.makedirs(output_dir,  exist_ok=True)
         

            for i in range(len(explainations)):
                path = paths[i]
                
                

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": f"{path}"},
                            {"type": "text", "text": "Which direction the horse is heading"}, #prompt
                        ]
                    },
                ]

                
                #inputs = processor.apply_chat_template(
                #        messages,
                #        return_tensors="pt",
                #        add_generation_prompt=True,
                #        tokenize=True,
                #        return_dict=True,
                #        num_frames=30,).to(model.device)
#
                #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                #    output = model.generate(**inputs, max_new_tokens=128)
                #decoded_output = processor.batch_decode(
                #    output,
                #    skip_special_tokens=True,
                #)[0].split("assistant\n")[-1]
                #print(decoded_output)
                #exit(1)
                
                

                #processor.image_processor.video_sampling["max_frames"] = 1
                #print(processor)
                #print("\n\n")

                inputs, indices = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        extended        = args.extended,
                        return_frame_indices = True,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                    )
                
                
                inputs = inputs.to(model.device, dtype=torch.bfloat16)
                #visualize(inputs,pred=normalized_bbox,indices=indices)
               
                
                if use_cfg:
                   
                    input_shape = inputs["pixel_values"].shape[-1]
                    
                    inner_means, outer_means, diffs, pred_visibility = get_gt("t", indices, path, input_shape, return_raw=False,modified=True)
                    
                    inputs["movement_vectors"] = diffs
                    #inputs["pred_visibility"] = pred_visibility

                    
                    
                    
                #model.eval()
               
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)

                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                explaination = generated_texts[0].split("Assistant: ")[-1]

               
                print(explaination,flush = True)
                exit(1)
                
                
                #visualize(inputs,pred=parse_loc_string(explaination),indices=indices,vis_pred=True)

              
                explainations[i] = explaination

                if sample not in inputs_dict:
                    inputs_dict[sample] = []
                    indices_dict[sample] = []
                inputs_dict[sample].append(inputs)
                indices_dict[sample].append(indices)

           
            explainations_compare_dict[model_type][sample] = explainations

        #if "BB" not in args.mode and model_type!="finetuned":
        #    create_video_with_text(paths, explainations, f"{output_dir}/final_vid_{model_type}.mp4")
        
        
        #TODO  adjust visualize for the optical flow
        #TODO visualize in a different way BB and optical flow for blurry vs. not blurry
        #TODO run the optical flow variant vs. standard
        



        #explainations_compare.append(explainations[0])
        #paths = [paths[0],paths[0]]
    #print("\n\n")
    #explainations_compare[0] = f"Finetuned: {explainations_compare[0]}"
    #explainations_compare[1] = f"Standard: {explainations_compare[1]}"

    #print(paths)

    # FOR COMPARING BETWEEN VIDEOS
    #create_video_with_text(paths, explainations_compare, f"{output_dir}/final_vid_compare.mp4")

    #exit(1)
    
    # FOR COMPARING MODELS

    

    if "BB" not in args.mode:
        for sample in arr_to_test:
            for j in range(len(explainations)):
                ext = "_blurry" if "blur" in videos[j] else ""

                explain_baseline_original =  f"Standard: {explainations_compare_dict['orig'][sample][j]}" 
                explain_finetuned_original = f"Finetuned: {explainations_compare_dict['finetuned'][sample][j]}" 

                if False:
                    exit(1)
                    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,compare_mode= "optFlow",indices_dict=indices_dict)
                else:
                    explainations_models = [explain_baseline_original, explain_finetuned_original]
                    
                    #exit(1)
                    
                    create_video_with_text([paths[j],paths[j]], explainations_models, f"{output_dir}/compare_models{ext}.mp4")
    
        #if "optFlow" in args.mode:
        #    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,compare_mode="optFlow")
    else:
        print("CALLED")
        gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,indices_dict=indices_dict)



    #if "BB" in args.mode:
    #    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict)







def collate_fn_QA(examples,image_token_id,model,processor, extended, num_frames, open_cap,cap, with_dim, ker, des_prompt, is_det, is_internVL, return_video_id):
    videoPath = "video_path"
    instances = []
    answers = []
    for example in examples:

        print(example[videoPath])

        if open_cap == False:
            question = example['question']
            actual_question =question.split("\n")[0]
            choices = question.split("\n")[1:]
            choices = "\n".join(choices)
            #print(choices)
            actual_question =  f"Question: {actual_question}\nPossible answers:\n"
            question = actual_question + choices + f"\n\n{des_prompt}"

            
            
        else:
            question = cap

       

       # _,indices =  processor.apply_chat_template(dummy_input, add_generation_prompt=False, extended = extended, num_frames = num_frames,tokenize=True, return_dict=True, return_tensors="pt", return_frame_indices=True)
        if open_cap:
            user_content = [{"type": "video", "path": example[videoPath]}]
            user_content.append({"type": "text", "text": question})
            
        
        else:
            user_content = [{"type": "video", "path": example[videoPath]}]
            user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "user", "content": user_content},
        ]

        if is_internVL:
            instance = processor.apply_chat_template(messages,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        num_frames=30,).to(model.device, dtype=torch.bfloat16)
                    
            #print(processor.batch_decode(instance["input_ids"],skip_special_tokens=True,)[0])
            #print("----------------")
        else:
            instance = processor.apply_chat_template(messages, add_generation_prompt=True, extended = extended, num_frames = num_frames,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        
        dim = 0 if is_internVL else 1

        if instance["pixel_values"].shape[dim] < 30:
            
            #return {}
            frames_to_add = 30 - instance["pixel_values"].shape[dim]

            if is_internVL:
                instance["pixel_values"] = torch.cat([instance["pixel_values"], instance["pixel_values"][-1:].repeat(frames_to_add, 1, 1, 1, 1)], dim=0)
                
                instance["attention_mask"] = torch.cat([
                    instance["attention_mask"], 
                    instance["attention_mask"][-1:].repeat(frames_to_add, 1,  1, 1)  # Only 4 dimensions here
                ], dim=0)
            
            else:
                instance["pixel_values"] = torch.cat([instance["pixel_values"], instance["pixel_values"][:, -1:].repeat(1, frames_to_add, 1, 1, 1)], dim=1)
            
                instance["pixel_attention_mask"] = torch.cat([
                    instance["pixel_attention_mask"], 
                    instance["pixel_attention_mask"][:, -1:].repeat(1, frames_to_add, 1, 1)  # Only 4 dimensions here
                ], dim=1)


        
        if open_cap or is_det:
        
            instance["pixel_values"] = instance["pixel_values"].unsqueeze(0)
            
            print(instance["attention_mask"].shape)
            #instance["attention_mask"] = instance["attention_mask"].unsqueeze(0)
            #instance["input_ids"] = instance["input_ids"].unsqueeze(0)
           
            video_tensor = instance["pixel_values"]
            
            batch_size, num_frames, channels, height, width = video_tensor.shape
            original_size = (height, width)
            downsample_factors = [1, 2, 4, 8, 16, 32]  # Original + 4 downsampled versions
            
            # List to store all video versions
            video_versions = []

            if ker:
                blur_kernel_sizes = [1, 15, 25, 35, 45, 75]
                for kernel_size in blur_kernel_sizes:
                    if kernel_size == 1:
                        # Keep original video as is
                        video_versions.append(video_tensor)
                    else:
                        # Calculate sigma for Gaussian blur (rule of thumb: sigma = kernel_size / 6)
                        sigma = kernel_size / 6.0
                        
                        # Reshape to process all frames together: [1*30, 3, 384, 384]
                        video_reshaped = video_tensor.view(-1, channels, height, width)
                        
                        # Apply Gaussian blur
                        # Note: We need to apply blur to each channel separately if using conv2d
                        # Alternative: use torchvision.transforms.functional.gaussian_blur if available
                        
                        # Create Gaussian kernel
                        kernel_1d = torch.exp(-0.5 * ((torch.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
                        kernel_1d = kernel_1d / kernel_1d.sum()
                        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
                        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).to(video_tensor.device, dtype=video_tensor.dtype)
                        
                        # Apply convolution with padding to maintain size
                        padding = kernel_size // 2
                        video_blurred = F.conv2d(
                            video_reshaped, 
                            kernel_2d, 
                            padding=padding, 
                            groups=channels
                        )
                        
                        # Reshape back to original format: [1, 30, 3, 384, 384]
                        video_processed = video_blurred.view(batch_size, num_frames, channels, height, width)
                        video_versions.append(video_processed)
            else:
                # Process the original video tensor for each downsample factor
                for downsample_factor in downsample_factors:
                    if downsample_factor == 1:
                        # Keep original video as is
                        video_versions.append(video_tensor)
                    else:
                        # Calculate new size after downsampling
                        new_height = height // downsample_factor
                        new_width = width // downsample_factor
                        new_size = (new_height, new_width)

                        # Reshape to process all frames together: [1*30, 3, 384, 384]
                        video_reshaped = video_tensor.view(-1, channels, height, width)

                        # Downsample
                        video_downsampled = F.interpolate(
                            video_reshaped, 
                            size=new_size, 
                            mode='bilinear', 
                            align_corners=False
                        )

                        # Upsample back to original size
                        video_upsampled = F.interpolate(
                            video_downsampled, 
                            size=original_size, 
                            mode='bilinear', 
                            align_corners=False
                        )

                        # Reshape back to original format: [1, 30, 3, 384, 384]
                        video_processed = video_upsampled.view(batch_size, num_frames, channels, height, width)
                        video_versions.append(video_processed)
            
            # Stack all versions along batch dimension: [5, 30, 3, 384, 384]
            video_final = torch.cat(video_versions, dim=0)
            instance["pixel_values"] = video_final
            
            # Expand other tensors to match the new batch size
            num_versions = len(downsample_factors)  # 5
            
            #instance["attention_mask"] = instance["attention_mask"].repeat(num_versions, 1, 1, 1)
            
            # Repeat input_ids along batch dimension - check if 2D or 1D
            if instance["input_ids"].dim() == 2:  # [1, seq_len]
                instance["input_ids"] = instance["input_ids"].repeat(num_versions, 1)
            else:  # [seq_len]
                instance["input_ids"] = instance["input_ids"].unsqueeze(0).repeat(num_versions, 1)
            
            # Repeat attention_mask along batch dimension - check if 2D or 1D
            if instance["attention_mask"].dim() == 2:  # [1, seq_len]
                instance["attention_mask"] = instance["attention_mask"].repeat(num_versions, 1)
            else:  # [seq_len]
                instance["attention_mask"] = instance["attention_mask"].unsqueeze(0).repeat(num_versions, 1)
       
        
        '''
        print(video_final.shape)
        video_final = video_final[2]
        print(video_final.shape)
        
        for i in range(video_final.shape[0]):

            x = video_final[i,:,:,:]
            x = x.permute(1, 2, 0)
            x = x.cpu().float().numpy()
            x = (x - x.min()) / (x.max() - x.min())
            x = np.float32(x)
            x =  np.uint8(255 * x)
            x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
       
            plt.imsave(f"to_del4/imgX_{i}.png", x)
        
        exit(1)'''

       
        res = {"input": instance,
        "gt": example['answer']}

        if with_dim and is_det == False:
            res["dim"] = example['dim']
        if  return_video_id:
            res["video_id"] = example[videoPath]
        return res
        


        instances.append(instance)
        answers.append(example['answer'])


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

    #labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gt": answers
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






def eval(args):

    if args.eval:
        modelTXT = SentenceTransformer('all-MiniLM-L6-v2')

        d = {}
     
        model_path = args.orig_dir if "baseline" in args.mode else args.finetuned_dir
        prompt_finetune = args.prompt_orig if "baseline" in args.mode else args.prompt_finetune
        if args.model_size == "internVL":
            processor_path        = "OpenGVLab/InternVL3-1B-hf"
            processor = InternVLProcessor.from_pretrained(processor_path)

        else:
            processor_path        = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            processor = SmolVLMProcessor.from_pretrained(processor_path)


        
        downsample_factors = [1, 2, 4, 8, 16, 32] if "ker" not in args.mode else [1, 15, 25, 35, 45, 75]

 

        #processor.padding_side = "right"
        #processor.tokenizer.padding_side = "right"

        

        if args.model_size == "internVL":
            if args.dir_type != "distilled_models":
                model = InternVLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                use_emph = ("emph" in args.mode),
                #_attn_implementation="flash_attention_2",
                    ).to("cuda")

            else:
                model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
            
                    ).to("cuda")

        else:
            if args.dir_type == "distilled_models":
                
                model = SmolVLMForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                #_attn_implementation="flash_attention_2",
            ).to("cuda")

            else:

                if "joint_learning" in args.mode and args.dir_type=="finetuned_models":
                    model = model_jl.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    use_mask = "mask" in args.mode,
                    use_emph = ("mask" in args.mode) and ("emph" in args.mode),
                    foc   = ("foc" in args.mode),
                    l2    = ("l2" in args.mode)
                    #_attn_implementation="flash_attention_2",
                    ).to("cuda")
        
        image_token_id = None
        if args.model_size != "internVL":
            image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        d["num_total"] = 0

        des_prompt = None
        
        if args.eval == 'directions':
            
            d["num_correct_precise"]= 0
            d["num_correct"]= 0

            args.dataset_csv =  "dataset/tempcomp/filtered_data.csv"
        elif 'blur_video' in args.eval: 
            for i in downsample_factors:
                d[i] = {}
                d[i]["sim"]= 0
            if 'sorted' in args.eval:
                args.dataset_csv =  "dataset/got10k/sorted_output_file.csv"
            else:
                args.dataset_csv =  "dataset/got10k/video_data_Eval.csv"
    

        elif args.eval == 'tempcomp':
            args.dataset_csv =  "dataset/tempcomp/all_data.csv"
            des_prompt = "Answer only with the option from the given choices directly." if args.model_size == "internVL" else "Answer with the letter.\n"
        
        elif args.eval == 'tempcomp_motion_bench':
            args.dataset_csv =  "dataset/motion_bench/all_data.csv"
            des_prompt = "Answer with the option's letter from the given choices directly." if args.model_size == "internVL" else "Answer with the letter.\n"

        elif args.eval == 'tempcomp_motion_bench_sports':
            args.dataset_csv =  "dataset/motion_bench/all_data_sports_fixed.csv"
            des_prompt = "Answer only with the option from the given choices directly."


        elif args.eval == 'tempcomp_mvbench':
            args.dataset_csv =  "dataset/mvbench/all_data.csv"
            des_prompt = "Answer with the option's letter from the given choices directly." if args.model_size == "internVL" else "Answer with the letter.\n"

        elif args.eval == 'tempcomp_caption_matching':
            args.dataset_csv =  "dataset/tempcomp/all_data_caption_matching.csv"
            des_prompt = "Answer only with the option from the given choices directly." if args.model_size == "internVL" else ""


        elif args.eval == 'tempcomp_yes_no':
            args.dataset_csv =  "dataset/tempcomp/all_data_yes_no.csv"
            des_prompt = "Answer yes or no.\n"

        
        elif 'tempcomp_det' in args.eval:
            args.dataset_csv =  "dataset/activity/all_data.csv"
            des_prompt = "Answer with the option's letter from the given choices directly." if args.model_size == "internVL" else "Answer with the letter.\n"  
            
        
      
        print(prompt_finetune)
        
        data_collator = partial(collate_fn_QA, image_token_id=image_token_id, 
        model=model, processor=processor, extended=True, 
        num_frames=30,open_cap = ('blur_video' in args.eval), 
        is_det = ('tempcomp_det' in args.eval),
        cap = prompt_finetune,
        with_dim = ('tempcomp' in args.eval),
        ker = (args.eval =='blur_video_ker_sorted') or (args.eval =='blur_video_ker') or (args.eval =='tempcomp_det_ker_check'),
        des_prompt=des_prompt,
        is_internVL = (args.model_size == "internVL"))  

        
        
        dataset = load_dataset("csv", data_files=args.dataset_csv)["train"]
        #if True:
        #    dataset = dataset.select(range(25))

       
        if 'blur_video' in args.eval:
            if 'sorted' in args.eval:
                dataset = dataset.select(range(1000))
            else:
                dataset = dataset.shuffle(seed=42).select(range(1500))
        
        
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=data_collator,
                shuffle=False
        )

        tot_len = len(dataloader)
        for idx, batch in enumerate(dataloader):
            print(f"{idx}/{tot_len}")

            '''prompt = "What is the direction of the man?\nPossible answers:\nA. moving towards the camera\nB. moving from left to right\nC. moving away from the camera\n\nAnswer with the letter (A,B, or C)."
            
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": "dataset/tempcomp/videos/1034419625.mp4"},
                            {"type": "text", "text": prompt}
                        ]
                    },
                ]

            inputs, indices = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        extended        = args.extended,
                        return_frame_indices = True,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                    )
                
                
            inputs = inputs.to(model.device, dtype=torch.bfloat16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)

            generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            explaination = generated_texts[0]

               
            print(explaination,flush = True)
            gt = batch["gt"]

            print(gt)
            exit(1)'''
            if batch == {}:
                continue
            gt = batch["gt"]
            dim = None
            if ('tempcomp' in args.eval) and ("tempcomp_det" not in args.eval):
                dim = batch["dim"]


            inputs =  {k: v for k, v in batch.items() if (k != "gt" and k != "dim")} 
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs["input"], do_sample=False, max_new_tokens=128)
            generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

   
            
            if args.model_size == "internVL":

                #print(generated_texts[0])
                #exit(1)

                explainations = [ex.split("assistant\n")[-1] for ex in generated_texts] 
                explainations = [ex.split("Answer:")[-1] for ex in explainations]
                explainations = [ex.split("Answer: ")[-1] for ex in explainations] 
                #explainations = generated_texts
                #print("")

                 
            else:
                explainations = [ex.split("Assistant: ")[-1] for ex in generated_texts] 
            for i, ex in enumerate(explainations):

                if args.eval == 'directions' or 'tempcomp' in args.eval:
                    if 'tempcomp' in args.eval:
                        if dim is None:
                            dim = downsample_factors[i]
                        if dim not in d:
                            d[dim] = {}
                            d[dim]["num_correct"] = 0
                            d[dim]["num_correct_precise"] = 0
                            d[dim]["num_total"] = 0
                        d[dim]["num_total"]+=1

                    if  'tempcomp_det' in args.eval:
                        #print("\n")
                        gt_letter = gt.split(".")[0]
                        if ex ==gt:
                            d[downsample_factors[i]]["num_correct"]+=1
                       
                        elif ex.split(".")[0] == gt_letter:
                            d[downsample_factors[i]]["num_correct_precise"]+=1
                            d[downsample_factors[i]]["num_correct"]+=1
                        
                        
                        else:
                            print("FAIL")
                            print(ex)
                            print(gt)
                            print("-----------------")

                
                        continue
                    
                    gt_letter = gt.split(".")[0]
                    if args.eval == 'tempcomp_yes_no':
                        ex = ex.lower()

                    if ex.split(".")[0] == gt_letter:
                        if 'tempcomp' in args.eval:
                            d[dim]["num_correct_precise"]+=1
                            d[dim]["num_correct"]+=1
                        else:
                            d["num_correct_precise"]+=1
                            d["num_correct"]+=1
                            print("SUCCESS")
                    
                    elif (gt in ex) or ((gt.split(":")[0].split(" ")[-1] == ex.split(":")[0].split(" ")[-1]) and (args.eval == 'tempcomp_caption_matching')):

                
                        if 'tempcomp' in args.eval:
                            d[dim]["num_correct"]+=1
                        else:
                            d["num_correct"]+=1
                        
                        
                        print("SUCCESS2")
                       

                    
                    else:
                        #print(generated_texts)
                        print("FAIL")
                        print(ex)
                        print(gt)
                        #exit(1)
                        print("-----------------")
                    
                else:
                  
                    original_embedding = modelTXT.encode([ex])
                    degraded_embedding = modelTXT.encode([explainations[0]])

                    similarity = cosine_similarity(original_embedding, degraded_embedding)[0][0]
                                    
                    d[downsample_factors[i]]["sim"]+=float(similarity)
                    
                
                d["num_total"]+=1
        
        
        print(args.eval_dir)
        


        update_json(f"{args.eval_dir}/res_{args.eval}_{args.epoch}.json",d)
        exit(1)


            

            



























def eval_check_qual(args):

    if True:
        modelTXT = SentenceTransformer('all-MiniLM-L6-v2')

        d = {}
        dd = {}
     
        model_path = args.orig_dir if "baseline" in args.mode else args.finetuned_dir
        prompt_finetune = args.prompt_orig if "baseline" in args.mode else args.prompt_finetune
        if args.model_size == "internVL":
            processor_path        = "OpenGVLab/InternVL3-1B-hf"
            processor = InternVLProcessor.from_pretrained(processor_path)

        else:
            processor_path        = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            processor = SmolVLMProcessor.from_pretrained(processor_path)


        
        downsample_factors = [1, 2, 4, 8, 16, 32] if "ker" not in args.mode else [1, 15, 25, 35, 45, 75]


        if False:
            pass
           

        else:
            if args.dir_type == "distilled_models":
                model = SmolVLMForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                #_attn_implementation="flash_attention_2",
            ).to("cuda")

            else:

                if "joint_learning" in args.mode and args.dir_type=="finetuned_models":
                    model = model_jl.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    use_mask = "mask" in args.mode,
                    use_emph = ("mask" in args.mode) and ("emph" in args.mode),
                    foc   = ("foc" in args.mode),
                    l2    = ("l2" in args.mode)
                    #_attn_implementation="flash_attention_2",
                    ).to("cuda")
        
        image_token_id = None
        if args.model_size != "internVL":
            image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
            ]

 

        des_prompt = "Answer yes or no.\n"
        args.dataset_csv =  "dataset/tempcomp/all_data_yes_no.csv"

        
        
        
        data_collator = partial(collate_fn_QA, image_token_id=image_token_id, 
        model=model, processor=processor, extended=True, 
        num_frames=30,open_cap = ('blur_video' in args.eval), 
        is_det = ('tempcomp_det' in args.eval),
        cap = prompt_finetune,
        with_dim = ('tempcomp' in args.eval),
        ker = (args.eval =='blur_video_ker_sorted') or (args.eval =='blur_video_ker') or (args.eval =='tempcomp_det_ker_check'),
        des_prompt=des_prompt,
        is_internVL = (args.model_size == "internVL"),
        return_video_id = True)  

        
        
        dataset = load_dataset("csv", data_files=args.dataset_csv)["train"]

       
        
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=data_collator,
                shuffle=False
        )

        tot_len = len(dataloader)
        for idx, batch in enumerate(dataloader):
            print(f"{idx}/{tot_len}")

          
            if batch == {}:
                continue
            gt = batch["gt"]
            dim = None
            dim = batch["dim"]
            video_id = batch["video_id"]
            p_dir = video_id.split(".")[0]


            if os.path.isdir(p_dir) == False:
                print(f"Directory  {p_dir} no exists")
                continue
            real_vidID = (video_id.split(".")[0]).split("/")[-1]
            if real_vidID not in dd:
                dd[real_vidID] = 0
            dd[real_vidID]+=1
            real_vidID =f"{real_vidID}##{dd[real_vidID]}"
      
        

            inputs =  {k: v for k, v in batch.items() if (k != "gt" and k != "dim")} 
            inputs["input"]["video_id"] = real_vidID
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs["input"], do_sample=False, max_new_tokens=128)
            generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

   
            explainations = [ex.split("Assistant: ")[-1] for ex in generated_texts] 
            
            
            for i, ex in enumerate(explainations):
                

    
                gt_letter = gt.split(".")[0]
                ex = ex.split(".")[0]
                if args.eval == 'tempcomp_yes_no':
                    ex = ex.lower()


                sym = "C" if gt_letter == ex else "W"
                if sym == "W":
                    print(gt_letter)
                    print(ex)
                    
                open(f"check_res3/{real_vidID}_${dim}$_{sym}.txt", "w").close()

            



def extract_info(args):
    from collections import defaultdict

     # Paths
    base_dir = "check_res3"       # contains .txt and flow_*.pt
    gt_dir = "dataset/tempcomp/videos"  # ground truth subfolders by ID

    results = defaultdict(list)

    for fname in os.listdir(base_dir):
        if not fname.endswith(".txt"):
            continue

        # Parse filename
        # Example: 1034419625##1_animals$_W.txt
        stem = fname[:-4]
        try:
            id_i, category_letter = stem.split("_", 1)
            category, letter = category_letter.split("$_")
            letter = letter  # "W" or "C"
            category = category.split("$")[-1]
            print(id_i, category, letter)
          
        except ValueError:
            print("Skipping malformed:", fname)
            continue

        # Prediction file
        flow_path = os.path.join(base_dir, f"flow_{id_i}.pt")
        if not os.path.exists(flow_path):
            print("Missing flow:", flow_path)
            continue
        pred = torch.load(flow_path)  # [29,2]

        # Ground truth file: gt_dir/{ID}/*.pt
        ID = id_i.split("##")[0]
        gt_subdir = os.path.join(gt_dir, ID, "tracks_cuts_grid")
        if not os.path.isdir(gt_subdir):
            print("Missing gt dir:", gt_subdir)
            continue
        gt_files = [f for f in os.listdir(gt_subdir) if f.endswith("pred_tracks.pt")]
        if not gt_files:
            print("No gt in:", gt_subdir)
            continue
        gt = torch.load(os.path.join(gt_subdir, gt_files[0]))  # assume first

        # Compute L1
        l1 = torch.abs(pred - gt).mean().item()
        results[(category, letter)].append(l1)

    # Report averages
    for (category, letter), vals in results.items():
        avg_l1 = sum(vals) / len(vals)
        print(f"Category={category}, {letter}: Avg L1 = {avg_l1:.6f} over {len(vals)} samples")








if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)

    #extract_info(args)
    #exit(1)
#
#
    #eval_check_qual(args)
    #exit(1)

    if args.compare_mode == "vis":
        vis(args)
    else:
        eval(args)




#30 29 18 15 27 7