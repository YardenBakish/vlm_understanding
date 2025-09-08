'''
TODO
1. set up initial framework for distillation - for now do it even on very small numbered examples
2. set up framework to blur objects and create entire datasets

'''
import os
from pathlib import Path

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



import config
import argparse

import matplotlib.font_manager as fm
from moviepy.editor import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from processing_smolvlm import SmolVLMProcessor



SAMPLES_TO_TEST = [

#"GOT-10k_Val_000018",
"GOT-10k_Val_000030",

#"GOT-10k_Val_000022",
#"GOT-10k_Val_000003",
#"GOT-10k_Val_000006",
#"GOT-10k_Val_000007",
#"GOT-10k_Val_000015",
#"GOT-10k_Val_000027",
#"GOT-10k_Val_000029",
#"GOT-10k_Val_000085",
#"GOT-10k_Val_000107",
#"GOT-10k_Val_000132",
#"GOT-10k_Val_000014",
#"GOT-10k_Val_000017",
#"GOT-10k_Val_000022",

]

SAMPLES_EMOTIONS_TO_TEST =[
    "vid1",
    "vid2",
    "vid3",
    "vid4",
    "vid5",
]



def parse_args():
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=[
                                        'distil_BB_extended', 'distil_optFlow_extended','distil_optFlow_extended_coarse',
                                        'finetune_cfg_cap', 'finetune_cfg_BB',
                                        'finetune_cfg_optflow_cap', 'finetune_cfg_optflow_BB',
                                        'finetune_joint_learning_extended_mask',
                                        'finetune_cfg_cap_simple_extended'
                                        ],
                                default = 'distil_BB_extended',
                                help='')
    
    parser.add_argument('--compare-mode', type=str,
                                choices=['vis', 'metric'],
                                default = 'vis',
                                help='')

    parser.add_argument('--model-size', type=str,
                                choices=['500M', '2.2B', '2.2B_quant'],
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
    eval_dir         = f"eval/{work_dir}" if args.compare_mode == "vis" else f"eval/seqVSsingle/{work_dir}"
    os.makedirs(eval_dir,exist_ok=True)
    args.eval_dir = eval_dir

    if "BB" in args.mode:
        args.orig_dir = get_last_checkpoint_dir(f'{args.paths["distilled_models"]}/BB_extended/{args.model_size}')
    else:
        args.orig_dir = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    args.finetuned_dir =  get_last_checkpoint_dir(finedtuned_dir) #  #args.orig_dir

    
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
    model_types           = [ "finetuned"  ,"orig" ,    ]
    processor_path        = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
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
       

       
        processor = SmolVLMProcessor.from_pretrained(processor_path)

       
        if args.dir_type == "distilled_models":
            model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            #_attn_implementation="flash_attention_2",
        ).to("cuda")

            

        else:
            if "joint_learning" in args.mode and model_type=="finetuned":
                model = model_jl.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_mask = "mask" in args.mode
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

            videos        = ["video_original.mp4",  "video_blur_full.mp4",  ] #"video_blur_object.mp4",
            explainations = [None for i in range(len(videos))]
            paths         = [f"{pref_dir}/{sample}/{videos[i]}" for i in range(len(videos))]
            #paths         = [f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_001148/{videos[i]}" for i in range(len(videos))]


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
                            {"type": "text", "text": prompt}
                        ]
                    },
                ]
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

                    
                    
                    

               
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)

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
        for sample in SAMPLES_TO_TEST:
            for j in range(len(explainations)):
                ext = "_blurry" if "blur" in videos[j] else ""

                explain_baseline_original =  f"Standard: {explainations_compare_dict['orig'][sample][j]}" 
                explain_finetuned_original = f"Finetuned: {explainations_compare_dict['finetuned'][sample][j]}" 

                if True:
                    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,compare_mode= "optFlow",indices_dict=indices_dict)
                else:
                    explainations_models = [explain_baseline_original, explain_finetuned_original]
                    create_video_with_text([paths[j],paths[j]], explainations_models, f"{output_dir}/compare_models{ext}.mp4")
    
        #if "optFlow" in args.mode:
        #    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,compare_mode="optFlow")
    else:
        print("CALLED")
        gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict,indices_dict=indices_dict)



    #if "BB" in args.mode:
    #    gen_visualizations_sampled_frames(args,explainations_compare_dict,inputs_dict)







#BB - blurry vs. standard
#optical - blurry vs. standard
# optical+blurry vs. standard
#optical+blurry vs. standard

#rest - finetuned

def eval(args):

    d_eval = {}

    test_dir = Path("dataset/GOT10KVAL_teacher")
    subdirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    video_files_list = []
    video_blur_files_list = []

    for subdir in subdirs:
        file_video_path = subdir / "video_original.mp4"
        file_video_blur_path = subdir / "video_blur_full.mp4"
        if file_video_path.exists():
            video_files_list.append(str(f"{test_dir}/{file_video_path.relative_to(test_dir)}"))
            video_blur_files_list.append(str(f"{test_dir}/{file_video_blur_path.relative_to(test_dir)}"))

   
    
    all_video_files_list = [video_files_list, video_blur_files_list]

    if args.mode == "optFlow" not in args.mode:
        model_types           = [ "finetuned",    ]
    else:
        model_types           = ["finetuned",   "orig",    ]

    processor_path        = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
  
    for model_type in model_types:
        ops = configure_options(args, model_type)
        model_path  = ops["model_path"]
        prompt      = ops["prompt"]
        use_cfg     = ops["use_cfg"]
        use_optflow = ops["use_optflow"]
       
        processor = SmolVLMProcessor.from_pretrained(processor_path)

        if args.dir_type == "distilled_models":
            model = AutoModelForImageTextToText.from_pretrained(
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
            ).to("cuda")
        

        for i in range(len(all_video_files_list)):
            if i == 0:
                video_type = "standard"
            else:
                video_type = "blur"
            
            d_eval[video_type] = {}
            d_eval[video_type]["num"] = 0
            if "BB" in args.mode:
                d_eval[video_type]["iou"] = 0
            else:
                d_eval[video_type]["inner"] = 0
                d_eval[video_type]["outer"] = 0





            videos_lst = all_video_files_list[i]
            for video_path in videos_lst:
                
               
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": f"{video_path}"},
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
                
                indices[-1] = indices[-1] -1

                
                inputs = inputs.to(model.device, dtype=torch.bfloat16)

                if use_cfg:
                    pred_tracks, pred_visibility = collect_tracks(None, f"debug_tracks/{sample}", 
                                          indices, 
                                          inputs.pixel_values.shape[-1], 
                                          inputs.pixel_values.shape[-2],
                                          skip_parse=True)

                    inputs['pred_tracks'] = pred_tracks
                    inputs['pred_visibility'] = pred_visibility
                
                generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
                
                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                explaination = generated_texts[0].split("Assistant: ")[-1]

                eval_performance(d_eval,video_type, explaination, indices, args.mode, video_path, inputs["pixel_values"].shape[-1] )
                #gt_data =  get_gt(args.mode, indices, video_path, inputs["pixel_values"].shape[-1])
                #gt_data = np.array(gt_data)


                #model_pred =  parse_loc_string(explaination)
                #model_pred = np.array(model_pred)

                #gt_data    = gt_data[indices]

                #for j in range(min(len(gt_data), len(model_pred))):
                #    d_eval[video_type]["num"]+=1
                #    eval_performance_per_frame(d_eval,video_type,args.mode, model_pred[j], gt_data[j])
                #    if j == 1:
                #        break
                break
            
            
            
    print(d_eval)
    






if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)
    if args.compare_mode == "vis":
        vis(args)
    else:
        eval(args)




#30 29 18 15 27 7