'''
TODO
1. set up initial framework for distillation - for now do it even on very small numbered examples
2. set up framework to blur objects and create entire datasets

'''
import os
os.environ['MPLCONFIGDIR'] = '/scratch200/yardenbakish/'

import config
import argparse
import matplotlib.pyplot as plt

import re
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText ,TrainingArguments, Trainer, TrainerCallback
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file
import torch
from convert_utils import *
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
#from model import SmolVLMForConditionalGeneration
from model_optical_flow import SmolVLMForConditionalGeneration





import config
import argparse

import matplotlib.font_manager as fm
'''
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for path in font_paths:
    print(path)
exit(1)

'''
from moviepy.editor import *


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


from processing_smolvlm import SmolVLMProcessor



SAMPLES_TO_TEST = [
#"GOT-10k_Val_000001",
#"GOT-10k_Val_000003",
#"GOT-10k_Val_000006",
#"GOT-10k_Val_000007",
#"GOT-10k_Val_000015",
#"GOT-10k_Val_000018",
#"GOT-10k_Val_000027",
#"GOT-10k_Val_000029",
"GOT-10k_Val_000030",
#"GOT-10k_Val_000085",
#"GOT-10k_Val_000107",
#"GOT-10k_Val_000132",


#"GOT-10k_Val_000014",
#"GOT-10k_Val_000017",
#"GOT-10k_Val_000022",






]



def get_last_checkpoint_dir(directory):
    largest_number = -1
    largest_file = None

    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern 'file_<number>'
        match = re.match(r'checkpoint-(\d+)', filename)
        if match:
            # Extract the number part from the filename and convert it to an integer
            number = int(match.group(1))
            # Update the largest file if the current number is larger
            if number > largest_number:
                largest_number = number
                largest_file = filename

    if largest_file:
        return os.path.join(directory, largest_file)
    else:
        print("No suitable file found")

        return False  # No file found that matches the pattern


def parse_args():
    parser = argparse.ArgumentParser(description='Distillation of SmolVLM2-500M-Video-Instruct for poor-apperance sequence')

    parser.add_argument('--mode', type=str,
                                choices=['uniform_blur', 
                                         'uniform_blur_extended',  
                                         'BB_w_blur','BB_w_blur2', 
                                         'BB_w_blur_extended', 
                                         'cfg',
                                         'caption_optical_flow',
                                         'BB_optical_flow'],
                                default = 'uniform_blur',
                                help='')
    
    parser.add_argument('--compare-mode', type=str,
                                choices=['standard', 'single_frame'],
                                default = 'standard',
                                help='')

    parser.add_argument('--model-size', type=str,
                                choices=['500M', '2.2B', '2.2B_quant'],
                                default = '500M',
                                help='')
   

    
    args = parser.parse_args()
    config.get_config(args)

   
    work_dir      = f'{args.mode}/{args.model_size}'
    distilled_dir = f'{args.paths["distilled_models"]}/{work_dir}'
    eval_dir      = f"eval/{work_dir}" if args.compare_mode == "standard" else f"eval/seqVSsingle/{work_dir}"
    os.makedirs(eval_dir,exist_ok=True)
    args.eval_dir = eval_dir
    if args.mode == "BB_w_blur":
        args.orig_dir = get_last_checkpoint_dir(f'{args.paths["distilled_models"]}/BB/{args.model_size}')
    else:
        args.orig_dir = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    args.distiled_dir = args.orig_dir # get_last_checkpoint_dir(distilled_dir)
    


    args.prompt_orig = "Describe this video in detail" if "BB" not in args.mode else "Return xyxy coordinates for the object in the video"
    args.prompt_distil = "Caption the video." if "BB" not in args.mode else "Return xyxy coordinates for the object in the video"


    args.extended = True
    args.num_frames = 20
    args.extend_frames = None

    if "extended" in args.mode:
        args.extended = True
        args.num_frames = 20 if "500M" in args.model_size else 20
        args.extend_frames = True


    return args




def gen_comparisons_w_BB(args,explainations_compare_dict,inputs_dict,compare_mode="standard"):

    model_type1 = "orig" if compare_mode=="standard" else "per_frame"
    model_type2 = "distilled" if compare_mode=="standard" else "sequence"
    
    text_row1 = "Original" if compare_mode=="standard" else "per_frame"
    text_row2 = "Distilled" if compare_mode=="standard" else "sequence"


    videos_types = ["original", "blur_object","blur_full"]
    rev_d = {}
    for model_type in explainations_compare_dict.keys():
        for sample in explainations_compare_dict[model_type]:
            if sample not in rev_d:
                rev_d[sample] = {}
            rev_d[sample][model_type] = explainations_compare_dict[model_type][sample]

    for sample in rev_d:
        output_dir = f'{args.eval_dir}/{sample}'
        for i in range(len(videos_types)):
            orig_pred_images      =  visualize(inputs_dict[sample][i],bbox=parse_loc_string(rev_d[sample][model_type1][i]),indices=None,vis_pred=True, save_im = False)
            distilles_pred_images =   visualize(inputs_dict[sample][i],bbox=parse_loc_string(rev_d[sample][model_type2][i]),indices=None,vis_pred=True, save_im = False)
            
            if orig_pred_images == -1 or distilles_pred_images == -1:
                continue
            
            save_image_grid(orig_pred_images, distilles_pred_images, f"{output_dir}/{videos_types[i]}_compare.png", text_row1=text_row1, text_row2=text_row2)



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
    font = ImageFont.truetype("FreeSansBold.ttf", font_size)

    
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




def visualize(inputs,bbox=None, indices = None,vis_pred=False, save_im = True):
    inputs = inputs.pixel_values.squeeze(0)
    H, W = inputs.shape[-2], inputs.shape[-1]
    bbox = np.array(bbox)
    #bbox_len = len(bbox)
    if vis_pred == False:
        indices[-1] = min(indices[-1], len(bbox)-1) 
        bbox = bbox[indices]
   
    else:
        if len(bbox) < inputs.shape[0]:
            print("NOT ENOUGH BBOXES")
            return -1
   
    images_w_BB = []
    for i in range(inputs.shape[0]):
        
        numbers = re.findall(r'<loc(\d+)>', bbox[i])
        y_min, x_min, y_max, x_max = map(int, numbers)
        print(x_min)
        x_min *= (W / 1024)
        y_min *= (H / 1024)
        x_max *= (W / 1024)
        y_max *= (H / 1024)
        
        x_min =  int(x_min)
        y_min =  int(y_min)
        x_max =  int(x_max)
        y_max =  int(y_max)


        x = inputs[i,:,:,:]
       
        x = x.permute(1, 2, 0)
        x = x.cpu().float().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        x = np.float32(x)
        x =  np.uint8(255 * x)
        x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)

        
        cv2.rectangle(x, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        images_w_BB.append(x)
        if save_im:
            plt.imsave(f"to_del/img_{i}.png", x)
    return images_w_BB




def eval(args):
    model_types = ["orig",   "distilled",  ]
    processor_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    explainations_compare = []
    video_to_compare      = ["video_blur_full.mp4"]
    prompt                = ""
    normalized_bbox                  = None

    explainations_compare_dict   = {"distilled": {}, "orig": {}}
    inputs_dict                 = {}
    ext = "labels_w_BB.csv" if "BB" in args.mode else "labels.csv"
    for model_type in model_types:
        if model_type == "orig": #False:
            model_path = args.orig_dir
            prompt = args.prompt_orig
        else:
            model_path = args.distiled_dir
            prompt = args.prompt_distil
        
        if "BB" in args.mode:
            processor = SmolVLMProcessor.from_pretrained(processor_path)
        else:
            processor = SmolVLMProcessor.from_pretrained(processor_path)

        #config = AutoConfig.from_pretrained(model_path)
        #config.pixel_shuffle_factor = 15
        #config.scale_factor = 15

 

        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_cfg = True,

         #   config = config
            #_attn_implementation="flash_attention_2"
        ).to("cuda")

        
        if True:
            for name, param in model.model.vision_model.named_parameters():
                if 'attentional_splatting.W_out' in name:
                    param.data.zero_()


          
       
        if True:
            flow_component =  create_gmflow_model(load_weights=True)#create_optical_flow_model()
            model.model.vision_model.optical_flow = flow_component
     
        
        #config = AutoConfig.from_pretrained(model_path)
        #model = SmolVLMForConditionalGeneration(config).to("cuda").to(torch.bfloat16)
        #state_dict = load_file(f"{model_path}/model.safetensors")
        #model.load_state_dict(state_dict)
        #model = model.to(torch.bfloat16).to("cuda")
        #exit(1)
      
        for sample in SAMPLES_TO_TEST:

            explainations = [None,None,None]
            videos = ["video_original.mp4", "video_blur_object.mp4","video_blur_full.mp4"]
            paths = [f"dataset/GOT10KVAL_teacher/{sample}/{videos[i]}" for i in range(3)]

            normalized_bbox = extract_bbox(f"dataset/GOT10KVAL_teacher/{sample}/{ext}") if "BB" in args.mode else None
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
                if "BB" in args.mode:
                    inputs, indices = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        return_frame_indices = True,
                        extended        = args.extended,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                    )
                   
                    print(indices)

                else:
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
                #visualize(inputs,bbox=normalized_bbox,indices=indices)
                #exit(1)
                #print(inputs.pixel_values.shape)
                #exit(1)
                
                
                pred_tracks = torch.load("conditioned_models/co-tracker/pred_tracks.pt")
                pred_visibility = torch.load("conditioned_models/co-tracker/pred_visibility.pt")
                
                print(pred_tracks.shape)
                print(pred_tracks[:,indices,:,:].shape)
                


                inputs['pred_tracks'] = pred_tracks[:,indices,:,:].to(dtype=torch.bfloat16)
                inputs['pred_visibility'] = pred_visibility

               

                generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)

                
                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                #print("HREE")
                explaination = generated_texts[0].split("Assistant: ")[-1]
                print(explaination)
                exit(1)
                
                #visualize(inputs,bbox=parse_loc_string(explaination),indices=indices,vis_pred=True)

              
                #print(explaination)
                #exit(1)
               
                explainations[i] = explaination

                #print(explaination)
                #exit(1)
                if sample not in inputs_dict:
                    inputs_dict[sample] = []
                inputs_dict[sample].append(inputs)

           
            explainations_compare_dict[model_type][sample] = explainations

            if "BB" not in args.mode:
                create_video_with_text(paths, explainations, f"{output_dir}/final_vid_{model_type}.mp4")

        #explainations_compare.append(explainations[0])
        #paths = [paths[0],paths[0]]
    #print("\n\n")
    #explainations_compare[0] = f"Distilled: {explainations_compare[0]}"
    #explainations_compare[1] = f"Standard: {explainations_compare[1]}"

    #print(paths)

    #create_video_with_text(paths, explainations_compare, f"{output_dir}/final_vid_compare.mp4")
    if "BB" in args.mode:
        gen_comparisons_w_BB(args,explainations_compare_dict,inputs_dict)








def eval_sequence_vs_single(args):
    model_types = ["per_frame", "sequence", ]
    processor_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    prompt                = ""

    explainations_compare_dict   = {"sequence": {}, "per_frame": {}}
    inputs_dict                 = {}
    ext = "labels_w_BB.csv" if "BB" in args.mode else "labels.csv"
    for model_type in model_types:
        model_path = args.distiled_dir
        prompt = args.prompt_distil
        
        if "BB" in args.mode:
            processor = SmolVLMProcessor.from_pretrained(processor_path)
        else:
            processor = AutoProcessor.from_pretrained(processor_path)

        

        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
         #   config = config
            #_attn_implementation="flash_attention_2"
        ).to("cuda")
     
        for sample in SAMPLES_TO_TEST:

            explainations = [None,None,None]
            videos = ["video_original.mp4", "video_blur_object.mp4","video_blur_full.mp4"]
            paths = [f"dataset/GOT10KVAL_teacher/{sample}/{videos[i]}" for i in range(3)]

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
                        return_frame_indices = True,
                        extended        = args.extended,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                )



                if model_type == "per_frame":
                    BBs_per_frame = []
                    for idx in indices:
                 
                        inputs_per_frame = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        selected_index       = idx,
                        extended        = args.extended,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                        )



                        if args.extend_frames:
                            inputs_per_frame.pixel_values = inputs_per_frame.pixel_values.repeat(1, 10, 1, 1, 1)

                        
                        inputs_per_frame = inputs_per_frame.to(model.device, dtype=torch.bfloat16)
                        print("REACHED")

                        generated_ids = model.generate(**inputs_per_frame, do_sample=False, max_new_tokens=1024)
                        generated_texts = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                        )
                        explaination_per_frame = generated_texts[0].split("Assistant: ")[-1]
                        explaination_per_frame = explaination_per_frame.split(";")[0]
                        BBs_per_frame.append(explaination_per_frame)
                        
                    explaination = ";".join(BBs_per_frame)
                else:

                   
               
                
                    inputs = inputs.to(model.device, dtype=torch.bfloat16)
                    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
                    generated_texts = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    explaination = generated_texts[0].split("Assistant: ")[-1]

                explainations[i] = explaination

 
                if sample not in inputs_dict:
                    inputs_dict[sample] = []
                inputs_dict[sample].append(inputs)

           
            explainations_compare_dict[model_type][sample] = explainations

            if "BB" not in args.mode:
                create_video_with_text(paths, explainations, f"{output_dir}/final_vid_{model_type}.mp4")

        #explainations_compare.append(explainations[0])
        #paths = [paths[0],paths[0]]
    #print("\n\n")
    #explainations_compare[0] = f"Distilled: {explainations_compare[0]}"
    #explainations_compare[1] = f"Standard: {explainations_compare[1]}"

    #print(paths)

    #create_video_with_text(paths, explainations_compare, f"{output_dir}/final_vid_compare.mp4")
    if "BB" in args.mode:
        gen_comparisons_w_BB(args,explainations_compare_dict,inputs_dict, compare_mode="single_frame")






if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)
    if args.compare_mode == "standard":
        eval(args)
    elif args.compare_mode == "single_frame":
        eval_sequence_vs_single(args)



#30 29 18 15 27 7