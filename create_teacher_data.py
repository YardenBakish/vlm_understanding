# given directory, sort all directories by name and numerical order
# manually set some interval for which videos to process
# create a new dir with original video + prediction, and blurry video/ partial and full

import glob

import config
import argparse
import os

import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from convert_utils import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd

FILTERED_DATASET = 'dataset/got10k/filtered_got10_train.txt'
PARTITIONS       = 10

def parse_args():
    parser = argparse.ArgumentParser(description='generte predictions on privillaged information')

    parser.add_argument('--partition', type=int,
                                choices=[i for i in range(PARTITIONS)],
                                default = 0,
                                help='')
    parser.add_argument('--mode', default='create', choices = ['create', 'create_movement_reps', 'create_tracks', 'unite', 'check_for_missing', 'shorten', 'add_BB'])
    parser.add_argument('--type', default='uniform_blur', choices = ['uniform_blur', 'uniform_blur_light'])
    parser.add_argument('--orig-dir',type=str)
    parser.add_argument('--teacher-dir',type=str)
    parser.add_argument('--op',type=str)
    parser.add_argument('--skip-teacher-data', action='store_true',
                        default=False,
                        help='')


    parser.add_argument('--random', action='store_true', help='use small model')
    
    parser.add_argument('--cuts', action='store_true', help='use small model')


    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    args = parser.parse_args()
    return args
    



def generate_teacher_data(model, processor,teacher_sample_path):

    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": f"{teacher_sample_path}/video_original.mp4"},
                    {"type": "text", "text": "Describe this video in detail"}
                ]
            },
        ]
    inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

  
    explaination = generated_texts[0].split("Assistant: ")[-1]
    data = [
    {"origin_video_path":      f"{teacher_sample_path}/video_original.mp4", 
     "blur_object_video_path": f"{teacher_sample_path}/video_blur_object.mp4",
     "blur_full_video_path":   f"{teacher_sample_path}/video_blur_full.mp4",
     "generated_prompt":       f"{explaination}"
    },
    ]
    df = pd.DataFrame(data)
    df.to_csv(f"{teacher_sample_path}/labels.csv", index=False)



def check_for_missing(args):
    missing_dirs = []

    #count_valid_dirs = 0
    #files_to_check = ["video_blur_full.mp4","video_blur_object.mp4","video_original.mp4","labels.csv"]
    #dir_to_check   =  args.output_dir
    #len_dirs_to_check =  len(os.listdir(dir_to_check))
    #for i, dir in enumerate(os.listdir(dir_to_check)):
    #    print(f"{i} / {len_dirs_to_check}")
    #    current_dir = os.path.join(dir_to_check, dir)
    #    valid = True
    #    if os.path.isdir(current_dir):
    #        files_curr_dir = os.listdir(current_dir)
    #        for file in files_to_check:
    #            if file not in files_curr_dir:
    #                missing_dirs.append(dir)
    #                valid = False
    #                break
    #        if valid:
    #            count_valid_dirs+=1
    #print("number of valid dirs:")
    #print(count_valid_dirs)
   
    with open(FILTERED_DATASET, 'r') as file:
        subdirs = file.readlines()
    subdirs_needed = [row.strip().split("/")[-1] for row in subdirs]
    dirs_ready     = set(os.listdir(args.output_dir))
    for dir_needed in subdirs_needed:
        if (dir_needed not in missing_dirs) and (dir_needed not in dirs_ready):
            missing_dirs.append(dir_needed)


    
    
    print(len(subdirs_needed))
    print(len(missing_dirs))

    print("missing dirs:")
    print(missing_dirs)

    #main(args,missing_subdirs=missing_dirs)

        

def get_normalized_BB(groundTruthPath, metaInfoPath):
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

                            coord = f"<loc{y_min:04d}><loc{x_min:04d}><loc{y_max:04d}><loc{x_max:04d}>"

                            normalized_boxes.append(coord)
                     
                        return normalized_boxes

            
                    



def add_BB(orig_dir, teacher_dir):

    pattern = re.compile(r'.*_(\d+)$')
    subdirs = [name for name in os.listdir(teacher_dir) if (os.path.isdir(os.path.join(teacher_dir, name)) and pattern.match(name))]
    sorted_teacher_subdirs = sorted(subdirs, key=lambda x: int(pattern.search(x).group(1)))

    #orig_subdirs = set([name for name in os.listdir(orig_dir) if (os.path.isdir(os.path.join(orig_dir, name)) and pattern.match(name))])

    dataframes = []
    len_subdirs = len(sorted_teacher_subdirs)
    for i, teacher_subdir in enumerate(sorted_teacher_subdirs):
        print(f"{i} / {len_subdirs}")
        curr_orig_dir = os.path.join(orig_dir, teacher_subdir)
        if not (os.path.isdir(curr_orig_dir)):
            print("PROBLEM WITH DIRS")
            exit(1)
        groundTruthPath =  os.path.join(orig_dir, teacher_subdir,"groundtruth.txt")
        metaInfoPath =  os.path.join(orig_dir, teacher_subdir,"meta_info.ini")

        if (not os.path.isfile(groundTruthPath)) or (not os.path.isfile(metaInfoPath)):
            print(f"Error: groundtruth.txt not found at {groundTruthPath}")
            exit(1)
        
        
  
        bboxes = get_normalized_BB(groundTruthPath, metaInfoPath)
        
        
        labelsPath        =  os.path.join(teacher_dir, teacher_subdir,"labels.csv")
        if (not os.path.isfile(labelsPath)):
            continue
        output_labelsPath =  os.path.join(teacher_dir, teacher_subdir,"labels_w_BB.csv")

        df = pd.read_csv(labelsPath)
        df['bbox'] = str(bboxes)

        df.to_csv(output_labelsPath, index=False)
        


    
        

    # Concatenate all dataframes
    #combined_df = pd.concat(dataframes, ignore_index=True)
    #output_path = os.path.join(args.output_dir, 'combined.csv')
    #combined_df.to_csv(output_path, index=False)
    #print(f"Combined CSV saved to: {output_path}")




def main(args,missing_subdirs = None):
    pattern = re.compile(r'.*_(\d+)$')

    dataset_dir = args.paths['dataset_dir'] #    "dataset/GOT10KVAL"                        
    output_dir  = args.output_dir    #"dataset/GOT10KVAL_teacher"  #

    #subdirs = [name for name in os.listdir(args.paths['dataset_dir']) if (os.path.isdir(os.path.join(args.paths['dataset_dir'], name)) and pattern.match(name))]
    if missing_subdirs == None:
        with open(FILTERED_DATASET, 'r') as file:
            subdirs = file.readlines()
        
        #DELETE THIS ROW IF YOU WANT FILTERED
        #subdirs = [name for name in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, name)) and pattern.match(name))]
        
        
        subdirs = [row.strip().split("/")[-1] for row in subdirs]
        sorted_subdirs = sorted(subdirs, key=lambda x: int(pattern.search(x).group(1)))
        num_subdirs_per_partition = int(len(sorted_subdirs) // PARTITIONS)
        paritioned_subdirs = sorted_subdirs[args.partition * num_subdirs_per_partition : (args.partition+1)*num_subdirs_per_partition]

    else:
        paritioned_subdirs = sorted(missing_subdirs, key=lambda x: int(pattern.search(x).group(1)))
    
   
    
       

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        #_attn_implementation="flash_attention_2"
    ).to("cuda")
    count = 0
    
  
    for subdir in paritioned_subdirs:
     
        full_path = os.path.join(dataset_dir, subdir)
        
        images2pixelatedVid(full_path, output_dir, args.type)
        if args.skip_teacher_data == False:
            generate_teacher_data(model, processor, os.path.join(output_dir, subdir))
        #count+=1
        #if count == 1:
        #    break
        
        
    
def unite(args):
    pattern = re.compile(r'.*_(\d+)$')
    subdirs = [name for name in os.listdir(args.output_dir) if (os.path.isdir(os.path.join(args.output_dir, name)) and pattern.match(name))]
    sorted_subdirs = sorted(subdirs, key=lambda x: int(pattern.search(x).group(1)))

    dataframes = []
    labels_path = 'labels.csv' if args.op == None else 'labels_w_BB.csv' 
    len_subdirs = len(sorted_subdirs)
    for i, subdir in enumerate(sorted_subdirs):
        print(f"{i} / {len_subdirs}")

        subdir_path = os.path.join(args.output_dir, subdir)
        for file in os.listdir(subdir_path):
            
            if file == labels_path:
                
            
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                dataframes.append(df)
       
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    ext = 'combined.csv' if args.op == None else 'combined_w_BB.csv'
    output_path = os.path.join(args.output_dir, ext)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")







def create_movement_reps(args):
    #iterate whatever folders you want, send the correct sample

    pattern = re.compile(r'.*_(\d+)$')

    dataset_dir =       args.paths['dataset_dir']     #      args.paths['dataset_dir']               
    output_dir  = args.output_dir    #  # args.output_dir

    with open(FILTERED_DATASET, 'r') as file:
            subdirs = file.readlines()


    subdirs = [name for name in os.listdir(output_dir) if (os.path.isdir(os.path.join(dataset_dir, name)) and pattern.match(name))] 
    
    subdirs = [row.strip().split("/")[-1] for row in subdirs]
    sorted_subdirs = sorted(subdirs, key=lambda x: int(pattern.search(x).group(1)))
    num_subdirs_per_partition = int(len(sorted_subdirs) // PARTITIONS)
    paritioned_subdirs = sorted_subdirs[args.partition * num_subdirs_per_partition : (args.partition+1)*num_subdirs_per_partition]
    
    paritioned_subdirs  = sorted_subdirs
    
    step =1
    count = 0
    for dir in paritioned_subdirs:
        sample_dir     = f"{dataset_dir}/{dir}"
        output_rep_dir = f"{output_dir}/{dir}/movement_flow_{step}"

        


        images_sample_dir = glob.glob(os.path.join(sample_dir, '*.png')) + \
                 glob.glob(os.path.join(sample_dir, '*.jpg'))
        
        images_rep_dir = glob.glob(os.path.join(output_rep_dir, '*.flo')) 
        
       
        if len(images_sample_dir) == (len(images_rep_dir) +1):
            continue
    
       
        print(output_rep_dir)

        os.makedirs(output_rep_dir, exist_ok=True)
        generate_movement_reps(args, sample_dir, output_rep_dir,step)

        








def create_tracks(args):
    #iterate whatever folders you want, send the correct sample

    pattern = re.compile(r'.*_(\d+)$')

    dataset_dir = args.paths['dataset_dir']     #      args.paths['dataset_dir']               
    output_dir  =  "dataset/GOT10KVAL_teacher"              #  # args.output_dir "dataset/GOT10KVAL_teacher" 

    with open(FILTERED_DATASET, 'r') as file:
        subdirs = file.readlines()


    subdirs = [name for name in os.listdir(output_dir) if (pattern.match(name))] 
    
    subdirs = [row.strip().split("/")[-1] for row in subdirs]
    sorted_subdirs = sorted(subdirs, key=lambda x: int(pattern.search(x).group(1)))
    num_subdirs_per_partition = int(len(sorted_subdirs) // PARTITIONS)
    paritioned_subdirs = sorted_subdirs[args.partition * num_subdirs_per_partition : (args.partition+1)*num_subdirs_per_partition]
    
    paritioned_subdirs  = sorted_subdirs
    

    pref_dir = "tracks_cuts" if args.cuts  else "tracks"
    
    
    output_dir_ext = "random" if args.random else "grid"
    for dir in paritioned_subdirs:

        if os.path.exists(f"{output_dir}/{dir}/video_original.mp4") == False:
            continue

        output_rep_dir = f"{output_dir}/{dir}/{pref_dir}_{output_dir_ext}"

        if args.cuts:
             if os.path.exists(f"{output_rep_dir}/pred_tracks.pt") and os.path.exists(f"{output_rep_dir}/pred_visibility.pt"):
                continue


        

        video_path = f"{output_dir}/{dir}/video_original.mp4"

        if os.path.exists(f"{output_rep_dir}/pred_tracks.pt") and os.path.exists(f"{output_rep_dir}/pred_visibility.pt"):
            continue
        if os.path.exists(f"{output_rep_dir}/pred_tracks_online.pt") and os.path.exists(f"{output_rep_dir}/pred_visibility_online.pt"):
            continue
        print(output_rep_dir,flush=True)
       

        os.makedirs(output_rep_dir, exist_ok=True)
       
        generate_track_reps(video_path, output_rep_dir,cuts = args.cuts, is_random=False)
        
       
        
        #generate_movement_reps(args, sample_dir, output_rep_dir,step)







        

#'dataset/GOT10KVAL_teacher/combined.csv'
def shorten(args,path = 'dataset/got10k/teacher/train/uniform_blur/combined.csv'):
    folder, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(folder, f"{name}_short_full{ext}")
    df = pd.read_csv(path)
    df['generated_prompt'] = df['generated_prompt'].apply(lambda x: str(x).split('.')[0]+ '.' if '.' in str(x) else str(x))
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args          = parse_args()
    config.get_config(args)
    args.output_dir = f"{args.paths['dataset_dir_teacher']}/{args.type}"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "create":
        main(args)
    elif args.mode == "unite":
        unite(args)    
    elif args.mode == "create_movement_reps":
        create_movement_reps(args)   
    elif args.mode == "create_tracks":
        print("MADE IT")
        create_tracks(args)   

    elif args.mode == "check_for_missing":
        check_for_missing(args)
    elif args.mode == "add_BB":
        if args.orig_dir == None or args.teacher_dir == None:
            print("spcify both orig_dir and teacher_dir")
            exit(1)
        add_BB(args.orig_dir, args.teacher_dir)
    elif args.mode == "shorten":
        shorten(args)