
import argparse
from pathlib import Path
import re
import cv2
import os
from moviepy.editor import *
import ast
import pandas as pd
from conditioned_models.FlowFormerPlusPlus.visualize_flow import extract_images,visualize_flow,build_model
from conditioned_models.cotracker.generate_track import generate_track
from conditioned_models.cotracker.generate_track_online import generate_track_online


from modules.FlowFormer.cfg import get_cfg
import torch
from modules.FlowFormer.FlowFormer import build_flowformer
from modules.gmflow.gmflow.gmflow import GMFlow
import numpy as np


import subprocess
import matplotlib.pyplot as plt

'''

SAMPLES_TO_TEST = [
"dataset/GOT10KVAL/GOT-10k_Val_000003",
"dataset/GOT10KVAL/GOT-10k_Val_000006",
"dataset/GOT10KVAL/GOT-10k_Val_000007",
"dataset/GOT10KVAL/GOT-10k_Val_000015",
"dataset/GOT10KVAL/GOT-10k_Val_000018",
"dataset/GOT10KVAL/GOT-10k_Val_000030",
]


'''



def natural_sort_key(s):
    """Sort strings with numbers in a natural way."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def parse_groundtruth(file_path):
    """Parse groundtruth.txt file containing bounding box coordinates."""
    bbox_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse x1,y1,x2,y2 format
                coords = list(map(float, line.split(',')))
                if len(coords) >= 4:
                    # Convert to int for pixel coordinates
                    bbox_list.append([int(c) for c in coords[:4]])
    return bbox_list

def apply_heavy_blur(image, bbox, kernel_size, blur_intensity):
    """Apply a very heavy blur to the region specified by the bounding box."""
    x1, y1, dx, dy = bbox
    x2 = x1+dx
    y2 = y1+dy
    
    # Ensure coordinates are within image boundaries
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Extract the region to blur
    region = image[y1:y2, x1:x2].copy()
    
    # Apply heavy blur (use large kernel size)
    blurred_region =   cv2.GaussianBlur(region, kernel_size, blur_intensity) #(99, 99) 30
    
    # Replace the region in the original image
    image[y1:y2, x1:x2] = blurred_region
    
    return image

def process_frame(image_path, bbox, output_path, kernel_size, blur_intensity, full=False):
    """Process a single frame with the provided bounding box."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image {image_path}")
        return False
    
    if full:
        bbox = (0,0,img.shape[1],img.shape[0])
    img_blurred = apply_heavy_blur(img, bbox, kernel_size, blur_intensity)
    
    # Save the processed frame
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.imwrite(output_path, img_blurred)

def create_video(frame_dir, output_video_path, fps=30):
    """Create an MP4 video from frames using MoviePy."""
    import os
    
    # Get all jpg files in the output directory
    frames = [f for f in os.listdir(frame_dir) if f.lower().endswith('.jpg')]
    
    # Sort frames naturally
    frames.sort(key=natural_sort_key)
    
    if not frames:
        print("No frames found to create video")
        return False
    
    # Get full paths to the frames
    frame_paths = [os.path.join(frame_dir, frame) for frame in frames]
    
    # Create a clip from the image sequence
    clip = ImageSequenceClip(frame_paths, fps=fps)
    
    # Write the result to an MP4 file
    clip.write_videofile(
        output_video_path,
        codec='libx264',
        audio=False,
        verbose=False,
        logger=None
    )
    
    print(f"Video created at: {output_video_path}")
    return True


def extract_bbox(path):
    df = pd.read_csv(path)

    df['bbox'] = df['bbox'].apply(ast.literal_eval)

    return df['bbox'].iloc[0]
   


def is_valid_loc(loc_str):
    # Check if the <loc[number]> format is valid and the number is within range
    pattern = r"<loc(\d{4})>"
    match = re.match(pattern, loc_str)
    if match:
        number = int(match.group(1))  # Extract the number part
        return 0 <= number <= 1024  # Check if the number is in the valid range
    return False


def parse_loc_string(loc_string):
    elements = loc_string.split(";")
    valid_elements = []
    for element in elements:
        # Split the element into parts based on <loc[number]> pattern
        locs = re.findall(r"<loc\d{4}>", element)
        
        # Ensure that there are exactly 4 valid <loc[number]> in each element
        if len(locs) == 4 and all(is_valid_loc(loc) for loc in locs):
            valid_elements.append(element)  # Add to the list if valid
    
    return valid_elements






def save_image_grid(images_row1, images_row2, filename, 
                   text_row1="Row 1", text_row2="Row 2", 
                   figsize=None, dpi=100):
    """
    Save a grid of images with two rows and text labels below each row.
    
    Args:
        images_row1: List/array of images for the first row
        images_row2: List/array of images for the second row
        filename: Output filename (e.g., "combined_image.png")
        text_row1: Text label for first row
        text_row2: Text label for second row
        figsize: Tuple for figure size (width, height), auto-calculated if None
        dpi: Resolution for saved image
    """
    n_images = len(images_row1)
    assert len(images_row2) == n_images, "Both arrays must have same number of images"
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_images * 3, 8)  # 3 inches per image width, 8 inches height
    
    fig, axes = plt.subplots(2, n_images, figsize=figsize)
    
    # Handle case where there's only one image
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    # Display images in first row
    for i in range(n_images):
        axes[0, i].imshow(images_row1[i])
        axes[0, i].axis('off')
    
    # Display images in second row
    for i in range(n_images):
        axes[1, i].imshow(images_row2[i])
        axes[1, i].axis('off')

    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0.4)
    
    # Add text labels below each row
    fig.text(0.5, 0.5, text_row1, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.08, text_row2, ha='center', va='center', fontsize=14, fontweight='bold')
    
   
    
    # Save the combined image
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()  # Close to free



'''

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
    parser.add_argument('--mode', required=True, choices = ['imgseq', 'video'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--sample', action='store_true')


    args = parser.parse_args()
    return args


'''


def vid2pixaletdvid(path):
    pass






def parse_meta_info(file_path):
    """Parse groundtruth.txt file containing bounding box coordinates."""
    with open(file_path, 'r') as f:
        for line in f:
            if 'anno_fps' in line:     
                _, fps = line.strip().split(':')
                fps = fps.strip()
                fps = fps.split("Hz")[0]
                return int(fps)
    print(f"NO FPS at {file_path}")
    exit(1)
   




def images2pixelatedVid(imageSequencePath, output_dir_root, blur_type):

    if blur_type == 'uniform_blur':
        kernel_size, blur_intensity = (99, 99), 30
    else:
        kernel_size, blur_intensity = (45, 45), 15


    sample_name = imageSequencePath.split("/")[-1]
    output_dir = Path(f'{output_dir_root}/{sample_name}')

    output_partial_blur = Path(f"{output_dir}/blur_object")
    output_full_blur = Path(f"{output_dir}/blur_full")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_partial_blur, exist_ok=True)
    os.makedirs(output_full_blur, exist_ok=True)

    input_dir = Path(imageSequencePath)



    groundtruth_path = input_dir / "groundtruth.txt"
    if not groundtruth_path.exists():
        print(f"Error: groundtruth.txt not found at {groundtruth_path}")
        return 1
    bboxes = parse_groundtruth(groundtruth_path)

    meta_info_path = input_dir / "meta_info.ini"
    if not meta_info_path.exists():
        print(f"Error: groundtruth.txt not found at {meta_info_path}")
        return 1
    
    fps =  30   # parse_meta_info(meta_info_path)
    

    print(f"Loaded {len(bboxes)} bounding boxes from groundtruth.txt")
    frames = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    frames.sort(key=natural_sort_key)
    print(f"Found {len(frames)} frames to process")
    if len(frames) == 0:
        print("No frames found in input directory")
        return 1
    
    for i, frame in enumerate(frames):
        if i >= len(bboxes):
            print(f"Skipping frame {frame}: No corresponding bounding box")
            continue
        
        input_path = input_dir / frame
        output_path_partial = output_partial_blur / frame
        output_path_full = output_full_blur / frame

        
        print(f"Processing frame {i+1}/{len(frames)}: {frame}")
        success1 = process_frame(str(input_path), bboxes[i], str(output_path_partial), kernel_size, blur_intensity)
        success2 = process_frame(str(input_path), bboxes[i], str(output_path_full), kernel_size, blur_intensity, full=True)

        
        if not success1 or not success2:
            print(f"Failed to process frame: {frame}")
       
    
    # Check if we have enough bounding boxes for all frames
    if len(bboxes) < len(frames):
        print(f"Warning: Number of bounding boxes ({len(bboxes)}) is less than number of frames ({len(frames)})")


    partial_blurred_video_output_path = output_dir / "video_blur_object.mp4"
    full_blurred_video_output_path = output_dir / "video_blur_full.mp4"
    original_video_output_path = output_dir / "video_original.mp4"
    


    create_video(str(output_partial_blur), str(partial_blurred_video_output_path), fps=fps) #fps=30
    create_video(str(output_full_blur), str(full_blurred_video_output_path), fps=fps)     #fps=30
    create_video(str(input_dir), str(original_video_output_path), fps=fps)          #fps=30
 

   
   
    try:
        subprocess.run(f"rm -rf {output_partial_blur}", check=True, shell=True)
        subprocess.run(f"rm -rf {output_full_blur}", check=True, shell=True)

        print(f"generated visualizations")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)






def generate_movement_reps(args, input_dir,output_dir,step):
   
    img_pairs = extract_images(input_dir)
    model     = build_model()

    with torch.no_grad():
        visualize_flow('.', output_dir, model, img_pairs, False)



    #FOR RAFT
    '''
    motion_reps = compute_optical_flow(args, input_dir,step)
    print("reached")
    for i in range(len(motion_reps)):
        plt.imsave( f'{output_dir}/image_{i}.jpg'  ,motion_reps[i])
    '''


def generate_movement_reps(args, input_dir,output_dir,step):
   
    img_pairs = extract_images(input_dir)
    model     = build_model()

    with torch.no_grad():
        visualize_flow('.', output_dir, model, img_pairs, False)



    #FOR RAFT
    '''
    motion_reps = compute_optical_flow(args, input_dir,step)
    print("reached")
    for i in range(len(motion_reps)):
        plt.imsave( f'{output_dir}/image_{i}.jpg'  ,motion_reps[i])
    '''
def generate_track_reps(video_path, output_dir, is_random=False):

    duration = VideoFileClip(video_path).duration
    if duration > 20:
       
        generate_track_online(video_path, output_dir, is_random=False)
        

    else:
        generate_track(video_path, output_dir, is_random=False)








def create_optical_flow_model():
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    #model.eval()

    return model




def create_gmflow_model(load_weights = False):
    model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                ).cuda()
    
    if load_weights:
        checkpoint_path = "modules/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth"
        checkpoint = torch.load(checkpoint_path)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(weights,strict=False)
    return model




def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel




def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image



def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


'''



if __name__ == "__main__":
    args                   = parse_args()
    if args.sample:
        for path in SAMPLES_TO_TEST:
            print(path)
            images2pixelatedVid(path=path)
        exit(1)

    if args.mode == "video":
        vid2pixaletdvid(path=args.path)
        exit(1)

    if args.mode == "imgseq":
        images2pixelatedVid(path=args.path)
        exit(1)

'''


def collect_optical_flow(examples,example, videoPath, indices, target_W, target_H):
    example_flow_maps = []

    flo_path = example[videoPath].split("/")[:-1]
    flo_path = "/".join(flo_path)
    flo_path = f"{flo_path}/movement_flow_1"
    flo_paths = np.array(sorted(glob(f'{flo_path}/*.flo'), key=lambda p: int(p.split('/')[-1].split('_')[1].split('.')[0])))
    
    if indices[-1] >= len(flo_paths):
        indices[-1] =len(flo_paths) -1
    flo_paths = flo_paths[indices]
    flo_paths = flo_paths[:-1]
    
    co = 0
    for i, flo_file_path in enumerate(flo_paths):
        with open(flo_file_path, 'rb') as f:
            # Read magic number
            magic = np.frombuffer(f.read(4), dtype=np.float32)[0]
            if magic != 202021.25:
                raise ValueError("Not a valid .flo file")
            # Read width, height
            w, h = np.frombuffer(f.read(8), dtype=np.int32)
            # Read flow data
            flow = np.frombuffer(f.read(), dtype=np.float32)
            flow = flow.reshape(h, w, 2)
            flow = cv2.resize(flow, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
            '''
            if "001148" in example[videoPath]:
               req_flow = torch.from_numpy(flow).permute(2, 0, 1).float().permute(1, 2, 0)
               req_flow = req_flow.cpu().numpy()
       
               flow_img = flow_to_image(req_flow)
               cv2.imwrite(f"to_del/img{i}.png", flow_img[:, :, [2,1,0]])
              '''
           
    
            example_flow_maps.append(torch.from_numpy(flow).permute(2, 0, 1).float().to("cuda"))  
    return example_flow_maps
    


def pad_optical_flow(all_flow_maps):
    max_flow_frames = max(len(flow_maps) for flow_maps in all_flow_maps) if all_flow_maps else 0
    
    if max_flow_frames > 0:
        # Get dimensions from first flow map
        sample_flow = all_flow_maps[0][0]
        flow_channels, flow_h, flow_w = sample_flow.shape
        
        # Create padded tensor for all flow maps
        padded_flow_maps = torch.zeros(
            (len(examples), max_flow_frames, flow_channels, flow_h, flow_w),
            dtype=sample_flow.dtype,
            device=sample_flow.device
        )
        
        # Fill in the actual flow data
        for batch_idx, example_flows in enumerate(all_flow_maps):
            for frame_idx, flow_map in enumerate(example_flows):
                padded_flow_maps[batch_idx, frame_idx] = flow_map
        
        optical_flow_maps = padded_flow_maps
    else:
        optical_flow_maps = torch.empty(0)
    return optical_flow_maps


def collect_tracks(example, videoPath, indices, target_W, target_H):
    

    tracks_path = example[videoPath].split("/")[:-1]
    tracks_path = "/".join(tracks_path)
    tracks_path = f"{tracks_path}/tracks_grid"
    pred_tracks     = torch.load(f"{tracks_path}/pred_tracks.pt")
    pred_visibility = torch.load(f"{tracks_path}/pred_visibility.pt")


    

    original_h, original_w = (512,512)


    scale_x = target_W / original_w
    scale_y = target_H / original_h

    scale_factors = torch.tensor([scale_x, scale_y], 
                                device=pred_tracks.device, 
                                dtype=pred_tracks.dtype)

    pred_tracks = pred_tracks * scale_factors


    pred_tracks = pred_tracks[:,indices,:,:].to(dtype=torch.bfloat16)
    pred_visibility = pred_visibility[:,indices,:]

    return pred_tracks, pred_visibility

    
    