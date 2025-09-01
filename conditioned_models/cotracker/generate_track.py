# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
#os.environ['MPLCONFIGDIR'] = '/home/ai_center/ai_users/yardenbakish/'

import torch
import argparse
import numpy as np
import torch.nn.functional as F
import sys
import cv2
from PIL import Image
from conditioned_models.cotracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
#from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)





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













def generate_points(T: int, H: int, W: int, n_points: int = 900):
    """
    Generate n_points evenly spread in HxW grid (padding = grid spacing),
    then assign each point a random time index from 0..T-1.

    Returns:
        points: Tensor of shape [n_points, 3] with [t, x, y]
    """
    assert H == W, "Frame must be square (H == W) to evenly place points."

    # Number of grid steps (square root of n_points)
    side = int(n_points ** 0.5)
    assert side * side == n_points, "n_points must be a perfect square (e.g., 900 = 30x30)."

    # Step size so padding equals step
    step = (H - 1) / (side + 1)

    # Generate grid coordinates
    xs = torch.linspace(step, W - 1 - step, side)
    ys = torch.linspace(step, H - 1 - step, side)

    # Create meshgrid
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1).float(), grid_y.reshape(-1).float()], dim=-1).round().long()  # [900, 2]

    # Sample random times
    times = torch.randint(0, T, (n_points, 1))

    # Combine: [t, x, y]
    points = torch.cat([times, coords], dim=-1)


    return points.float()

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def generate_track(video_path, output_dir, is_random=False):
    grid_size = 30
    grid_query_frame = 0
    #video_path = "conditioned_models/cotracker/assets/movie1.mp4"
    print(is_random)
    print(video_path)

    # load the input video frame by frame
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)

    # Resize
    video_resized = F.interpolate(video_reshaped, size=(512, 512), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B, T, C, 512, 512)


  
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    

    if is_random:
        points = generate_points(T, 512, 512,  n_points=576).cuda()
        if torch.cuda.is_available():
            queries = points.cuda()
        pred_tracks, pred_visibility = model(
        video,
        queries = queries[None],
        backward_tracking=True,
    )
    else:
        pred_tracks, pred_visibility = model(
            video,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
   
    print(output_dir)
    print("-----------------------")
    torch.save(pred_tracks,     f"{output_dir}/pred_tracks.pt")
    torch.save(pred_visibility, f"{output_dir}/pred_visibility.pt")

    '''
    # save a video with predicted tracks
    seq_name = video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=50, linewidth=1)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if is_random else grid_query_frame,
    )
    '''





def generate_mov_from_track(video_path, tracks_path, BB_path, output_dir):

    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)

    # Resize
    video_resized = F.interpolate(video_reshaped, size=(512, 512), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B,T, C, 512, 512)


    

    try:
        pred_tracks     = torch.load(f"{tracks_path}/pred_tracks.pt")
        pred_visibility = torch.load(f"{tracks_path}/pred_visibility.pt")
    except:
        pred_tracks     = torch.load(f"{tracks_path}/pred_tracks_online.pt")
        pred_visibility = torch.load(f"{tracks_path}/pred_visibility_online.pt")
    


    original_h, original_w = (512,512)
    with open(BB_path, "r") as f:
        bboxes = []
        for line in f:
            vals = line.strip().split(",")
            if len(vals) >= 4:
                x, y, w, h = map(float, vals[:4])

                x1, y1 = x, y 
                x2, y2 = x + w, y + h
                bboxes.append([x1, y1, x2, y2])


    bboxes = np.array(bboxes)

    sx = original_w / W
    sy = original_h / H

    bboxes[:, [0, 2]] *= sx
    bboxes[:, [1, 3]] *= sy

    bboxes = bboxes.astype(np.int32)


    num_samples = 30
    frame_indices = np.linspace(0, T-1, num_samples, dtype=int)
    if frame_indices[-1] + 1 >= video.shape[1]:
        frame_indices[-1] -=1
    print(frame_indices)
    print(video.shape)
    video = video[:, frame_indices]
    pred_tracks = pred_tracks[:, frame_indices]         # [1, 30, N, 2]
    pred_visibility = pred_visibility[:, frame_indices] # [1, 30, N]
    bboxes = bboxes[frame_indices] 

    diffs = pred_tracks[:, 1:] - pred_tracks[:, :-1]  # [1, 29, N, 2]
    inner_means = []  # per step t
    outer_means = []

    for t in range(diffs.shape[1]):
        vb = (pred_visibility[0, t] & pred_visibility[0, t + 1])
        x1_t, y1_t, x2_t, y2_t = bboxes[t]
        x1_n, y1_n, x2_n, y2_n = bboxes[t + 1]
    
    # coords at t and t+1 (float)
        xy_t   = pred_tracks[0, t]     # [N,2]
        xy_n   = pred_tracks[0, t + 1] # [N,2]
        diff_t = diffs[0, t]        # [N,2]

        # in-bbox tests (inclusive)
        in_t = (xy_t[:, 0] >= x1_t) & (xy_t[:, 0] <= x2_t) & \
               (xy_t[:, 1] >= y1_t) & (xy_t[:, 1] <= y2_t)
        in_n = (xy_n[:, 0] >= x1_n) & (xy_n[:, 0] <= x2_n) & \
               (xy_n[:, 1] >= y1_n) & (xy_n[:, 1] <= y2_n)

        inner_mask = (vb & in_t & in_n)         # [N] bool
        outer_mask = ~inner_mask                # rest

        # means (fallback to zero vector if empty)
        if inner_mask.any():
            inner_mean = diff_t[inner_mask].mean(dim=0)
        else:
            inner_mean = torch.zeros(2, dtype=torch.float32)
        if outer_mask.any():
            outer_mean = diff_t[outer_mask].mean(dim=0)
        else:
            outer_mean = torch.zeros(2, dtype=torch.float32)

        inner_means.append(inner_mean)
        outer_means.append(outer_mean)

    for t in range(diffs.shape[1]):
        x1_t, y1_t, x2_t, y2_t = bboxes[t]
        im_inner = inner_means[t].cpu().numpy().astype(np.float32)  # (2,)
        im_outer = outer_means[t].cpu().numpy().astype(np.float32)  # (2,)

        flow = np.empty((original_h, original_w, 2), dtype=np.float32)
        flow[:, :, 0] = im_outer[0]
        flow[:, :, 1] = im_outer[1]
        # overwrite bbox region with inner mean
        if (x2_t > x1_t) and (y2_t > y1_t):
            flow[y1_t:y2_t, x1_t:x2_t, 0] = im_inner[0]
            flow[y1_t:y2_t, x1_t:x2_t, 1] = im_inner[1]

        flow_img = flow_to_image(flow)
        frame = video[0, t].permute(1, 2, 0).cpu().numpy()  # [H,W,C]
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Ensure both are same color ordering
        if flow_img.shape[2] == 3 and frame.shape[2] == 3:
            # Convert frame to BGR for OpenCV consistency
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Flow is usually in RGB — convert to BGR if needed
            flow_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

            # Blend
            blended = cv2.addWeighted(frame_bgr, 1 - 0.6, flow_bgr, 0.6, 0)

            cv2.imwrite(f"{output_dir}/img{t}.png", blended)
    

    print(pred_tracks.shape)
    print(flow.shape)

    exit(1)

    out_path = os.path.join(f"{output_dir}/flow_averages.txt")
    with open(out_path, "w") as out_f:
            out_f.write("frame,inside_u,inside_v,outside_u,outside_v\n")

'''
num = "001148"
video_path = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}/video_original.mp4"
tracks_path = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}/tracks_grid"
BB_path     = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}/groundtruth.txt"
output_dir  = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}/tracks_mov_grid"
output_dir  = "to_del2"
generate_mov_from_track(video_path, tracks_path, BB_path, output_dir)

'''

#video_path = "./assets/movie1.mp4"
#output_dir = "saved_videos"
#generate_track(video_path, output_dir, is_random=True)