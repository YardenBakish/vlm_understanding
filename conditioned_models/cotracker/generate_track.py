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

from PIL import Image
from conditioned_models.cotracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
#from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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


#video_path = "./assets/movie1.mp4"
#output_dir = "saved_videos"
#generate_track(video_path, output_dir, is_random=True)