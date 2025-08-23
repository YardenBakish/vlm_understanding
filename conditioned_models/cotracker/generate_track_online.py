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
import imageio.v3 as iio
from PIL import Image
from conditioned_models.cotracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
#from cotracker.predictor import CoTrackerPredictor
#from conditioned_models.cotracker.cotracker.predictor import CoTrackerOnlinePredictor
import cv2


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

def generate_track_online(video_path, output_dir, is_random=False):
    grid_size = 30
    grid_query_frame = 0
    #video_path = "conditioned_models/cotracker/assets/movie1.mp4"


    # load the input video frame by frame
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)

    # Resize
    video_resized = F.interpolate(video_reshaped, size=(512, 512), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B, T, C, 512, 512)


  
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    #model.step = 20
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    window_frames = []
    
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            video_path,
            plugin="FFMPEG",
        )
    ):

        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)

        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=30,
                grid_query_frame=0,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=30,
        grid_query_frame=0,
    )

    torch.save(pred_tracks,     f"{output_dir}/pred_tracks_online.pt")
    torch.save(pred_visibility, f"{output_dir}/pred_visibility_online.pt")

    '''
    # save a video with predicted tracks
    seq_name = video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=50, linewidth=1)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=0
    )'''
    


#video_path = "conditioned_models/cotracker/assets/movie1.mp4"
#output_dir = "saved_videos"
#generate_track(video_path, output_dir, is_random=True)