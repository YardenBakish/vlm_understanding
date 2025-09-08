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
import matplotlib.pyplot as plt

from PIL import Image
from conditioned_models.cotracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
#from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def visualize_track(video_dir,tracks_path, output_dir, is_random=False):

    video_path = f"{video_dir}/video_original.mp4"
    debug_dir = f'debug_tracks/{video_dir.split("/")[-1]}' 

    pred_tracks = torch.load(f"{tracks_path}/pred_tracks.pt", map_location=torch.device('cpu'))
    pred_visibility = torch.load(f"{tracks_path}/pred_visibility.pt", map_location=torch.device('cpu'))

    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)

    # Resize
    video_resized = F.interpolate(video_reshaped, size=(512, 512), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B, T, C, 512, 512)
 
    seq_name = video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=50, linewidth=1)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if is_random else grid_query_frame,
    )
   


def visualize_track_cuts(video_dir,tracks_path, output_dir, is_random=False):

    video_path = f"{video_dir}/video_original.mp4"

    pred_tracks = torch.load(f"{tracks_path}/pred_tracks.pt", map_location=torch.device('cpu'))
    pred_visibility    = torch.load(f"{tracks_path}/pred_visibility.pt", map_location=torch.device('cpu'))
    
    

    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)
    indices = np.linspace(0, T-1, 30, dtype=int)
    
    # Resize
    video_resized = F.interpolate(video_reshaped, size=(384, 384), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B, T, C, 384, 384)

    grid_size = 27
    patch_h = 384 // grid_size
    patch_w = 384 // grid_size
    y_centers = np.arange(patch_h//2, 384, patch_h)
    x_centers = np.arange(patch_w//2, 384, patch_w)
    grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing="ij")
    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # shape [729, 2]


    for i, frame_idx in enumerate(indices[:-1]):
        curr_frame = video[0, frame_idx].permute(1, 2, 0).byte().numpy()  # [H, W, C]

        # Predicted displacement vectors for this frame [729, 2]
        displacements = pred_tracks[i].numpy()
        curr_vis      = pred_visibility[i].numpy()

        displacements = displacements[curr_vis]

        # Compute magnitudes for coloring
        magnitudes = np.linalg.norm(displacements, axis=1)
        
        curr_center = centers[curr_vis]

        print(magnitudes.shape)
        print(curr_center.shape)



        plt.figure(figsize=(6, 6))
        plt.imshow(curr_frame)
        plt.quiver(
            curr_center[:, 0], curr_center[:, 1],   # X, Y start points
            displacements[:, 0], displacements[:, 1],  # U, V (dx, dy)
            magnitudes,  # color by magnitude
            angles="xy", scale_units="xy", scale=1.5, cmap="turbo", width=0.003
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{i:03d}.png", dpi=150)
        plt.close()
 
    


num = "001148"
#num="000030"
video_dir = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}"
tracks_path = f"dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_{num}/tracks_cuts_grid"

output_dir = "saved_videos"
visualize_track_cuts(video_dir,tracks_path, output_dir, is_random=True)