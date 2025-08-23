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


def visualize_track(video_dir, output_dir, is_random=False):

    video_path = f"{video_dir}/video_original.mp4"
    pred_tracks = torch.load(f"{video_dir}/tracks_grid/pred_tracks.pt", map_location=torch.device('cpu'))
    pred_visibility = torch.load(f"{video_dir}/tracks_grid/pred_visibility.pt", map_location=torch.device('cpu'))

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
   


video_dir = "dataset/got10k/teacher/train/uniform_blur/GOT-10k_Train_001985"
output_dir = "saved_videos"
visualize_track(video_dir, output_dir, is_random=True)