# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ['MPLCONFIGDIR'] = '/home/ai_center/ai_users/yardenbakish/'

import torch
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

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

    print(points)

    return points.float()

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/movie5_blur.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=30, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    B, T, C, H, W = video.shape
    video_reshaped = video.view(B * T, C, H, W)

    # Resize
    video_resized = F.interpolate(video_reshaped, size=(512, 512), mode='bilinear', align_corners=False)

    # Reshape back to original [B, T, C, H, W] format
    video = video_resized.view(B, T, C, 512, 512)

    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    points = generate_points(T, 512, 512,  n_points=576).cuda()
    #points = torch.tensor([
    #    [0., 400., 350.],  # point tracked from the first frame
    #    [10., 100., 200.], # frame number 10
    #    [20., 50., 90.], # ...
    #    [30., 450., 450.]
    #])

    if torch.cuda.is_available():
        queries = points.cuda()

    print("REACHED")
    pred_tracks, pred_visibility = model(
        video,
        
        #queries = queries[None],
        #backward_tracking=True,

        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        #backward_tracking=args.backward_tracking,
     
    )
    #print(pred_tracks)
    #print(pred_visibility)

    print(pred_tracks.max())
    print(pred_tracks.min())

    print(pred_tracks.shape)
    print(pred_visibility.shape)

    torch.save(pred_tracks, "pred_tracks.pt")
    torch.save(pred_visibility, "pred_visibility.pt")


    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=50, linewidth=1)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
