"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
import argparse
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, rendering


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # dnerf
            "bouncingballs",
            "hellwarrior",
            "hook",
            "jumpingjacks",
            "lego",
            "mutant",
            "standup",
            "trex",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--auto_aabb",
        action="store_true",
        help="whether to automatically compute the aabb",
    )
    parser.add_argument(
        '-f',
        "--use_feat_predict",
        action="store_true",
        help="use a mlp to predict the hash feature",
    )
    parser.add_argument(
        '-w',
        "--use_weight_predict",
        action="store_true",
        help="use a mlp to predict the weight feature",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        '-d',
        "--distortion_loss",
        action="store_true",
        help="use a distortion loss",
    )
    parser.add_argument(
        "--rec_loss",
        type=str,
        default="huber",
        choices=[
            "huber",
            "mse",
            "smooth_l1",
        ],
    )
    parser.add_argument(
        '-o',
        "--use_opacity_loss",
        action="store_true",
        help="use a opacity loss",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="",
    )

    parser.add_argument(
        "--test_print",
        action="store_true",
        help="run evaluation during training",
    )
    
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
