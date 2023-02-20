"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
import argparse
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import (
    OccupancyGrid, ray_marching, 
    rendering, pack_info, ray_aabb_intersect,
    rendering_test, rendering_test_v2,
    ray_marching_test_v3
)


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
    extra_info = []
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
        rgb, opacity, depth, weight = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
        # packed_info = pack_info(ray_indices, n_rays=len(chunk_rays.origins))
        extra_info.append((weight, t_starts, t_ends, ray_indices))
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        extra_info,
    ) 


@torch.no_grad()
def render_image_test_v3(
    max_samples: int,
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
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
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
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
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

    N_rays = rays.origins.shape[0]
    device = rays.origins.device
    opacities = torch.zeros(N_rays, 1, device=device)
    depths = torch.zeros(N_rays, 1, device=device)
    rgbs = torch.zeros(N_rays, 3, device=device)

    alive_indices = torch.arange(N_rays, device=device)

    min_samples = 1 if cone_angle==0 else 4

    samples = total_samples = 0

    # logic for t_min and t_max:
    # 1. if t_min and t_max are given, use them with highest priority.
    # 2. if t_min and t_max are not given, but scene_aabb is given, use
    # ray_aabb_intersect to compute t_min and t_max.
    # 3. if t_min and t_max are not given, and scene_aabb is not given,
    # set t_min to 0.0, and t_max to 1e10. (the case of unbounded scene)
    # 4. always clip t_min with near_plane and t_max with far_plane if given.
    if scene_aabb is not None:
        # print('rays.origins: ', rays.origins.dtype)
        # print('rays.viewdirs: ', rays.viewdirs.dtype)
        # print('scene_aabb: ', scene_aabb.dtype)
        t_min, t_max = ray_aabb_intersect(rays.origins, rays.viewdirs, scene_aabb)
    else:
        t_min = torch.zeros_like(rays.origins[..., 0])
        t_max = torch.ones_like(rays.viewdirs[..., 0]) * 1e10

    if near_plane is not None:
        t_min = torch.clamp(t_min, min=near_plane)
    if far_plane is not None:
        t_max = torch.clamp(t_max, max=far_plane)

    # print("-"*20, "new_frame", "-"*20)
    # T_thr = 1-1e-3

    while samples < max_samples:
        # mask = total_alive_indices >= 0
        # alive_indices = total_alive_indices[mask]

        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        # debug_str = f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples} | "

        # chunk_rays = namedtuple_map(lambda r: r, rays)
        (
            ray_indices, 
            t_starts, 
            t_ends, 
            t_min_mask, 
            temp_pack_info, 
            # debug_str_1
        ) = ray_marching_test_v3(
            N_alive,
            N_samples,
            rays.origins[alive_indices],
            rays.viewdirs[alive_indices],
            t_min[alive_indices],
            t_max[alive_indices],
            grid=occupancy_grid,
            render_step_size=render_step_size,
            cone_angle=cone_angle,
        )
        # temp_pack_info = pack_info(ray_indices, N_alive)
        # debug_str += f"t_min[0]: {t_min[0]} | " + debug_str_1
        rgbs, depths, opacities, alive_indices = rendering_test_v2(
            rgbs.contiguous(),
            depths.contiguous(), 
            opacities.contiguous(), 
            temp_pack_info.contiguous(), 
            alive_indices.contiguous(), 
            ray_indices.contiguous(),
            t_starts.contiguous(),
            t_ends.contiguous(),
            rgb_sigma_fn=rgb_sigma_fn,
            early_stop_eps=early_stop_eps,
            alpha_threshold=alpha_thre,
        )
        # rgbs[mask] += rgb
        # opacities[mask] += opacity
        # depths[mask] += depth

        # update rays status
        total_samples += ray_indices.shape[0]
        t_min[alive_indices] = t_min_mask
        alive_indices = alive_indices[alive_indices>=0]
        # alive_indices[opacities[mask, 0]>(1-early_stop_eps)] = -1
        # alive_indices[N_sample_per_rays < N_samples] = -1
        # total_alive_indices[mask] = alive_indices
        # print(debug_str)

    
    rgbs = rgbs + render_bkgd * (1.0 - opacities)

    return (
        rgbs.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        total_samples,
    )