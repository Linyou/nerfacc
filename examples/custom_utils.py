
import random
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn

import nerfacc.cuda as _C

import torch.nn.functional as F
from torch.cuda.amp import autocast

from nerfacc import OccupancyGrid, ray_marching, rendering, pack_info, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays

from radiance_fields import trunc_exp


def reduce_along_rays(
    ray_indices: Tensor,
    values: Tensor,
    n_rays: Optional[int] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    Note:
        This function is only differentiable to `weights` and `values`.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples,).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        values: The values to be accmulated. Tensor with shape (n_samples, D). If \
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If \
            None, it will be inferred from `ray_indices.max() + 1`.  If specified \
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then we return \
            the accumulated weights, in which case D == 1.
    """
    assert ray_indices.dim() == 1 and values.dim() == 2
    if not values.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if weights is not None:
        assert (
            values.dim() == 2 and values.shape[0] == weights.shape[0]
        ), "Invalid shapes: {} vs {}".format(values.shape, weights.shape)
        src = weights*values
    else:
        src = values

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=values.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    # assert n_rays > ray_indices.max()

    ray_indices = ray_indices.int()
    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=values.device, dtype=src.dtype)
    outputs.scatter_reduce_(0, index, src, reduce="mean")
    return outputs

def accumulate_along_rays_no_weight(
    ray_indices: Tensor,
    values: Tensor,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    Note:
        This function is only differentiable to `weights` and `values`.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples,).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        values: The values to be accmulated. Tensor with shape (n_samples, D). If \
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If \
            None, it will be inferred from `ray_indices.max() + 1`.  If specified \
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then we return \
            the accumulated weights, in which case D == 1.

    """
    assert ray_indices.dim() == 1

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, values.shape[-1]), device=values.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    # assert n_rays > ray_indices.max()

    ray_indices = ray_indices.int()
    index = ray_indices[:, None].long().expand(-1, values.shape[-1])
    outputs = torch.zeros((n_rays, values.shape[-1]), device=values.device)
    outputs.scatter_add_(0, index, values)
    return outputs

def custom_rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can 
    be used for gradient-based optimization.

    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided. 

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        n_rays: Total number of rays. This will decide the shape of the ouputs.
        rgb_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and density \
            values (N, 1). 
        rgb_alpha_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and opacity \
            values (N, 1).
        render_bkgd: Optional. Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1) and depths (n_rays, 1).
    """
    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas, extra = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
            return_trans=True
        )
        # trans = render_transmittance_from_density(
        #     t_starts,
        #     t_ends,
        #     sigmas,
        #     ray_indices=ray_indices,
        #     n_rays=n_rays,
        # )
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices.long())
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )


    extra_reduce = []
    feat_loss, p_weight, selector, move_norm = extra

    # move_norm_view = move_norm[:, None]
    w_dim = sigmas.dim()
    m_dim = move_norm.dim()
    assert w_dim == m_dim, f"sigmas: {w_dim} and move :{m_dim} not equal!"

    with torch.no_grad():
        render_move = render_weight_from_density(
            t_starts,
            t_ends,
            move_norm,
            ray_indices=ray_indices,
            n_rays=n_rays
        )
        final_move = accumulate_along_rays(
            render_move, ray_indices, values=None, n_rays=n_rays
        )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )
    # move_norm_render = accumulate_along_rays(
    #     weights.detach(), ray_indices, values=move_norm.unsqueeze(-1), n_rays=n_rays
    # )*10

    if feat_loss is not None:
        extra_reduce.append(
            reduce_along_rays(
                ray_indices,
                values=feat_loss,
                n_rays=n_rays,
                # weights=render_move,
            )
        )
    else:
        extra_reduce.append(None)

    if p_weight is not None:
        weight_loss = F.huber_loss(p_weight, weights, reduction='none') * selector[..., None]
        extra_reduce.append(
            reduce_along_rays(
                ray_indices,
                values=weight_loss,
                n_rays=n_rays,
                # weights=render_move,
            )
        ) 
    else:
        extra_reduce.append(None)


    # extra_reduce.append(
    #     reduce_along_rays(
    #         ray_indices,
    #         values=loss_times,
    #         n_rays=n_rays,
    #         weights=render_move,
    #     )
    # )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, final_move, weights, extra_reduce


def custom_render_image(
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
        ray_indices = ray_indices.long()
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
        ray_indices = ray_indices.long()
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
    extra_results = []
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
        rgb, opacity, depth, final_move, weight, extra = custom_rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, final_move, len(t_starts)]
        results.append(chunk_results)
        extra_results.append(extra)
        extra_info.append((weight, t_starts, t_ends, ray_indices, pack_info(ray_indices, chunk_rays.origins.shape[0])))
    colors, opacities, depths, final_moves, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    extra = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r 
        for r in zip(*extra_results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        final_moves.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        extra,
        extra_info,
    )