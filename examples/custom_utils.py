
import argparse
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

from nerfacc import OccupancyGrid, ray_marching, rendering, pack_info, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays, ContractionType

from radiance_fields import trunc_exp
from radiance_fields.ngp import NGPradianceField
import math

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
            # 3d unbounded
            "coffee_martini",
            "cook_spinach", 
            "cut_roasted_beef", 
            "flame_salmon_1",
            "flame_steak", 
            "sear_steak",
            # mipnerf360 unbounded
            "garden",
            "bicycle",
            "bonsai",
            "counter",
            "kitchen",
            "room",
            "stump",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        # default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        default="-1.0,-1.0,-1.0,1.0,1.0,1.0",
        # default="-16.0,-16.0,-16.0,16.0,16.0,16.0",
        # default="-64.0,-64.0,-64.0,64.0,64.0,64.0",
        # default="-8.0,-8.0,-8.0,8.0,8.0,8.0",
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
        '-rl',
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
        "--pretrained_model_path",
        default="",
    )

    parser.add_argument(
        '-tp',
        "--test_print",
        action="store_true",
        help="run evaluation during training",
    )

    parser.add_argument(
        '-df',
        "--use_dive_offsets",
        action="store_true",
        help="predict offsets for the DIVE method",
    )

    parser.add_argument(
        '-te',
        "--use_time_embedding",
        action="store_true",
        help="predict density with time embedding",
    )

    parser.add_argument(
        '-ta',
        "--use_time_attenuation",
        action="store_true",
        help="use time attenuation in time embedding",
    )

    parser.add_argument(
        '-ms',
        "--moving_step",
        type=float,
        default=4096.,
    )

    parser.add_argument(
        '-lv',
        "--log_visdom",
        action="store_true",
        help="print to visdom",
    )

    parser.add_argument(
        '-hl',
        "--hash_level",
        type=int,
        default=0,
        choices=[
            0,
            1,
            2,
        ],
    )
        

    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    return args

def get_feat_dir_and_loss(args):

    feat_dir = 'hl' + str(args.hash_level) + '_'

    feat_dir += 'pf' if args.use_feat_predict else 'nopf'

    if args.use_weight_predict:
        feat_dir += '_pw'
    else:
        feat_dir += '_nopw'

    if args.rec_loss == 'huber':
        rec_loss_fn = F.huber_loss
        feat_dir += '_l-huber'
    elif args.rec_loss == 'mse':
        rec_loss_fn = F.mse_loss
        feat_dir += '_l-mse'
    else:
        feat_dir += '_l-sml1'
        rec_loss_fn = F.smooth_l1_loss

    if args.distortion_loss:
        feat_dir += "_distor"

    if args.use_dive_offsets:
        feat_dir += "_dive" + str(args.moving_step)

    if args.use_opacity_loss:
        feat_dir += "_op"

    if args.use_time_embedding:
        feat_dir += "_te"
        if args.use_time_attenuation:
            feat_dir += "_ta"

    

    return feat_dir, rec_loss_fn


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
        rgbs, sigmas, extra = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays
        )
        # trans = render_transmittance_from_density(
        #     t_starts,
        #     t_ends,
        #     sigmas,
        #     ray_indices=ray_indices,
        #     n_rays=n_rays,
        # )
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
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
    feat_loss, p_weight, selector = extra

    # move_norm_view = move_norm[:, None]
    # w_dim = sigmas.dim()
    # m_dim = move_norm.dim()
    # assert w_dim == m_dim, f"sigmas: {w_dim} and move :{m_dim} not equal!"

    # with torch.no_grad():
    #     render_move = render_weight_from_density(
    #         t_starts,
    #         t_ends,
    #         move_norm,
    #         ray_indices=ray_indices,
    #         n_rays=n_rays
    #     )
    #     final_move = accumulate_along_rays(
    #         render_move, ray_indices, values=None, n_rays=n_rays
    #     )

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
                # weights=weights,
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
                weights=weights,
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

    return colors, opacities, depths, selector, weights, extra_reduce, rgbs


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
    idx: Optional[torch.Tensor] = None,
    use_sigma_fn: bool = True,
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

    if use_sigma_fn:
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
    
    else:
        sigma_fn = None

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
            if idx is not None:
                idxx = (
                    idx[ray_indices]
                    if radiance_field.training
                    else idxx.expand_as(positions[:, :1])   
                )
                return radiance_field(positions, t, t_dirs, idxx)
            else:
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
        # print(len(t_starts))
        rgb, opacity, depth, final_move, weight, extra, rgbs = custom_rendering(
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
        extra_info.append((weight, t_starts, t_ends, ray_indices, rgbs))
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
        final_moves,
        sum(n_rendering_samples),
        extra,
        extra_info,
    )


def get_ngp_args(args):

    device = "cuda:0"

    render_n_samples = 1024

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/360_v2/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256
    else:
        from datasets.nerf_360_v2 import SubjectLoader

        # data_root_fp = "/home/loyot/workspace/Datasets/NeRF/nerf_synthetic/"
        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/360_v2/"
        target_sample_batch_size = 1 << 18
        grid_resolution = 128

        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    # train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    # test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    if args.auto_aabb:
        camera_locs = torch.cat(
            [train_dataset.camtoworlds, test_dataset.camtoworlds]
        )[:, :3, -1]
        args.aabb = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
        print("Using auto aabb", args.aabb)

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e2
        render_step_size = 1e-2
        alpha_thre = 1e-2
        # alpha_thre = 0
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float16, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0
        near_plane = 0.2
        # far_plane = 1e2
        render_step_size = 1e-2
        alpha_thre = 1e-2

    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    root = '/home/loyot/workspace/code/training_results/nerfacc/checkpoints'
    model_path = root + f'/ngp_nerf_{args.scene}_20000.pth'
    # state_dict = torch.load(args.pretrained_model_path)

    state_dict = torch.load(model_path)
    radiance_field.load_state_dict(state_dict["radiance_field"])
    occupancy_grid.load_state_dict(state_dict["occupancy_grid"])


    gui_args = {
        'train_dataset': train_dataset, 
        'test_dataset': test_dataset, 
        'radiance_field': radiance_field, 
        'occupancy_grid': occupancy_grid,
        'contraction_type': contraction_type,
        'scene_aabb': scene_aabb,
        'near_plane': near_plane,
        'far_plane': far_plane,
        'alpha_thre': alpha_thre,
        'cone_angle': args.cone_angle,
        'test_chunk_size': args.test_chunk_size,
        'render_bkgd': torch.ones(3, device=device),
        'render_n_samples': render_n_samples,
        'render_step_size': render_step_size,
        'args_aabb': np.array(args.aabb),
    }

    return gui_args


def get_ngp_multires_args(args):

    device = "cuda:0"

    device = "cuda:0"
    target_sample_batch_size = 1 << 18  # train with 1M samples per batch
    grid_resolution = 128  # resolution of the occupancy grid
    grid_nlvl = 4  # number of levels of the occupancy grid
    render_step_size = 1e-3  # render step size
    alpha_thre = 1e-2  # skipping threshold on alpha
    max_steps = 20000  # training steps
    aabb_scale = 1 << (grid_nlvl - 1)  # scale up the the aabb as pesudo unbounded
    near_plane = 0.02

    from datasets.nerf_360_v2 import SubjectLoader

    # setup the dataset
    data_root_fp = "/home/loyot/workspace/Datasets/NeRF/360_v2/"
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="train",
        num_rays=16384,  # initial number of rays
        color_bkgd_aug="random",
        factor=4,
        device=device,
    )
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        factor=4,
        device=device,
    )


    def enlarge_aabb(aabb, factor: float) -> torch.Tensor:
        center = (aabb[:3] + aabb[3:]) / 2
        extent = (aabb[3:] - aabb[:3]) / 2
        return torch.cat([center - extent * factor, center + extent * factor])

    # The region of interest of the scene has been normalized to [-1, 1]^3.
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    aabb_bkgd = enlarge_aabb(aabb, aabb_scale)


    radiance_field = NGPradianceField(aabb=aabb_bkgd).to(device)
    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)

    # root = '/home/loyot/workspace/code/training_results/nerfacc/checkpoints'
    # model_path = root + f'/ngp_nerf_{args.scene}_20000.pth'
    # state_dict = torch.load(args.pretrained_model_path)

    state_dict = torch.load('multires.pth')
    radiance_field.load_state_dict(state_dict["radiance_field"])
    occupancy_grid.load_state_dict(state_dict["occupancy_grid"])


    gui_args = {
        'train_dataset': train_dataset, 
        'test_dataset': test_dataset, 
        'radiance_field': radiance_field, 
        'occupancy_grid': occupancy_grid,
        'scene_aabb': aabb_bkgd,
        'near_plane': near_plane,
        'far_plane': None,
        'alpha_thre': alpha_thre,
        'cone_angle': args.cone_angle,
        # 'test_chunk_size': args.test_chunk_size,
        'render_bkgd': torch.zeros(3, device=device),
        'render_step_size': render_step_size,
        'args_aabb': None,
        'contraction_type': None,
    }

    return gui_args