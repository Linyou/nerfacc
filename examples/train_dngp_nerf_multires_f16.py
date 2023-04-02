"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import time
import math
import imageio
import nerfvis
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor
from einops import rearrange
# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from radiance_fields.ngp import NGPradianceField
from radiance_fields.dngp import NGPradianceField as DNGPradianceField
# from radiance_fields.custom_ngp import NGPDradianceField
from utils import render_image, set_random_seed, render_image_test_v3, namedtuple_map
from custom_utils import custom_render_image

from nerfacc import OccupancyGrid

from show_gui_unbound import NGPGUI, render_gui

import apex
from torch_efficient_distloss import flatten_eff_distloss, eff_distloss

import pdb

from logger import Logger

import taichi as ti
ti.init(arch=ti.cuda, offline_cache=True)

def enlarge_aabb(aabb, factor: float) -> torch.Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])



def distortion(
    ray_ids: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
) -> Tensor:

    interval = t_ends - t_starts
    tmid = (t_starts + t_ends) / 2

    return flatten_eff_distloss(weights.squeeze(-1), tmid.squeeze(-1), interval.squeeze(-1), ray_ids)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene",
    type=str,
    default="coffee_martini",
    choices=[
        # 3d unbounded
        "coffee_martini",
        "cook_spinach", 
        "cut_roasted_beef", 
        "flame_salmon_1",
        "flame_salmon_2",
        "flame_salmon_3",
        "flame_salmon_4",
        "flame_steak", 
        "sear_steak",
        # hyperNerf
        'interp_aleks-teapot',
        'interp_chickchicken',
        'interp_cut-lemon',
        'interp_hand',
        'interp_slice-banana',
        'interp_torchocolate',
        'misc_americano',
        'misc_cross-hands',
        'misc_espresso',
        'misc_keyboard',
        'misc_oven-mitts',
        'misc_split-cookie',
        'misc_tamping',
        'vrig_3dprinter',
        'vrig_broom',
        'vrig_chicken',
        'vrig_peel-banana',
    ],
    help="which scene to use",
)
parser.add_argument(
    '-d',
    "--distortion_loss",
    action="store_true",
    help="use a distortion loss",
)
parser.add_argument(
    "--gui_only",
    action="store_true",
)
parser.add_argument(
    "--vis_nerf",
    action="store_true",
)
parser.add_argument(
    "--vis_blocking",
    action="store_true",
)
parser.add_argument(
    '-df',
    "--use_dive_offsets",
    action="store_true",
    help="predict offsets for the DIVE method",
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
    '-wr',
    "--weight_rgbper",
    action="store_true",
    help="use weighted rgbs for rgb",
)
parser.add_argument(
    '-ae',
    "--acc_entorpy_loss",
    action="store_true",
    help="use accumulated opacites as entropy loss",
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
    default=1e-3,
)

parser.add_argument(
    "--data_factor",
    type=int,
    default=4.,
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
)
# load checkpoint path
parser.add_argument(
    "--load_model",
    type=str,
    default='multires_dngp.pth',
)
# number of levels of the occupancy grid
parser.add_argument(
    '-gn',
    "--grid_nlvl",
    type=int,
    default=1,
)

# max steps
parser.add_argument(
    "--max_steps",
    type=int,
    default=20000,
)
# dataset, hybernerf or 3vdnerf
parser.add_argument(
    '-ds',
    "--dataset",
    type=str,
    default="hypernerf",
    choices=[
        "hypernerf",
        "3dnerf",
    ],
    help="which dataset to use",
)

parser.add_argument("--cone_angle", type=float, default=0.004)
args = parser.parse_args()

# for reproducibility
set_random_seed(42)

# hyperparameters
# 310s 24.57psnr
device = "cuda:0"
target_sample_batch_size = 1 << 18  # train with 1M samples per batch
grid_resolution = 128  # resolution of the occupancy grid
grid_nlvl = args.grid_nlvl  # number of levels of the occupancy grid
render_step_size = 1e-3  # render step size
# render_step_size = 0.0016914558667675782
alpha_thre = 1e-2  # skipping threshold on alpha
max_steps = args.max_steps  # training steps
aabb_scale = 1 << (grid_nlvl - 1)  # scale up the the aabb as pesudo unbounded
near_plane = 0.02
batch_size = 4096
cone_angle = args.cone_angle
use_sigma_fn = True


# The region of interest of the scene has been normalized to [-1, 1]^3.
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
aabb_bkgd = enlarge_aabb(aabb, aabb_scale)


# aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
# use synthetic
# render_step_size = (
#     (aabb[3:] - aabb[:3]).max()
#     * math.sqrt(3)
#     / 1024
# ).item()



# setup the dataset
add_cam = False
if args.dataset == "hypernerf":
    from datasets.hypernerf import SubjectLoader
    data_root_fp = "/home/loyot/workspace/Datasets/NeRF/HyberNeRF/"
    add_cam = True if 'vrig' in args.scene else False
    extra_kwargs = {
        "add_cam": add_cam,
    }
    dst_hash_resolution = 4096
elif args.dataset == "3dnerf":
    from datasets.dnerf_3d_video import SubjectLoader
    data_root_fp = "/home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets/"
    extra_kwargs = {}
    dst_hash_resolution = 4096*2

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="train",
    num_rays=batch_size,  # initial number of rays
    color_bkgd_aug="black",
    factor=args.data_factor,
    device=device,
    **extra_kwargs,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
    # pin_memory=True
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="test",
    num_rays=None,
    color_bkgd_aug="black",
    factor=args.data_factor,
    device=device,
    **extra_kwargs,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
)

grad_scaler = torch.cuda.amp.GradScaler()

# setup the radiance field we want to train.
radiance_field = DNGPradianceField(
    aabb=aabb_bkgd,
    log2_hashmap_size=19,
    n_levels=16,
    n_features_per_level=2,
    base_resolution=16,
    dst_resolution=dst_hash_resolution,
    moving_step=args.moving_step,
    # moving_step=args.moving_step,
    use_dive_offsets=args.use_dive_offsets,
    use_time_embedding=args.use_time_embedding,
    use_time_attenuation=args.use_time_attenuation,
    use_feat_predict=args.use_feat_predict,
    use_weight_predict=args.use_weight_predict,

).to(device)
# radiance_field.loose_move = True
# optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
lr = args.lr
optimizer = apex.optimizers.FusedAdam(radiance_field.parameters(), lr=lr, eps=1e-15)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     # milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
#     milestones=[
#         max_steps // 2,
#         max_steps * 3 // 4,
#         max_steps * 5 // 6,
#         max_steps * 9 // 10,
#     ],
#     gamma=0.33,
# )
one_epoch = 1000
max_epoch = max_steps // one_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    max_epoch,
    lr/30
)

occupancy_grid = OccupancyGrid(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)
camera_fuse = torch.cat([
    train_dataset.camtoworlds,
    # test_dataset.camtoworlds,
])

if args.gui_only:
    state_dict = torch.load(args.load_model)
    radiance_field.load_state_dict(state_dict["radiance_field"])
    occupancy_grid.load_state_dict(state_dict["occupancy_grid"])

    print("grid before mask: ", occupancy_grid.occs[occupancy_grid.occs>=0].shape)
    if add_cam:
        # for i in range(train_dataset.K.shape[0]):
        #     occupancy_grid.mark_invisible_cells(
        #         train_dataset.K[i].to(device), 
        #         camera_fuse[i:i+1].to(device), 
        #         [train_dataset.width, train_dataset.height],
        #         near_plane,
        #     )
        occupancy_grid.mark_invisible_cells(
            train_dataset.K[30].to(device), 
            camera_fuse.to(device), 
            [train_dataset.width, train_dataset.height],
            near_plane,
        )
    else:
        occupancy_grid.mark_invisible_cells(
            train_dataset.K.to(device), 
            camera_fuse.to(device), 
            [train_dataset.width, train_dataset.height],
            near_plane,
        )
    print("grid after mask: ", occupancy_grid.occs[occupancy_grid.occs>=0].shape)

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

    ngp = NGPGUI(
        render_kwargs=gui_args, 
        use_time=True, 
        hyber=True if args.dataset == "hypernerf" else False,
    )
    render_gui(ngp)
    exit()
else:
    print("grid before mask: ", occupancy_grid.occs[occupancy_grid.occs>=0].shape)
    if add_cam:
        # for i in range(train_dataset.K.shape[0]):
        #     occupancy_grid.mark_invisible_cells(
        #         train_dataset.K[i].to(device), 
        #         camera_fuse[i:i+1].to(device), 
        #         [train_dataset.width, train_dataset.height],
        #         near_plane,
        #     )
        occupancy_grid.mark_invisible_cells(
            train_dataset.K[30].to(device), 
            camera_fuse.to(device), 
            [train_dataset.width, train_dataset.height],
            near_plane,
        )
    else:
        occupancy_grid.mark_invisible_cells(
            train_dataset.K.to(device), 
            camera_fuse.to(device), 
            [train_dataset.width, train_dataset.height],
            near_plane,
        )
    print("grid after mask: ", occupancy_grid.occs[occupancy_grid.occs>=0].shape)


if args.vis_nerf:
    # setup visualizer to inspect camera and aabb
    vis = nerfvis.Scene("nerf")

    # for i in range(train_dataset.K.shape[0]):
    #     vis.add_camera_frustum(
    #         f"train_camera_{i}",
    #         focal_length=train_dataset.K[i, 0, 0].item(),
    #         image_width=train_dataset.width,
    #         image_height=train_dataset.height,
    #         z=0.05,
    #         r=train_dataset.camtoworlds[i, :3, :3],
    #         t=train_dataset.camtoworlds[i, :3, -1],
    #     )
    vis.add_camera_frustum(
        "train_camera",
        focal_length=train_dataset.focals[0, 0].item(),
        image_width=train_dataset.images.shape[2],
        image_height=train_dataset.images.shape[1],
        z=0.05,
        r=train_dataset.camtoworlds[:, :3, :3],
        t=train_dataset.camtoworlds[:, :3, -1],
    )

    # occ_non_vis = occupancy_grid.get_non_visiable()
    # for level, occ_i in enumerate(occ_non_vis):
    #     occ_i = occ_i.cpu().numpy()
    #     print(occ_i.shape)
    #     vis.add_points(
    #         f"occ_{level}",
    #         occ_i,
    #         point_size=2**(level),  
    #     )

    p1 = aabb[:3].cpu().numpy()
    p2 = aabb[3:].cpu().numpy()
    verts, segs = [
        [p1[0], p1[1], p1[2]],
        [p1[0], p1[1], p2[2]],
        [p1[0], p2[1], p2[2]],
        [p1[0], p2[1], p1[2]],
        [p2[0], p1[1], p1[2]],
        [p2[0], p1[1], p2[2]],
        [p2[0], p2[1], p2[2]],
        [p2[0], p2[1], p1[2]],
    ], [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    vis.add_lines(
        "aabb",
        np.array(verts).astype(dtype=np.float32),
        segs=np.array(segs),
    )

    p1 = aabb_bkgd[:3].cpu().numpy()
    p2 = aabb_bkgd[3:].cpu().numpy()
    verts, segs = [
        [p1[0], p1[1], p1[2]],
        [p1[0], p1[1], p2[2]],
        [p1[0], p2[1], p2[2]],
        [p1[0], p2[1], p1[2]],
        [p2[0], p1[1], p1[2]],
        [p2[0], p1[1], p2[2]],
        [p2[0], p2[1], p2[2]],
        [p2[0], p2[1], p1[2]],
    ], [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    vis.add_lines(
        "aabb_bkgd",
        np.array(verts).astype(dtype=np.float32),
        segs=np.array(segs),
    )

    vis.display(port=8889, serve_nonblocking=(not args.vis_blocking))


logger_obj = Logger(args)

# training
step = 0
epochs = 1
tic = time.time()
progress_bar = tqdm.tqdm(total=one_epoch, desc=f'{epochs}/{max_epoch}')
# for _ in range(max_steps + 1):
aabb_bkgd = aabb_bkgd.to(torch.float16)
# while step < (max_steps + 1):

for _ in range(10000000):
    for data in train_dataloader:
        radiance_field.train()

        # i = torch.randint(0, len(train_dataset), (1,)).item()
        # data = train_dataset[i]

        render_bkgd = data["color_bkgd"].to(torch.float16).to(device)
        # rays = data["rays"]
        rays = namedtuple_map(lambda r: r.to(torch.float16).to(device), data["rays"])
        pixels = data["pixels"].to(torch.float16).to(device)
        timestamps = data["timestamps"].to(torch.float16).to(device)


        def occ_eval_fn(x):
            step_size = render_step_size
            idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
            t = timestamps[idxs]
            density = radiance_field.query_density(x, t)
            return density * step_size

        with torch.autocast(device_type='cuda', dtype=torch.float16):

            # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

            if args.use_feat_predict or args.use_weight_predict:
                radiance_field.return_extra = True
                rgb, acc, depth, move, n_rendering_samples, extra, extra_info = custom_render_image(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    scene_aabb=aabb_bkgd,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=args.cone_angle,
                    # cone_angle=0,
                    alpha_thre=alpha_thre,
                    timestamps=timestamps,
                    # idx=idx,
                    use_sigma_fn=use_sigma_fn,
                )
            else:
                # render
                rgb, acc, depth, n_rendering_samples, extra_info = render_image(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    scene_aabb=aabb_bkgd,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    timestamps=timestamps,
                )
            if n_rendering_samples == 0:
                continue
            # dynamic batch size for rays to keep sample batch size constant.
            # num_rays = len(pixels)
            # num_rays = int(
            #     num_rays * (target_sample_batch_size / float(n_rendering_samples))
            # )
            # train_dataset.update_num_rays(num_rays)

            # compute loss
            # loss = F.smooth_l1_loss(rgb, pixels)
            # loss = F.huber_loss(rgb, pixels)
            # loss = F.mse_loss(rgb, pixels)
            loss = (rgb - pixels).pow(2).mean()

            # o = acc
            # loss_o = (-o*torch.log(o)).mean()*1e-4
            # loss += loss_o

            if args.acc_entorpy_loss:
                T_last = 1 - acc
                T_last = T_last.clamp(1e-6, 1-1e-6)
                entropy_loss = -(T_last*torch.log(T_last) + (1-T_last)*torch.log(1-T_last)).mean()
                loss += entropy_loss*1e-2

            if args.use_feat_predict:
                loss_extra = (extra[0]).mean()
                loss += loss_extra

            if args.use_weight_predict:
                loss_weight = (extra[1]).mean()
                loss += loss_weight

            if args.distortion_loss or args.weight_rgbper:
                loss_distor = 0.
                loss_weight_rgbper = 0.
                for (weight, t_starts, t_ends, ray_indices, rgbs) in extra_info:

                    if args.weight_rgbper:
                        rgbper = (rgbs - pixels[ray_indices]).pow(2).sum(dim=-1)
                        loss_weight_rgbper += (rgbper * weight[:, 0].detach()).sum() / pixels.shape[0] * 1e-2

                    if args.distortion_loss:
                        loss_distor += distortion(ray_indices, weight, t_starts, t_ends) * 1e-3

                loss += loss_distor + loss_weight_rgbper

        optimizer.zero_grad()
        # (loss * 1024).backward()
        grad_scaler.scale(loss).backward()
        # optimizer.step()
        # scheduler.step()
        grad_scaler.step(optimizer)
        scale = grad_scaler.get_scale()
        grad_scaler.update()

        if step % 1000 == 0:

            if not scale > grad_scaler.get_scale():
                scheduler.step()
            else:
                continue

            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            loss_log = f"loss={loss:.5f} | "
            if args.use_feat_predict:
                loss_log += f"extra={loss_extra:.7f} | "
            if args.use_weight_predict:
                loss_log += f"weight={loss_weight:.7f} | "
            if args.distortion_loss:
                loss_log += f"dist={loss_distor:.7f} | "
            if args.weight_rgbper:
                loss_log += f"rgbper={loss_weight_rgbper:.7f} | "
            if args.acc_entorpy_loss:
                loss_log += f"entro={entropy_loss:.7f} | "

            prog = (
                f"eta={elapsed_time:.2f}s | step={step} | "
                f'{loss_log}'
                f"n_s={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"s/ray={n_rendering_samples/len(pixels):.2f}"
            )
            logger_obj.log(prog)
            if step != 0:
                progress_bar.set_postfix_str(prog)
                progress_bar.close()
                # progress_bar.set_description_str(f'epoch: {epochs}/{max_epoch}')
                if step != max_steps:
                    epochs+=1
                    progress_bar = tqdm.tqdm(total=one_epoch, desc=f'{epochs}/{max_epoch}')

        prog = (
            # f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d}"
        )

        progress_bar.set_postfix_str(prog)

        if step != max_steps:
            progress_bar.update()

        if step >= 0 and step % max_steps == 0 and step > 0:

            logger_obj.save_state(radiance_field, occupancy_grid, step)

            torch.cuda.empty_cache()
            val_ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
            val_msssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
            val_lpips = LearnedPerceptualImagePatchSimilarity('vgg').to(device)
            progress_bar.close()
            # evaluation
            radiance_field.eval()
            # test_dataset = test_dataset.to(device)

            psnrs = []
            ssims = []
            msssims = []
            lpips = []
            aabb_bkgd = aabb_bkgd.to(torch.float32)
            # timer = time.time()
            with torch.no_grad():
                # for i in tqdm.tqdm(range(len(test_dataset))):
                #     data = test_dataset[i]
                progress_bar = tqdm.tqdm(total=len(test_dataset), desc=f'evaluating: ')
                for test_step, data in enumerate(test_dataloader):
                    progress_bar.update()
                    # render_bkgd = data["color_bkgd"]
                    # rays = data["rays"]
                    # pixels = data["pixels"]
                    # timestamps = data["timestamps"]
                    render_bkgd = data["color_bkgd"].to(device)
                    rays = namedtuple_map(lambda r: r.to(device), data["rays"])
                    pixels = data["pixels"].to(device)
                    timestamps = data["timestamps"].to(device)

                    # rendering
                    rgb, acc, depth, _ = render_image_test_v3(
                        1024,
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb=aabb_bkgd,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                        timestamps=timestamps,
                        # test options
                        # test_chunk_size=16384,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    rgb_pred = rearrange(rgb, 'h w c -> 1 c h w', h=rgb.shape[0])
                    rgb_gt = rearrange(pixels, 'h w c -> 1 c h w', h=pixels.shape[0])

                    val_ssim(rgb_pred, rgb_gt)
                    ssim = val_ssim.compute()
                    ssims.append(ssim)
                    val_ssim.reset()

                    val_msssim(rgb_pred, rgb_gt)
                    msssim = val_msssim.compute()
                    msssims.append(msssim)
                    val_msssim.reset()

                    val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                                torch.clip(rgb_gt*2-1, -1, 1))
                    lpip = val_lpips.compute()
                    lpips.append(lpip)
                    val_lpips.reset()

                    logger_obj.save_image(rgb, depth, pixels=pixels, step=test_step)

                # elapsed_time = time.time() - timer
                # per_i = elapsed_time/len(test_dataset)
                # print(f"total training time: {elapsed_time:.2f}, per_image: {per_i:.2f}")

                progress_bar.close()
            
                    # if step != max_steps:

                if args.vis_nerf:
                    # occ_non_vis = occupancy_grid.get_non_visiable()
                    # for level, occ_i in enumerate(occ_non_vis):
                    #     occ_i = occ_i.cpu().numpy()
                    #     print(occ_i.shape)
                    #     vis.add_points(
                    #         f"occ_{level}",
                    #         occ_i,
                    #         point_size=2**(5 - level),  
                    #     )

                    def nerfvis_eval_fn(x, dirs):
                        t = torch.zeros(*x.shape[:-1], 1, device=x.device)
                        density, embedding = radiance_field.query_density(
                            x, t, return_feat=True
                        )
                        embedding = embedding.expand(-1, dirs.shape[1], -1)
                        dirs = dirs.expand(embedding.shape[0], -1, -1)
                        rgb = radiance_field._query_rgb(
                            dirs, embedding=embedding, apply_act=False
                        )
                        return rgb, density

                    vis.remove("nerf")
                    vis.add_nerf(
                        name="nerf",
                        eval_fn=nerfvis_eval_fn,
                        center=((aabb[3:] + aabb[:3]) / 2.0).tolist(),
                        radius=((aabb[3:] - aabb[:3]) / 2.0).max().item(),
                        use_dirs=True,
                        reso=128,
                        sigma_thresh=1.0,
                    )
                    vis.display(port=8889, serve_nonblocking=True)

                # imageio.imwrite(
                #     "rgb_test.png",
                #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                # )
                # imageio.imwrite(
                #     "rgb_error.png",
                #     (
                #         (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #     ).astype(np.uint8),
                # )
                # break

            psnr_avg = sum(psnrs) / len(psnrs)
            ssim_avg = sum(ssims) / len(ssims)
            msssim_avg = sum(msssims) / len(msssims)
            lpip_avg = sum(lpips) / len(lpips)
            perf=f"evaluation: psnr_avg={psnr_avg}, ssim_avg={ssim_avg}, msssim_avg={msssim_avg}, lpip_avg={lpip_avg}"
            logger_obj.log(perf)
            print(perf)
            

            torch.save(
                {
                    "radiance_field": radiance_field.state_dict(),
                    "occupancy_grid": occupancy_grid.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "psnr_avg": psnr_avg,
                },
                "multires_dngp.pth"
            )

        if step == max_steps:
            print("training stops")
            exit()

        step += 1
