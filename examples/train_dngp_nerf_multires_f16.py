"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import time

import imageio
import nerfvis
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor
from datasets.dnerf_3d_video import SubjectLoader
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed, render_image_test_v3, namedtuple_map

from nerfacc import OccupancyGrid

from show_gui_unbound import NGPGUI, render_gui

import apex

import taichi as ti
ti.init(arch=ti.cuda, offline_cache=True)

def enlarge_aabb(aabb, factor: float) -> torch.Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])

from torch_efficient_distloss import flatten_eff_distloss, eff_distloss


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
        "flame_steak", 
        "sear_steak",
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
parser.add_argument("--cone_angle", type=float, default=0.004)
args = parser.parse_args()

# for reproducibility
set_random_seed(42)

# hyperparameters
# 310s 24.57psnr
device = "cuda:0"
target_sample_batch_size = 1 << 18  # train with 1M samples per batch
grid_resolution = 128  # resolution of the occupancy grid
grid_nlvl = 4  # number of levels of the occupancy grid
render_step_size = 1e-3  # render step size
# render_step_size = 0.0016914558667675782
alpha_thre = 1e-2  # skipping threshold on alpha
max_steps = 20000  # training steps
aabb_scale = 1 << (grid_nlvl - 1)  # scale up the the aabb as pesudo unbounded
near_plane = 0.02

# setup the dataset
data_root_fp = "/home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets/"
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="train",
    num_rays=8192,  # initial number of rays
    color_bkgd_aug="black",
    factor=4,
    device=device,
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="test",
    num_rays=None,
    color_bkgd_aug="black",
    factor=4,
    device=device,
)

# The region of interest of the scene has been normalized to [-1, 1]^3.
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
aabb_bkgd = enlarge_aabb(aabb, aabb_scale)

grad_scaler = torch.cuda.amp.GradScaler()

# setup the radiance field we want to train.
radiance_field = NGPradianceField(aabb=aabb_bkgd).to(device)
# optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
lr = 1e-2
optimizer = apex.optimizers.FusedAdam(radiance_field.parameters(), lr=lr, eps=1e-15)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
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
occupancy_grid.mark_invisible_cells(
    train_dataset.K.to(device), 
    camera_fuse.to(device), 
    [train_dataset.width, train_dataset.height],
    near_plane,
)

if args.gui_only:
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

    ngp = NGPGUI(render_kwargs=gui_args)
    render_gui(ngp)
    exit()


if args.vis_nerf:
    # setup visualizer to inspect camera and aabb
    vis = nerfvis.Scene("nerf")

    vis.add_camera_frustum(
        "train_camera",
        focal_length=train_dataset.K[0, 0].item(),
        image_width=train_dataset.images.shape[2],
        image_height=train_dataset.images.shape[1],
        z=0.1,
        r=train_dataset.camtoworlds[:, :3, :3],
        t=train_dataset.camtoworlds[:, :3, -1],
    )

    occ_non_vis = occupancy_grid.get_non_visiable()
    for level, occ_i in enumerate(occ_non_vis):
        occ_i = occ_i.cpu().numpy()
        print(occ_i.shape)
        vis.add_points(
            f"occ_{level}",
            occ_i,
            point_size=2**(level),  
        )

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


# training
step = 0
epochs = 1
tic = time.time()
progress_bar = tqdm.tqdm(total=one_epoch, desc=f'epoch: {epochs}/{max_epoch}')
# for _ in range(max_steps + 1):
aabb_bkgd = aabb_bkgd.to(torch.float16)
while step < (max_steps + 1):
    radiance_field.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"].to(torch.float16)
    # rays = data["rays"]
    rays = namedtuple_map(lambda r: r.to(torch.float16), data["rays"])
    pixels = data["pixels"].to(torch.float16)

    def occ_eval_fn(x):
        step_size = render_step_size
        density = radiance_field.query_density(x)
        return density * step_size

    # update occupancy grid
    occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

    with torch.autocast(device_type='cuda', dtype=torch.float16):

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
            cone_angle=args.cone_angle,
            alpha_thre=alpha_thre,
        )

        # dynamic batch size for rays to keep sample batch size constant.
        # num_rays = len(pixels)
        # num_rays = int(
        #     num_rays * (target_sample_batch_size / float(n_rendering_samples))
        # )
        # train_dataset.update_num_rays(num_rays)

        # compute loss
        # loss = F.smooth_l1_loss(rgb, pixels)
        # loss = F.huber_loss(rgb, pixels)
        loss = F.mse_loss(rgb, pixels)

        if args.distortion_loss:
            loss_distor = 0.
            for (weight, t_starts, t_ends, ray_indices) in extra_info:
                loss_distor += distortion(ray_indices, weight, t_starts, t_ends) * 1e-3
            loss += loss_distor

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
        if args.distortion_loss:
            loss_log += f"dist={loss_distor:.7f} | "
        prog = (
            f"e_time={elapsed_time:.2f}s | step={step} | "
            f'{loss_log}'
            f"n_s={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"s/ray={n_rendering_samples/len(pixels):.2f}"
        )
        if step != 0:
            progress_bar.set_postfix_str(prog)
            progress_bar.close()
            # progress_bar.set_description_str(f'epoch: {epochs}/{max_epoch}')
            if step != max_steps:
                epochs+=1
                progress_bar = tqdm.tqdm(total=one_epoch, desc=f'epoch: {epochs}/{max_epoch}')

    prog = (
        # f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
        f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d}"
    )

    progress_bar.set_postfix_str(prog)

    if step != max_steps:
        progress_bar.update()

    if step >= 0 and step % max_steps == 0 and step > 0:
        progress_bar.close()
        # evaluation
        radiance_field.eval()

        psnrs = []
        aabb_bkgd = aabb_bkgd.to(torch.float32)
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

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
                    cone_angle=args.cone_angle,
                    alpha_thre=alpha_thre,
                    # test options
                    # test_chunk_size=16384,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())

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
                    density, embedding = radiance_field.query_density(
                        x, return_feat=True
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

            imageio.imwrite(
                "rgb_test.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                "rgb_error.png",
                (
                    (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                ).astype(np.uint8),
            )
            # break

        psnr_avg = sum(psnrs) / len(psnrs)
        print(f"evaluation: psnr_avg={psnr_avg}")

        torch.save(
            {
                "radiance_field": radiance_field.state_dict(),
                "occupancy_grid": occupancy_grid.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "psnr_avg": psnr_avg,
            },
            "multires.pth"
        )

    step += 1
