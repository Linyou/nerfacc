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
from datasets.nerf_360_v2 import SubjectLoader
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed

from nerfacc import OccupancyGrid


def enlarge_aabb(aabb, factor: float) -> torch.Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


# arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene",
    type=str,
    default="garden",
    choices=[
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
alpha_thre = 1e-2  # skipping threshold on alpha
max_steps = 20000  # training steps
aabb_scale = 1 << (grid_nlvl - 1)  # scale up the the aabb as pesudo unbounded
near_plane = 0.02

# setup the dataset
data_root_fp = "/home/ruilongli/data/360_v2/"
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="train",
    num_rays=1024,  # initial number of rays
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

# The region of interest of the scene has been normalized to [-1, 1]^3.
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
aabb_bkgd = enlarge_aabb(aabb, aabb_scale)

# setup the radiance field we want to train.
radiance_field = NGPradianceField(aabb=aabb_bkgd).to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
    gamma=0.33,
)
occupancy_grid = OccupancyGrid(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# # setup visualizer to inspect camera and aabb
# vis = nerfvis.Scene("nerf")

# vis.add_camera_frustum(
#     "train_camera",
#     focal_length=train_dataset.K[0, 0].item(),
#     image_width=train_dataset.images.shape[2],
#     image_height=train_dataset.images.shape[1],
#     z=0.3,
#     r=train_dataset.camtoworlds[:, :3, :3],
#     t=train_dataset.camtoworlds[:, :3, -1],
# )

# p1 = aabb[:3].cpu().numpy()
# p2 = aabb[3:].cpu().numpy()
# verts, segs = [
#     [p1[0], p1[1], p1[2]],
#     [p1[0], p1[1], p2[2]],
#     [p1[0], p2[1], p2[2]],
#     [p1[0], p2[1], p1[2]],
#     [p2[0], p1[1], p1[2]],
#     [p2[0], p1[1], p2[2]],
#     [p2[0], p2[1], p2[2]],
#     [p2[0], p2[1], p1[2]],
# ], [
#     [0, 1],
#     [1, 2],
#     [2, 3],
#     [3, 0],
#     [4, 5],
#     [5, 6],
#     [6, 7],
#     [7, 4],
#     [0, 4],
#     [1, 5],
#     [2, 6],
#     [3, 7],
# ]
# vis.add_lines(
#     "aabb",
#     np.array(verts).astype(dtype=np.float32),
#     segs=np.array(segs),
# )

# vis.display(port=8889, serve_nonblocking=True)


# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        step_size = render_step_size
        density = radiance_field.query_density(x)
        return density * step_size

    # update occupancy grid
    occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

    # render
    rgb, acc, depth, n_rendering_samples = render_image(
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
    num_rays = len(pixels)
    num_rays = int(
        num_rays * (target_sample_batch_size / float(n_rendering_samples))
    )
    train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    (loss * 1024).backward()
    optimizer.step()
    scheduler.step()

    if step % 10000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"depth={depth.max():.3f} | "
        )

    if step >= 0 and step % max_steps == 0 and step > 0:
        # evaluation
        radiance_field.eval()

        psnrs = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image(
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
                    test_chunk_size=8192,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())

                # if step != max_steps:

                #     def nerfvis_eval_fn(x, dirs):
                #         density, embedding = radiance_field.query_density(
                #             x, return_feat=True
                #         )
                #         embedding = embedding.expand(-1, dirs.shape[1], -1)
                #         dirs = dirs.expand(embedding.shape[0], -1, -1)
                #         rgb = radiance_field._query_rgb(
                #             dirs, embedding=embedding, apply_act=False
                #         )
                #         return rgb, density

                #     vis.remove("nerf")
                #     vis.add_nerf(
                #         name="nerf",
                #         eval_fn=nerfvis_eval_fn,
                #         center=((aabb[3:] + aabb[:3]) / 2.0).tolist(),
                #         radius=((aabb[3:] - aabb[:3]) / 2.0).max().item(),
                #         use_dirs=True,
                #         reso=128,
                #         sigma_thresh=1.0,
                #     )
                #     vis.display(port=8889, serve_nonblocking=True)

                #     imageio.imwrite(
                #         "rgb_test.png",
                #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                #     )
                #     imageio.imwrite(
                #         "rgb_error.png",
                #         (
                #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #         ).astype(np.uint8),
                #     )
                #     break

        psnr_avg = sum(psnrs) / len(psnrs)
        print(f"evaluation: psnr_avg={psnr_avg}")

        # torch.save(
        #     {
        #         "radiance_field": radiance_field.state_dict(),
        #         "occupancy_grid": occupancy_grid.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scheduler": scheduler.state_dict(),
        #         "step": step,
        #     },
        #     "ckpt.pt"
        # )
