"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.custom_ngp import NGPDradianceField
from utils import render_image, set_random_seed
from custom_utils import custom_render_image
from visdom import Visdom
from radiance_fields.mlp import DNeRFRadianceField

from nerfacc import ContractionType, OccupancyGrid, loss_distortion
from nerfacc.taichi_modules import distortion

import taichi as ti
ti.init(arch=ti.cuda)

def total_variation_loss(x):
    # Get resolution
    tv_x = torch.pow(x[1:,:,:,:]-x[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(x[:,1:,:,:]-x[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()

    return (tv_x.mean() + tv_y.mean() + tv_z.mean())/3

if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

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
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    render_n_samples = 1024

    results_root = '/home/loyot/workspace/code/training_results/nerfacc'

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = "/home/ruilongli/data/360_v2/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256
    else:
        from datasets.dnerf_synthetic import SubjectLoader

        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/dynamic_data/"
        target_sample_batch_size = 1 << 18
        grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    ).to(device)

    train_dataset_test = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=None,
        **train_dataset_kwargs,
    ).to(device)
    train_dataset_test.training = False

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    ).to(device)

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
        far_plane = 1e4
        render_step_size = 1e-2
        alpha_thre = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0

    # setup the radiance field we want to train.
    max_steps = 20000
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    viz = Visdom()
    # dnerf_radiance_field = DNeRFRadianceField().to(device)
    # dnerf_radiance_field.load_state_dict(torch.load('checkpoints/dnerf_lego_30000.pth', map_location=torch.device('cuda')))
    radiance_field = NGPDradianceField(
        # dnerf=dnerf_radiance_field,
        logger=viz,
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=1e-2, eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    

    # training
    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            timestamps = data["timestamps"]

            def occ_eval_fn(x):
                if args.cone_angle > 0.0:
                    # randomly sample a camera for computing step size.
                    camera_ids = torch.randint(
                        0, len(train_dataset), (x.shape[0],), device=device
                    )
                    origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                    t = (origins - x).norm(dim=-1, keepdim=True)
                    # compute actual step size used in marching, based on the distance to the camera.
                    step_size = torch.clamp(
                        t * args.cone_angle, min=render_step_size
                    )
                    # filter out the points that are not in the near far plane.
                    if (near_plane is not None) and (far_plane is not None):
                        step_size = torch.where(
                            (t > near_plane) & (t < far_plane),
                            step_size,
                            torch.zeros_like(step_size),
                        )
                else:
                    step_size = render_step_size
                # compute occupancy
                idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
                t = timestamps[idxs]
                density = radiance_field.query_density(x, t)
                return density * step_size


            # # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

            # render
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            rgb, acc, depth, n_rendering_samples, extra = custom_render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                alpha_thre=alpha_thre,
                timestamps=timestamps,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss

            # grid_feat = radiance_field.get_grid()
            # tv = total_variation_loss(grid_feat.view(64, 64, 64, -1))

            # loss_times = F.smooth_l1_loss(times_acc[alive_ray_mask], timestamps[alive_ray_mask])
            o = acc[alive_ray_mask]
            loss_o = (-o*torch.log(o)).mean()*1e-2 
            loss_extra = (extra[0][alive_ray_mask]).mean()*1e-1
            # loss_distor = (extra[1][alive_ray_mask] * 0.01).mean()
            # for k in extra:
            #     loss_extra += (k[alive_ray_mask]*0.01).mean()
            rec_loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
            loss = rec_loss + loss_o + loss_extra

            # loss_extra = 0.
            # # loss_distor = 0.
            # for (predict, hash_feat, weight, t_starts, t_ends, ray_indices, packed_info) in extra:
            #     alive_samples_mask = alive_ray_mask[ray_indices.long()]
            #     loss_extra += F.smooth_l1_loss(predict[alive_samples_mask], hash_feat[alive_samples_mask])*0.1
            #     # loss_distor += distortion(packed_info, weight, t_starts, t_ends).mean() * 0.01
            
            # loss += loss_extra
                # loss += loss_distor

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            # grad_scaler.step(optimizer)
            scheduler.step()

            # grad_scaler.update()

            if step % 1000 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | " f"loss_extra={loss_extra:.7f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                )

            if step >= 0 and step % max_steps == 0 and step > 0:
               # save the model
                os.makedirs(f'{results_root}/checkpoints', exist_ok=True)
                torch.save(
                    radiance_field.state_dict(),
                    f"{results_root}/checkpoints/ngp_dnerf_{args.scene}_{step}.pth",
                )
                # evaluation
                radiance_field.eval()

                with torch.no_grad():
                    psnrs = []
                    print("save train image")
                    for i in tqdm.tqdm(range(len(train_dataset_test))):
                        data = train_dataset_test[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]
                        timestamps = data["timestamps"]

                        # rendering
                        rgb, acc, depth, _, = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            alpha_thre=alpha_thre,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                            timestamps=timestamps,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        os.makedirs(f'{results_root}/ngp_dnerf/{args.scene}/train', exist_ok=True)
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/train/acc_{i}.png",
                            ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                        )
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/train/depth_{i}.png",
                            (lambda x: (x - x.min()) / (x.max() - x.min()) * 255)(depth.cpu().numpy()).astype(np.uint8),
                        )
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/train/rgb_{i}.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                        # break
                    psnr_avg = sum(psnrs) / len(psnrs)
                    print(f"evaluation: psnr_avg={psnr_avg}")
                    psnrs = []
                    for i in tqdm.tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]
                        timestamps = data["timestamps"]

                        # rendering
                        rgb, acc, depth, _, = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            alpha_thre=alpha_thre,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                            timestamps=timestamps,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        os.makedirs(f'{results_root}/ngp_dnerf/{args.scene}', exist_ok=True)
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/acc_{i}.png",
                            ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                        )
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/depth_{i}.png",
                            (lambda x: (x - x.min()) / (x.max() - x.min()) * 255)(depth.cpu().numpy()).astype(np.uint8),
                        )
                        imageio.imwrite(
                            f"{results_root}/ngp_dnerf/{args.scene}/rgb_{i}.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                    psnr_avg = sum(psnrs) / len(psnrs)
                    print(f"evaluation: psnr_avg={psnr_avg}")

                # psnr_avg = sum(psnrs) / len(psnrs)
                # print(f"evaluation: psnr_avg={psnr_avg}")
                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()

            step += 1
