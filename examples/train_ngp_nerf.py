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
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed, namedtuple_map, render_image_test_v3
from tqdm import tqdm

from nerfacc import ContractionType, OccupancyGrid, unpack_data

from torch import Tensor
from torch_efficient_distloss import flatten_eff_distloss
import apex

import cv2
from loguru import logger

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def distortion(
    ray_ids: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
) -> Tensor:
    """Distortion loss from Mip-NeRF 360 paper, Equ. 15.

    Args:
        packed_info: Packed info for the samples. (n_rays, 2)
        weights: Weights for the samples. (all_samples,)
        t_starts: Per-sample start distance. Tensor with shape (all_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (all_samples, 1).

    Returns:
        Distortion loss. (n_rays,)
    """

    # （all_samples, 1) -> (n_rays, n_samples)
    # w = unpack_data(ray_ids, weights).squeeze(-1)
    # t1 = unpack_data(ray_ids, t_starts).squeeze(-1)
    # t2 = unpack_data(ray_ids, t_ends).squeeze(-1)
    # print("interval: ", interval.shape)
    # print("tmid: ", tmid.shape)
    # print("weights: ", weights.shape)
    # print("ray_ids: ", ray_ids.shape)

    interval = t_ends - t_starts
    tmid = (t_starts + t_ends) / 2

    # interval = t2 - t1
    # tmid = (t1 + t2) / 2

    # return eff_distloss(w, tmid, interval)
    return flatten_eff_distloss(weights.squeeze(-1), tmid.squeeze(-1), interval.squeeze(-1), ray_ids)


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
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
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
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        # default="-16.0,-16.0,-16.0,16.0,16.0,16.0",
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
        from datasets.nerf_synthetic import SubjectLoader
        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/nerf_synthetic/"
        target_sample_batch_size = 1 << 18
        grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )

    if args.auto_aabb:
        camera_locs = torch.cat(
            [train_dataset.camtoworlds, test_dataset.camtoworlds]
        )[:, :3, -1]
        args.aabb = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
        print("Using auto aabb", args.aabb)

    # train_dataset.images = train_dataset.images.to(device)
    # train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    # train_dataset.K = train_dataset.K.to(device)

    train_dataset.update_num_rays(8192)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=16,
        persistent_workers=True,
        batch_size=None,
        pin_memory=True
    )
    camtoworlds = train_dataset.camtoworlds.to(device)

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.02
        far_plane = 100
        render_step_size = 1e-2
        alpha_thre = 1e-2
        # alpha_thre = 0.0
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        # render_step_size = (
        #     (scene_aabb[3:] - scene_aabb[:3]).max()
        #     * math.sqrt(3)
        #     / render_n_samples
        # ).item()
        # near_plane = 0.2
        # far_plane = 1e2
        # render_step_size = 1e-2
        # alpha_thre = 1e-2
        # alpha_thre = 0.0

    # setup the radiance field we want to train.
    max_steps = 20000
    one_epoch = 1000
    max_epoch = max_steps // one_epoch
    epochs = 1
    grad_scaler = torch.cuda.amp.GradScaler()
    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)
    # optimizer = torch.optim.Adam(
    #     radiance_field.parameters(), lr=1e-2, eps=1e-15
    # )
    lr = 1e-2
    optimizer = apex.optimizers.FusedAdam(radiance_field.parameters(), lr, eps=1e-15)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
    #     gamma=0.33,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max_epoch,
        lr/30
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
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
    # logger file
    results_root = '/home/loyot/workspace/code/training_results/nerfacc'
    logger.remove(0)
    logger.add(os.path.join(results_root, 'logs', f'ngp_nerf_{args.scene}_'+"{time}.log"))


    # training
    step = 0
    tic = time.time()

    progress_bar = tqdm(total=one_epoch, desc=f'epoch: {epochs}/{max_epoch}')

    for epoch in range(10000000):
        # for i in range(len(train_dataset)):
        for i, data in enumerate(train_dataloader):
            radiance_field.train()
            # data = train_dataset[i]

            render_bkgd = data["color_bkgd"].to(device)
            # rays = data["rays"]
            # rays = namedtuple_map(lambda r: r.to(torch.float16), data["rays"])
            rays = namedtuple_map(lambda r: r.to(device), data["rays"])
            pixels = data["pixels"].to(device)

            def occ_eval_fn(x):
                if args.cone_angle > 0.0:
                    # randomly sample a camera for computing step size.
                    camera_ids = torch.randint(
                        0, len(train_dataset), (x.shape[0],), device=device
                    )
                    origins = camtoworlds[camera_ids, :3, -1]
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
                density = radiance_field.query_density(x)
                return density * step_size

            with torch.autocast(device_type='cuda', dtype=torch.float16):

                # update occupancy grid
                occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

                # render
                rgb, acc, depth, n_rendering_samples, extra = render_image(
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
                )
                if n_rendering_samples == 0:
                    continue

                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays
                    * ((target_sample_batch_size) / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)
                alive_ray_mask = acc.squeeze(-1) > 0

                # compute loss
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
    
                # o = acc[alive_ray_mask]
                # loss += (-o*torch.log(o)).mean()*1e-3

                loss_d = 0.
                for (weight, t_starts, t_ends, ray_indices) in extra:
                    loss_d += distortion(ray_indices, weight, t_starts, t_ends) * 1e-3

                loss = loss + loss_d
            
            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            scale = grad_scaler.get_scale()
            grad_scaler.update()

            # prevent scheduler.step() before optimizer.step()
            # if not scale > grad_scaler.get_scale():
            #     scheduler.step()

            # optimizer.step()
            # scheduler.step()

            if step % 1000 == 0:

                if not scale > grad_scaler.get_scale():
                    scheduler.step()

                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                prog = (
                    f"e_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | " 
                    # f"loss_o={loss_o:.5f} | "
                    f"loss_d={loss_d:.5f} | "
                    f"alive={alive_ray_mask.long().sum():d} | "
                    f"n_s={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"s/ray={n_rendering_samples/len(pixels):.2f}"
                )
                logger.info(prog)
                if step != 0:
                    progress_bar.set_postfix_str(prog)
                    progress_bar.close()
                    # progress_bar.set_description_str(f'epoch: {epochs}/{max_epoch}')
                    if step != max_steps:
                        epochs+=1
                        progress_bar = tqdm(total=one_epoch, desc=f'epoch: {epochs}/{max_epoch}')

            prog = (
                # f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d}"
            )

            progress_bar.set_postfix_str(prog)

            if step != max_steps:
                progress_bar.update()


            if step >= 0 and step % max_steps == 0 and step > 0:
                progress_bar.close()

                os.makedirs(f'{results_root}/checkpoints', exist_ok=True)
                torch.save(
                        {
                            "radiance_field": radiance_field.state_dict(),
                            "occupancy_grid": occupancy_grid.state_dict(),
                        },
                        f"{results_root}/checkpoints/ngp_nerf_{args.scene}_{step}.pth",
                )

                image_root = f'{results_root}/ngp_nerf/{args.scene}/'
                print('image_root: ', image_root)

                # evaluation
                import taichi as ti
                ti.init(arch=ti.cuda, offline_cache=True)

                test_dataset.images = test_dataset.images.to(device)
                test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
                test_dataset.K = test_dataset.K.to(device)
                radiance_field.eval()

                psnrs = []
                with torch.no_grad():
                    for i in tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        # rays = namedtuple_map(lambda r: r.to(torch.float16), data["rays"])
                        pixels = data["pixels"]

                        # rendering
                        rgb, _, depth, n_rendering_samples = render_image_test_v3(
                            1024,
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
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        os.makedirs(f'{image_root}', exist_ok=True)
                        assert os.path.exists(f'{image_root}'), f"test images saving path dose not exits! path: {image_root}"
                        imageio.imwrite(
                            f"{image_root}/depth_{i}.png",
                            depth2img(depth),
                        )
                        imageio.imwrite(
                            f"{image_root}/rgb_{i}.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: psnr_avg={psnr_avg}")
                logger.info(f"evaluation: psnr_avg={psnr_avg}")
                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()

            step += 1
