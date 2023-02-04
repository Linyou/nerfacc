"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import math
import os
import time

import apex
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from radiance_fields.custom_ngp import NGPDradianceField
from utils import render_image, set_random_seed, namedtuple_map, render_image_test_v3
from custom_utils import custom_render_image, get_feat_dir_and_loss, get_opts
from visdom import Visdom
from radiance_fields.mlp import DNeRFRadianceField

from nerfacc import ContractionType, OccupancyGrid, loss_distortion
# from nerfacc.taichi_modules import distortion
from warmup_scheduler import GradualWarmupScheduler

import cv2
from torch import Tensor
import torch.distributed as dist
from loguru import logger

from torch_efficient_distloss import flatten_eff_distloss, eff_distloss

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

import taichi as ti
ti.init(arch=ti.cuda, offline_cache=True)

def total_variation_loss(x):
    # Get resolution
    tv_x = torch.pow(x[1:,:,:,:]-x[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(x[:,1:,:,:]-x[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()

    return (tv_x.mean() + tv_y.mean() + tv_z.mean())/3

@torch.no_grad()
def send_dist(rad, occ, is_async=False):
    rad_params = rad.parameters()
    for pn, p in enumerate(rad_params):
        send_temp = p.clone().detach()
        if is_async:
            dist.isend(tensor=send_temp, dst=1, tag=pn)
        else:
            dist.send(tensor=send_temp, dst=1, tag=pn)
    occ_params = occ.parameters()
    for pn, p in enumerate(occ_params):
        send_temp = p.clone().detach()
        if is_async:
            dist.isend(tensor=send_temp, dst=1, tag=50+pn)
        else:
            dist.send(tensor=send_temp, dst=1, tag=50+pn)

def main(args, gui=False):

    device = "cuda:0"
    set_random_seed(42)

    # args = get_opts()

    render_n_samples = 1024

    results_root = '/home/loyot/workspace/code/training_results/nerfacc'

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.dnerf_3d_video import SubjectLoader

        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 2}
        test_dataset_kwargs = {"factor": 2}
        grid_resolution = 256
    else:
        from datasets.dnerf_synthetic import SubjectLoader

        data_root_fp = "/home/loyot/workspace/Datasets/NeRF/dynamic_data/"
        target_sample_batch_size = 1 << 18
        grid_resolution = 128

        # data_root_fp = "/home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets/"
        # target_sample_batch_size = 1 << 20
        # train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 2}
        # test_dataset_kwargs = {"factor": 2}
        # grid_resolution = 256

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    # train_dataset_test = SubjectLoader(
    #     subject_id=args.scene,
    #     root_fp=data_root_fp,
    #     split=args.train_split,
    #     num_rays=None,
    #     **train_dataset_kwargs,
    # )
    # train_dataset_test.training = False

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )

    if args.auto_aabb:
        camera_locs = torch.cat(
            [train_dataset.camtoworlds,]
        )[:, :3, -1]
        # args.aabb = torch.cat(
        #     [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        # ).tolist()
        aabb_temp = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
        print("Using auto aabb", aabb_temp)

    scale = torch.tensor([
        3.0, 3.0, 2.0,
        3.0, 3.0, 4.0
    ])
    offset = torch.tensor([
        0.0,  0.0, 1.0,
        0.0,  0.0, 1.0
    ])
    print("before scene aabb: ", args.aabb)
    args.aabb = (torch.tensor(args.aabb)*scale + offset).tolist()
    print("after scene aabb: ", args.aabb)

    # if args.unbounded:
    train_dataset.update_num_rays(8192)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=16,
        persistent_workers=True,
        batch_size=None,
        pin_memory=True
    )
    camtoworlds = train_dataset.camtoworlds.to(device)

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     num_workers=8,
    #     persistent_workers=True,
    #     batch_size=None,
    #     pin_memory=False
    # )

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

        # near_plane = 0.2
        # # # far_plane = 1e2
        # render_step_size = 1e-2
        # alpha_thre = 1e-2
    
    # setup the radiance field we want to train.
    # max_steps = 20000
    max_steps = 20000
    one_epoch = 1000
    max_epoch = max_steps // one_epoch
    epochs = 1
    grad_scaler = torch.cuda.amp.GradScaler()
    if args.log_visdom:
        viz = Visdom()
    else:
        viz = None
    # dnerf_radiance_field = DNeRFRadianceField().to(device)
    # dnerf_radiance_field.load_state_dict(torch.load('checkpoints/dnerf_lego_30000.pth', map_location=torch.device('cuda')))
    radiance_field = NGPDradianceField(
        # dnerf=dnerf_radiance_field,
        logger=viz,
        aabb=args.aabb,
        unbounded=args.unbounded,
        use_feat_predict=args.use_feat_predict,
        use_weight_predict=args.use_weight_predict,
        moving_step=args.moving_step,
        use_dive_offsets=args.use_dive_offsets,
        use_time_embedding=args.use_time_embedding,
        use_time_attenuation=args.use_time_attenuation,
        hash_level=args.hash_level,
    ).to(device)

    lr = args.lr

    optimizer = apex.optimizers.FusedAdam(radiance_field.parameters(), lr, eps=1e-15)
    # optimizer = torch.optim.Adam(
    #     radiance_field.parameters(), lr=lr, eps=1e-15
    # )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
    #     # milestones=[max_steps // 2, max_steps * 9 // 10],
    #     gamma=0.33,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max_epoch,
        lr/30
    )

    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler)


    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[
    #         max_steps // 2,
    #         max_steps * 3 // 4,
    #         max_steps * 5 // 6,
    #         max_steps * 9 // 10,
    #     ],
    #     gamma=0.33,
    # )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # gui_args = {
    #     'train_dataset': train_dataset, 
    #     'test_dataset': test_dataset, 
    #     'radiance_field': radiance_field, 
    #     'occupancy_grid': occupancy_grid,
    #     'contraction_type': contraction_type,
    #     'scene_aabb': scene_aabb,
    #     'near_plane': near_plane,
    #     'far_plane': far_plane,
    #     'alpha_thre': alpha_thre,
    #     'cone_angle': args.cone_angle,
    #     'test_chunk_size': args.test_chunk_size,
    #     'render_bkgd': torch.ones(3, device=device),
    # }

    # mp.set_start_method('spawn')
    # size = 1
    # rank = 0
    # p = mp.Process(target=init_process, args=(rank, size, render_gui, args))
    # p.start()

    # logger file
    results_root = '/home/loyot/workspace/code/training_results/nerfacc'
    logger.remove(0)
    logger.add(os.path.join(results_root, 'logs', f'ngp_dnerf_{args.scene}_'+"{time}.log"))

    if gui:
        send_dist(radiance_field, occupancy_grid, is_async=False)


    # generate log fir for test image and psnr
    feat_dir, rec_loss_fn = get_feat_dir_and_loss(args)

    # training
    step = 0
    tic = time.time()
    # radiance_field.loose_move = True

    progress_bar = tqdm(total=one_epoch, desc=f'epoch: {epochs}/{max_epoch}')

    for epoch in range(10000000):
        # for i in range(len(train_dataset)):
        for i, data in enumerate(train_dataloader):
            radiance_field.train()
            # data = train_dataset[i]

            render_bkgd = data["color_bkgd"].to(device)
            # rays = data["rays"]
            rays = namedtuple_map(lambda r: r.to(device), data["rays"])
            pixels = data["pixels"].to(device)
            timestamps = data["timestamps"].to(device)
            idx = data["idx"].to(device)

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
                idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
                t = timestamps[idxs]
                density = radiance_field.query_density(x, t)
                return density * step_size


            # render
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # # update occupancy grid
                occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)


                if args.use_feat_predict or args.use_weight_predict:
                    radiance_field.return_extra = True
                    rgb, acc, depth, move, n_rendering_samples, extra, extra_info = custom_render_image(
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
                        # idx=idx,
                    )
                else:
                    rgb, acc, depth, n_rendering_samples, extra_info = render_image(
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
                # if not args.unbounded:
                #     num_rays = len(pixels)
                #     num_rays = int(
                #         num_rays
                #         * (target_sample_batch_size / float(n_rendering_samples))
                #     )
                #     train_dataset.update_num_rays(num_rays)

                alive_ray_mask = acc.squeeze(-1) > 0

                # compute loss
                loss = rec_loss_fn(rgb[alive_ray_mask], pixels[alive_ray_mask])

                if args.use_opacity_loss:
                    o = acc[alive_ray_mask]
                    loss_o = (-o*torch.log(o)).mean()*1e-4
                    loss += loss_o

                if args.use_feat_predict:
                    loss_extra = (extra[0][alive_ray_mask]).mean()
                    loss += loss_extra
                else:
                    loss_extra = 0.

                if args.use_weight_predict:
                    loss_weight = (extra[1][alive_ray_mask]).mean()
                    loss += loss_weight
                else:
                    loss_weight = 0.

                if args.distortion_loss:
                    loss_distor = 0.
                    for (weight, t_starts, t_ends, ray_indices) in extra_info:
                        # alive_samples_mask = alive_ray_mask[ray_indices.long()]
                        # loss_extra += F.smooth_l1_loss(predict[alive_samples_mask], hash_feat[alive_samples_mask])*0.1
                        loss_distor += distortion(ray_indices, weight, t_starts, t_ends) * 1e-4
                
                    loss += loss_distor

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            # grad_scaler.unscale_(optimizer)
            if (args.use_feat_predict or args.use_weight_predict) and args.log_visdom:
                radiance_field.log_grad(step)
            # optimizer.step()
            grad_scaler.step(optimizer)
            scale = grad_scaler.get_scale()
            grad_scaler.update()


            if gui:
                send_dist(radiance_field, occupancy_grid, is_async=True)

            if step % 1000 == 0:

                if not scale > grad_scaler.get_scale():
                    scheduler.step()

                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                loss_log = f"loss={loss:.5f} | "
                if args.use_feat_predict:
                    loss_log += f"extra={loss_extra:.7f} | "

                if args.use_weight_predict:
                    loss_log += f"weight={loss_weight:.7f} | "
                if args.distortion_loss:
                     loss_log += f"dist={loss_distor:.7f} | "
                if args.use_opacity_loss:
                    loss_log += f"opac={loss_o:.7f} | "
                prog = (
                    f"e_time={elapsed_time:.2f}s | step={step} | "
                    f'{loss_log}'
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


                str_lr = str(lr).replace('.', '-')
                # evaluation
                radiance_field.eval()
                image_root = f'{results_root}/ngp_dnerf/{args.scene}/lr_{str_lr}/{feat_dir}/training'

                if args.test_print:
                    # print('image_root: ', image_root)

                    with torch.no_grad():

                        # for i in tqdm.tqdm(range(len(test_dataset))):
                        i_id = 15
                        data = test_dataset[i_id]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]
                        timestamps = data["timestamps"]

                        # rendering
                        rgb, acc, depth, _, _ = render_image(
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

                        os.makedirs(f'{image_root}', exist_ok=True)
                        assert os.path.exists(f'{image_root}'), f"test images saving path dose not exits! path: {image_root}"
                        # print("depth shape: ", depth.shape)
                        imageio.imwrite(
                            f"{image_root}/depth_{i_id}_{step}.png",
                            depth2img(depth),
                        )
                        imageio.imwrite(
                            f"{image_root}/rgb_{i_id}_{step}.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                        print(f"test {i_id} for {step} iter: {psnr}")
                        # train_dataset.change_split('train')

            prog = (
                # f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d}"
            )

            progress_bar.set_postfix_str(prog)

            if step != max_steps:
                progress_bar.update()

            if step >= 0 and step % max_steps == 0 and step > 0:
                str_lr = str(lr).replace('.', '-')
                # save the model
                os.makedirs(f'{results_root}/checkpoints', exist_ok=True)
                torch.save(
                    {
                        "radiance_field": radiance_field.state_dict(),
                        "occupancy_grid": occupancy_grid.state_dict(),
                    },
                    f"{results_root}/checkpoints/ngp_dnerf_lr_{str_lr}_{feat_dir}_{args.scene}_{step}.pth",
                )

                # evaluation
                radiance_field.eval()
                image_root = f'{results_root}/ngp_dnerf/{args.scene}/lr_{str_lr}/{feat_dir}'
                print('image_root: ', image_root)

                # test_dataset.images = test_dataset.images.to(device)
                # test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
                # test_dataset.K = test_dataset.K.to(device)


                with torch.no_grad():

                    if args.unbounded:

                        psnrs = []
                        print("save train image")

                        os.makedirs(f'{image_root}/train', exist_ok=True)
                        assert os.path.exists(f'{image_root}/train'), f"train images saving path dose not exits! path: {image_root}/train"
                        train_dataset.training = False
                        for i in tqdm(range(50)):
                            data = train_dataset[i]
                            render_bkgd = data["color_bkgd"].to(device)
                            rays = namedtuple_map(lambda r: r.to(device), data["rays"])
                            pixels = data["pixels"].to(device)
                            timestamps = data["timestamps"].to(device)

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
                                timestamps=timestamps,
                            )
                            mse = F.mse_loss(rgb, pixels)
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)
                            psnrs.append(psnr.item())
                            imageio.imwrite(
                                f"{image_root}/train/depth_{i}.png",
                                depth2img(depth),
                            )
                            imageio.imwrite(
                                f"{image_root}/train/rgb_{i}.png",
                                (rgb.cpu().numpy() * 255).astype(np.uint8),
                            )
                            # break
                        psnr_avg = sum(psnrs) / len(psnrs)
                        print(f"evaluation: psnr_avg={psnr_avg}")

                        metrics_file = f"{image_root}/train/psnr.txt"
                        psnr_str = [f'{p:.5f}\n' for p in psnrs] + [f'mean: {psnr_avg}']
                        with open(metrics_file, 'w') as f:
                            f.writelines(psnr_str)

                    print('save test image')
                    psnrs = []

                    for i in tqdm(range(len(test_dataset))):
                    # for data in test_dataloader:
                        data = test_dataset[i]
                        
                        render_bkgd = data["color_bkgd"].to(device)
                        # rays = data["rays"]
                        rays = namedtuple_map(lambda r: r.to(device), data["rays"])
                        pixels = data["pixels"].to(device)
                        timestamps = data["timestamps"].to(device)
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
                            timestamps=timestamps,
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

                    metrics_file = f"{image_root}/psnr.txt"
                    psnr_str = [f'{p:.5f}\n' for p in psnrs] + [f'mean: {psnr_avg}']
                    with open(metrics_file, 'w') as f:
                        f.writelines(psnr_str)

                    # print('save no move image')

                    # os.makedirs(f'{image_root}/no_move', exist_ok=True)
                    # assert os.path.exists(f'{image_root}/no_move'), f"test images saving path dose not exits! path: {image_root}/no_move"

                    # radiance_field.loose_move = True

                    # for i in tqdm(range(len(test_dataset))):
                    # # for data in test_dataloader:
                    #     data = test_dataset[i]
                    #     render_bkgd = data["color_bkgd"].to(device)
                    #     # rays = data["rays"]
                    #     rays = namedtuple_map(lambda r: r.to(device), data["rays"])
                    #     pixels = data["pixels"].to(device)
                    #     timestamps = data["timestamps"].to(device)

                    #     # rendering
                    #     rgb, acc, depth, _, _ = render_image(
                    #         radiance_field,
                    #         occupancy_grid,
                    #         rays,
                    #         scene_aabb,
                    #         # rendering options
                    #         near_plane=near_plane,
                    #         far_plane=far_plane,
                    #         render_step_size=render_step_size,
                    #         render_bkgd=render_bkgd,
                    #         cone_angle=args.cone_angle,
                    #         alpha_thre=alpha_thre,
                    #         # test options
                    #         test_chunk_size=args.test_chunk_size,
                    #         timestamps=timestamps,
                    #     )
                    #     imageio.imwrite(
                    #         f"{image_root}/no_move/depth_{i}.png",
                    #         (lambda x: (x - x.min()) / (x.max() - x.min()) * 255)(depth.cpu().numpy()).astype(np.uint8),
                    #     )
                    #     imageio.imwrite(
                    #         f"{image_root}/no_move/rgb_{i}.png",
                    #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                    #     )

                # psnr_avg = sum(psnrs) / len(psnrs)
                # print(f"evaluation: psnr_avg={psnr_avg}")
                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()

            step += 1


if __name__ == "__main__":
    args = get_opts()
    main(args)