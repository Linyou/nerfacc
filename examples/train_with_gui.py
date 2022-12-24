"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.custom_ngp import NGPDradianceField
from utils import render_image, set_random_seed, get_opts
from custom_utils import custom_render_image
from visdom import Visdom
from radiance_fields.mlp import DNeRFRadianceField

from nerfacc import ContractionType, OccupancyGrid, loss_distortion
# from nerfacc.taichi_modules import distortion
from warmup_scheduler import GradualWarmupScheduler

import taichi as ti
from show_gui import NGPGUI, write_buffer
# ti.init(arch=ti.cuda, offline_cache=True)

def total_variation_loss(x):
    # Get resolution
    tv_x = torch.pow(x[1:,:,:,:]-x[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(x[:,1:,:,:]-x[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()

    return (tv_x.mean() + tv_y.mean() + tv_z.mean())/3

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
    # max_steps = 20000
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
        use_feat_predict=args.use_feat_predict,
        use_weight_predict=args.use_weight_predict,
    ).to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=lr, eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        # milestones=[max_steps // 2, max_steps * 9 // 10],
        gamma=0.33,
    )

    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler)


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

    # generate log fir for test image and psnr
    feat_dir = 'pf' if args.use_feat_predict else 'nopf'
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
    }

    ngp = NGPGUI(render_kwargs=gui_args)

    ti.init(arch=ti.cuda, offline_cache=True)

    W, H = ngp.W, ngp.H
    final_pixel = ti.Vector.field(3, dtype=float, shape=(W, H))

    window = ti.ui.Window('Window Title', (W, H),)
    canvas = window.get_canvas()
    gui = window.get_gui()


    # GUI controls variables
    last_orbit_x = None
    last_orbit_y = None

    last_inner_x = None
    last_inner_y = None

    timestamps_gui = 0.0
    last_timestamps = 0.0

    playing = False

    test_view = 0
    train_view = 0
    last_train_view = 0
    last_test_view = 0

    train_views_size = ngp.train_dataset.images.shape[0]-1
    test_views_size = ngp.test_dataset.images.shape[0]-1

    # training
    step = 0
    tic = time.time()
    # radiance_field.loose_move = True
    while window.running:
        for i in range(len(train_dataset)):
            if step != max_steps:
                radiance_field.train()
                data = train_dataset[i]

                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                timestamps = data["timestamps"]
                idx = data["idx"]

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
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if args.use_feat_predict or args.use_weight_predict or args.distortion_loss:
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
                        rgb, acc, depth, n_rendering_samples = render_image(
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

                    rec_loss = rec_loss_fn(rgb[alive_ray_mask], pixels[alive_ray_mask], reduction='none')
                        
                    loss = rec_loss.mean()

                    if args.use_opacity_loss:
                        o = acc[alive_ray_mask]
                        loss_o = (-o*torch.log(o)).mean()*1e-2
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
                        for (weight, t_starts, t_ends, ray_indices, packed_info) in extra_info:
                            alive_samples_mask = alive_ray_mask[ray_indices.long()]
                            # loss_extra += F.smooth_l1_loss(predict[alive_samples_mask], hash_feat[alive_samples_mask])*0.1
                            loss_distor += distortion(packed_info, weight, t_starts, t_ends)[alive_ray_mask].mean() * 1e-2
                    
                        loss += loss_distor

                optimizer.zero_grad()
                # do not unscale it because we are using Adam.
                grad_scaler.scale(loss).backward()
                # grad_scaler.unscale_(optimizer)
                if args.use_feat_predict or args.use_weight_predict:
                    radiance_field.log_grad(step)
                # optimizer.step()
                grad_scaler.step(optimizer)
                scale = grad_scaler.get_scale()
                grad_scaler.update()

                if not scale > grad_scaler.get_scale():
                    scheduler.step()

                # scheduler.step()

                if step % 1000 == 0:
                    elapsed_time = time.time() - tic
                    loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    loss_log = f"loss={loss:.5f} | "
                    if args.use_feat_predict:
                        loss_log += f"extra={loss_extra:.7f} | "

                    # loss_log += f"time={loss_time:.7f} | "

                    if args.use_weight_predict:
                        loss_log += f"weight={loss_weight:.7f} | "
                    if args.distortion_loss:
                        loss_log += f"distor={loss_distor:.7f} | "
                    if args.use_opacity_loss:
                        loss_log += f"opac={loss_o:.7f} ||"
                    print(
                        f"elapsed_time={elapsed_time:.2f}s | step={step} ||",
                        loss_log,
                        f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                        f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                    )

            step += 1

            ngp.radiance_field.eval()

            if window.is_pressed(ti.ui.RMB):
                curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
                if last_orbit_x is None or last_orbit_y is None:
                    last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
                else:
                    dx = curr_mouse_x - last_orbit_x
                    dy = curr_mouse_y - last_orbit_y
                    ngp.cam.orbit(dx, -dy)
                    last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y

            elif window.is_pressed(ti.ui.MMB):
                curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
                if last_inner_x is None or last_inner_y is None:
                    last_inner_x, last_inner_y = curr_mouse_x, curr_mouse_y
                else:
                    dx = curr_mouse_x - last_inner_x
                    dy = curr_mouse_y - last_inner_y
                    ngp.cam.inner_orbit(dx, -dy)
                    last_inner_x, last_inner_y = curr_mouse_x, curr_mouse_y
            else:
                last_orbit_x = None
                last_orbit_y = None

                last_inner_x = None
                last_inner_y = None

            if window.is_pressed('w'):
                ngp.cam.scale(0.2)
            if window.is_pressed('s'):
                ngp.cam.scale(-0.2)
            if window.is_pressed('a'):
                ngp.cam.pan(-500, 0.)
            if window.is_pressed('d'):
                ngp.cam.pan(500, 0.)
            if window.is_pressed('e'):
                ngp.cam.pan(0., -500)
            if window.is_pressed('q'):
                ngp.cam.pan(0., 500)

            with gui.sub_window("Options", 0.01, 0.01, 0.4, 0.3) as w:
                ngp.cam.rotate_speed = w.slider_float('rotate speed', ngp.cam.rotate_speed, 0.1, 1.)

                timestamps_gui = w.slider_float('timestamps', timestamps_gui, 0., 1.)
                if last_timestamps != timestamps_gui:
                    last_timestamps = timestamps_gui
                    ngp.timestamps[0] = timestamps_gui

                if gui.button('play'):
                    playing = True
                if gui.button('pause'):
                    playing = False

                if playing:
                    timestamps += 0.01
                    if timestamps > 1.0:
                        timestamps = 0.0

                ngp.img_mode = w.checkbox("show depth", ngp.img_mode)

                train_view = w.slider_int('train view', train_view, 0, train_views_size)
                test_view = w.slider_int('test view', test_view, 0, test_views_size)

                if last_train_view != train_view:
                    last_train_view = train_view
                    ngp.cam.reset(ngp.train_dataset.camtoworlds[train_view])

                if last_test_view != test_view:
                    last_test_view = test_view
                    ngp.cam.reset(ngp.test_dataset.camtoworlds[test_view])

                w.text(f'samples per rays: {ngp.mean_samples} s/r')
                w.text(f'render times: {1000*ngp.dt:.2f} ms')

            render_buffer = ngp.render_frame()
            write_buffer(W, H, render_buffer, final_pixel)
            canvas.set_image(final_pixel)
            window.show()


if __name__ == "__main__":
    args = get_opts()
    main(args)