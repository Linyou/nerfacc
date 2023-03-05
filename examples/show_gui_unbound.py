import torch
import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation as R
import time
import cv2

from custom_utils import get_opts, get_feat_dir_and_loss, get_ngp_args, get_ngp_multires_args

import math
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from radiance_fields.custom_ngp import NGPDradianceField
from utils import render_image, Rays, render_image_test_v3
from visdom import Visdom

import taichi as ti

import torch.distributed as dist

from nerfacc import ContractionType, OccupancyGrid

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img.astype(np.float32)

import warnings; warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_rays(K, pose, width, height, opengl=True, hyber=False):

    if hyber:
        K = K[0]

    x, y = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float16),
        torch.arange(height, device=device, dtype=torch.float16),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    # generate rays
    c2w = pose[None, ...]  # (num_rays, 3, 4)
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1]
                * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [num_rays, 3]

    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # print(c2w[:, :3, -1])
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    # origins = torch.reshape(origins, (height, width, 3)).to(torch.float16)
    # viewdirs = torch.reshape(viewdirs, (height, width, 3)).to(torch.float16)
    # directions = torch.reshape(directions, (height, width, 3)).to(torch.float16)
    origins = torch.reshape(origins, (height, width, 3))
    viewdirs = torch.reshape(viewdirs, (height, width, 3))
    directions = torch.reshape(directions, (height, width, 3))

    rays = Rays(origins=origins, viewdirs=viewdirs)

    return rays


class OrbitCamera:
    def __init__(self, K, img_wh, pose, r, center=None):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        pose_np = pose.cpu().numpy()
        if center is not None:
            self.center = center
        else:
            self.center = np.zeros(3)
        self.rot = np.eye(3)
        # self.center = pose_np[20][:3, 3]
        # self.rot = pose_np[50][:3, :3]
        self.res_defalut = pose_np[20]
        self.rotate_speed = 0.8

        self.inner_rot = np.eye(3)

    def reset(self, pose=None, aabb=None):
        self.rot = np.eye(3)
        self.inner_rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 1.5
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] += self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # inner rotate
        rot = np.eye(4)
        rot[:3, :3] = self.inner_rot
        res = res @ rot
        # translate
        res[:3, 3] += self.center
        # return res

        # print("res_defalut: ", self.res_defalut)
        # print("res: ", res)
        # return self.res_defalut
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(-100*self.rotate_speed * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100*self.rotate_speed * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def inner_orbit(self, dx, dy):
        rotvec_x = self.inner_rot[:, 1] * np.radians(-100*self.rotate_speed * dx)
        rotvec_y = self.inner_rot[:, 0] * np.radians(-100*self.rotate_speed * dy)
        self.inner_rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                         R.from_rotvec(rotvec_x).as_matrix() @ \
                         self.inner_rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, radius=1.5, render_kwargs=None, use_time=False, opengl=True, hyber=False):

        device = "cuda:0"

        self.train_dataset = render_kwargs['train_dataset']
        self.test_dataset = render_kwargs['test_dataset']

        self.radiance_field = render_kwargs['radiance_field']
        self.occupancy_grid = render_kwargs['occupancy_grid']

        self.contraction_type = render_kwargs['contraction_type']
        self.scene_aabb = render_kwargs['scene_aabb']
        self.near_plane = render_kwargs['near_plane']
        self.far_plane = render_kwargs['far_plane']
        self.render_step_size = render_kwargs['render_step_size']
        self.alpha_thre = render_kwargs['alpha_thre']

        self.cone_angle = render_kwargs['cone_angle']
        # self.test_chunk_size = render_kwargs['test_chunk_size']
        self.render_bkgd = render_kwargs['render_bkgd']
        self.args_aabb = render_kwargs['args_aabb']

        self.get_rays_func = get_rays


        self.radiance_field.eval()
        self.occupancy_grid.eval()

        K, img_wh, pose = self.test_dataset.K, (self.test_dataset.width, self.test_dataset.height), self.test_dataset.camtoworlds

        self.cam = OrbitCamera(K, img_wh, pose, r=radius)
        self.W, self.H = img_wh

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        if use_time:
            self.timestamps = torch.tensor([0.0], device=device)
        else:
            self.timestamps = None
        self.max_samples = 100

        self.opengl = opengl
        self.hyber = hyber

    @torch.no_grad()
    def render_cam(self, cam):
        t = time.time()
        # print(cam.pose)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            rays = self.get_rays_func(cam.K, torch.cuda.FloatTensor(cam.pose), self.W, self.H, opengl=self.opengl, hyber=self.hyber)
            # ngp test rendering v3 (rays level)
            rgb, _, depth, n_rendering_samples = render_image_test_v3(
                self.max_samples,
                self.radiance_field,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                # rendering options
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                render_bkgd=self.render_bkgd,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
                # test options
                # test_chunk_size=self.test_chunk_size,
                timestamps=self.timestamps,
                # early_stop_eps=0.01,
            )

            depth = depth.squeeze(-1)
            self.dt = time.time()-t
            self.mean_samples = n_rendering_samples/(self.W * self.H)

            if self.img_mode == 0:
                return rgb
            elif self.img_mode == 1:
                return depth2img(depth)/255.0

    def render_frame(self):
        return self.render_cam(self.cam)


@ti.kernel
def write_buffer(W:ti.i32, H:ti.i32, x: ti.types.ndarray(), final_pixel:ti.template()):
    for i, j in ti.ndrange(W, H):
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[H-1-j, i, p]


@torch.no_grad()
def rev_param(ngp, is_async=False):
    rad_params = ngp.radiance_field.parameters()
    for pn, p in enumerate(rad_params):
        # recv_temp = torch.zeros_like(p)
        if is_async:
            dist.irecv(tensor=p[:], src=0, tag=pn)
        else:
            dist.recv(tensor=p[:], src=0, tag=pn)
        # p[:] = recv_temp

    occ_params = ngp.occupancy_grid.parameters()
    for pn, p in enumerate(occ_params):
        recv_temp = torch.zeros_like(p)
        if is_async:
            dist.irecv(tensor=recv_temp, src=0, tag=50+pn)
        else:
            dist.recv(tensor=recv_temp, src=0, tag=50+pn)
        p[:] = recv_temp


def render_gui(ngp=None, args=None):

    ti.init(arch=ti.cuda, offline_cache=True)

    if args is not None:
        ngp = NGPGUI(args=args)
        rev_param(ngp, is_async=False)


    W, H = ngp.W, ngp.H
    print("W:", type(W))
    final_pixel = ti.Vector.field(3, dtype=float, shape=(W, H))

    window = ti.ui.Window('Window Title', (W, H),)
    canvas = window.get_canvas()
    gui = window.get_gui()


    # GUI controls variables
    last_orbit_x = None
    last_orbit_y = None

    last_inner_x = None
    last_inner_y = None

    timestamps = 0.0
    last_timestamps = 0.0

    playing = False

    test_view = 0
    train_view = 0
    last_train_view = 0
    last_test_view = 0

    ref_c2w = ngp.train_dataset.camtoworlds[train_view].cpu().numpy()

    train_views_size = ngp.train_dataset.images.shape[0]-1
    test_views_size = ngp.test_dataset.images.shape[0]-1

    ngp.radiance_field.eval()
    while window.running:

        if args is not None:
            rev_param(ngp, is_async=True)

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

            timestamps = w.slider_float('timestamps', timestamps, 0., 1.)
            if last_timestamps != timestamps:
                last_timestamps = timestamps
                ngp.timestamps[0] = timestamps

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
                ref_c2w = ngp.train_dataset.camtoworlds[train_view].cpu().numpy()
                ngp.cam.reset(
                    ngp.train_dataset.camtoworlds[train_view],
                    aabb=ngp.args_aabb
                )

            if last_test_view != test_view:
                last_test_view = test_view
                ngp.cam.reset(
                    ngp.test_dataset.camtoworlds[test_view],
                    aabb=ngp.args_aabb
                )

            cam_pose = ngp.cam.pose
            w.text(f'samples per rays: {ngp.mean_samples} s/r')
            w.text(f'render times: {1000*ngp.dt:.2f} ms')
            w.text(f'radius: {ngp.cam.radius}')
            w.text(f'pose:')
            w.text(f'{ngp.cam.rot[0]}')
            w.text(f'{ngp.cam.rot[1]}')
            w.text(f'{ngp.cam.rot[2]}')
            w.text(f'c2w:')
            w.text(f'{cam_pose[0]}')
            w.text(f'{cam_pose[1]}')
            w.text(f'{cam_pose[2]}')
            w.text(f'{cam_pose[2]}')
            w.text(f'ref c2w:')
            w.text(f'{ref_c2w[0]}')
            w.text(f'{ref_c2w[1]}')
            w.text(f'{ref_c2w[2]}')
            w.text(f'{ref_c2w[2]}')

        render_buffer = ngp.render_frame()
        write_buffer(W, H, render_buffer, final_pixel)
        canvas.set_image(final_pixel)
        window.show()


if __name__ == "__main__":

    hparams = get_opts()
    args= get_ngp_args(hparams)
    ngp = NGPGUI(render_kwargs=args)

    # ngp = NGPGUI(args=hparams)

    render_gui(ngp)

