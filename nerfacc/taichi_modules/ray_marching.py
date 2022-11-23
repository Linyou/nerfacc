from re import T
import taichi as ti
from taichi.math import ivec3, vec3
import torch

vec6 = ti.types.vector(6, ti.f32)
thread = 512

ContractionType_AABB = 0

@ti.func
def calc_dt(t, cone_angle, dt_min, dt_max):
    return ti.math.clamp(t*cone_angle, dt_min, dt_max)

@ti.func
def grid_idx_at(xyz_unit, grid_res):
    ixyz_ = ivec3(xyz_unit * ti.cast(grid_res, ti.f32))
    ixyz = ti.math.clamp(ixyz_, ivec3(0,0,0), grid_res - 1)
    grid_offset = ivec3(grid_res.y * grid_res.z, grid_res.z, 1);
    idx = ixyz.dot(grid_offset)
    return idx
    # return ixyz

@ti.func
def roi_to_unit(xyz, roi_min, roi_max):
    return (xyz - roi_min) / (roi_max - roi_min)

@ti.func
def grid_occupied_at(xyz, roi_min, roi_max, type, grid_res):
    return_value = -1
    if not ((type == ContractionType_AABB) and \
       (xyz.x < roi_min.x or xyz.x > roi_max.x or \
        xyz.y < roi_min.y or xyz.y > roi_max.y or \
        xyz.z < roi_min.z or xyz.z > roi_max.z)):

        # uint
        xyz_unit = roi_to_unit(xyz, roi_min, roi_max)
        idx = grid_idx_at(xyz_unit, grid_res)
        return_value = idx

    return return_value


@ti.func
def distance_to_next_voxel(xyz, dir, inv_dir, roi_min, roi_max, grid_res):
    _occ_res = ti.cast(grid_res, ti.f32)
    _xyz = roi_to_unit(xyz, roi_min, roi_max) * _occ_res
    txyz = ((ti.floor(_xyz+0.5+0.5*ti.math.sign(dir)) - _xyz) * inv_dir) / _occ_res * (roi_max - roi_min)
    return ti.max(txyz.min(), 0.0)


@ti.func
def advance_to_next_voxel(t, dt_min, xyz, dir, inv_dir, roi_min, roi_max, grid_res):
    t_target = t + distance_to_next_voxel(xyz, dir, inv_dir, roi_min, roi_max, grid_res)
    _t = t + dt_min
    while _t < t_target:
        _t += dt_min
    return _t

@ti.kernel
def ray_marching_kernel(
    # rays
    n_rays: ti.i32,
    rays_o: ti.types.ndarray(field_dim=2),
    rays_d: ti.types.ndarray(field_dim=2),
    t_min: ti.types.ndarray(field_dim=1),
    t_max: ti.types.ndarray(field_dim=1),
    # occupancy grid & contraction
    roi: ti.types.ndarray(field_dim=1),
    grid_res: ivec3,
    grid_binary: ti.types.ndarray(field_dim=3),
    type: ti.i32,
    # sampling
    step_size: ti.f32,
    cone_angle: ti.f32,
    # output
    num_steps: ti.types.ndarray(field_dim=1),
    # blocks_thread: ti.i32
    ):

    # ti.loop_config(block_dim=thread)
    for i in range(n_rays):
        # if i < n_rays:

            # locate
            origin = vec3(rays_o[i, 0], rays_o[i, 1], rays_o[i, 2])
            dir = vec3(rays_d[i, 0], rays_d[i, 1], rays_d[i, 2])
            near = t_min[i]
            far = t_max[i]

            inv_dir = 1.0 / dir

            roi_min = vec3(roi[0], roi[1], roi[2])
            roi_max = vec3(roi[3], roi[4], roi[5])


            dt_min = step_size
            dt_max = 1e10

            j = 0
            t0 = near
            dt = calc_dt(t0, cone_angle, dt_min, dt_max)
            t1 = t0 + dt
            t_mid = (t0+t1)*0.5

            while t_mid < far:
                xyz = origin + t_mid * dir
                idx = grid_occupied_at(xyz, roi_min, roi_max, type, grid_res)
                if idx > 0 and grid_binary[idx] == 1:
                    j+=1
                    t0 = t1
                    t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max)
                    t_mid = (t0+t1)*0.5
                else:
                    if type == ContractionType_AABB:
                        t_mid = advance_to_next_voxel(t_mid, dt_min, xyz, dir, inv_dir, roi_min, roi_max, grid_res)
                        dt = calc_dt(t_mid, cone_angle, dt_min, dt_max)
                        t0 = t_mid - dt*0.5
                        t1 = t_mid + dt*0.5
                    else:
                        t0 = t1
                        t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max)
                        t_mid = (t0+ t1)*0.5


            num_steps[i] = j


@ti.kernel
def ray_marching_kernel_pack(
    # rays
    n_rays: ti.i32,
    rays_o: ti.types.ndarray(field_dim=2),
    rays_d: ti.types.ndarray(field_dim=2),
    t_min: ti.types.ndarray(field_dim=1),
    t_max: ti.types.ndarray(field_dim=1),
    # occupancy grid & contraction
    roi: ti.types.ndarray(field_dim=1),
    grid_res: ivec3,
    grid_binary: ti.types.ndarray(field_dim=3),
    type: ti.i32,
    # sampling
    step_size: ti.f32,
    cone_angle: ti.f32,
    packed_info: ti.types.ndarray(field_dim=2),
    # output
    t_starts: ti.types.ndarray(field_dim=2),
    t_ends: ti.types.ndarray(field_dim=2),
    # blocks_thread: ti.i32
    ):

    # ti.loop_config(block_dim=thread)
    for i in range(n_rays):
        # if i < n_rays:

            base = packed_info[i, 0]
            # steps = packed_info[i*2 + 1]

            # locate
            origin = vec3(rays_o[i, 0], rays_o[i, 1], rays_o[i, 2])
            dir = vec3(rays_d[i, 0], rays_d[i, 1], rays_d[i, 2])
            near = t_min[i]
            far = t_max[i]

            inv_dir = 1.0 / dir

            roi_min = vec3(roi[0], roi[1], roi[2])
            roi_max = vec3(roi[3], roi[4], roi[5])


            dt_min = step_size
            dt_max = 1e10

            j = 0
            t0 = near
            dt = calc_dt(t0, cone_angle, dt_min, dt_max)
            t1 = t0 + dt
            t_mid = (t0+t1)*0.5

            while t_mid < far:
                xyz = origin + t_mid * dir
                idx = grid_occupied_at(xyz, roi_min, roi_max, type, grid_res)
                if idx > 0 and grid_binary[idx] == 1:
                    t_starts[base+j, 0] = t0
                    t_ends[base+j, 0] = t1
                    j+=1
                    t0 = t1
                    t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max)
                    t_mid = (t0+t1)*0.5
                else:
                    if type == ContractionType_AABB:
                        t_mid = advance_to_next_voxel(t_mid, dt_min, xyz, dir, inv_dir, roi_min, roi_max, grid_res)
                        dt = calc_dt(t_mid, cone_angle, dt_min, dt_max)
                        t0 = t_mid - dt*0.5
                        t1 = t_mid + dt*0.5
                    else:
                        t0 = t1
                        t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max)
                        t_mid = (t0+ t1)*0.5


def ray_marching(    
    # rays
    rays_o,
    rays_d,
    t_min,
    t_max,
    # occupancy grid & contraction
    roi,
    grid_binary,
    type,
    # sampling
    step_size,
    cone_angle):

    n_rays = rays_o.size(0)
    grid_res = ivec3(grid_binary.size(0), grid_binary.size(1), grid_binary.size(2))

    # blocks_thread = ((n_rays + thread - 1) // thread) * thread
    num_steps = torch.empty(n_rays, dtype=torch.int32, device=rays_o.device)

    grid_binary = grid_binary.to(torch.uint8).flatten()
    type = int(type)

    ray_marching_kernel(
        # rays
        n_rays,
        rays_o,
        rays_d,
        t_min,
        t_max,
        # occupancy grid & contraction
        roi,
        grid_res,
        grid_binary,
        type,
        # sampling
        step_size,
        cone_angle,
        # output
        num_steps,
        # blocks_thread
    )

    cum_steps = num_steps.cumsum(0, dtype=torch.int32)
    packed_info = torch.stack([cum_steps - num_steps, num_steps], dim=1)

    total_steps = cum_steps[-1]

    t_starts = torch.empty(total_steps, 1, dtype=torch.float32, device=rays_o.device)
    t_ends = torch.empty(total_steps, 1, dtype=torch.float32, device=rays_o.device)

    ray_marching_kernel_pack(
        # rays
        n_rays,
        rays_o,
        rays_d,
        t_min,
        t_max,
        # occupancy grid & contraction
        roi,
        grid_res,
        grid_binary,
        type,
        # sampling
        step_size,
        cone_angle,
        packed_info,
        # output
        t_starts,
        t_ends,
        # blocks_thread
    )

    return packed_info, t_starts, t_ends