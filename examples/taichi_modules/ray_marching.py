import taichi as ti
from taichi.math import vec3
from utils import calc_dt

@ti.kernel
def raymarching_test_kernel(
              rays_o: ti.types.ndarray(field_dim=2),
              rays_d: ti.types.ndarray(field_dim=2),
              hits_t: ti.types.ndarray(field_dim=2),
       alive_indices: ti.types.ndarray(field_dim=1),
    density_bitfield: ti.types.ndarray(field_dim=1),
    cascades: int, grid_size: int, scale: float, exp_step_factor: float,
           N_samples: int, max_samples: int,
                xyzs: ti.types.ndarray(field_dim=2),
                dirs: ti.types.ndarray(field_dim=2),
              deltas: ti.types.ndarray(field_dim=1),
                  ts: ti.types.ndarray(field_dim=1),
       N_eff_samples: ti.types.ndarray(field_dim=1),):

    for n in alive_indices:
        r = alive_indices[n]
        grid_size3 = grid_size**3
        grid_size_inv = 1.0/grid_size

        ray_o = vec3(rays_o[r, 0], rays_o[r, 1], rays_o[r, 2])
        ray_d = vec3(rays_d[r, 0], rays_d[r, 1], rays_d[r, 2])
        d_inv = 1.0/ray_d

        t = hits_t[r, 0]
        t2 = hits_t[r, 1]

        s = 0

        while (0<=t) & (t<t2) & (s<N_samples):
            xyz = ray_o + t*ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            # mip = ti.max(mip_from_pos(xyz, cascades),
            #             mip_from_dt(dt, grid_size, cascades))


            mip_bound = 0.5
            mip_bound_inv = 1/mip_bound

            nxyz = ti.math.clamp(0.5*(xyz*mip_bound_inv+1)*grid_size, 0.0, grid_size-1.0)
            # nxyz = ti.ceil(nxyz)

            idx =  calc_dt(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx//8)] & (1 << ti.u32(idx%8))

            if occ:
                xyzs[n, s, 0] = xyz[0]
                xyzs[n, s, 1] = xyz[1]
                xyzs[n, s, 2] = xyz[2]
                dirs[n, s, 0] = ray_d[0]
                dirs[n, s, 1] = ray_d[1]
                dirs[n, s, 2] = ray_d[2]
                ts[n, s] = t
                deltas[n, s] = dt
                t += dt
                hits_t[r, 0] = t
                s += 1

            else:
                txyz = (((nxyz+0.5+0.5*ti.math.sign(ray_d))*grid_size_inv*2-1)*mip_bound-xyz)*d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)

        N_eff_samples[n] = s