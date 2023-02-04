import taichi as ti
from taichi.math import vec3
import torch
# from torch.cuda.amp import custom_fwd, custom_bwd

@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]

@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]

@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]

@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]

@ti.kernel
def composite_test(
           sigmas: ti.types.ndarray(ndim=2),
             rgbs: ti.types.ndarray(ndim=2),
          t_start: ti.types.ndarray(ndim=2),
            t_end: ti.types.ndarray(ndim=2),
        pack_info: ti.types.ndarray(ndim=2),
    alive_indices: ti.types.ndarray(ndim=1),
      T_threshold: float,
  aplha_threshold: float,
          opacity: ti.types.ndarray(ndim=2),
            depth: ti.types.ndarray(ndim=2),
              rgb: ti.types.ndarray(ndim=2)):
    
    ti.loop_config(block_dim=256)
    for n in alive_indices:
        start_idx = pack_info[n, 0]
        steps = pack_info[n, 1]
        ray_idx = alive_indices[n]
        if steps == 0:
            alive_indices[n] = -1
        else:
            T = 1 - opacity[ray_idx, 0]

            rgb_temp = vec3(0.0)
            depth_temp = 0.0
            opacity_temp = 0.0

            for s in range(steps):
              s_n = start_idx + s
              t1 = t_start[s_n, 0]
              t2 = t_end[s_n, 0]
              delta = t2 - t1
              a = 1.0 - ti.exp(-sigmas[s_n, 0]*delta)

              if a > aplha_threshold:

                w = a * T
                tmid = (t1 + t2) / 2
                rgbs_vec3 = vec3(
                  rgbs[s_n, 0], rgbs[s_n, 1], rgbs[s_n, 2]
                )
                rgb_temp += w * rgbs_vec3
                depth_temp += w * tmid
                opacity_temp += w
                T *= 1.0 - a

                if T <= T_threshold:
                    alive_indices[n] = -1
                    break

            rgb[ray_idx, 0] += rgb_temp[0]
            rgb[ray_idx, 1] += rgb_temp[1]
            rgb[ray_idx, 2] += rgb_temp[2]
            depth[ray_idx, 0] += depth_temp
            opacity[ray_idx, 0] += opacity_temp


data_type = ti.f32

@ti.kernel
def composite_train_fw(
           sigmas: ti.template(),
             rgbs: ti.template(),
           deltas: ti.template(),
               ts: ti.template(),
           rays_a: ti.template(),
      T_threshold: ti.template(),
    total_samples: ti.template(),
          opacity: ti.template(),
            depth: ti.template(),
              rgb: ti.template(),
               ws: ti.template(),
                B: ti.i32):

    for n in opacity:
        ray_idx = rays_a[n, 0]
        start_idx = rays_a[n, 1]
        N_samples = rays_a[n, 2]
        thr = T_threshold[0]

        T = 1.0
        samples = 0
        # while samples<N_samples:
        for sample in range(N_samples):
          if T>=thr:
            s = start_idx + sample
            a = 1.0 - ti.exp(-sigmas[s]*deltas[s])
            w = a*T

            rgb[ray_idx, 0] += w*rgbs[s, 0]
            rgb[ray_idx, 1] += w*rgbs[s, 1]
            rgb[ray_idx, 2] += w*rgbs[s, 2]
            depth[ray_idx] += w*ts[s]
            opacity[ray_idx] += w
            ws[s] = w
            T *= 1.0-a

            samples += 1

        total_samples[ray_idx] = samples



class VolumeRender(torch.nn.Module):

    def __init__(self, ):
        super(VolumeRender, self).__init__()
        # samples level
        self.sigmas_fields = ti.field(dtype=ti.f16, shape=(8192*1024, 3), needs_grad=True)
        self.rgbs_fields = ti.field(dtype=ti.f16, shape=(8192*1024, 3), needs_grad=True)
        self.deltas_fields = ti.field(dtype=ti.f16, shape=(8192*1024, 3), needs_grad=True)
        self.ts_fields = ti.field(dtype=ti.f16, shape=(8192*1024, 3), needs_grad=True)
        # rays level
        self.rays_a_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)
        self.total_samples_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)
        self.opacity_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)
        self.depth_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)
        self.rgb_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)
        self.ws_fields = ti.field(dtype=ti.f16, shape=(8192, 3), needs_grad=True)


        class _module_function(torch.autograd.Function):
            @staticmethod
            # @custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, input_dir):
                # If no output gradient is provided, no need to
                # automatically materialize it as torch.zeros.

                # ctx.set_materialize_grads(False) # maybe not needed
                input_dir = input_dir.to(torch.float16)
                ctx.save_for_backward(input_dir)
                output_embedding = torch.zeros(
                    input_dir.shape[0], 16, dtype=torch.float16, device=input_dir.device
                )

                # ti.sync()
                torch2ti(self.input_fields, input_dir.contiguous())
                composite_train_fw(
                    self.input_fields,
                    self.output_fields,
                    input_dir.shape[0]
                )
                ti2torch(self.output_fields, output_embedding)
                # ti.sync()

                return output_embedding

            @staticmethod
            # @custom_bwd
            def backward(ctx, doutput):
                if doutput is None:
                    print("all None")
                    return None

                if not doutput.is_cuda:
                    print("TAICHI WARNING: doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
                    doutput = doutput.cuda()

                input_dir = ctx.saved_tensors
                grad = torch.zeros_like(input_dir)

                # ti.sync()
                torch2ti_grad(self.output_fields, doutput.contiguous())
                dir_encoder.grad(
                    self.input_fields, 
                    self.output_fields, 
                    doutput.shape[0]
                )
                ti2torch_grad(self.input_fields, grad)
                # ti.sync()

                return grad

        self._module_function = _module_function


    def forward(self, positions):

        embedding = self._module_function.apply(positions)

        return embedding