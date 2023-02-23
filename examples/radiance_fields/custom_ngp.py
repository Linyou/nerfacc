from typing import Callable, List, Union

import math
import functools
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from .ngp import NGPradianceField, trunc_exp
from .mlp import SinusoidalEncoder, MLP

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x, mask


MOVING_STEP = 1/4096
class SinusoidalEncoderWithExp(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg), self.x_dim],
        )
        x_var_b = torch.reshape(
            (x_var[Ellipsis, None, :] * self.scales[:, None]),
            list(x_var.shape[:-1]) + [(self.max_deg - self.min_deg)],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1)) * torch.exp(-6e1 * x_var_b)[..., None]
        latent = torch.reshape(latent, list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim * 2])
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class CanonicalWarper(torch.nn.Module):
    def __init__(
        self,
        num_dim: int = 3,
        moving_step: float = MOVING_STEP,
        use_dive_offsets: bool = False,
    ) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.MOVING_STEP = moving_step
        self.use_dive_offsets = use_dive_offsets

        if self.use_dive_offsets:
            out_dim = self.num_dim*2
        else:
            out_dim = self.num_dim
        self.fre = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "degree": 4,
                    },
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                ],
            },
        )
        self.posi_encoder = tcnn.Network(
            n_input_dims=self.fre.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3,
            },
        )
        # self.posi_encoder = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=num_dim,
        #     encoding_config={
        #         "otype": "Composite",
        #         "nested": [
        #             {
        #                 "n_dims_to_encode": 3, # Spatial dims
        #                 "otype": "TriangleWave",
        #                 "n_frequencies": 12
        #             },
        #             # {
        #             #     "n_dims_to_encode": 3,
        #             #     "otype": "HashGrid",
        #             #     "n_levels": 16,
        #             #     "n_features_per_level": 2,
        #             #     "log2_hashmap_size": 19,
        #             #     "base_resolution": 16,
        #             #     "per_level_scale": 1.4472692012786865,
        #             # }
        #         ]
        #     },
        #     n_output_dims=out_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 3,
        #     },
        # )
        # self.posi_encoder = MLP(
        #     input_dim=3,
        #     output_dim=3,
        #     net_depth=3,
        #     net_width=64,
        #     skip_layer=4,
        #     output_enabled=True,
        # )

    def forward(self, x, timestamps, time_id=None, aabb=None, mask=None):
        if not x.size(0) == 0:
        # if mask != None:
        #     x_ori = x
        #     x = x[mask]
        #     timestamps = timestamps[mask]

        # offsets = self.posi_encoder(torch.cat([x, timestamps], dim=-1))
        # if self.use_dive_offsets:
        #     grid_move = offsets[:, 0:3]*self.MOVING_STEP
        #     fine_move = (torch.special.expit(offsets[:, 3:])*2 - 1)*self.MOVING_STEP
        #     move = grid_move + fine_move
        #     # fine_move = F.tanh(offsets[:, 3:])*self.MOVING_STEP
        # else:
        #     grid_move = offsets*self.MOVING_STEP
        #     fine_move = torch.zeros_like(grid_move)

        #     move = grid_move 

        # if mask != None:
        #     x_ori[mask] += move
        #     x = x_ori
        #     move_norm = torch.zeros_like(x[:, 0:1])
        #     move_norm[mask] += move.norm(dim=-1)[:, None]
        # else:
        #     x = move + x
        #     move_norm = move.norm(dim=-1)[:, None]
        # print("x shape: ", x.shape)
        # if not x.size(0) == 0:
            offsets = self.posi_encoder(self.fre(x))
            x = offsets + x
            print("grad x: ", x.requires_grad)
            print("grad offsets: ", offsets.requires_grad)
        else:
            print("zeros x")
            offsets = torch.zeros_like(x)
            x = offsets + x

        # else:
        #     move = torch.zeros_like(x)
        #     move_norm = torch.zeros_like(x[:, 0:1])
        #     grid_move = torch.zeros_like(x)
        #     fine_move = torch.zeros_like(x)


        # return x.view(-1, self.num_dim), move, move_norm, grid_move, fine_move
        return x, None, None, None, None




class NGPDradianceField(NGPradianceField):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        # dnerf,
        aabb: Union[torch.Tensor, List[float]],
        logger = None,
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 15,
        base_resolution: int = 16,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        use_feat_predict: bool = False,
        use_weight_predict: bool = False,
        moving_step: float = MOVING_STEP,
        use_dive_offsets: bool = False,
        use_time_embedding: bool = False,
        use_time_attenuation: bool = False,
        hash_level: int = 0,
    ) -> None:
        super().__init__(
            aabb, 
            num_dim, 
            use_viewdirs, 
            density_activation, 
            unbounded, 
            geo_feat_dim, 
            n_levels, 
            log2_hashmap_size
        )

        self.return_extra = False
        self.loose_move = False

        # self.aabb = self.aabb.to(torch.float16)
        # b = np.exp(np.log(2048/16)/(n_levels-1))
        # per_level_scale = b
        if hash_level == 0:
            per_level_scale = 1.3195079565048218 # 1024
        elif hash_level == 1:
            per_level_scale = 1.3819128274917603 # 2048
        elif hash_level == 2:
            per_level_scale = 1.4472692012786865 # 4096

        per_level_scale = math.exp(
            math.log(4096 * 1 / base_resolution) / (n_levels - 1)
        )  # 1.4472692012786865

        print('--NGPDradianceField configuration--')
        print(f'  moving_step: {moving_step}')
        print(f'  hash_level: {hash_level}, b: {per_level_scale:6f}')
        print(f'  use_dive_offsets: {use_dive_offsets}')
        print(f'  use_feat_predict: {use_feat_predict}')
        print(f'  use_weight_predict: {use_weight_predict}')
        print(f'  use_time_embedding: {use_time_embedding}')
        print(f'  use_time_attenuation: {use_time_attenuation}')
        print('-----------------------------------')


        self.use_feat_predict = use_feat_predict
        self.use_weight_predict = use_weight_predict
        self.use_time_embedding = use_time_embedding
        self.use_time_attenuation = use_time_attenuation

        # per_level_scale = 1.4472692012786865
        # per_level_scale = 1.3819128274917603
        # per_level_scale = 1.3195079565048218
        self.xyz_wrap = CanonicalWarper(
            3, 
            moving_step=1./moving_step, 
            use_dive_offsets=use_dive_offsets
        )


        # self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        if self.use_time_embedding:
            self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
            self.time_encoder_feat = SinusoidalEncoderWithExp(1, 0, 6, True)

        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            # n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        input_dim4base = 32
        if self.use_time_embedding:
            if self.use_time_attenuation:
                input_dim4base += self.time_encoder_feat.latent_dim
            else:
                input_dim4base += self.time_encoder.latent_dim

        self.mlp_base = tcnn.Network(
            n_input_dims=input_dim4base,
            # n_input_dims=32,
            # n_input_dims=64,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        if self.use_feat_predict:
            self.mlp_feat_prediction = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim+1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3
                },
                n_output_dims=32,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            

        if self.use_weight_predict:
            self.mlp_time_to_weight = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim+1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3
                },
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

        self.viz = logger
        if logger is not None:
            self.logger_count = 1

            self.name = ['x', 'y', 'z']
            self.grad_name = ['mean', 'min', 'max']

            self.viz.line([0.], [0.], win='move_mean', opts=dict(title='move_mean', legend=self.name ))
            self.viz.line([0.], [0.], win='move_min', opts=dict(title='move_min', legend=self.name ))
            self.viz.line([0.], [0.], win='move_max', opts=dict(title='move_max', legend=self.name ))
            self.viz.line([0.], [0.], win='move_norm', opts=dict(title='move_norm', legend=self.name))

            self.viz.line([0.], [0.], win='grid_move_mean', opts=dict(title='grid_move_mean', legend=self.name ))
            self.viz.line([0.], [0.], win='grid_move_min', opts=dict(title='grid_move_min', legend=self.name ))
            self.viz.line([0.], [0.], win='grid_move_max', opts=dict(title='grid_move_max', legend=self.name ))

            self.viz.line([0.], [0.], win='fine_move_mean', opts=dict(title='fine_move_mean', legend=self.name ))
            self.viz.line([0.], [0.], win='fine_move_min', opts=dict(title='fine_move_min', legend=self.name ))
            self.viz.line([0.], [0.], win='fine_move_max', opts=dict(title='fine_move_max', legend=self.name ))

            self.viz.line([0.], [0.], win='grad_feat_predict', opts=dict(title='grad_feat_predict', legend=self.grad_name))
        
            self.viz.line([0.], [0.], win='grad_weight_predict', opts=dict(title='grad_weight_predict', legend=self.grad_name))

            self.viz.line([0.], [0.], win='grad_wrap1_predict', opts=dict(title='grad_wrap1_predict', legend=self.grad_name))

    def log_move(self, move, grid, fine):

        if self.logger_count % 50 == 0 and (move.shape[0] > 0 and grid.shape[0] > 0 and fine.shape[0] > 0):
            i = self.logger_count
            t_move = move
            move_norm = move.norm(dim=-1)
            for j in range(3):
                j_move = t_move[:, j]
                self.viz.line([torch.mean(j_move).item()], [i], win='move_mean', update='append', name=self.name[j])
                j_move = torch.abs(t_move[:, j])
                self.viz.line([torch.min(j_move).item()], [i], win='move_min', update='append', name=self.name[j])
                self.viz.line([torch.max(j_move).item()], [i], win='move_max', update='append', name=self.name[j])

            for j in range(3):
                j_move = grid[:, j]
                self.viz.line([torch.mean(j_move).item()], [i], win='grid_move_mean', update='append', name=self.name[j])
                j_move = torch.abs(grid[:, j])
                self.viz.line([torch.min(j_move).item()], [i], win='grid_move_min', update='append', name=self.name[j])
                self.viz.line([torch.max(j_move).item()], [i], win='grid_move_max', update='append', name=self.name[j])

            for j in range(3):
                j_move = fine[:, j]
                self.viz.line([torch.mean(j_move).item()], [i], win='fine_move_mean', update='append', name=self.name[j])
                j_move = torch.abs(fine[:, j])
                self.viz.line([torch.min(j_move).item()], [i], win='fine_move_min', update='append', name=self.name[j])
                self.viz.line([torch.max(j_move).item()], [i], win='fine_move_max', update='append', name=self.name[j])
                
            self.viz.line([torch.mean(move_norm).item()], [i], win='move_norm', update='append')

        self.logger_count += 1

    def log_grad(self, step):
        if step % 50 == 0:
            if self.use_feat_predict:
                parameters_grad = [p.grad.reshape(-1) for p in self.mlp_feat_prediction.parameters()]
                grad = torch.cat(parameters_grad, dim=-1)
                data_info = [torch.mean(grad).item(), torch.min(grad).item(), torch.max(grad).item()]

                for j in range(3):
                    self.viz.line([data_info[j]], [step], win='grad_feat_predict', update='append', name=self.grad_name[j])

            if self.use_weight_predict:
                parameters_grad = [p.grad.reshape(-1) for p in self.mlp_time_to_weight.parameters()]
                grad = torch.cat(parameters_grad, dim=-1)
                data_info = [torch.mean(grad).item(), torch.min(grad).item(), torch.max(grad).item()]

                for j in range(3):
                    self.viz.line([data_info[j]], [step], win='grad_weight_predict', update='append', name=self.grad_name[j])


    def query_density(self, x, timestamps, time_id: torch.Tensor = None, return_feat: bool = False, dir: torch.Tensor = None):

        if self.unbounded:
            x, mask = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
            mask=None
            # x_move = (x_move - aabb_min) / (aabb_max - aabb_min)
            # move = ((move - aabb_min) / (aabb_max - aabb_min)
        if not self.loose_move:
            # print("x shape: ", x.shape)
            # print("timestamps shape: ", timestamps.shape)
            x_move, move, move_norm, grid_move, fine_move = self.xyz_wrap(x, timestamps, time_id=time_id, aabb=self.aabb, mask=None)
            # print("x_move shape: ", x_move.shape)
        else:
            x_move = x
            move_norm = torch.zeros_like(x[:, 0:1])

        selector = ((x_move > 0.0) & (x_move < 1.0)).all(dim=-1)
        static_feat = self.xyz_encoder(x_move.view(-1, self.num_dim))

        if self.use_time_embedding:
            if self.use_time_attenuation:
                time_encode = self.time_encoder_feat(timestamps, move_norm.detach())
            else:
                time_encode = self.time_encoder(timestamps)

            cat_x = torch.cat([static_feat, time_encode], dim=-1)
        else:
            cat_x = static_feat

        x = (
            self.mlp_base(cat_x)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )


        if return_feat:
            if self.training:

                if self.viz is not None:
                    self.log_move(move, grid_move, fine_move)
                
                if self.use_feat_predict or self.use_weight_predict:
                    temp_feat = torch.cat([x_move, timestamps], dim=-1)

                    if self.use_feat_predict:
                        if not x_move.size(0) == 0:
                            predict = self.mlp_feat_prediction(temp_feat)
                        else:
                            predict = torch.zeros_like(static_feat)
                        loss_feat = F.huber_loss(predict, static_feat, reduction='none') * selector[..., None]
                    else:
                        loss_feat = None

                    if self.use_weight_predict:
                        if not x_move.size(0) == 0:
                            predict_weight = self.mlp_time_to_weight(temp_feat)
                        else:
                            predict_weight = torch.zeros_like(density)
                    else:
                        predict_weight = None 

                    return density, base_mlp_out, [loss_feat, predict_weight, selector]
                elif self.return_extra:
                    return density, base_mlp_out, [None, None, None]
                else:
                    return density, base_mlp_out
            else:
                return density, base_mlp_out
        else:
            return density

    def forward(
        self,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        directions: torch.Tensor = None,
        time_id: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            directions = directions / torch.linalg.norm(
                directions, dim=-1, keepdims=True
            )
            if self.training and (self.return_extra):
                density, embedding, extra = self.query_density(positions, timestamps, time_id=time_id, dir=directions, return_feat=True)
                rgb = self._query_rgb(directions, embedding=embedding)
                return rgb, density, extra
            else:
                density, embedding = self.query_density(positions, timestamps, dir=directions, return_feat=True)
                rgb = self._query_rgb(directions, embedding=embedding)
                return rgb, density


from einops import rearrange
def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (3) or (B, 3)
    
    Outputs:
        R: (3, 3) or (B, 3, 3)
    """
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R

class ExtrPose(nn.Module):
    def __init__(
        self,
        training_size,
        device,
    ) -> None:
        super().__init__()

        N = training_size
        self.register_parameter('dR',
            nn.Parameter(torch.zeros(N, 3, device=device)))
        self.register_parameter('dT',
            nn.Parameter(torch.zeros(N, 3, device=device)))

    def forward(self, img_idx, poses):
        dR = axisangle_to_R(self.dR[img_idx])
        poses[..., :3] = dR @ poses[..., :3]
        poses[..., 3] += self.dT[img_idx]