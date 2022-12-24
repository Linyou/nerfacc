from typing import Callable, List, Union

import math
import functools
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from .ngp import NGPradianceField, contract_to_unisphere, trunc_exp
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
        # dnerf,
        num_dim: int = 3,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        moving_step: float = 1/1024,
    ) -> None:
        super().__init__()
        self.num_dim = num_dim

        # per_level_scale = 1.3195079565048218

        self.sin_posi_encoder = SinusoidalEncoder(3, 0, 1, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)

        # self.xyz_scale = torch.nn.Parameter(torch.tensor([1e-1, ]))
        # self.time_size = 200
        # self.time_embedding = nn.Embedding(200, self.time_encoder.latent_dim)

        self.MOVING_STEP = moving_step

        self.posi_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim+1,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 4
            },
            n_output_dims=self.num_dim*2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3,
            },
        )


        # constants
        # dst_res = 32
        # base_res = 2
        # b = np.exp(np.log(dst_res/base_res)/(n_levels-1))
        # per_level_scale = b
        # self.posi_encoder = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=num_dim+1,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": n_levels,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": base_res,
        #         "per_level_scale": per_level_scale,
        #     },
        #     n_output_dims=self.num_dim*2,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )

        # self.posi_encoder = tcnn.Encoding(
        #     n_input_dims=num_dim+1,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 4,
        #         "n_features_per_level": 8,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": 2,
        #         "per_level_scale": per_level_scale,
        #     }
        # )

        # b = np.exp(np.log(256/2)/(n_levels-1))
        # per_level_scale = b
        # self.final_encoder = tcnn.Encoding(
        #     n_input_dims=num_dim,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": 2,
        #         "per_level_scale": per_level_scale,
        #     },
        # )
        # self.posi_encoder = tcnn.Encoding(
        #     n_input_dims=num_dim,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 4
        #     },
        # )
        # self.time_encoder = tcnn.Encoding(
        #     n_input_dims=1,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 4
        #     },
        # )
        print("canonical out dim: ", self.posi_encoder.n_output_dims)
        print("time out dim: ", self.time_encoder.latent_dim)
        # self.canonical_head = MLP(
        #     input_dim=2,
        #     output_dim=3,
        #     net_depth=1,
        #     net_width=16,
        #     output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        # )
        # self.fine_head = MLP(
        #     input_dim=self.posi_encoder.latent_dim+self.time_encoder.latent_dim,
        #     output_dim=3,
        #     net_depth=1,
        #     net_width=64,
        #     output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        # )
        # self.canonical_head = tcnn.Network(
        #     n_input_dims=(
        #         self.posi_encoder.latent_dim
        #         + self.time_encoder.latent_dim
        #     ),
        #     n_output_dims=self.num_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #     },
        # )
        # self.fine_head = tcnn.Network(
        #     n_input_dims=(
        #         self.sin_posi_encoder.latent_dim
        #         + self.time_encoder.latent_dim
        #     ),
        #     n_output_dims=self.num_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )
        # self.canonical_head = tcnn.Network(
        #     n_input_dims=self.sin_posi_encoder.latent_dim+32,
        #     n_output_dims=self.num_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #     },
        # )
        # self.final_head = tcnn.Network(
        #     n_input_dims=32,
        #     n_output_dims=self.num_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )
        # self.canonical_head = dnerf

        # print("param_precision: ", self.canonical_head.native_tcnn_module.param_precision()) # Precision.Fp16
        # print("output_precision: ", self.canonical_head.native_tcnn_module.output_precision()) # Precision.Fp16

    @torch.no_grad()
    def no_grad_forward(self, x, timestamps, aabb=None):
        # if aabb is not None:
        #     aabb_min, aabb_max = torch.split(aabb, self.num_dim, dim=-1)
        #     x = (x - aabb_min) / (aabb_max - aabb_min)ß
        x = self.posi_encoder(x)
        # padding zeros so the dimension is align with 16
        # t = torch.cat([timestamps, timestamps**2], dim=-1)
        x = torch.cat([x, self.time_encoder(timestamps.to(torch.float16))], dim=-1)
        x = self.canonical_head.warp(x)
        return x.view(-1, self.num_dim)


    def forward(self, x, timestamps, time_id=None, aabb=None):
        if not x.size(0) == 0:
            # if aabb is not None:
            #     aabb_min, aabb_max = torch.split(aabb, self.num_dim, dim=-1)
            #     x_n = (x - aabb_min) / (aabb_max - aabb_min)
            # x_n = torch.cat([x_n, timestamps], dim=-1)
            # flatten_xyz = self.xyz_encoding(x_n)
            offsets = self.posi_encoder(torch.cat([x, timestamps], dim=-1))
            grid_move =offsets[:, 0:3]*self.MOVING_STEP
            fine_move = (torch.special.expit(offsets[:, 3:])*2 - 1)*self.MOVING_STEP
            # xyz_feat = self.final_encoder(x_n)
            # t_enc = self.time_encoder(timestamps)
            # padding zeros so the dimension is align with 16
            # t = torch.cat([timestamps, timestamps**2], dim=-1)
            # x_enc = torch.cat([move_sin, move], dim=-1)
            # move = self.canonical_head(x_enc)
            # x_enc = torch.cat([x_enc, timestamps], dim=-1)
            # x = move*MOVING_STEP + x_n
            # grid_move = 0.
            # fine_move = 0.
            move = grid_move + fine_move
            x = move + x
            move_norm = move.norm(dim=-1)[:, None]
            # print("move:", move.shape)
            # print("x_n:", x_n.shape)
            # print("move_norm:", move_norm.shape)
            # x = x_enc*0.1 + x

            # x_enc = torch.cat([self.sin_posi_encoder(x), t_enc], dim=-1)
            # x = self.fine_head(x_enc)*0.01 + x
            # if aabb is not None:
            #     aabb_min, aabb_max = torch.split(aabb, self.num_dim, dim=-1)
            #     x_n = (x - aabb_min) / (aabb_max - aabb_min)
            # move = self.final_encoder(torch.cat([x_n, timestamps], dim=-1))
            # x = move*0.1 + x
        else:
            move = torch.zeros_like(x)
            move_norm = torch.zeros_like(x[:, 0:1])
            grid_move = torch.zeros_like(x)
            fine_move = torch.zeros_like(x)


        return x.view(-1, self.num_dim), move, move_norm, grid_move, fine_move




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
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        use_feat_predict: bool = False,
        use_weight_predict: bool = False,
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

        self.use_feat_predict = use_feat_predict
        self.use_weight_predict = use_weight_predict
        # self.aabb = self.aabb.to(torch.float16)
        # b = np.exp(np.log(2048/16)/(n_levels-1))
        # per_level_scale = b
        # per_level_scale = 1.4472692012786865
        # per_level_scale = 1.3819128274917603
        per_level_scale = 1.3195079565048218
        # self.wrap = CanonicalWarper(16)
        self.xyz_wrap = CanonicalWarper(3)

        # self.xyz_can_encoder = tcnn.Encoding(
        #     n_input_dims=num_dim,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": n_levels,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": 16,
        #         "per_level_scale": per_level_scale,
        #     },
        # )
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.posi_encoder_feat = SinusoidalEncoderWithExp(3, 0, 4, True)
        self.time_encoder_feat = SinusoidalEncoderWithExp(1, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)

        # self.time_embedding = nn.Embedding(32, 8)

        # self.posi_encoder = tcnn.Encoding(
        #     n_input_dims=num_dim,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 4
        #     },
        # )
        # self.time_encoder = tcnn.Encoding(
        #     n_input_dims=1,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 4
        #     },
        # )

        self.loose_move = False
        self.return_extra = False


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
        self.move_encoder = tcnn.Encoding(
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

        self.mlp_base = tcnn.Network(
            n_input_dims=32+self.time_encoder_feat.latent_dim,
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

        # self.mlp_time = tcnn.Network(
        #     n_input_dims=self.time_encoder.latent_dim,
        #     n_output_dims=32,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )

        if self.use_feat_predict:
            # self.mlp_feat_prediction = tcnn.Network(
            #     n_input_dims=self.posi_encoder.latent_dim,
            #     n_output_dims=32,
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "None",
            #         "n_neurons": 64,
            #         "n_hidden_layers": 1,
            #     },
            # )
            self.mlp_feat_prediction = MLP(
                input_dim=self.posi_encoder.latent_dim + self.time_encoder.latent_dim,
                output_dim=32,
                net_depth=1,
                net_width=64,
                output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
            )

        if self.use_weight_predict:
            self.mlp_time_to_weight = tcnn.Network(
                n_input_dims=self.posi_encoder.latent_dim + self.time_encoder.latent_dim,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

        # self.mlp_feat_norm= MLP(
        #     input_dim=32,
        #     output_dim=1,
        #     net_depth=1,
        #     net_width=64,
        #     output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        # )
        # self.mlp_feat_norm = tcnn.Network(
        #     n_input_dims=32,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )
        

        # self.mlp_time_to_density = tcnn.Network(
        #     n_input_dims=15+8,
        #     n_output_dims=15,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )
        # self.mlp_time_to_density = MLP(
        #     input_dim=32+9,
        #     output_dim=1,
        #     net_depth=1,
        #     net_width=64,
        #     output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        # )

        self.mlp_head = tcnn.Network(
            n_input_dims=(
                (
                    self.direction_encoding.n_output_dims
                    if self.use_viewdirs
                    else 0
                )
                + self.geo_feat_dim
            ),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        # xs, ys, zs = np.meshgrid(
        #         np.arange(64, dtype=np.int32),
        #         np.arange(64, dtype=np.int32),
        #         np.arange(64, dtype=np.int32), 
        #         indexing='xy'
        #     )

        # xs = torch.from_numpy(xs) / 64
        # ys = torch.from_numpy(ys) / 64
        # zs = torch.from_numpy(zs) / 64

        # grid_xyz = torch.from_numpy(np.concatenate([xs[:,None], ys[:,None], zs[:,None]]))
        # grid_xyz.requires_grad = False
        # self.register_buffer("grid_xyz", grid_xyz)

        if logger is not None:
            self.viz = logger
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

            # self.viz.line([0.], [0.], win='grad_time_predict', opts=dict(title='grad_time_predict', legend=self.grad_name))

            self.viz.line([0.], [0.], win='grad_wrap1_predict', opts=dict(title='grad_wrap1_predict', legend=self.grad_name))

            # self.viz.line([0.], [0.], win='grad_wrap2_predict', opts=dict(title='grad_wrap2_predict', legend=self.grad_name))
            # self.viz.line([0.], [0.], win='x_scale', opts=dict(title='x_scale', legend=['x', 'y', 'z']))
            # self.viz.line([0.], [0.], win='y_scale', opts=dict(title='y_scale', legend=['x', 'y', 'z']))
            # self.viz.line([0.], [0.], win='z_scale', opts=dict(title='z_scale', legend=['x', 'y', 'z']))

    def log_move(self, move, grid, fine):

        if self.logger_count % 50 == 0:
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
                # j_bmove = self.xyz_scale
                # self.viz.line([j_bmove[0].item()], [i], win='x_scale', update='append', name=self.name[j])
                # self.viz.line([j_bmove[1].item()], [i], win='y_scale', update='append', name=self.name[j])
                # self.viz.line([j_bmove[2].item()], [i], win='z_scale', update='append', name=self.name[j])
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

            # parameters_grad = [p.grad.reshape(-1) for p in self.mlp_feat_norm.parameters()]
            # grad = torch.cat(parameters_grad, dim=-1)
            # data_info = [torch.mean(grad).item(), torch.min(grad).item(), torch.max(grad).item()]

            # for j in range(3):
            #     self.viz.line([data_info[j]], [step], win='grad_time_predict', update='append', name=self.grad_name[j])

            # parameters_grad = [p.grad.reshape(-1) for p in self.xyz_wrap.canonical_head.parameters()]
            # grad = torch.cat(parameters_grad, dim=-1)
            # data_info = [torch.mean(grad).item(), torch.min(grad).item(), torch.max(grad).item()]

            # for j in range(3):
            #     self.viz.line([data_info[j]], [step], win='grad_wrap1_predict', update='append', name=self.grad_name[j])

            # parameters_grad = [p.grad.reshape(-1) for p in self.xyz_wrap.canonical_head_2.parameters()]
            # grad = torch.cat(parameters_grad, dim=-1)
            # data_info = [torch.mean(grad).item(), torch.min(grad).item(), torch.max(grad).item()]

            # for j in range(3):
            #     self.viz.line([data_info[j]], [step], win='grad_wrap2_predict', update='append', name=self.grad_name[j])


    def query_density(self, x, timestamps, time_id: torch.Tensor = None, return_feat: bool = False, dir: torch.Tensor = None):
        # move = self.xyz_wrap(x, timestamps).detach()
        # feat = self.wrap(x, timestamps)
        # move_norm = move.norm(dim=-1)
        # timestamps = timestamps * 2
        # x_ori = x
        # x = x.to(torch.float16)
        # timestamps = timestamps.to(torch.float16)
        # aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)

        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
            # x_move = (x_move - aabb_min) / (aabb_max - aabb_min)
            # move = ((move - aabb_min) / (aabb_max - aabb_min)

        if not self.loose_move:
            x_move, move, move_norm, grid_move, fine_move = self.xyz_wrap(x, timestamps, time_id=time_id, aabb=self.aabb)
            # move = grid_move + fine_move
            # grid_move = grid_move * (aabb_max - aabb_min) + aabb_min 
            # fine_move = fine_move * (aabb_max - aabb_min) + aabb_min 
            # m_factor = (torch.exp(0.5*(move_norm))).detach()
        else:
            x_move = x
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x_move = (x_move - aabb_min) / (aabb_max - aabb_min)
            move_norm = torch.zeros_like(x[:, 0:1])
            # m_factor = 0.
        # move = torch.clamp(move, -1.5, 1.5)
        # if not self.loose_move:
        #     x_move = x + move #+ torch.randn_like(move[:, :1])*0.001
        # else:
        #     x_move = x 
        # x_move = x_move + self.mlp_fine(x_move)

        selector = ((x_move > 0.0) & (x_move < 1.0)).all(dim=-1)
        # selector_move = ((x_move > 0.0) & (x_move < 1.0)).all(dim=-1)[:, None]
        # selector_ori = ((x_ori > 0.0) & (x_ori < 1.0)).all(dim=-1)[:, None]
        # static_feat = self.xyz_encoder(x_ori.view(-1, self.num_dim))
        # dynimc_feat = self.move_encoder(x_move.view(-1, self.num_dim))
        # cond = move_norm > 1e-5
        # x_final = torch.where(cond, x_move, x_ori)
        static_feat = self.xyz_encoder(x_move.view(-1, self.num_dim))


        # static_feat = static_feat.reshape(-1, 32, 2)
        # print("dynimc_feat: ", dynimc_feat.shape)
        # print("static_feat: ", static_feat.shape)
        # print("selector_move: ", selector_move.shape)
        # print("selector_ori: ", selector_ori.shape)
        # new_feat = torch.where(cond, dynimc_feat, static_feat)
        # selector = torch.where(cond, selector_move, selector_ori)[:, 0]
        # feat = self.xyz_encoder(x.view(-1, self.num_dim))
        # move_feat = self.xyz_encoder(x_ori.view(-1, self.num_dim))
        # dynimic_feat_delta = self.posi_encoder(move)
        # dynimic_feat_p = self.posi_encoder(x.view(-1, self.num_dim) + move)
        # print(move_norm.shape)
        # static_feat_shape = static_feat.reshape(-1, 32, 2)
        
        # new_static = static_feat_shape[..., 0] * (1-move_norm) + move_norm*static_feat_shape[..., 1]

        time_encode = self.time_encoder_feat(timestamps, move_norm.detach())
        # t_l = torch.floor(timestamps * 31).long()
        # t_h = torch.ceil(timestamps * 31).long()
        # interval = timestamps - t_l
        # t_l_emb = self.time_embedding(t_l.squeeze(-1))
        # t_h_emb = self.time_embedding(t_h.squeeze(-1))
        # time_encode = t_l_emb * (1-interval) + interval * t_h_emb
        # time_encode = self.time_encoder(timestamps)
        # pos_encode_feat = self.posi_encoder_feat(x_move, move_norm)
        # if not time_encode.size(0) == 0:
        #     t_feat = self.mlp_time(time_encode)
        # else:
        #     t_feat = torch.zeros_like(static_feat)

        cat_x = torch.cat([static_feat, time_encode], dim=-1)

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

        # base_mlp_out = torch.cat([base_mlp_out, time_encode], dim=-1)
        # base_mlp_out = self.mlp_time_to_density(torch.cat([base_mlp_out, time_encode], dim=-1))

        if return_feat:
            if self.training:
                self.log_move(move, grid_move, fine_move)
                if self.use_feat_predict or self.use_weight_predict:
                    # predict = self.mlp_time_prediction(cat_x)
                    # feat_times = torch.cat([static_feat, self.xyz_wrap.time_encoder(timestamps)], dim=-1)
                    # print(feat_times.shape)
                    # predict_density = self.mlp_time_to_density(feat_times)
                    # target = torch.cat([timestamps, x_move, move], dim=-1)
                    # x_enc = torch.where(cond, self.posi_encoder(x_move), self.posi_encoder(x_ori))
                    x_enc = self.posi_encoder(x_move)
                    temp_feat = torch.cat([
                        x_enc, 
                        # self.posi_encoder(dir),
                        self.time_encoder(timestamps)
                    ], dim=-1)
                    # temp_feat = self.posi_encoder(x_move, move_norm)

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
                    # time_predict = self.mlp_feat_norm(static_feat)
                    # w_dim = time_predict.dim()
                    # m_dim = timestamps.dim()
                    # assert w_dim == m_dim, f"time_predict: {w_dim} and timestamps :{m_dim} not equal!"
                    # loss_times = F.smooth_l1_loss(time_predict, timestamps, reduction='none')


                    return density, base_mlp_out, [loss_feat, predict_weight, selector]
                    # loss_density = F.smooth_l1_loss(predict_density, density, reduction='none')
                elif self.return_extra:
                    return density, base_mlp_out, [None, None, None]
                else:
                    return density, base_mlp_out
            else:
                return density, base_mlp_out
        else:
            return density

    # def get_grid(self):
    #     grid_feat = self.xyz_encoder(self.grid_xyz.view(-1, self.num_dim))
    #     return grid_feat



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