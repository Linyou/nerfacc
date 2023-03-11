"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import math
from typing import Callable, List, Union
from torch import nn
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
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


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


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
        return x

MOVING_STEP = 1/(4096*1)
class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 15,
        base_resolution: int = 16,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        dst_resolution: int = 4096,
        log2_hashmap_size: int = 19,
        use_feat_predict: bool = False,
        use_weight_predict: bool = False,
        moving_step: float = MOVING_STEP,
        use_dive_offsets: bool = False,
        use_time_embedding: bool = False,
        use_time_attenuation: bool = False,
        hash4motion: bool = False,    
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded

        self.box_scale = (aabb[3:] - aabb[:3]).max() / 2
        print("self.box_scale: ", self.box_scale)

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = math.exp(
            math.log(dst_resolution * 1 / base_resolution) / (n_levels - 1)
        )  # 1.4472692012786865
        # per_level_scale = 1.4472692012786865

        print('--NGPDradianceField configuration--')
        print(f'  moving_step: {moving_step}')
        print(f'  hash b: {per_level_scale:6f}')
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
        self.MOVING_STEP = moving_step
        self.use_dive_offsets = use_dive_offsets

        self.loose_move = False

        self.motion_input_dim = 3 + 1
        self.motion_output_dim = 3 * 2 if use_dive_offsets else 3

        self.return_extra = False

        if hash4motion:
            # hash table for time encoding
            self.xyz_wrap = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.motion_input_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 4,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": 8,
                    "per_level_scale":  math.exp(
                        math.log(64 / 8) / (4 - 1)
                    )  # 1.4472692012786865,
                },
                n_output_dims=self.motion_output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
        else:
            self.xyz_wrap = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.motion_input_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 4,
                            "otype": "Frequency",
                            "n_frequencies": 8
                        },
                        # {
                        #     "n_dims_to_encode": 1,
                        #     "otype": "Frequency",
                        #     # "n_frequencies": 8
                        # },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
                n_output_dims=self.motion_output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 3,
                },
            )


        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )



        self.hash_encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        input_dim4base = self.hash_encoder.n_output_dims

        if self.use_time_embedding:
            self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
            self.time_encoder_feat = SinusoidalEncoderWithExp(1, 0, 6, True)

            if self.use_time_attenuation:
                input_dim4base += self.time_encoder_feat.latent_dim
            else:
                input_dim4base += self.time_encoder.latent_dim

        self.mlp_base = tcnn.Network(
            n_input_dims=input_dim4base,
            n_output_dims=1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
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
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )


        if self.use_feat_predict:
            self.mlp_feat_prediction = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim+1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3
                },
                n_output_dims=self.hash_encoder.n_output_dims,
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

    def query_move(self, x, t):
        offsets = self.xyz_wrap(torch.cat([x, t], dim=-1))
        if self.use_dive_offsets:
            grid_move = offsets[:, 0:3]*self.MOVING_STEP
            fine_move = (torch.special.expit(offsets[:, 3:])*2 - 1)*self.MOVING_STEP
            # fine_move = torch.tanh(offsets[:, 3:])*self.MOVING_STEP

        else:
            grid_move = offsets*self.MOVING_STEP
            fine_move = 0

        move = grid_move + fine_move
        move_norm = move.norm(dim=-1)[:, None]

        return x + move, move_norm

    def query_density(self, x, t, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)

        if (not self.loose_move) and x.shape[0] > 0:
            x_move, move_norm = self.query_move(
                x.view(-1, self.num_dim), 
                t.view(-1, 1)
            )
        else:
            x_move = x.view(-1, self.num_dim)
            move_norm = torch.zeros_like(x_move[:, :1])

        x = x_move.view_as(x)

        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        hash_feat = self.hash_encoder(x_move)

        if self.use_time_embedding:
            if self.use_time_attenuation:
                time_encode = self.time_encoder_feat(t.view(-1, 1), move_norm.detach())
            else:
                time_encode = self.time_encoder(t.view(-1, 1))

            cat_feat = torch.cat([hash_feat, time_encode], dim=-1)
        else:
            cat_feat = hash_feat


        x = (
            self.mlp_base(cat_feat)
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
                if self.use_feat_predict or self.use_weight_predict:
                    temp_feat = torch.cat([x_move, t], dim=-1)

                    if self.use_feat_predict:
                        if x.size(0) > 0:
                            predict_feat = self.mlp_feat_prediction(temp_feat)
                            loss_feat = F.huber_loss(predict_feat, hash_feat, reduction='none') * selector[..., None]
                        else:
                            loss_feat = torch.zeros_like(hash_feat)
                    else:
                        loss_feat = None

                    if self.use_weight_predict:
                        predict_weight = self.mlp_time_to_weight(temp_feat)
                    else:
                        predict_weight = None

                    return density, base_mlp_out, [loss_feat, predict_weight, selector]
                else:
                    return density, base_mlp_out
            else:
                return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = dir / torch.linalg.norm(
                dir, dim=-1, keepdims=True
            )

            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
        # density, embedding = self.query_density(positions, t, return_feat=True)
        # rgb = self._query_rgb(directions, embedding=embedding)

        if self.training and (self.return_extra):
            density, embedding, extra = self.query_density(positions, t, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
            return rgb, density, extra
        else:
            density, embedding = self.query_density(positions, t, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
            return rgb, density
