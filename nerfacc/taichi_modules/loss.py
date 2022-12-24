import taichi as ti
import torch
from torch import Tensor


@ti.kernel
def prefix_sums_kernel(
    packed_info: ti.types.ndarray(), 
    weights: ti.types.ndarray(), 
    weights_ts: ti.types.ndarray(), 
    ws_inclusive_scan: ti.types.ndarray(), 
    ws_exclusive_scan: ti.types.ndarray(), 
    wts_inclusive_scan: ti.types.ndarray(), 
    wts_exclusive_scan: ti.types.ndarray(), 
):
    for i in ti.ndrange(packed_info.shape[0]):
        start_idx = packed_info[i,0]
        N_samples = packed_info[i,1]

        ws_temp = 0.
        wst_temp = 0.
        for n in range(N_samples):
            idx = start_idx + n

            #exclusive_scan
            ws_exclusive_scan[idx] = ws_temp
            wts_exclusive_scan[idx] = wst_temp

            ws_temp+=weights[idx]
            wst_temp+=weights_ts[idx]

            #inclusive_scan
            ws_inclusive_scan[idx] = ws_temp
            wts_inclusive_scan[idx] = wst_temp

@ti.kernel
def _loss_kernel(
    _loss: ti.types.ndarray(), 
    interval: ti.types.ndarray(), 
    weights: ti.types.ndarray(), 
    ws_inclusive_scan: ti.types.ndarray(), 
    ws_exclusive_scan: ti.types.ndarray(), 
    wts_inclusive_scan: ti.types.ndarray(), 
    wts_exclusive_scan: ti.types.ndarray(), 
):
    for i in ti.ndrange(weights.shape[0]):
        ws = weights[i]
        deltas = interval[i]
        wts_inc = wts_inclusive_scan[i]
        ws_exc = ws_exclusive_scan[i]
        ws_inc = ws_inclusive_scan[i]
        wts_exc = wts_exclusive_scan[i]

        _loss[i] = 2.*(wts_inc*ws_exc-ws_inc*wts_exc) + 1./3.*ws*ws*deltas


@ti.kernel
def distortion_loss_fw_kernel(
    packed_info: ti.types.ndarray(), 
    _loss: ti.types.ndarray(), 
    loss: ti.types.ndarray(), 
):

    for i in ti.ndrange(packed_info.shape[0]):
        start_idx = packed_info[i,0]
        N_samples = packed_info[i,1]

        loss_temp = 0.
        for n in range(N_samples):
            idx = start_idx + n

            loss_temp += _loss[idx]

        loss[i] = loss_temp

@ti.kernel
def distortion_loss_bw_kernel(
    dL_dloss: ti.types.ndarray(), 
    interval: ti.types.ndarray(), 
    weights: ti.types.ndarray(), 
    tmid: ti.types.ndarray(), 
    ws_inclusive_scan: ti.types.ndarray(), 
    wts_inclusive_scan: ti.types.ndarray(), 
    packed_info: ti.types.ndarray(), 
    dL_dws: ti.types.ndarray(), 
):
    for i in ti.ndrange(packed_info.shape[0]):

        start_idx = packed_info[i,0]
        N_samples = packed_info[i,1]

        end_idx = start_idx + N_samples - 1

        ws_sum = ws_inclusive_scan[end_idx]
        wts_sum = wts_inclusive_scan[end_idx]

        for n in range(N_samples):
            idx = n + N_samples

            selector = 0 if idx == start_idx else tmid[idx]*ws_inclusive_scan[idx-1]-wts_inclusive_scan[idx-1]

            dL_dws[idx] = dL_dloss[i] * 2 * selector + (wts_sum-wts_inclusive_scan[idx]-tmid[idx]*(ws_sum-ws_inclusive_scan[idx]))

            dL_dws[idx] += dL_dloss[i] * 2./3.*weights[idx]*interval[idx]


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, interval, timd, packed_info):
        loss = torch.zeros(packed_info.size(0), dtype=ws.dtype, device=ws.device)

        ws_inclusive_scan = torch.zeros_like(ws)
        ws_exclusive_scan = torch.zeros_like(ws)
        wts_inclusive_scan = torch.zeros_like(ws)
        wts_exclusive_scan = torch.zeros_like(ws)

        _loss = torch.zeros_like(ws)

        wts = ws*timd

        wts = wts.contiguous()
        ws = ws.contiguous()
        packed_info = packed_info.contiguous()
        interval = interval.contiguous()

        prefix_sums_kernel(
            packed_info, ws, wts, 
            ws_inclusive_scan, 
            ws_exclusive_scan, 
            wts_inclusive_scan, 
            wts_exclusive_scan
        )

        # _loss = 2.*(wts_inclusive_scan*ws_exclusive_scan-
        #             ws_inclusive_scan*wts_exclusive_scan) + 1./3.*ws*ws*interval
    
        _loss_kernel(
            _loss, interval, ws, 
            ws_inclusive_scan, 
            ws_exclusive_scan,
            wts_inclusive_scan, 
            wts_exclusive_scan
        )

        distortion_loss_fw_kernel(packed_info, _loss, loss)

        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, interval, timd, packed_info)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors

        dL_dws = torch.zeros_like(ws)

        distortion_loss_bw_kernel(
            dL_dloss, deltas, ws, ts,
            ws_inclusive_scan, wts_inclusive_scan,
            rays_a, dL_dws
        )

        return dL_dws, None, None, None


@torch.cuda.amp.autocast(dtype=torch.float32)
def distortion(
    packed_info: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
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

    weights = weights.squeeze(-1)
    t_starts = t_starts.squeeze(-1)
    t_ends = t_ends.squeeze(-1)

    interval = t_ends - t_starts
    tmid = (t_starts + t_ends) / 2
    
    return DistortionLoss.apply(weights, interval, tmid, packed_info)
