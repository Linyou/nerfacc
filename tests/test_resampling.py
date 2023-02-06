import pytest
import torch

from nerfacc import pack_info, ray_marching, ray_resampling

device = "cuda:0"
batch_size = 128
eps = torch.finfo(torch.float32).eps


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_resampling():
    batch_size = 1024
    num_bins = 128
    num_samples = 128

    t = torch.randn((batch_size, num_bins + 1), device=device)
    t = torch.sort(t, dim=-1).values
    w_logits = torch.randn((batch_size, num_bins), device=device) * 0.1
    w = torch.softmax(w_logits, dim=-1)
    masks = w_logits > 0
    w_logits[~masks] = -torch.inf

    from nerfacc.reference.multinerf.stepfun import sample

    t_samples = sample(
        False,
        t.clone(),
        w_logits.clone(),
        num_samples + 1,
        deterministic_center=True,
    )

    t_starts = t[:, :-1][masks].unsqueeze(-1)
    t_ends = t[:, 1:][masks].unsqueeze(-1)
    w_logits = w_logits[masks].unsqueeze(-1)
    w = w[masks].unsqueeze(-1)
    num_steps = masks.long().sum(dim=-1)
    cum_steps = torch.cumsum(num_steps, dim=0)
    packed_info = torch.stack([cum_steps - num_steps, num_steps], dim=-1).int()

    _, t_starts, t_ends = ray_resampling(
        packed_info, t_starts, t_ends, w, num_samples
    )
    assert torch.allclose(
        t_starts.view(batch_size, num_samples), t_samples[:, :-1], atol=1e-3
    )
    assert torch.allclose(
        t_ends.view(batch_size, num_samples), t_samples[:, 1:], atol=1e-3
    )


def test_pdf_query():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    ray_indices, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=1e-3,
    )
    packed_info = pack_info(ray_indices, n_rays=batch_size)
    weights = torch.rand((t_starts.shape[0], 1), device=device)
    packed_info, t_starts, t_ends = ray_resampling(
        packed_info, t_starts, t_ends, weights, n_samples=32
    )
    assert t_starts.shape == t_ends.shape == (batch_size * 32, 1)


if __name__ == "__main__":
    test_resampling()
    test_pdf_query()
