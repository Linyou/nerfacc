import torch

from nerfacc.cuda import invert_cdf
from nerfacc.reference.multinerf.stepfun import invert_cdf as invert_cdf_py

device = "cuda:0"


def test_invert_cdf():
    n_bins = 1280
    w_logits = torch.rand((1, n_bins), device=device)
    t = torch.sort(torch.randn((1, n_bins + 1), device=device), dim=-1).values
    n_bins_inv = 320
    u = torch.sort(
        torch.rand((1, n_bins_inv + 1), device=device), dim=-1
    ).values
    t_new = invert_cdf_py(u, t, w_logits)

    src_bins = torch.tensor([[0, n_bins]], device=device).int()
    s0 = t[:, :-1]
    s1 = t[:, 1:]
    w = torch.softmax(w_logits, dim=-1)
    tgt_bins = torch.tensor([[0, n_bins_inv]], device=device).int()
    cdf_u0 = u[:, :-1]
    cdf_u1 = u[:, 1:]
    t0, t1 = invert_cdf(src_bins, s0, s1, w, tgt_bins, cdf_u0, cdf_u1)

    print((t0 - t_new[:, :-1]).abs().max())
    print((t1 - t_new[:, 1:]).abs().max())

    # import tqdm
    # torch.cuda.synchronize()
    # for _ in tqdm.tqdm(range(10000)):
    #     t_new = invert_cdf_py(u, t, w_logits)
    #     # torch.cuda.synchronize()
    # torch.cuda.synchronize()
    # for _ in tqdm.tqdm(range(10000)):
    #     t0, t1 = invert_cdf(src_bins, s0, s1, w, tgt_bins, cdf_u0, cdf_u1)
    #     # torch.cuda.synchronize()


if __name__ == "__main__":
    test_invert_cdf()
