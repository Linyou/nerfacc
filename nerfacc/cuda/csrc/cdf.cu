/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"


template <typename scalar_t>
// Invert the CDF defined by (t, w) at the points specified by u in [0, 1).
__global__ void invert_cdf_kernel(
    const uint32_t N,    // batch size.
    const int *src_bins, // batch info for input bins: {first bin id, #bins}
    const scalar_t *s0,  // source interval of the input bins
    const scalar_t *s1,
    const scalar_t *w,    // weights of the input bins, not necessarily normalized
    const int *tgt_bins,  // batch info for inverted bins: {first bin id, #bins}
    const scalar_t *cdf_u0, // target cdf interval of the inverted bins
    const scalar_t *cdf_u1,
    scalar_t *t0, // target interval of the inverted bins
    scalar_t *t1)
{
    CUDA_GET_THREAD_ID(i, N);

    // locate
    const int src_base = src_bins[i * 2 + 0];
    const int src_steps = src_bins[i * 2 + 1];
    const int tgt_base = tgt_bins[i * 2 + 0];
    const int tgt_steps = tgt_bins[i * 2 + 1];

    if (tgt_steps == 0)
        // skip this one.
        return;
    if (src_steps == 0)
        // nothing to be inverted. raise error in host code.
        return;

    s0 += src_base;
    s1 += src_base;
    w += src_base;

    cdf_u0 += tgt_base;
    cdf_u1 += tgt_base;
    t0 += tgt_base;
    t1 += tgt_base;

    // normalize weights to sum to one.
    scalar_t w_sum = 0.0f;
    for (int j = 0; j < src_steps; j++)
        w_sum += w[j];

    // if almost zero, pad every bin a uniform weight
    scalar_t padding = 0.0f;
    if (w_sum < 1e-20f) {
        w_sum += 1.0f;
        padding = 1.0f / src_steps;
    }

    // invert cdf sampling
    int idx = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = (w[idx] + padding) / w_sum;
    scalar_t scaling;
    for (int j = 0; j < tgt_steps; j++)
    {   
        // NOTE: we assume the intervals are sorted and non-overlapping.
        // I.E. for all j, cdf_u0[j + 1] >= cdf_u1[j].

        // compute t0
        while (idx < src_steps - 1 && cdf_u0[j] >= cdf_next)
        {
            idx += 1;
            cdf_prev = cdf_next;
            cdf_next += (w[idx] + padding) / w_sum;
        }
        scaling = (s1[idx] - s0[idx]) / (cdf_next - cdf_prev);
        t0[j] = (cdf_u0[j] - cdf_prev) * scaling + s0[idx];

        // compute t1
        while (idx < src_steps - 1 && cdf_u1[j] >= cdf_next)
        {
            idx += 1;
            cdf_prev = cdf_next;
            cdf_next += (w[idx] + padding) / w_sum;
        }
        scaling = (s1[idx] - s0[idx]) / (cdf_next - cdf_prev);
        t1[j] = (cdf_u1[j] - cdf_prev) * scaling + s0[idx];
    }
    return;
}

template <typename scalar_t>
__global__ void pdf_query_kernel(
    const uint32_t n_rays,
    // query
    const int *packed_info, // input ray & point indices.
    const scalar_t *starts, // input start t
    const scalar_t *ends,   // input end t
    const scalar_t *pdfs,   // pdf to be queried
    // resample
    const int *resample_packed_info, // input ray & point indices.
    const scalar_t *resample_starts, // input start t, sorted
    const scalar_t *resample_ends,   // input end t, sorted
    // output
    scalar_t *resample_pdfs) // should be zero-initialized
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];                    // point idx start.
    const int steps = packed_info[i * 2 + 1];                   // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];  // point idx start.
    const int resample_steps = resample_packed_info[i * 2 + 1]; // point idx shift.

    if (resample_steps == 0) // nothing to query
        return;

    if (steps == 0) // nothing to be queried: set pdfs to 0
        return;

    starts += base;
    ends += base;
    pdfs += base;
    resample_starts += resample_base;
    resample_ends += resample_base;
    resample_pdfs += resample_base;

    // which interval is resample_start (t0) located
    int t0_id = 0;
    scalar_t t0_start = starts[0], t0_end = ends[0];
    scalar_t cdf0_start = 0.0f, cdf0_end = pdfs[0];
    // which interval is resample_end (t1) located
    int t1_id = 0;
    scalar_t t1_start = starts[0], t1_end = ends[0];
    scalar_t cdf1_start = 0.0f, cdf1_end = pdfs[0];
    // go!
    for (int j = 0; j < resample_steps; ++j)
    {
        scalar_t t0 = resample_starts[j];
        while (t0 > t0_end & t0_id < steps - 1)
        {
            t0_id++;
            t0_start = starts[t0_id];
            t0_end = ends[t0_id];
            cdf0_start = cdf0_end;
            cdf0_end += pdfs[t0_id];
        }
        // if (t0 > t0_end) {
        //     resample_pdfs[j] = 0.0f;
        //     continue;
        // }
        scalar_t pct0 = 0.0f; // max(t0 - t0_start, 0.0f) / max(t0_end - t0_start, 1e-10f);
        scalar_t resample_cdf_start = cdf0_start + pct0 * (cdf0_end - cdf0_start);

        scalar_t t1 = resample_ends[j];
        while (t1 > t1_end & t1_id < steps - 1)
        {
            t1_id++;
            t1_start = starts[t1_id];
            t1_end = ends[t1_id];
            cdf1_start = cdf1_end;
            cdf1_end += pdfs[t1_id];
        }
        // if (t1 > t1_end) {
        //     resample_pdfs[j] = cdf1_end - resample_cdf_start;
        //     continue;
        // }
        scalar_t pct1 = 1.0f; // max(t1 - t1_start, 0.0f) / max(t1_end - t1_start, 1e-10f);
        scalar_t resample_cdf_end = cdf1_start + pct1 * (cdf1_end - cdf1_start);

        // compute pdf of [t0, t1]
        resample_pdfs[j] = resample_cdf_end - resample_cdf_start;
    }

    return;
}

template <typename scalar_t>
__global__ void cdf_resampling_kernel(
    const uint32_t n_rays,
    const int *packed_info, // input ray & point indices.
    const scalar_t *starts, // input start t
    const scalar_t *ends,   // input end t
    const scalar_t *w,      // transmittance weights
    const int *resample_packed_info,
    scalar_t *resample_starts,
    scalar_t *resample_ends)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    w += base;
    resample_starts += resample_base;
    resample_ends += resample_base;

    // normalize weights **per ray**
    scalar_t w_sum = 0.0f;
    for (int j = 0; j < steps; j++)
        w_sum += w[j];
    scalar_t padding = fmaxf(1e-10f - w_sum, 0.0f);
    scalar_t padding_step = padding / steps;
    w_sum += padding;

    int num_endpoints = resample_steps + 1;
    scalar_t cdf_pad = 1.0f / (2 * num_endpoints);
    scalar_t cdf_step_size = (1.0f - 2 * cdf_pad) / resample_steps;

    int idx = 0, j = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = (w[idx] + padding_step) / w_sum;
    scalar_t cdf_u = cdf_pad;
    while (j < num_endpoints)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t t = (cdf_u - cdf_prev) * scaling + starts[idx];
            // if (j == 100) {
            //     printf(
            //         "cdf_u: %.10f, cdf_next: %.10f, cdf_prev: %.10f, scaling: %.10f, t: %.10f, starts[idx]: %.10f, ends[idx]: %.10f\n",
            //         cdf_u, cdf_next, cdf_prev, scaling, t, starts[idx], ends[idx]);
            // }
            if (j < num_endpoints - 1)
                resample_starts[j] = t;
            if (j > 0)
                resample_ends[j - 1] = t;
            // going further to next resample
            // cdf_u += cdf_step_size;
            j += 1;
            cdf_u = j * cdf_step_size + cdf_pad;
        }
        else
        {
            // going to next interval
            idx += 1;
            if (idx >= steps)
            {
                printf("Error: idx: %d, steps: %d, j: %d, num_endpoints: %d, cdf_u: %f, cdf_next: %f, cdf_prev: %f\n", idx, steps, j, num_endpoints, cdf_u, cdf_next, cdf_prev);
                return;
            }
            cdf_prev = cdf_next;
            cdf_next += (w[idx] + padding_step) / w_sum;
        }
    }
    if (j != num_endpoints)
    {
        printf("Error: %d %d %f\n", j, num_endpoints, w_sum);
    }
    return;
}

// template <typename scalar_t>
// __global__ void cdf_resampling_kernel(
//     const uint32_t n_rays,
//     const int *packed_info,   // input ray & point indices.
//     const scalar_t *starts,   // input start t
//     const scalar_t *ends,     // input end t
//     const scalar_t *weights,  // transmittance weights
//     const int *resample_packed_info,
//     scalar_t *resample_starts,
//     scalar_t *resample_ends)
// {
//     CUDA_GET_THREAD_ID(i, n_rays);

//     // locate
//     const int base = packed_info[i * 2 + 0];  // point idx start.
//     const int steps = packed_info[i * 2 + 1]; // point idx shift.
//     const int resample_base = resample_packed_info[i * 2 + 0];
//     const int resample_steps = resample_packed_info[i * 2 + 1];
//     if (steps == 0)
//         return;

//     starts += base;
//     ends += base;
//     weights += base;
//     resample_starts += resample_base;
//     resample_ends += resample_base;

//     scalar_t cdf_step_size = 1.0f / resample_steps;

//     // normalize weights **per ray**
//     scalar_t weights_sum = 0.0f;
//     for (int j = 0; j < steps; j++)
//         weights_sum += weights[j];

//     scalar_t padding = fmaxf(1e-5f - weights_sum, 0.0f);
//     scalar_t padding_step = padding / steps;
//     weights_sum += padding;

//     int idx = 0, j = 0;
//     scalar_t cdf_prev = 0.0f, cdf_next = (weights[idx] + padding_step) / weights_sum;
//     scalar_t cdf_u = 0.5f * cdf_step_size;
//     while (cdf_u < 1.0f)
//     {
//         if (cdf_u < cdf_next)
//         {
//             // resample in this interval
//             scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
//             scalar_t resample_mid = (cdf_u - cdf_prev) * scaling + starts[idx];
//             scalar_t resample_half_size = cdf_step_size * scaling * 0.5;
//             resample_starts[j] = fmaxf(resample_mid - resample_half_size, starts[idx]);
//             resample_ends[j] = fminf(resample_mid + resample_half_size, ends[idx]);
//             // going further to next resample
//             cdf_u += cdf_step_size;
//             j += 1;
//         }
//         else
//         {
//             // go to next interval
//             idx += 1;
//             if (idx == steps)
//                 break;
//             cdf_prev = cdf_next;
//             cdf_next += (weights[idx] + padding_step) / weights_sum;
//         }
//     }
//     if (j != resample_steps)
//     {
//         printf("Error: %d %d %f\n", j, resample_steps, weights_sum);
//     }
//     return;
// }

std::vector<torch::Tensor> ray_resampling(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    // torch::Tensor resample_num_steps = torch::clamp(num_steps, 0, steps);
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_starts = torch::zeros({total_steps, 1}, starts.options());
    torch::Tensor resample_ends = torch::zeros({total_steps, 1}, ends.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               weights.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_starts.data_ptr<scalar_t>(),
               resample_ends.data_ptr<scalar_t>()); }));

    return {resample_packed_info, resample_starts, resample_ends};
}

torch::Tensor ray_pdf_query(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor pdfs,
    torch::Tensor resample_packed_info,
    torch::Tensor resample_starts,
    torch::Tensor resample_ends)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(pdfs);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(pdfs.ndimension() == 2 & pdfs.size(1) == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_resamples = resample_starts.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor resample_pdfs = torch::zeros({n_resamples, 1}, pdfs.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        pdfs.scalar_type(),
        "pdf_query",
        ([&]
         { pdf_query_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               pdfs.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               resample_starts.data_ptr<scalar_t>(),
               resample_ends.data_ptr<scalar_t>(),
               // outputs
               resample_pdfs.data_ptr<scalar_t>()); }));

    return resample_pdfs;
}



std::vector<torch::Tensor> invert_cdf(
    torch::Tensor src_bins,
    torch::Tensor s0,
    torch::Tensor s1,
    torch::Tensor w,
    torch::Tensor tgt_bins,
    torch::Tensor cdf_u0,
    torch::Tensor cdf_u1)
{
    DEVICE_GUARD(src_bins);

    TORCH_CHECK(src_bins.size(0) == tgt_bins.size(0));

    const uint32_t n_rays = src_bins.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    auto t0 = torch::empty_like(cdf_u0);
    auto t1 = torch::empty_like(cdf_u1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        s0.scalar_type(),
        "invert_cdf",
        ([&]
         { invert_cdf_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               src_bins.data_ptr<int>(),
               s0.data_ptr<scalar_t>(),
               s1.data_ptr<scalar_t>(),
               w.data_ptr<scalar_t>(),
               tgt_bins.data_ptr<int>(),
               cdf_u0.data_ptr<scalar_t>(),
               cdf_u1.data_ptr<scalar_t>(),
               // outputs
               t0.data_ptr<scalar_t>(),
               t1.data_ptr<scalar_t>()); }));

    return {t0, t1};
}