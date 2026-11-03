// moe_dev.cuh -- the MoE backward's two fills on the GPU, the host's bits by construction (oz_bf.cuh, _rn operations in the
// host loops' order); the building blocks of the GPU-resident MoE backward (README, "Next"). llm/fill_check.cu proves them.
//  k_fill_down: rows i of an expert (or of a group, rows one after another): x_i = bfb(w_i g_{r_i}) into the down product's
//    input (D columns), and dw_i = sum_c g_{r_i,c} y_{i,c} in ascending c -- jais_lora.cu's fill_down.
//  k_fill_gu: x_i = [bfb(dg) | bfb(du)] of swiglu_back from gg, uu (FP32) and the down product's FP32 outputs -- fill_gu.
#pragma once
#include "oz_bf.cuh"
__global__ void k_fill_down(const double *g, const int *rows, const double *w, const float *y, int n, int D, uint16_t *xb, double *dw) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)n * D; t += (size_t)gridDim.x * blockDim.x) {
        size_t i = t / D, c = t % D;
        xb[t] = oz_bfb(__dmul_rn(w[i], g[(size_t)rows[i] * D + c]));
    }
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n; i += (size_t)gridDim.x * blockDim.x) {
        const double *gr = g + (size_t)rows[i] * D; const float *yi = y + i * D; double d = 0;
        for (int c = 0; c < D; c++) d = __dadd_rn(d, __dmul_rn(gr[c], (double)yi[c]));
        dw[i] = d;
    }
}
__global__ void k_fill_gu(const float *gg, const float *uu, const float *yd, int n, int F, uint16_t *xb) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)n * F; t += (size_t)gridDim.x * blockDim.x) {
        size_t i = t / F, c = t % F; uint16_t a, b;
        oz_swiglu_bf(gg[t], uu[t], yd[t], a, b);
        xb[i * 2 * F + c] = a; xb[i * 2 * F + F + c] = b;
    }
}
