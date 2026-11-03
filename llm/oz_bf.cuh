// oz_bf.cuh -- BF16 rounding and swiglu_back's two outputs as BF16 bits, one source for the host and the GPU (with
// oz_exp.cuh's exp). oz_bfb is ultra.cu's bf() followed by jais_lora.cu's bfb(): the double rounded once to BF16, nearest even
// (the significand's bits for |v| >= 2^-126, frexp/ldexp/rint below), then its upper 16 bits. oz_swiglu_bf is fill_gu's
// formula: da = 0 + y (as the dgrad add made it: -0 becomes +0), s = 1 / (1 + exp(-g)),
// dg = ((da u) s)(1 + g (1 - s)), du = (da g) s, every operation one IEEE operation in that order. llm/fill_check.cu proves
// the GPU's bits equal the host's, and the host's equal jais_lora.cu's fill_gu.
#pragma once
#include "oz_exp.cuh"
#include <cfloat>
#ifdef __CUDA_ARCH__
#define OZX_DIV(a, b) __ddiv_rn(a, b)
#else
#define OZX_DIV(a, b) ((a) / (b))
#endif

__host__ __device__ inline uint16_t oz_bfb(double v) {
    double a = fabs(v); float f;
    if (a >= 0x1p-126 && a <= DBL_MAX) {
        uint64_t u = ozx_bits(v); u += ((1ULL << 44) - 1) + ((u >> 45) & 1); u &= ~((1ULL << 45) - 1); f = (float)ozx_dbl(u);
    } else if (v == 0 || !(a <= DBL_MAX)) f = (float)v;   // +-0, +-inf, NaN
    else { int ex; frexp(v, &ex); int q = ex - 8 > -133 ? ex - 8 : -133; f = (float)ldexp(rint(ldexp(v, -q)), q); }
    uint32_t b;
#ifdef __CUDA_ARCH__
    b = __float_as_uint(f);
#else
    memcpy(&b, &f, 4);
#endif
    return (uint16_t)(b >> 16);
}

__host__ __device__ inline void oz_swiglu_bf(float gf, float uf, float yf, uint16_t &xg, uint16_t &xu) {
    double g = gf, da = OZX_ADD(0.0, (double)yf), s = OZX_DIV(1.0, OZX_ADD(1.0, oz_exp(-g)));
    xg = oz_bfb(OZX_MUL(OZX_MUL(OZX_MUL(da, (double)uf), s), OZX_ADD(1.0, OZX_MUL(g, OZX_SUB(1.0, s)))));
    xu = oz_bfb(OZX_MUL(OZX_MUL(da, g), s));
}
