// oz_exp.cuh -- exp(x) as ONE fixed sequence of IEEE-754 binary64 operations: the same bits on the host and on the GPU, and
// on any host, whatever its libm. The sequence is the one glibc 2.39 runs on x86-64 with FMA and AVX2 (__ieee754_exp_fma,
// read instruction by instruction from Ubuntu's libm.so.6 2.39-0ubuntu8.9, the library the training's host code calls):
//   kd = fma(x, 128/ln2, 0x1.8p52), ki = bits(kd), kd -= 0x1.8p52
//   r  = fma(kd, -ln2lo/128, fma(kd, -ln2hi/128, x))
//   tmp = fma(r2*r2, fma(r, C5, C4), fma(fma(r, C3, C2), r2, r + tail)),  r2 = r*r
//   exp(x) = fma(scale, tmp, scale), scale = 2^(k/128) from the table; near the overflow/underflow limits glibc's
//   specialcase, its subnormal branch without fma.
// Algorithm and table: Szabolcs Nagy, Arm optimized-routines v21.02, math/exp.c and math/exp_data.c (MIT License,
// Copyright (c) 1999-2019 Arm Limited; the notice is in llm/oz_exp.LICENSE). The table below is that file's N = 128 table,
// word for word the 256 words inside the libm.so.6 named above (both compared 03-11-2026).
// llm/oz_exp_check.cu proves: host oz_exp = the host's exp() and GPU oz_exp = host oz_exp, for every one of the 2^32 FP32
// inputs the MoE backward gives it (x = -(double)f), plus 2^32 random doubles.
// Device code uses the _rn intrinsics so nvcc cannot contract or reorder; host code calls std::fma (exact by IEEE) and
// plain + - * (the training's host build has no FMA instructions, so nothing is contracted either).
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

#define OZX_TAB { \
    0x0000000000000000ull, 0x3ff0000000000000ull, 0x3c9b3b4f1a88bf6eull, 0x3feff63da9fb3335ull, \
    0xbc7160139cd8dc5dull, 0x3fefec9a3e778061ull, 0xbc905e7a108766d1ull, 0x3fefe315e86e7f85ull, \
    0x3c8cd2523567f613ull, 0x3fefd9b0d3158574ull, 0xbc8bce8023f98efaull, 0x3fefd06b29ddf6deull, \
    0x3c60f74e61e6c861ull, 0x3fefc74518759bc8ull, 0x3c90a3e45b33d399ull, 0x3fefbe3ecac6f383ull, \
    0x3c979aa65d837b6dull, 0x3fefb5586cf9890full, 0x3c8eb51a92fdeffcull, 0x3fefac922b7247f7ull, \
    0x3c3ebe3d702f9cd1ull, 0x3fefa3ec32d3d1a2ull, 0xbc6a033489906e0bull, 0x3fef9b66affed31bull, \
    0xbc9556522a2fbd0eull, 0x3fef9301d0125b51ull, 0xbc5080ef8c4eea55ull, 0x3fef8abdc06c31ccull, \
    0xbc91c923b9d5f416ull, 0x3fef829aaea92de0ull, 0x3c80d3e3e95c55afull, 0x3fef7a98c8a58e51ull, \
    0xbc801b15eaa59348ull, 0x3fef72b83c7d517bull, 0xbc8f1ff055de323dull, 0x3fef6af9388c8deaull, \
    0x3c8b898c3f1353bfull, 0x3fef635beb6fcb75ull, 0xbc96d99c7611eb26ull, 0x3fef5be084045cd4ull, \
    0x3c9aecf73e3a2f60ull, 0x3fef54873168b9aaull, 0xbc8fe782cb86389dull, 0x3fef4d5022fcd91dull, \
    0x3c8a6f4144a6c38dull, 0x3fef463b88628cd6ull, 0x3c807a05b0e4047dull, 0x3fef3f49917ddc96ull, \
    0x3c968efde3a8a894ull, 0x3fef387a6e756238ull, 0x3c875e18f274487dull, 0x3fef31ce4fb2a63full, \
    0x3c80472b981fe7f2ull, 0x3fef2b4565e27cddull, 0xbc96b87b3f71085eull, 0x3fef24dfe1f56381ull, \
    0x3c82f7e16d09ab31ull, 0x3fef1e9df51fdee1ull, 0xbc3d219b1a6fbffaull, 0x3fef187fd0dad990ull, \
    0x3c8b3782720c0ab4ull, 0x3fef1285a6e4030bull, 0x3c6e149289cecb8full, 0x3fef0cafa93e2f56ull, \
    0x3c834d754db0abb6ull, 0x3fef06fe0a31b715ull, 0x3c864201e2ac744cull, 0x3fef0170fc4cd831ull, \
    0x3c8fdd395dd3f84aull, 0x3feefc08b26416ffull, 0xbc86a3803b8e5b04ull, 0x3feef6c55f929ff1ull, \
    0xbc924aedcc4b5068ull, 0x3feef1a7373aa9cbull, 0xbc9907f81b512d8eull, 0x3feeecae6d05d866ull, \
    0xbc71d1e83e9436d2ull, 0x3feee7db34e59ff7ull, 0xbc991919b3ce1b15ull, 0x3feee32dc313a8e5ull, \
    0x3c859f48a72a4c6dull, 0x3feedea64c123422ull, 0xbc9312607a28698aull, 0x3feeda4504ac801cull, \
    0xbc58a78f4817895bull, 0x3feed60a21f72e2aull, 0xbc7c2c9b67499a1bull, 0x3feed1f5d950a897ull, \
    0x3c4363ed60c2ac11ull, 0x3feece086061892dull, 0x3c9666093b0664efull, 0x3feeca41ed1d0057ull, \
    0x3c6ecce1daa10379ull, 0x3feec6a2b5c13cd0ull, 0x3c93ff8e3f0f1230ull, 0x3feec32af0d7d3deull, \
    0x3c7690cebb7aafb0ull, 0x3feebfdad5362a27ull, 0x3c931dbdeb54e077ull, 0x3feebcb299fddd0dull, \
    0xbc8f94340071a38eull, 0x3feeb9b2769d2ca7ull, 0xbc87deccdc93a349ull, 0x3feeb6daa2cf6642ull, \
    0xbc78dec6bd0f385full, 0x3feeb42b569d4f82ull, 0xbc861246ec7b5cf6ull, 0x3feeb1a4ca5d920full, \
    0x3c93350518fdd78eull, 0x3feeaf4736b527daull, 0x3c7b98b72f8a9b05ull, 0x3feead12d497c7fdull, \
    0x3c9063e1e21c5409ull, 0x3feeab07dd485429ull, 0x3c34c7855019c6eaull, 0x3feea9268a5946b7ull, \
    0x3c9432e62b64c035ull, 0x3feea76f15ad2148ull, 0xbc8ce44a6199769full, 0x3feea5e1b976dc09ull, \
    0xbc8c33c53bef4da8ull, 0x3feea47eb03a5585ull, 0xbc845378892be9aeull, 0x3feea34634ccc320ull, \
    0xbc93cedd78565858ull, 0x3feea23882552225ull, 0x3c5710aa807e1964ull, 0x3feea155d44ca973ull, \
    0xbc93b3efbf5e2228ull, 0x3feea09e667f3bcdull, 0xbc6a12ad8734b982ull, 0x3feea012750bdabfull, \
    0xbc6367efb86da9eeull, 0x3fee9fb23c651a2full, 0xbc80dc3d54e08851ull, 0x3fee9f7df9519484ull, \
    0xbc781f647e5a3ecfull, 0x3fee9f75e8ec5f74ull, 0xbc86ee4ac08b7db0ull, 0x3fee9f9a48a58174ull, \
    0xbc8619321e55e68aull, 0x3fee9feb564267c9ull, 0x3c909ccb5e09d4d3ull, 0x3feea0694fde5d3full, \
    0xbc7b32dcb94da51dull, 0x3feea11473eb0187ull, 0x3c94ecfd5467c06bull, 0x3feea1ed0130c132ull, \
    0x3c65ebe1abd66c55ull, 0x3feea2f336cf4e62ull, 0xbc88a1c52fb3cf42ull, 0x3feea427543e1a12ull, \
    0xbc9369b6f13b3734ull, 0x3feea589994cce13ull, 0xbc805e843a19ff1eull, 0x3feea71a4623c7adull, \
    0xbc94d450d872576eull, 0x3feea8d99b4492edull, 0x3c90ad675b0e8a00ull, 0x3feeaac7d98a6699ull, \
    0x3c8db72fc1f0eab4ull, 0x3feeace5422aa0dbull, 0xbc65b6609cc5e7ffull, 0x3feeaf3216b5448cull, \
    0x3c7bf68359f35f44ull, 0x3feeb1ae99157736ull, 0xbc93091fa71e3d83ull, 0x3feeb45b0b91ffc6ull, \
    0xbc5da9b88b6c1e29ull, 0x3feeb737b0cdc5e5ull, 0xbc6c23f97c90b959ull, 0x3feeba44cbc8520full, \
    0xbc92434322f4f9aaull, 0x3feebd829fde4e50ull, 0xbc85ca6cd7668e4bull, 0x3feec0f170ca07baull, \
    0x3c71affc2b91ce27ull, 0x3feec49182a3f090ull, 0x3c6dd235e10a73bbull, 0x3feec86319e32323ull, \
    0xbc87c50422622263ull, 0x3feecc667b5de565ull, 0x3c8b1c86e3e231d5ull, 0x3feed09bec4a2d33ull, \
    0xbc91bbd1d3bcbb15ull, 0x3feed503b23e255dull, 0x3c90cc319cee31d2ull, 0x3feed99e1330b358ull, \
    0x3c8469846e735ab3ull, 0x3feede6b5579fdbfull, 0xbc82dfcd978e9db4ull, 0x3feee36bbfd3f37aull, \
    0x3c8c1a7792cb3387ull, 0x3feee89f995ad3adull, 0xbc907b8f4ad1d9faull, 0x3feeee07298db666ull, \
    0xbc55c3d956dcaebaull, 0x3feef3a2b84f15fbull, 0xbc90a40e3da6f640ull, 0x3feef9728de5593aull, \
    0xbc68d6f438ad9334ull, 0x3feeff76f2fb5e47ull, 0xbc91eee26b588a35ull, 0x3fef05b030a1064aull, \
    0x3c74ffd70a5fddcdull, 0x3fef0c1e904bc1d2ull, 0xbc91bdfbfa9298acull, 0x3fef12c25bd71e09ull, \
    0x3c736eae30af0cb3ull, 0x3fef199bdd85529cull, 0x3c8ee3325c9ffd94ull, 0x3fef20ab5fffd07aull, \
    0x3c84e08fd10959acull, 0x3fef27f12e57d14bull, 0x3c63cdaf384e1a67ull, 0x3fef2f6d9406e7b5ull, \
    0x3c676b2c6c921968ull, 0x3fef3720dcef9069ull, 0xbc808a1883ccb5d2ull, 0x3fef3f0b555dc3faull, \
    0xbc8fad5d3ffffa6full, 0x3fef472d4a07897cull, 0xbc900dae3875a949ull, 0x3fef4f87080d89f2ull, \
    0x3c74a385a63d07a7ull, 0x3fef5818dcfba487ull, 0xbc82919e2040220full, 0x3fef60e316c98398ull, \
    0x3c8e5a50d5c192acull, 0x3fef69e603db3285ull, 0x3c843a59ac016b4bull, 0x3fef7321f301b460ull, \
    0xbc82d52107b43e1full, 0x3fef7c97337b9b5full, 0xbc892ab93b470dc9ull, 0x3fef864614f5a129ull, \
    0x3c74b604603a88d3ull, 0x3fef902ee78b3ff6ull, 0x3c83c5ec519d7271ull, 0x3fef9a51fbc74c83ull, \
    0xbc8ff7128fd391f0ull, 0x3fefa4afa2a490daull, 0xbc8dae98e223747dull, 0x3fefaf482d8e67f1ull, \
    0x3c8ec3bc41aa2008ull, 0x3fefba1bee615a27ull, 0x3c842b94c3a9eb32ull, 0x3fefc52b376bba97ull, \
    0x3c8a64a931d185eeull, 0x3fefd0765b6e4540ull, 0xbc8e37bae43be3edull, 0x3fefdbfdad9cbe14ull, \
    0x3c77893b4d91cd9dull, 0x3fefe7c1819e90d8ull, 0x3c5305c14160cc89ull, 0x3feff3c22b8f71f1ull, \
}
static __device__ const uint64_t ozx_tab_d[256] = OZX_TAB;
static const uint64_t ozx_tab_h[256] = OZX_TAB;

#ifdef __CUDA_ARCH__
#define OZX_FMA(a, b, c) __fma_rn(a, b, c)
#define OZX_MUL(a, b) __dmul_rn(a, b)
#define OZX_ADD(a, b) __dadd_rn(a, b)
#define OZX_SUB(a, b) __dsub_rn(a, b)
#define OZX_T ozx_tab_d
#else
#define OZX_FMA(a, b, c) std::fma(a, b, c)
#define OZX_MUL(a, b) ((a) * (b))
#define OZX_ADD(a, b) ((a) + (b))
#define OZX_SUB(a, b) ((a) - (b))
#define OZX_T ozx_tab_h
#endif

__host__ __device__ inline uint64_t ozx_bits(double x) {
#ifdef __CUDA_ARCH__
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t u; memcpy(&u, &x, 8); return u;
#endif
}
__host__ __device__ inline double ozx_dbl(uint64_t u) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double((long long)u);
#else
    double x; memcpy(&x, &u, 8); return x;
#endif
}

// near the limits: k > 0 may overflow the exponent of scale by <= 460, k < 0 may land in the subnormal range
__host__ __device__ inline double ozx_special(double tmp, uint64_t sbits, uint64_t ki) {
    if ((ki & 0x80000000u) == 0) {
        double scale = ozx_dbl(sbits - (1009ull << 52));
        return OZX_MUL(0x1p1009, OZX_FMA(scale, tmp, scale));
    }
    double scale = ozx_dbl(sbits + (1022ull << 52)), st = OZX_MUL(scale, tmp), y = OZX_ADD(scale, st);
    if (y < 1.0) {                                     // round to the right precision before scaling (no double rounding)
        double hi = OZX_ADD(y, 1.0), lo = OZX_ADD(OZX_SUB(scale, y), st);
        lo = OZX_ADD(OZX_ADD(OZX_SUB(1.0, hi), y), lo);
        y = OZX_SUB(OZX_ADD(lo, hi), 1.0);
        if (y == 0.0) y = 0.0;
    }
    return OZX_MUL(y, 0x1p-1022);
}

__host__ __device__ inline double oz_exp(double x) {
    uint64_t ix = ozx_bits(x);
    uint32_t abstop = (uint32_t)(ix >> 52) & 0x7ff;
    if (abstop - 0x3c9u >= 0x3fu) {                    // not 2^-54 <= |x| < 512
        if ((int32_t)(abstop - 0x3c9u) < 0) return OZX_ADD(1.0, x);     // |x| < 2^-54 (and +-0)
        if (abstop >= 0x409) {                         // |x| >= 1024, inf, nan
            if (ix == 0xfff0000000000000ull) return 0.0;
            if (abstop == 0x7ff) return OZX_ADD(1.0, x);
            return (ix >> 63) ? 0.0 : INFINITY;        // glibc's __math_uflow(0) / __math_oflow(0)
        }
        abstop = 0;                                    // 512 <= |x| < 1024: ozx_special
    }
    double kd = OZX_FMA(x, 0x1.71547652b82fep7, 0x1.8p52);
    uint64_t ki = ozx_bits(kd);
    kd = OZX_SUB(kd, 0x1.8p52);
    double r = OZX_FMA(kd, -0x1.cf79abc9e3b3ap-47, OZX_FMA(kd, -0x1.62e42fefa0000p-8, x));
    uint64_t i2 = 2 * (ki % 128), sbits = OZX_T[i2 + 1] + (ki << 45);
    double tail = ozx_dbl(OZX_T[i2]);
    double r2 = OZX_MUL(r, r);
    double t1 = OZX_FMA(OZX_FMA(r, 0x1.555555555543cp-3, 0x1.ffffffffffdbdp-2), r2, OZX_ADD(r, tail));
    double tmp = OZX_FMA(OZX_MUL(r2, r2), OZX_FMA(r, 0x1.1111167a4d017p-7, 0x1.55555cf172b91p-5), t1);
    if (abstop == 0) return ozx_special(tmp, sbits, ki);
    double scale = ozx_dbl(sbits);
    return OZX_FMA(scale, tmp, scale);
}
