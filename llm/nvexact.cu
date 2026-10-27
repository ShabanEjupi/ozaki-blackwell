// nvexact.cu -- correctly rounded NVFP4 x NVFP4 products (W4A4) on INT8 tensor cores, on the real weights of NVIDIA
// Nemotron 3 Ultra (560 524 578 816 parameters; weights, recipe and 14.8T tokens of its pre-training data are published,
// and it was pre-trained in NVFP4: the routed experts ship as E2M1 values with one E4M3 scale per 16 and an FP32 global
// scale, activations quantized the same way with a static global scale -- ModelOpt "group_1", 4 bits, group 16).
//
// Why it is exact. w = qw/2 * sW * gW and a = qa/2 * sA * gA with qw, qa in {0, +-1, 2, 3, 4, 6, 8, 12} (twice E2M1) and
// s = sigma 2^eps an E4M3 scale (sigma = 8 + mantissa, eps = exp - 10; subnormal: sigma = mantissa, eps = -9). Per block b
// of 16, p_b = sum qw qa is one m16n8k16 INT8 MMA (|p_b| <= 16 * 144 = 2304), c_b = sigmaW sigmaA <= 225 and
//     y = gW gA 2^-20 * sum_b c_b p_b 2^(epsW + epsA + 18),   0 <= epsW + epsA + 18 <= 28,
// so every term is below 2^47 and the sum of K/16 <= 320 terms below 2^56: an int64 holds it without loss. gW gA is an
// exact 48-bit integer times a power of two; the 104-bit product is rounded once to FP32. The result is the real number
// the quantized model defines, rounded once: it cannot depend on tiling, split-K, batch size or the GPU.
//
// The baseline is what block-scaled FP4 tensor cores and CUTLASS do: each block term (exact in FP32) is added into an
// FP32 accumulator in k order, split-K slices are reduced in order, and the epilogue multiplies by alpha = gW * gA
// rounded to FP32. Its bits change with the split.
//
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_nv llm/nvexact.cu
// out_nv <dir> <expert prefix, e.g. backbone.layers.1.mixer.experts.0> <N tokens> [seed] [outdir]
// One routed expert end to end: X = BF16(h fc1_latent^T), h ~ N(0,1) -> up_proj -> BF16(relu(.)^2) -> down_proj.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// E4M3 byte -> sigma | (eps + 32) << 8 (sign ignored: block scales are positive)
__host__ __device__ inline int e4m3_sig_eps(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? (m | ((-9 + 32) << 8)) : ((8 + m) | ((e - 10 + 32) << 8));
}
__host__ __device__ inline float e4m3_val(uint8_t b) { int v = e4m3_sig_eps(b); return ldexpf((float)(v & 255), (v >> 8) - 32); }

// float (>= 0) -> E4M3, round to nearest even, saturating at 448
__host__ __device__ inline uint8_t to_e4m3(float x) {
    if (!(x > 0.f)) return 0;
    if (x >= 448.f) return 0x7E;
    int ex; frexpf(x, &ex);                         // x = f 2^ex, f in [0.5, 1)
    int E = ex - 1;                                 // x in [2^E, 2^(E+1))
    int q = E < -6 ? -9 : E - 3;                    // ulp exponent: 2^(E-3) for normals, 2^-9 below 2^-6
    float t = ldexpf(x, -q);                        // exact: x has 24 bits, we only scale
    float r = rintf(t);                             // ties to even
    int s = (int)r;                                 // sigma: 0..16
    if (s == 0) return 0;
    if (q == -9 && s < 8) return (uint8_t)s;        // subnormal
    if (s == 16) { s = 8; q++; }
    int e = q + 10; if (e > 15 || (e == 15 && s > 14)) return 0x7E;
    return (uint8_t)((e << 3) | (s - 8));
}
// |v| -> 2 * E2M1 (0, 1, 2, 3, 4, 6, 8, 12), round to nearest even on the E2M1 grid, saturating at 6
__host__ __device__ inline int to_2e2m1(float v) {
    float a = fabsf(v), t2;
    if (a >= 6.f) t2 = 12.f;
    else if (a > 4.f) t2 = a < 5.f ? 8.f : (a > 5.f ? 12.f : 8.f);          // grid step 2 between 4 and 6; tie -> 4 (even)
    else if (a > 2.f) t2 = 2.f * rintf(a);                                  // step 1 from 2 to 4: ties to 2 and 4 (even codes)
    else t2 = rintf(2.f * a);                                               // step 0.5 up to 2: 2a rounded, ties to even
    int q = (int)t2;
    return v < 0 ? -q : q;
}

// ---- preprocessing ------------------------------------------------------------------------------------------------
// packed [M][K/2] (low nibble = even k) -> qw [M][K] int8 = 2 * E2M1
__global__ void unpack_w(const uint8_t *pk, int8_t *qw, size_t n2) {
    const int8_t T[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n2; i += (size_t)gridDim.x * blockDim.x) {
        uint8_t b = pk[i];
        int lo = b & 15, hi = b >> 4;
        qw[2 * i] = (lo & 8) ? -T[lo & 7] : T[lo & 7];
        qw[2 * i + 1] = (hi & 8) ? -T[hi & 7] : T[hi & 7];
    }
}
// the activations the expert sees: X = BF16(h fc1^T) for h ~ N(0,1) (the latent projection is real: fc1_latent_proj)
__global__ void latent(const __nv_bfloat16 *h, const __nv_bfloat16 *fc1, __nv_bfloat16 *X, int N, int K, int H) {
    __shared__ float hs[8][256];
    int k = blockIdx.x * blockDim.x + threadIdx.x, j0 = blockIdx.y * 8;
    float acc[8] = {0};
    for (int c = 0; c < H; c += 256) {
        __syncthreads();
        for (int t = threadIdx.x; t < 8 * 256; t += blockDim.x) hs[t / 256][t % 256] = __bfloat162float(h[(size_t)(j0 + t / 256) * H + c + t % 256]);
        __syncthreads();
        if (k < K) for (int u = 0; u < 256; u++) { float w = __bfloat162float(fc1[(size_t)k * H + c + u]); for (int j = 0; j < 8; j++) acc[j] += w * hs[j][u]; }
    }
    if (k < K) for (int j = 0; j < 8; j++) X[(size_t)(j0 + j) * K + k] = __float2bfloat16(acc[j]);
}
// ModelOpt NVFP4 activation quantization, dynamic per block of 16 with the static global scale gA:
// sA = E4M3(amax / (6 gA)), qa = 2 E2M1(x / (sA gA)). One thread per block.
__global__ void quant_x(const __nv_bfloat16 *X, int8_t *qa, uint8_t *sa, float gA, size_t nblk) {
    for (size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; b < nblk; b += (size_t)gridDim.x * blockDim.x) {
        float v[16], amax = 0.f;
        for (int i = 0; i < 16; i++) { v[i] = __bfloat162float(X[b * 16 + i]); amax = fmaxf(amax, fabsf(v[i])); }
        uint8_t s = to_e4m3(amax / (6.f * gA));
        float d = e4m3_val(s) * gA;
        sa[b] = s;
        for (int i = 0; i < 16; i++) qa[b * 16 + i] = s ? (int8_t)to_2e2m1(v[i] / d) : 0;
    }
}

// ---- 128-bit rounding: a * 2^X to nearest even FP32 -----------------------------------------------------------------
__device__ __host__ inline float round_i128(__int128 a, int X) {
    if (a == 0) return 0.f;
    bool neg = a < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-a) : (unsigned __int128)a;
    uint64_t hi = (uint64_t)(u >> 64), lo = (uint64_t)u;
#ifdef __CUDA_ARCH__
    int nb = hi ? 128 - __clzll(hi) : 64 - __clzll(lo);
#else
    int nb = hi ? 128 - __builtin_clzll(hi) : 64 - __builtin_clzll(lo);
#endif
    int Ev = nb - 1 + X, L = Ev - 23 > -149 ? Ev - 23 : -149, r = L - X;
    float f;
    if (r <= 0) f = ldexpf((float)(uint64_t)u, X);
    else if (r > 128) f = 0.f;
    else {
        unsigned __int128 q = r == 128 ? 0 : u >> r, rem = u - (r == 128 ? 0 : q << r), half = (unsigned __int128)1 << (r - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        f = ldexpf((float)(uint64_t)q, L);
    }
    return neg ? -f : f;
}

// ---- the exact product on tensor cores --------------------------------------------------------------------------------
// One warp: 16 rows x 32 tokens, one m16n8k16 INT8 MMA per (block of 16, 8 tokens) = the exact block sums p_b, then
// acc += (c_b p_b) << (epsW + epsA + 18) in int64. Block: 8 warps = 128 rows. Y [N][M] FP32, correctly rounded.
__constant__ int c_se[256];
__device__ inline void mma16816(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}
__global__ void __launch_bounds__(256) nv_exact(const int8_t *__restrict__ qw, const uint8_t *__restrict__ sw,
                                                const int8_t *__restrict__ qa, const uint8_t *__restrict__ sa,
                                                long long Gm, int Gx, float *Y, int M, int N, int K) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    int m0 = (blockIdx.x * 8 + warp) * 16, n0 = blockIdx.y * 32, B = K / 16;
    if (m0 >= M) return;
    long long acc[4][4] = {};
    for (int b = 0; b < B; b++) {
        uint32_t a0 = *(const uint32_t *)(qw + (size_t)(m0 + g) * K + b * 16 + t4 * 4);
        uint32_t a1 = *(const uint32_t *)(qw + (size_t)(m0 + g + 8) * K + b * 16 + t4 * 4);
        int sw0 = c_se[sw[(size_t)(m0 + g) * B + b]], sw1 = c_se[sw[(size_t)(m0 + g + 8) * B + b]];
#pragma unroll
        for (int s = 0; s < 4; s++) {
            uint32_t b0 = *(const uint32_t *)(qa + (size_t)(n0 + s * 8 + g) * K + b * 16 + t4 * 4);
            int d[4]; mma16816(d, a0, a1, b0);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = n0 + s * 8 + t4 * 2 + (i & 1);
                int x = c_se[sa[(size_t)col * B + b]], w = i < 2 ? sw0 : sw1;
                int c = (w & 255) * (x & 255), sh = (w >> 8) + (x >> 8) - 64 + 18;
                acc[s][i] += (long long)(c * d[i]) << sh;
            }
        }
    }
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = m0 + g + (i < 2 ? 0 : 8), col = n0 + s * 8 + t4 * 2 + (i & 1);
            Y[(size_t)col * M + row] = round_i128((__int128)acc[s][i] * Gm, Gx - 20);
        }
}
// the same exact sum, one thread per output, plain integer loops: a second, independent route on the GPU
__global__ void nv_exact_ref(const int8_t *qw, const uint8_t *sw, const int8_t *qa, const uint8_t *sa, long long Gm, int Gx,
                             float *Y, int M, int N, int K) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= (size_t)M * N) return;
    int i = o % M, j = o / M, B = K / 16;
    long long acc = 0;
    for (int b = 0; b < B; b++) {
        int p = 0;
        for (int k = 0; k < 16; k++) p += qw[(size_t)i * K + b * 16 + k] * qa[(size_t)j * K + b * 16 + k];
        int w = e4m3_sig_eps(sw[(size_t)i * B + b]), x = e4m3_sig_eps(sa[(size_t)j * B + b]);
        acc += (long long)((w & 255) * (x & 255) * p) << ((w >> 8) + (x >> 8) - 64 + 18);
    }
    Y[o] = round_i128((__int128)acc * Gm, Gx - 20);
}
// ---- baseline: FP32 accumulation of the block terms, split-K in S slices reduced in order, alpha = fp32(gW gA) -------
__global__ void nv_fp32(const int8_t *qw, const uint8_t *sw, const int8_t *qa, const uint8_t *sa, float alpha,
                        float *Y, int M, int N, int K, int S, long long Gm = 0, int Gx = 0) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= (size_t)M * N) return;
    int i = o % M, j = o / M, B = K / 16, per = B / S;
    float tot = 0.f;
    for (int z = 0; z < S; z++) {
        float acc = 0.f;
        for (int b = z * per; b < (z == S - 1 ? B : (z + 1) * per); b++) {
            int p = 0;
            for (int k = 0; k < 16; k++) p += qw[(size_t)i * K + b * 16 + k] * qa[(size_t)j * K + b * 16 + k];
            acc += (float)p * 0.25f * e4m3_val(sw[(size_t)i * B + b]) * e4m3_val(sa[(size_t)j * B + b]);   // exact term
        }
        tot += acc;
    }
    if (Gm) {                       // the same FP32 sum, but the epilogue exact: tot gW gA rounded once (isolates the summation)
        int ex; float fr = frexpf(tot, &ex);
        Y[o] = round_i128((__int128)(long long)ldexpf(fr, 24) * Gm, ex - 24 + Gx);
    } else Y[o] = tot * alpha;
}

// ---- decode: a few tokens, the weights read once as packed 4-bit -----------------------------------------------------
// Token generation reads every weight once per token: memory (or disk) bound. One warp per row; lane l takes the blocks
// l, l + 32, ... (8 packed bytes + 1 scale byte each, coalesced), unpacks the nibbles through a 256-entry table, does the
// 16-element block sum with 4 dp4a per token and adds (c_b p_b) << sh into an int64 per token; the warp reduction adds
// integers, so the order cannot matter. Bit-identical to the prefill path by construction (both are correctly rounded).
template <int NT>
__global__ void __launch_bounds__(256) nv_decode(const uint8_t *__restrict__ pk, const uint8_t *__restrict__ sw,
                                                 const int8_t *__restrict__ qa, const uint8_t *__restrict__ sa,
                                                 long long Gm, int Gx, float *Y, int M, int K) {
    __shared__ uint16_t lut[256];
    const int8_t T[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        int lo = b & 15, hi = b >> 4;
        lut[b] = (uint8_t)((lo & 8) ? -T[lo & 7] : T[lo & 7]) | ((uint16_t)(uint8_t)((hi & 8) ? -T[hi & 7] : T[hi & 7]) << 8);
    }
    __syncthreads();
    int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, B = K / 16;
    if (row >= M) return;
    long long acc[NT] = {};
    for (int b = lane; b < B; b += 32) {
        uint2 p = *(const uint2 *)(pk + (size_t)row * (K / 2) + b * 8);
        uint32_t w[4];
        w[0] = lut[p.x & 255] | (uint32_t)lut[(p.x >> 8) & 255] << 16; w[1] = lut[(p.x >> 16) & 255] | (uint32_t)lut[p.x >> 24] << 16;
        w[2] = lut[p.y & 255] | (uint32_t)lut[(p.y >> 8) & 255] << 16; w[3] = lut[(p.y >> 16) & 255] | (uint32_t)lut[p.y >> 24] << 16;
        int ws = c_se[sw[(size_t)row * B + b]];
#pragma unroll
        for (int t = 0; t < NT; t++) {
            uint4 a = *(const uint4 *)(qa + (size_t)t * K + b * 16);
            int d = __dp4a((int)w[0], (int)a.x, 0); d = __dp4a((int)w[1], (int)a.y, d);
            d = __dp4a((int)w[2], (int)a.z, d); d = __dp4a((int)w[3], (int)a.w, d);
            int xs = c_se[sa[(size_t)t * B + b]];
            acc[t] += (long long)((ws & 255) * (xs & 255) * d) << ((ws >> 8) + (xs >> 8) - 64 + 18);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        long long v = acc[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) Y[(size_t)t * M + row] = round_i128((__int128)v * Gm, Gx - 20);
    }
}
// what engines do for decode: dequantize in registers, FP32 multiply-add, warp reduction (alpha = fp32(gW gA))
template <int NT>
__global__ void __launch_bounds__(256) fp4_gemv(const uint8_t *__restrict__ pk, const uint8_t *__restrict__ sw,
                                                const int8_t *__restrict__ qa, const uint8_t *__restrict__ sa,
                                                float alpha, float *Y, int M, int K) {
    const float T[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, B = K / 16;
    if (row >= M) return;
    float acc[NT] = {};
    for (int b = lane; b < B; b += 32) {
        uint2 p = *(const uint2 *)(pk + (size_t)row * (K / 2) + b * 8);
        float s = e4m3_val(sw[(size_t)row * B + b]);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float st = s * e4m3_val(sa[(size_t)t * B + b]), part = 0.f;
#pragma unroll
            for (int k = 0; k < 16; k++) {
                int nib = ((k < 8 ? p.x : p.y) >> (4 * (k & 7))) & 15;
                part = fmaf((nib & 8 ? -T[nib & 7] : T[nib & 7]), 0.5f * qa[(size_t)t * K + b * 16 + k], part);
            }
            acc[t] = fmaf(part, st, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float v = acc[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) Y[(size_t)t * M + row] = v * alpha;
    }
}

static std::vector<uint8_t> load(const std::string &p, size_t n) {
    std::vector<uint8_t> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f || fread(v.data(), 1, n, f) != n) { fprintf(stderr, "cannot read %zu bytes of %s\n", n, p.c_str()); exit(1); }
    fclose(f);
    return v;
}
static float loadf(const std::string &p) { auto v = load(p, 4); float f; memcpy(&f, v.data(), 4); return f; }

// relu^2 of the exact up_proj output, rounded to BF16: the input of down_proj
__global__ void relu2(const float *Y, __nv_bfloat16 *X, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = fmaxf(Y[i], 0.f); X[i] = __float2bfloat16(v * v);
    }
}

// one projection: quantize X [N][K] with the checkpoint's input scale, exact product + baselines; Y [N][M] stays on the GPU
static float *project(const std::string &dir, const std::string &pre, int M, int K, int N, __nv_bfloat16 *dX, const std::string &out) {
    int B = K / 16;
    auto pk = load(dir + "/" + pre + ".weight.bin", (size_t)M * K / 2), sw = load(dir + "/" + pre + ".weight_scale.bin", (size_t)M * B);
    float gW = loadf(dir + "/" + pre + ".weight_scale_2.bin"), gA = loadf(dir + "/" + pre + ".input_scale.bin");
    uint8_t *dpk, *dsw, *dsa; int8_t *dqw, *dqa; float *Y, *Yr, *Y1, *Y4, *Yx, *Ys;
    size_t MN = (size_t)M * N;
    CK(cudaMalloc(&dpk, pk.size())); CK(cudaMalloc(&dsw, sw.size())); CK(cudaMalloc(&dqw, (size_t)M * K));
    CK(cudaMalloc(&dqa, (size_t)N * K)); CK(cudaMalloc(&dsa, (size_t)N * B));
    for (float **p : {&Y, &Yr, &Y1, &Y4, &Yx}) CK(cudaMalloc(p, MN * 4));
    CK(cudaMalloc(&Ys, (size_t)M * 32 * 4));
    CK(cudaMemcpy(dpk, pk.data(), pk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsw, sw.data(), sw.size(), cudaMemcpyHostToDevice));
    unpack_w<<<1024, 256>>>(dpk, dqw, pk.size());
    quant_x<<<1024, 256>>>(dX, dqa, dsa, gA, (size_t)N * B);
    CK(cudaGetLastError());
    int xW, xA; float fW = frexpf(gW, &xW), fA = frexpf(gA, &xA);          // gW gA = (mW mA) 2^(xW + xA - 48)
    long long Gm = (long long)ldexpf(fW, 24) * (long long)ldexpf(fA, 24); int Gx = xW + xA - 48;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto tc = [&] { nv_exact<<<dim3((M + 127) / 128, N / 32), 256>>>(dqw, dsw, dqa, dsa, Gm, Gx, Y, M, N, K); };
    tc(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < 10; r++) tc(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
    nv_exact_ref<<<(MN + 255) / 256, 256>>>(dqw, dsw, dqa, dsa, Gm, Gx, Yr, M, N, K);
    float alpha = gW * gA;
    nv_fp32<<<(MN + 255) / 256, 256>>>(dqw, dsw, dqa, dsa, alpha, Y1, M, N, K, 1);
    nv_fp32<<<(MN + 255) / 256, 256>>>(dqw, dsw, dqa, dsa, alpha, Y4, M, N, K, 4);
    nv_fp32<<<(MN + 255) / 256, 256>>>(dqw, dsw, dqa, dsa, alpha, Yx, M, N, K, 1, Gm, Gx);
    nv_exact<<<dim3((M + 127) / 128, 1), 256>>>(dqw, dsw, dqa, dsa, Gm, Gx, Ys, M, 32, K);   // the first 32 tokens alone
    CK(cudaDeviceSynchronize());
    std::vector<float> hY(MN), hR(MN), h1(MN), h4(MN), hx(MN), hs((size_t)M * 32);
    CK(cudaMemcpy(hY.data(), Y, MN * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hR.data(), Yr, MN * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h1.data(), Y1, MN * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h4.data(), Y4, MN * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), Ys, hs.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx.data(), Yx, MN * 4, cudaMemcpyDeviceToHost));
    size_t dref = 0, d1 = 0, d4 = 0, d14 = 0, dx = 0, nz = 0, dbatch = 0;
    for (size_t o = 0; o < MN; o++) {
        dref += memcmp(&hY[o], &hR[o], 4) != 0; d1 += memcmp(&hY[o], &h1[o], 4) != 0;
        d4 += memcmp(&hY[o], &h4[o], 4) != 0; d14 += memcmp(&h1[o], &h4[o], 4) != 0; dx += memcmp(&hY[o], &hx[o], 4) != 0; nz += hY[o] != 0.f;
    }
    for (size_t o = 0; o < hs.size(); o++) dbatch += memcmp(&hs[o], &hY[o], 4) != 0;
    printf("%s  M %d K %d N %d  gW %.6e gA %.6e  outputs %zu (nonzero %zu)\n", pre.c_str(), M, K, N, gW, gA, MN, nz);
    printf("  exact (INT8 MMA m16n8k16 + int64) %.3f ms = %.1f TOPS | integer loops: %zu differ | 32 tokens alone vs batch %d: %zu differ\n",
           ms, 2.0 * MN * K / ms / 1e9, dref, N, dbatch);
    printf("  FP32 accumulation of the block terms: split 1 not correctly rounded %zu (%.2f %%), split 4 %zu (%.2f %%), "
           "split 1 vs 4 differ %zu (%.2f %%); FP32 sum with an exact epilogue: %zu (%.2f %%)\n", d1, 100.0 * d1 / MN, d4, 100.0 * d4 / MN, d14, 100.0 * d14 / MN, dx, 100.0 * dx / MN);
    // decode: 1 and 8 tokens straight from the packed nibbles, timed against the FP32 GEMV; bits must equal the prefill path
    float *Yd; CK(cudaMalloc(&Yd, (size_t)M * 8 * 4));
    double wbytes = (double)M * K / 2 + (double)M * B;
    auto dec = [&](int nt, bool exact) {
        auto launch = [&] {
            if (exact) { if (nt == 1) nv_decode<1><<<(M + 7) / 8, 256>>>(dpk, dsw, dqa, dsa, Gm, Gx, Yd, M, K);
                         else nv_decode<8><<<(M + 7) / 8, 256>>>(dpk, dsw, dqa, dsa, Gm, Gx, Yd, M, K); }
            else { if (nt == 1) fp4_gemv<1><<<(M + 7) / 8, 256>>>(dpk, dsw, dqa, dsa, alpha, Yd, M, K);
                   else fp4_gemv<8><<<(M + 7) / 8, 256>>>(dpk, dsw, dqa, dsa, alpha, Yd, M, K); }
        };
        launch(); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for (int r = 0; r < 100; r++) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float t; cudaEventElapsedTime(&t, e0, e1); t /= 100;
        std::vector<float> h((size_t)M * nt); CK(cudaMemcpy(h.data(), Yd, h.size() * 4, cudaMemcpyDeviceToHost));
        size_t d = 0; for (size_t o = 0; o < h.size(); o++) d += memcmp(&h[o], &hY[o], 4) != 0;
        printf("    %s %d token%s: %.4f ms = %.0f GB/s of weights, %zu of %zu outputs differ from the exact prefill\n",
               exact ? "decode exact (dp4a + int64)" : "decode FP32 GEMV (engines)", nt, nt > 1 ? "s" : " ", t, wbytes / t / 1e6, d, h.size());
        return d;
    };
    size_t ddec = dec(1, true) + dec(8, true); dec(1, false); dec(8, false);
    CK(cudaFree(Yd));
    if (ddec) { fprintf(stderr, "DECODE DIFFERS FROM PREFILL\n"); exit(2); }
    if (dref || dbatch) { fprintf(stderr, "EXACT PATHS DISAGREE\n"); exit(2); }
    if (!out.empty()) {   // for llm/nv_check.py: quantized activations, their scales, the three outputs
        std::string d = out + "/" + pre.substr(pre.rfind('.') + 1);
        if (system(("mkdir -p " + d).c_str())) exit(1);
        auto dump = [&](const char *n, const void *p, size_t b) { FILE *f = fopen((d + "/" + n).c_str(), "wb"); fwrite(p, 1, b, f); fclose(f); };
        std::vector<int8_t> q((size_t)N * K); std::vector<uint8_t> s((size_t)N * B);
        CK(cudaMemcpy(q.data(), dqa, q.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s.data(), dsa, s.size(), cudaMemcpyDeviceToHost));
        dump("qa.i8", q.data(), q.size()); dump("sa.e4m3", s.data(), s.size());
        dump("Y.f32", hY.data(), MN * 4); dump("Yb1.f32", h1.data(), MN * 4); dump("Yb4.f32", h4.data(), MN * 4);
        FILE *f = fopen((d + "/meta.txt").c_str(), "w"); fprintf(f, "%d %d %d %s\n", M, K, N, pre.c_str()); fclose(f);
    }
    for (void *p : {(void *)dpk, (void *)dsw, (void *)dqw, (void *)dqa, (void *)dsa, (void *)Yr, (void *)Y1, (void *)Y4, (void *)Yx, (void *)Ys}) CK(cudaFree(p));
    return Y;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "out_nv <dir> <expert prefix> <N> [seed] [outdir]\n"); return 1; }
    std::string dir = argv[1], pre = argv[2];
    int N = (atoi(argv[3]) + 31) / 32 * 32; unsigned seed = argc > 4 ? atoi(argv[4]) : 1; std::string out = argc > 5 ? argv[5] : "";
    const int L = 2048, F = 5120, H = 8192;                                   // Ultra: latent, expert width, hidden
    int se[256]; for (int b = 0; b < 256; b++) se[b] = e4m3_sig_eps((uint8_t)b);
    CK(cudaMemcpyToSymbol(c_se, se, sizeof se));
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> h((size_t)N * H);
    for (auto &v : h) v = __float2bfloat16(nd(rng));
    auto fc1 = load(dir + "/" + pre.substr(0, pre.find("experts")) + "fc1_latent_proj.weight.bin", (size_t)L * H * 2);
    __nv_bfloat16 *dh, *df, *dX, *dX2;
    CK(cudaMalloc(&dh, h.size() * 2)); CK(cudaMalloc(&df, fc1.size())); CK(cudaMalloc(&dX, (size_t)N * L * 2)); CK(cudaMalloc(&dX2, (size_t)N * F * 2));
    CK(cudaMemcpy(dh, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(df, fc1.data(), fc1.size(), cudaMemcpyHostToDevice));
    latent<<<dim3(L / 256, N / 8), 256>>>(dh, df, dX, N, L, H);
    CK(cudaGetLastError()); CK(cudaFree(dh)); CK(cudaFree(df));
    float *Yup = project(dir, pre + ".up_proj", F, L, N, dX, out);
    relu2<<<1024, 256>>>(Yup, dX2, (size_t)N * F);
    float *Ydn = project(dir, pre + ".down_proj", L, F, N, dX2, out);
    if (!out.empty()) {
        std::vector<__nv_bfloat16> x((size_t)N * L);
        CK(cudaMemcpy(x.data(), dX, x.size() * 2, cudaMemcpyDeviceToHost));
        FILE *f = fopen((out + "/X_latent.bf16").c_str(), "wb"); fwrite(x.data(), 2, x.size(), f); fclose(f);
    }
    CK(cudaFree(Yup)); CK(cudaFree(Ydn));
    return 0;
}
