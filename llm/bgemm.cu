// bgemm.cu -- the linear maps of Jais-3 for many tokens at once on INT8 tensor cores, correctly rounded, so equal bit for
// bit to the per-token GEMV of ultra.cu: the batched evaluation of a benchmark (2500 questions x ~200 tokens) needs GEMMs,
// and correct rounding makes batching invisible. Two formats:
// FP8 (routed experts): W = E4M3 times a BF16 scale per 128x128 block; X = E4M3 times an FP32 scale per token and group of
// 128 (act_fp8g). Exact. An E4M3 value times 2^9 is an integer v, |v| <= 448 * 512 < 2^18: normal (8 + m) 2^(e-1), subnormal m.
// Three balanced base-128 digits in [-64, 63] hold it (v = d0 + 128 d1 + 16384 d2), so a block sum S_b = sum_k vw va over
// the 128 k of a block is 9 digit-pair sums, each a chain of 8 m16n8k16 INT8 MMAs (|d d'| <= 2^12, 128 terms < 2^19 in
// int32), recombined as sum P_pq << 7(p+q) in int64 (|S_b| <= 128 * 229376^2 < 2^43). The block term S_b sig(ws) sig(xs)
// 2^(e(ws) + e(xs) - 18) is below 2^75 times its power of two; the K/128 terms go into a 128-bit window anchored at the
// largest block exponent, bits below it counted as tail and the rounding certified (cert_round of ultra.cu).
// BF16 (attention, router, shared expert, dense MLP): exponents are free per element, so each block of 128 of a weight row
// and of a token is written as integers v = floor(x / 2^l) on a common grid l = (top bit of the block) - 19 (|v| < 2^20,
// three digits again). Elements with bits below l are counted; the dropped part is below one unit each, so the true block
// product differs from 2^(lw+lx) S_b by less than 2^(lw+lx) ((nw + nx) 2^20 + min(nw, nx)) -- that bound goes into the
// tail, and cert_round decides. An output it cannot certify comes back as NaN and is recomputed by the GEMV: the batched
// result is always the correctly rounded one; how often the GEMV is needed is the price, measured below.
// nvcc -O3 -arch=sm_120a -std=c++17 -o out_bgemm llm/bgemm.cu
// out_bgemm <model dir> <tensor without .weight> <M> <K> [N tokens, default 4096]: the batched product (format from the
// checkpoint) against the per-token GEMV on the first 256 tokens (BF16 and FP32 outputs), then its time alone.
#pragma once
#define ULTRA_LIB
#include "ultra.cu"

// E4M3 bytes -> three planes of balanced base-128 digits of v = value * 2^9
__global__ void fp8_digits(const uint8_t *q, int8_t *d, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint8_t b = q[i]; int e = (b >> 3) & 15, m = b & 7, v = e ? (8 + m) << (e - 1) : m;
        if (b & 0x80) v = -v;
        int d0 = ((v + 64) & 127) - 64; v = (v - d0) >> 7;
        int d1 = ((v + 64) & 127) - 64; v = (v - d1) >> 7;
        d[i] = (int8_t)d0; d[n + i] = (int8_t)d1; d[2 * n + i] = (int8_t)v;
    }
}
// act_fp8g on the GPU, one thread per token and group: the same IEEE operations, so the same bytes and scales
__global__ void quant_fp8g(const float *x, uint8_t *q, float *s, size_t ngroups) {
    for (size_t gi = blockIdx.x * (size_t)blockDim.x + threadIdx.x; gi < ngroups; gi += (size_t)gridDim.x * blockDim.x) {
        const float *xs = x + gi * 128; float amax = 0.f;
        for (int i = 0; i < 128; i++) amax = fmaxf(amax, fabsf(xs[i]));
        float sc = fmaxf(amax, 1e-10f) / 448.f; s[gi] = sc;
        for (int i = 0; i < 128; i++) { float t = xs[i] / sc; q[gi * 128 + i] = to_e4m3(fabsf(t)) | (t < 0 ? 0x80 : 0); }
    }
}
__device__ inline void mma16816acc(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
// One warp: 16 rows x 8 tokens; a block of 8 warps = the 128 rows of one scale row-block. wd/xd: 3 digit planes [M][K] /
// [N][K]; ws: BF16 scales [M/128][K/128]; xs: FP32 scales [N][K/128]. Y [N][M]: p = 8 BF16 values, p = 24 FP32.
__global__ void __launch_bounds__(256) fp8b_gemm(const int8_t *__restrict__ wd, const uint16_t *__restrict__ ws,
        const int8_t *__restrict__ xd, const float *__restrict__ xs, float *Y, int M, int N, int K, int p, int *nbad) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    int m0 = blockIdx.x * 128 + warp * 16, n0 = blockIdx.y * 8, KB = K / 128, rb = blockIdx.x;
    size_t PW = (size_t)M * K, PX = (size_t)N * K;
    auto wexp = [&](int b, int &sig) { uint16_t u = ws[(size_t)rb * KB + b]; int x = (u >> 7) & 255; sig = x ? 128 + (u & 127) : u & 127; return x ? x - 134 : -133; };
    auto xexp = [&](int tok, int b, int &sig) { int x; sig = (int)ldexpf(frexpf(xs[(size_t)tok * KB + b], &x), 24); return x - 24; };
    int tok[2] = {n0 + t4 * 2, n0 + t4 * 2 + 1}, hi[2] = {-100000, -100000};
    for (int b = 0; b < KB; b++) { int s1, s2; int ew = wexp(b, s1); for (int c = 0; c < 2; c++) hi[c] = max(hi[c], ew + xexp(tok[c], b, s2)); }
    int lsb[2] = {hi[0] - 18 + 81 - 125, hi[1] - 18 + 81 - 125};   // |sum| < 2^(hi - 18 + 75 + 6): K/128 <= 64 terms
    __int128 acc[4] = {}; int tail[4] = {};
    for (int b = 0; b < KB; b++) {
        int P[9][4] = {};
        for (int kk = b * 128; kk < b * 128 + 128; kk += 16) {
            uint32_t a0[3], a1[3], b0[3];
#pragma unroll
            for (int q = 0; q < 3; q++) {
                a0[q] = *(const uint32_t *)(wd + q * PW + (size_t)(m0 + g) * K + kk + t4 * 4);
                a1[q] = *(const uint32_t *)(wd + q * PW + (size_t)(m0 + g + 8) * K + kk + t4 * 4);
                b0[q] = *(const uint32_t *)(xd + q * PX + (size_t)(n0 + g) * K + kk + t4 * 4);
            }
#pragma unroll
            for (int pw = 0; pw < 3; pw++)
#pragma unroll
                for (int px = 0; px < 3; px++) mma16816acc(P[pw * 3 + px], a0[pw], a1[pw], b0[px]);
        }
        int sw; int ew = wexp(b, sw);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            long long S = 0;
#pragma unroll
            for (int pw = 0; pw < 3; pw++)
#pragma unroll
                for (int px = 0; px < 3; px++) S += (long long)P[pw * 3 + px][i] << (7 * (pw + px));
            if (!S) continue;
            int c = i & 1, sx; int e = ew + xexp(tok[c], b, sx) - 18 - lsb[c];
            __int128 t = (__int128)S * ((long long)sw * sx);
            if (e >= 0) acc[i] += t << e;
            else { tail[i]++; acc[i] += e > -127 ? t >> -e : (t < 0 ? -1 : 0); }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = m0 + g + (i < 2 ? 0 : 8), col = tok[i & 1]; float r;
        if (!cert_round(acc[i], tail[i], lsb[i & 1], 1, 0, p, r)) { atomicAdd(nbad, 1); r = __int_as_float(0x7FC00000); }
        Y[(size_t)col * M + row] = r;
    }
}

// BF16 bits [R][K] -> nd digit planes of v = floor(x / 2^l) per block of 128 on the grid l = top bit - (7 nd - 2) (lexp), and
// the count of elements with bits below l (ntr). One warp per row and block, four elements per lane, so loads and stores
// coalesce (one thread per block put every byte in a sector of its own: 27 of the 28 s of an MoE layer's backward, 30-10-2026).
static unsigned dig_grid(size_t nblk) { return (unsigned)std::min<size_t>((nblk + 7) / 8, 1u << 20); }   // bf16_digits: a warp per block
__global__ void bf16_digits(const uint16_t *src, int8_t *d, int16_t *lexp, uint8_t *ntr, int R, int K, int nd) {
    size_t KB = K / 128, nblk = (size_t)R * KB, n = (size_t)R * K; int lane = threadIdx.x & 31;
    for (size_t bi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32; bi < nblk; bi += (size_t)gridDim.x * blockDim.x / 32) {
        size_t base = (bi / KB) * K + (bi % KB) * 128 + lane * 4; uint2 raw = *(const uint2 *)(src + base);
        uint16_t us[4] = {(uint16_t)raw.x, (uint16_t)(raw.x >> 16), (uint16_t)raw.y, (uint16_t)(raw.y >> 16)}; int top = -100000, cnt = 0;
        for (int i = 0; i < 4; i++) { uint16_t u = us[i]; int ex = (u >> 7) & 255, f = u & 127; if (!(u & 0x7FFF)) continue;
            top = max(top, (ex ? ex - 134 : -133) + 31 - __clz(ex ? 128 + f : f)); }
        for (int o = 16; o; o >>= 1) top = max(top, __shfl_xor_sync(~0u, top, o));
        int l = top == -100000 ? -32768 : top - (7 * nd - 2);   // |v| < 2^(7 nd - 1): nd balanced base-128 digits; -32768: all zero
        uint32_t pk[8] = {};   // plane q's four bytes of this lane
        for (int i = 0; i < 4; i++) {
            uint16_t u = us[i]; int ex = (u >> 7) & 255, f = u & 127, sig = ex ? 128 + f : f, sh = (ex ? ex - 134 : -133) - l; long long v;   // up to 2^41 (6 digits)
            if (!(u & 0x7FFF)) v = 0;
            else if (sh >= 0) v = (u >> 15) ? -((long long)sig << sh) : (long long)sig << sh;
            else {   // floor: the dropped part is in [0, 1) units for both signs
                int q = sh > -9 ? sig >> -sh : 0, rem = sh > -9 ? (sig & ((1 << -sh) - 1)) != 0 : 1; cnt += rem;
                v = (u >> 15) ? -(q + rem) : q;
            }
            for (int q = 0; q < nd - 1; q++) { int dq = (int)((v + 64) & 127) - 64; pk[q] |= (uint32_t)(uint8_t)dq << (8 * i); v = (v - dq) >> 7; }
            pk[nd - 1] |= (uint32_t)(uint8_t)v << (8 * i);
        }
        for (int q = 0; q < nd; q++) *(uint32_t *)(d + q * n + base) = pk[q];
        for (int o = 16; o; o >>= 1) cnt += __shfl_xor_sync(~0u, cnt, o);
        if (!lane) { lexp[bi] = (int16_t)l; ntr[bi] = (uint8_t)cnt; }
    }
}
// One warp: 16 rows x 8 tokens; weights in DW digits (|vw| < 2^(7 DW - 1)), tokens in DX (|vx| < 2^(7 DX - 1)): activations spread
// over more binades than trained weights (3 digits left 3-8 % of the BF16 outputs uncertified, 4 digits ...).
// Y [N][M]: certified outputs, NaN where the bound on dropped bits leaves the rounding open.
template <int DW, int DX>
__global__ void __launch_bounds__(256) bf16_gemm(const int8_t *__restrict__ wd, const int16_t *__restrict__ wl, const uint8_t *__restrict__ wn,
        const int8_t *__restrict__ xd, const int16_t *__restrict__ xl, const uint8_t *__restrict__ xn, float *Y, int M, int N, int K, int p, int *nbad) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    int m0 = (blockIdx.x * 8 + warp) * 16, n0 = blockIdx.y * 8, KB = K / 128;
    if (m0 >= M) return;
    size_t PW = (size_t)M * K, PX = (size_t)N * K;
    int row[4], tok[4], lsb[4];
    for (int i = 0; i < 4; i++) {   // |S_b| < 128 2^(7 DW - 1) 2^(7 DX - 1), at most 64 blocks: |sum| < 2^(max(lw + lx) + 7 (DW + DX) + 11)
        row[i] = m0 + g + (i < 2 ? 0 : 8); tok[i] = n0 + t4 * 2 + (i & 1); int hi = -100000;
        for (int b = 0; b < KB; b++) hi = max(hi, wl[(size_t)row[i] * KB + b] + xl[(size_t)tok[i] * KB + b]);
        lsb[i] = hi + 7 * (DW + DX) + 11 - 125;
    }
    __int128 acc[4] = {}; unsigned __int128 tail[4] = {};
    for (int b = 0; b < KB; b++) {
        int P[DW * DX][4] = {};
        for (int kk = b * 128; kk < b * 128 + 128; kk += 16) {
            uint32_t a0[DW], a1[DW], b0[DX];
#pragma unroll
            for (int q = 0; q < DW; q++) {
                a0[q] = *(const uint32_t *)(wd + q * PW + (size_t)(m0 + g) * K + kk + t4 * 4);
                a1[q] = *(const uint32_t *)(wd + q * PW + (size_t)(m0 + g + 8) * K + kk + t4 * 4);
            }
#pragma unroll
            for (int q = 0; q < DX; q++) b0[q] = *(const uint32_t *)(xd + q * PX + (size_t)(n0 + g) * K + kk + t4 * 4);
#pragma unroll
            for (int pw = 0; pw < DW; pw++)
#pragma unroll
                for (int px = 0; px < DX; px++) mma16816acc(P[pw * DX + px], a0[pw], a1[pw], b0[px]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __int128 S = 0;   // |S| < 2^(7 (DW + DX) + 5): beyond int64 from 9 digits on
#pragma unroll
            for (int pw = 0; pw < DW; pw++)
#pragma unroll
                for (int px = 0; px < DX; px++) S += (__int128)P[pw * DX + px][i] << (7 * (pw + px));
            size_t wi = (size_t)row[i] * KB + b, xi = (size_t)tok[i] * KB + b;
            if (wl[wi] == -32768 || xl[xi] == -32768) continue;   // a zero block: its product is exactly 0, no bound (padding rows)
            int e = wl[wi] + xl[xi] - lsb[i], nw = wn[wi], nx = xn[xi];   // the bound, in units of 2^(lw + lx):
            unsigned long long bound = ((unsigned long long)nx << (7 * DW - 1)) + ((unsigned long long)nw << (7 * DX - 1)) + min(nw, nx);
            if (S) { if (e >= 0) acc[i] += S << e; else { tail[i]++; acc[i] += e > -127 ? S >> -e : (S < 0 ? -1 : 0); } }
            if (bound) tail[i] += e >= 0 ? (unsigned __int128)bound << e : ((unsigned __int128)bound >> min(-e, 127)) + 1;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float r;
        if (!cert_round(acc[i], tail[i], lsb[i], 1, 0, p, r)) { atomicAdd(nbad, 1); r = __int_as_float(0x7FC00000); }
        Y[(size_t)tok[i] * M + row[i]] = r;
    }
}
__device__ inline void mma16832acc(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// bf16_gemm with m16n8k32 (twice the work per instruction), the digit products summed per shift 7 (pw + px) inside a block
// of 128 (DW + DX - 1 accumulators instead of DW DX; |sum| < min(DW, DX) 2^19), and NT tiles of 8 tokens per warp on one
// load of the weight digits. The same exact block sums and the same bounds: bit for bit bf16_gemm.
template <int DW, int DX, int NT>
__global__ void __launch_bounds__(256) bf16_gemm2(const int8_t *__restrict__ wd, const int16_t *__restrict__ wl, const uint8_t *__restrict__ wn,
        const int8_t *__restrict__ xd, const int16_t *__restrict__ xl, const uint8_t *__restrict__ xn, float *Y, int M, int N, int K, int p, int *nbad) {
    constexpr int NS = DW + DX - 1;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    int m0 = (blockIdx.x * 8 + warp) * 16, n0 = blockIdx.y * 8 * NT, KB = K / 128;
    if (m0 >= M) return;
    size_t PW = (size_t)M * K, PX = (size_t)N * K;
    int row[4], tok[NT][4], lsb[NT][4]; bool live[NT];
    for (int i = 0; i < 4; i++) row[i] = m0 + g + (i < 2 ? 0 : 8);
    for (int t = 0; t < NT; t++) { live[t] = n0 + t * 8 < N;
        for (int i = 0; i < 4; i++) { tok[t][i] = n0 + t * 8 + t4 * 2 + (i & 1); int hi = -100000;
            if (live[t]) for (int b = 0; b < KB; b++) hi = max(hi, wl[(size_t)row[i] * KB + b] + xl[(size_t)tok[t][i] * KB + b]);
            lsb[t][i] = hi + 7 * (DW + DX) + 11 - 125; } }
    __int128 acc[NT][4] = {}; unsigned __int128 tail[NT][4] = {};
    for (int b = 0; b < KB; b++) {
        int P[NS][NT][4] = {};
        for (int kk = b * 128; kk < b * 128 + 128; kk += 32) {
            uint32_t a[DW][4];
#pragma unroll
            for (int q = 0; q < DW; q++) {
                const int8_t *w0 = wd + q * PW + (size_t)(m0 + g) * K + kk + t4 * 4, *w1 = w0 + (size_t)8 * K;
                a[q][0] = *(const uint32_t *)w0; a[q][1] = *(const uint32_t *)w1; a[q][2] = *(const uint32_t *)(w0 + 16); a[q][3] = *(const uint32_t *)(w1 + 16);
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (!live[t]) continue;
                uint32_t b0[DX], b1[DX];
#pragma unroll
                for (int q = 0; q < DX; q++) { const int8_t *x = xd + q * PX + (size_t)(n0 + t * 8 + g) * K + kk + t4 * 4;
                    b0[q] = *(const uint32_t *)x; b1[q] = *(const uint32_t *)(x + 16); }
#pragma unroll
                for (int pw = 0; pw < DW; pw++)
#pragma unroll
                    for (int px = 0; px < DX; px++) mma16832acc(P[pw + px][t], a[pw], b0[px], b1[px]);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (!live[t]) continue;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                __int128 S = 0;
#pragma unroll
                for (int s = 0; s < NS; s++) S += (__int128)P[s][t][i] << (7 * s);
                size_t wi = (size_t)row[i] * KB + b, xi = (size_t)tok[t][i] * KB + b;
                if (wl[wi] == -32768 || xl[xi] == -32768) continue;   // a zero block: its product is exactly 0, no bound (padding rows)
                int e = wl[wi] + xl[xi] - lsb[t][i], nw = wn[wi], nx = xn[xi];
                unsigned long long bound = ((unsigned long long)nx << (7 * DW - 1)) + ((unsigned long long)nw << (7 * DX - 1)) + min(nw, nx);
                if (S) { if (e >= 0) acc[t][i] += S << e; else { tail[t][i]++; acc[t][i] += e > -127 ? S >> -e : (S < 0 ? -1 : 0); } }
                if (bound) tail[t][i] += e >= 0 ? (unsigned __int128)bound << e : ((unsigned __int128)bound >> min(-e, 127)) + 1;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        if (!live[t]) continue;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float r;
            if (!cert_round(acc[t][i], tail[t][i], lsb[t][i], 1, 0, p, r)) { atomicAdd(nbad, 1); r = __int_as_float(0x7FC00000); }
            Y[(size_t)tok[t][i] * M + row[i]] = r;
        }
    }
}

// The outputs bf16_gemm could not certify (NaN), recomputed exactly on the CUDA cores, one warp each: the GEMV's sum (every
// product exact, a 128-bit window at the row's and token's top bits, truncated terms counted). The list comes from
// nan_list. Together the two kernels give every output correctly rounded.
__global__ void nan_list(const float *Y, size_t n, unsigned *list, unsigned *cnt, unsigned cap) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (Y[i] != Y[i]) { unsigned j = atomicAdd(cnt, 1u); if (j < cap) list[j] = (unsigned)i; }
}
__device__ inline void bf16_me(uint16_t u, int &m, int &e) { int ex = (u >> 7) & 255, f = u & 127; m = ex ? 128 + f : f; e = ex ? ex - 134 : -133; if (u >> 15) m = -m; }
__global__ void bf16_fixup(const uint16_t *w, const uint16_t *xb, const unsigned *list, const unsigned *cnt, unsigned cap,
                           float *Y, int M, int K, int p, int *nbad) {   // NaNs beyond cap stay NaN: the host GEMV takes them
    unsigned nl = min(*cnt, cap), lane = threadIdx.x & 31;
    for (unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) / 32; j < nl; j += gridDim.x * blockDim.x / 32) {
        size_t i = list[j], tok = i / M, row = i % M; const uint16_t *wr = w + row * K, *xr = xb + tok * K;
        int hp = -100000;   // the largest product's top bit (gemv_k: anchoring at max |w| + max |x| lost whole rows, 30-10-2026)
        for (int k = lane; k < K; k += 32) { int mw, ew, mx, ex; bf16_me(wr[k], mw, ew); bf16_me(xr[k], mx, ex);
                                             if (mw && mx) hp = max(hp, ew + ex + 64 - __clz(abs(mw)) - __clz(abs(mx))); }
        for (int o = 16; o; o >>= 1) hp = max(hp, __shfl_xor_sync(~0u, hp, o));
        if (hp == -100000) { if (!lane) Y[i] = 0.f; continue; }
        int lsb = hp + 16 - 125; __int128 acc = 0; int tail = 0;
        for (int k = lane; k < K; k += 32) {
            int mw, ew, mx, ex; bf16_me(wr[k], mw, ew); bf16_me(xr[k], mx, ex); if (!mw || !mx) continue;
            long long pr = (long long)mw * mx; int sh = ew + ex - lsb;
            if (sh >= 0) acc += (__int128)pr << sh; else { tail++; acc += sh > -63 ? (__int128)(pr >> -sh) : (__int128)(pr < 0 ? -1 : 0); }
        }
        for (int o = 16; o; o >>= 1) {
            unsigned long long lo = (unsigned long long)acc, hi = (unsigned long long)((unsigned __int128)acc >> 64);
            lo = __shfl_xor_sync(~0u, lo, o); hi = __shfl_xor_sync(~0u, hi, o);
            acc += (__int128)(((unsigned __int128)hi << 64) | lo); tail += __shfl_xor_sync(~0u, tail, o);
        }
        if (!lane) { float r; if (!cert_round(acc, tail, lsb, 1, 0, p, r)) { atomicAdd(nbad, 1); r = __int_as_float(0x7FC00000); } Y[i] = r; }   // NaN: the host GEMV
    }
}

constexpr int DWN = 3, DXN = 4;     // digits of a weight and of a token block (BF16), p = 8
constexpr int DW24 = 5, DX24 = 6;   // p = 24 (FP32 logits): 3 + 4 left most outputs to the fixup
// the BF16 product of N tokens (digit planes of the weights for the p in use already in wd): Y [N][M], NaN = uncertified.
// dw: the weights' digits -- wdig(p), or 4 at p = 24 for weights that are FP8 times a block scale (hi or lo of one: on the
// grid of 2^-9 times the scale, at most 26 bits, so 4 digits hold them exactly)
static void bf16_gemm_p(int p, int dw, const int8_t *wd, const int16_t *wl, const uint8_t *wn, const int8_t *xd, const int16_t *xl,
                        const uint8_t *xn, float *Y, int M, int N, int K, int *nbad) {
    static const bool k16 = getenv("OZ_K16");   // OZ_K16=1: the first kernel (m16n8k16, an accumulator per digit pair)
    if (k16) { dim3 g((M + 127) / 128, N / 8);
        if (p == 24 && dw == 4) bf16_gemm<4, DX24><<<g, 256>>>(wd, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad);
        else if (p == 24) bf16_gemm<DW24, DX24><<<g, 256>>>(wd, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad);
        else bf16_gemm<DWN, DXN><<<g, 256>>>(wd, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad);
        return; }
    auto go = [&](auto k1, auto k2) { if (N >= 64) k2<<<dim3((M + 127) / 128, (N + 15) / 16), 256>>>(wd, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad);
                                        else k1<<<dim3((M + 127) / 128, N / 8), 256>>>(wd, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad); };
    if (p == 24 && dw == 4) go(bf16_gemm2<4, DX24, 1>, bf16_gemm2<4, DX24, 2>);
    else if (p == 24) go(bf16_gemm2<DW24, DX24, 1>, bf16_gemm2<DW24, DX24, 2>);
    else go(bf16_gemm2<DWN, DXN, 1>, bf16_gemm2<DWN, DXN, 2>);
}
static int wdig(int p) { return p == 24 ? DW24 : DWN; }
static int xdig(int p) { return p == 24 ? DX24 : DXN; }
#ifndef BGEMM_LIB
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: out_bgemm <model dir> <tensor> <M> <K> [N]\n"); return 1; }
    DIR = argv[1]; std::string T = argv[2]; int M = atoi(argv[3]), K = atoi(argv[4]), N = argc > 5 ? atoi(argv[5]) : 4096, NREF = 256, KB = K / 128;
    read_index();
    bool fp8 = idx.at(T + ".weight").dt == "F8_E4M3";
    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)M * K * 2)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, K * 4)); CK(cudaMalloc(&dxe, K * 2)); CK(cudaMalloc(&dy, M * 4)); CK(cudaMalloc(&dbad, 4));
    DevW W = load_w(T + ".weight", fp8 ? F_E4M3B : F_BF16, M, K);
    if (fp8) { to_dev(T + ".weight_scale", dS); W.sw = (uint8_t *)dS; }
    auto act = [&](const std::vector<float> &v) { return fp8 ? act_fp8g(v) : act_plain(v); };
    // tokens: BF16 values, each token with its own scale and a few outliers (as hidden states have)
    std::mt19937_64 rng(28102026); std::normal_distribution<double> nd; std::vector<float> x((size_t)N * K);
    for (int n = 0; n < N; n++) { double sc = exp2((double)(rng() % 24) - 12);
        for (int k = 0; k < K; k++) x[(size_t)n * K + k] = bf(nd(rng) * sc * (rng() % 97 == 0 ? 64 : 1)); }
    std::vector<uint16_t> xb(x.size()); for (size_t i = 0; i < x.size(); i++) { uint32_t u; memcpy(&u, &x[i], 4); xb[i] = u >> 16; }
    int8_t *wd, *xd; uint8_t *xq, *wn, *xn; int16_t *wl, *xl; uint16_t *dxb; float *dx, *xsc, *Y; int *nb;
    CK(cudaMalloc(&wd, (size_t)DW24 * M * K)); CK(cudaMalloc(&xd, (size_t)DX24 * N * K)); CK(cudaMalloc(&xq, (size_t)N * K));
    CK(cudaMalloc(&dx, (size_t)N * K * 4)); CK(cudaMalloc(&xsc, (size_t)N * KB * 4)); CK(cudaMalloc(&Y, (size_t)N * M * 4)); CK(cudaMalloc(&nb, 4));
    unsigned *list, *lcnt, cap = (unsigned)((size_t)N * M / 8); int *nb2; CK(cudaMalloc(&list, (size_t)cap * 4)); CK(cudaMalloc(&lcnt, 4)); CK(cudaMalloc(&nb2, 4));
    CK(cudaMalloc(&dxb, (size_t)N * K * 2)); CK(cudaMalloc(&wl, (size_t)M * KB * 2)); CK(cudaMalloc(&wn, (size_t)M * KB)); CK(cudaMalloc(&xl, (size_t)N * KB * 2)); CK(cudaMalloc(&xn, (size_t)N * KB));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxb, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
    if (fp8) fp8_digits<<<1024, 256>>>((const uint8_t *)W.w, wd, (size_t)M * K);
    auto run = [&](int p) {
        if (!fp8) bf16_digits<<<dig_grid((size_t)M * KB), 256>>>((const uint16_t *)W.w, wd, wl, wn, M, K, wdig(p));
        if (fp8) {
            quant_fp8g<<<1024, 256>>>(dx, xq, xsc, (size_t)N * KB);
            fp8_digits<<<1024, 256>>>(xq, xd, (size_t)N * K);
            fp8b_gemm<<<dim3(M / 128, N / 8), 256>>>(wd, (const uint16_t *)dS, xd, xsc, Y, M, N, K, p, nb);
        } else {
            bf16_digits<<<dig_grid((size_t)N * KB), 256>>>(dxb, xd, xl, xn, N, K, xdig(p));
            bf16_gemm_p(p, wdig(p), wd, wl, wn, xd, xl, xn, Y, M, N, K, nb);
            CK(cudaMemset(lcnt, 0, 4)); nan_list<<<1024, 256>>>(Y, (size_t)N * M, list, lcnt, cap);
            bf16_fixup<<<1024, 256>>>((const uint16_t *)W.w, dxb, list, lcnt, cap, Y, M, K, p, nb2);
        }
    };
    for (int p : {8, 24}) {
        CK(cudaMemset(nb, 0, 4)); CK(cudaMemset(nb2, 0, 4)); run(p); CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t)N * M); CK(cudaMemcpy(y.data(), Y, y.size() * 4, cudaMemcpyDeviceToHost));
        int hb, hb2; CK(cudaMemcpy(&hb, nb, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hb2, nb2, 4, cudaMemcpyDeviceToHost));
        long long nan_tok = 0;   // the outputs the batched product could not certify: their tokens go through the GEMV
        for (int n = 0; n < N; n++) {
            bool any = false; for (int m = 0; m < M && !any; m++) any = y[(size_t)n * M + m] != y[(size_t)n * M + m];
            if (!any) continue; nan_tok++;
            std::vector<float> xv(x.begin() + (size_t)n * K, x.begin() + (size_t)(n + 1) * K); auto r = gemv(W, act(xv), p, 0);
            for (int m = 0; m < M; m++) if (y[(size_t)n * M + m] != y[(size_t)n * M + m]) y[(size_t)n * M + m] = r[m];
        }
        long long diff = 0; st_sup = 0;
        for (int n = 0; n < NREF; n++) {
            std::vector<float> xv(x.begin() + (size_t)n * K, x.begin() + (size_t)(n + 1) * K);
            auto r = gemv(W, act(xv), p, 0);
            for (int m = 0; m < M; m++) diff += r[m] != y[(size_t)n * M + m];
        }
        printf("%s %s %d x %d, %d tokens, p %d: batched vs per-token GEMV on %d tokens: %lld of %lld differ; tensor cores left %d outputs "
               "(%.4f%%) to the CUDA-core fixup (uncertified there %d), %lld tokens to the host GEMV; GEMV outputs finished on the host %lld\n", T.c_str(),
               fp8 ? "FP8" : "BF16", M, K, N, p, NREF, diff, (long long)NREF * M, hb, 100.0 * hb / ((double)N * M), hb2, nan_tok, st_sup);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int r = 0; r < 5; r++) run(8);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    printf("batched: %.2f ms for %d tokens (activation preparation + product + fixup) = %.2f TOPS\n", ms, N, 2.0 * M * K * N / (ms * 1e9));
    return 0;
}
#endif
