// tgemm.cu -- the batched products on the INT8 tensor cores, tiled (shared memory, cp.async, ldmatrix), every output the
// correctly rounded value of the exact sum, as bf16_gemm2 gives it, at a fraction of its time:
//  * a block of 256 threads computes 64 rows x 32 tokens, 8 warps of 16 x 16; K in tiles of BK through ST cp.async stages
//    (rows of BK + 16 bytes: ldmatrix without bank conflicts); m16n8k32 MMAs summed per digit shift (DW + DX - 1 int32
//    accumulators an output); after each block of 128 the block term goes into a 128-bit window with a tail (truncated
//    terms, the bound on dropped bits), cert_round decides, NaN = open. bf16_gemm2 loaded every operand from global memory
//    for each warp (the weights once per 16 tokens): ~6 % of the tensor cores.
//  * BF16 weights: bf16_digits per row and block of 128 (wl, wn); the window, the bound and so every output, NaN or not,
//    exactly those of bf16_gemm2.
//  * F8 (the backward's dX = dY W for the routed experts, jais_lora.cu): the weight stays FP8, Wt[r][m] = W[m][r] = v 2^-9 s,
//    v = E4M3 2^9 an integer (|v| <= 448 * 512 < 2^18: three balanced base-128 digits, exact) and s the BF16 scale of W's
//    128 x 128 block -- one per 128 rows of Wt and block of 128 in m. The hi|lo product it replaces ([dY | dY] [W_hi ; W_lo],
//    BF16 digits) doubled K and took 4 digits each (v s: 26 bits): 2 x 4 x 6 = 48 digit products a term, here 3 x 6 = 18.
//    dY in 6 digits a block (|vx| < 2^41), xn of its elements with bits below the grid: the block term is
//    s 2^(es - 9 + l) (S + E), S = sum v vx exactly, |E| < xn 2^18. f8t_fixup recomputes what stays open, the host the rest.
//  * XF8 (the forward of the routed experts, fp8b_gemm's product): W [M][K] as above but not transposed, X the E4M3 bytes of
//    act_fp8g in three digits too, each token's block of 128 with its FP32 scale sx = sig 2^(x - 24): the block term is
//    s sig 2^(es + x - 33) S exactly, |S| < 2^43, so below 2^75 times its power of two -- (S s) sig as a 64 x 32 -> 128-bit
//    product, shifted into the 63-bit window (always down: C = 75 + log2 blocks > 62). No dropped bits, the bound is the
//    truncations alone. f8x_fixup recomputes what stays open exactly as fp8b_gemm does: the outputs are fp8b_gemm's.
#pragma once

namespace tg {
constexpr int BM = 64, BN = 32, KBMAX = 128;
template <int DW, int DX, int BK, int ST> constexpr int smem() { return ST * (DW * BM + DX * BN) * (BK + 16); }
}

// W [Mo][Ko] E4M3 bytes -> the three digit planes of Wt [Ko][ld] (plane stride PW), columns off .. off + Mo
__global__ void tr_f8_digits(const uint8_t *s, int8_t *d, size_t PW, int Mo, int Ko, int ld, int off) {
    __shared__ uint8_t t[32][33];
    int k0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int m = m0 + i, k = k0 + threadIdx.x; if (m < Mo && k < Ko) t[i][threadIdx.x] = s[(size_t)m * Ko + k]; }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int k = k0 + i, m = m0 + threadIdx.x; if (m < Mo && k < Ko) {
        uint8_t b = t[threadIdx.x][i]; int e = (b >> 3) & 15, f = b & 7, v = e ? (8 + f) << (e - 1) : f; if (b & 0x80) v = -v;
        int d0 = ((v + 64) & 127) - 64; v = (v - d0) >> 7; int d1 = ((v + 64) & 127) - 64; v = (v - d1) >> 7;
        size_t o = (size_t)k * ld + off + m; d[o] = (int8_t)d0; d[PW + o] = (int8_t)d1; d[2 * PW + o] = (int8_t)v; } }
}
// W's block scales [Mo/128][Ko/128] -> Wt's: tsc[r/128][(off + m)/128]
__global__ void tr_f8_scales(const uint16_t *sc, uint16_t *tsc, int Mo, int Ko, int KB, int off) {
    int RB = Ko / 128, MB = Mo / 128;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < RB * MB; i += gridDim.x * blockDim.x)
        tsc[(size_t)(i / MB) * KB + off / 128 + i % MB] = sc[(size_t)(i % MB) * RB + i / MB];
}

__device__ inline void cp16(uint32_t dst, const void *src, int n) {   // n = 16, or 0: sixteen zero bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n));
}
__device__ inline void ldm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// XF8: X's block exponent with its scale's significand (fp8b_gemm's xexp, the E4M3 2^-9 included): sx = sig 2^(x - 24)
__device__ inline int xf8_exp(const float *xs, size_t i, int &sig) { int x; sig = (int)ldexpf(frexpf(xs[i], &x), 24); return x - 33; }
// Y [N][M] = X W^T: wd DW digit planes of W [M][K], xd/xl/xn the bf16_digits of X [N][K] (DX digits); F8: tsc, else wl/wn;
// XF8 (with F8, DW = DX = 3): xd the E4M3 digits of X and xs its scales [N][K/128] in place of xl/xn.
// M a multiple of 16 (rows past M load as zeros), K of BK and 128 (at most 128 blocks), N of 8.
template <int DW, int DX, bool F8, int BK, int ST, bool XF8 = false>
__global__ void __launch_bounds__(256) tgemm(const int8_t *__restrict__ wd, const uint16_t *__restrict__ tsc, const int16_t *__restrict__ wl,
        const uint8_t *__restrict__ wn, const int8_t *__restrict__ xd, const int16_t *__restrict__ xl, const uint8_t *__restrict__ xn,
        float *Y, int M, int N, int K, int p, int *nbad, const float *__restrict__ xs) {
    using namespace tg;
    static_assert(!XF8 || (F8 && DW == 3 && DX == 3), "XF8: E4M3 times E4M3, three digits each");
    constexpr int NS = DW + DX - 1, LD = BK + 16, WT = DW * BM * LD, XT = DX * BN * LD, CH = BK / 16;
    extern __shared__ __align__(16) uint8_t sm[];
    const int C = (XF8 ? 75 : F8 ? 74 : 7 * (DW + DX) + 5) + (K > 128 ? 32 - __clz(K / 128 - 1) : 0);   // |block term| < 2^(C - log2 blocks)
    __shared__ int ses[F8 ? KBMAX : 1], sss[F8 ? KBMAX : 1];   // F8: per block of this block's rows, the scale's exponent - 9 and significand
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3, wm = warp & 3, wq = warp >> 2;
    int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM, KB = K / 128, KT = K / BK;
    size_t PW = (size_t)M * K, PX = (size_t)N * K;
    if (F8) for (int b = tid; b < KB; b += 256) { uint16_t u = tsc[(size_t)(m0 / 128) * KB + b]; int x = (u >> 7) & 255;
        sss[b] = x ? 128 + (u & 127) : u & 127; ses[b] = (x ? x - 134 : -133) - 9; }
    __syncthreads();
    int row[2] = {m0 + wm * 16 + g, m0 + wm * 16 + g + 8}, tok[2][2], lsb[2][4];
    for (int t = 0; t < 2; t++) for (int c = 0; c < 2; c++) tok[t][c] = n0 + wq * 16 + t * 8 + t4 * 2 + c;
    for (int t = 0; t < 2; t++) for (int i = 0; i < 4; i++) {
        int r = row[i >> 1], tk = tok[t][i & 1], hi = -100000;
        if (tk < N && r < M) for (int b = 0; b < KB; b++) { int sg, l = XF8 ? xf8_exp(xs, (size_t)tk * KB + b, sg) : xl[(size_t)tk * KB + b],
                                                            w = F8 ? ses[b] : wl[(size_t)r * KB + b];
                                                            if (l != -32768 && w != -32768) hi = max(hi, w + l); }
        // |sum| < 2^(hi + C): F8 |s S| < 2^(8 + 18 + 41 + 7) a block, BF16 |S| < 2^(7 (DW + DX) + 5), times the blocks. A 63-bit
        // window (int64: bf16_gemm2's 125 bits cost as many instructions as the MMAs) holds the result to ~50 bits whenever it
        // is not far below its largest block: what it cannot certify goes to the fixup, as before
        lsb[t][i] = hi + C - 62;
    }
    uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
    auto load = [&](int kt, int st) {
        int k0 = kt * BK; uint32_t sw = sbase + st * (WT + XT), sx = sw + WT;
        for (int c = tid; c < DW * BM * CH; c += 256) { int q = c / (BM * CH), r = (c / CH) % BM, ch = c % CH; bool ok = m0 + r < M;
            cp16(sw + (q * BM + r) * LD + ch * 16, ok ? (const void *)(wd + q * PW + (size_t)(m0 + r) * K + k0 + ch * 16) : (const void *)wd, ok ? 16 : 0); }
        for (int c = tid; c < DX * BN * CH; c += 256) { int q = c / (BN * CH), r = (c / CH) % BN, ch = c % CH; bool ok = n0 + r < N;
            cp16(sx + (q * BN + r) * LD + ch * 16, ok ? (const void *)(xd + q * PX + (size_t)(n0 + r) * K + k0 + ch * 16) : (const void *)xd, ok ? 16 : 0); }
    };
    int P[NS][2][4] = {}; long long acc[2][4] = {}; unsigned long long tail[2][4] = {};
    for (int s = 0; s < ST - 1; s++) { if (s < KT) load(s, s); asm volatile("cp.async.commit_group;\n"); }
    int j8 = lane >> 3, r8 = lane & 7;
    for (int kt = 0; kt < KT; kt++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(ST - 2));
        __syncthreads();
        if (kt + ST - 1 < KT) load(kt + ST - 1, (kt + ST - 1) % ST);
        asm volatile("cp.async.commit_group;\n");
        uint32_t sw = sbase + (kt % ST) * (WT + XT), sx = sw + WT;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 32) {
            uint32_t a[DW][4], bx[DX][4];   // a: rows 0-7 / 8-15 x k 0-15 / 16-31; bx: tokens 0-7 k 0-15, k 16-31, tokens 8-15 ...
#pragma unroll
            for (int q = 0; q < DW; q++) ldm4(a[q], sw + (q * BM + wm * 16 + (j8 & 1) * 8 + r8) * LD + ks + (j8 >> 1) * 16);
#pragma unroll
            for (int q = 0; q < DX; q++) ldm4(bx[q], sx + (q * BN + wq * 16 + (j8 >> 1) * 8 + r8) * LD + ks + (j8 & 1) * 16);
#pragma unroll
            for (int t = 0; t < 2; t++)
#pragma unroll
                for (int pw = 0; pw < DW; pw++)
#pragma unroll
                    for (int px = 0; px < DX; px++) mma16832acc(P[pw + px][t], a[pw], bx[px][2 * t], bx[px][2 * t + 1]);
        }
        if ((kt + 1) * BK % 128) continue;
        int b = (kt + 1) * BK / 128 - 1, ss = F8 ? sss[b] : 1;   // the end of block b: its terms into the windows
        int lx[2][2], nxs[2][2], sxs[2][2], lw[2], nws[2];   // the grids and dropped counts of this block, per token and per row
        for (int t = 0; t < 2; t++) for (int c = 0; c < 2; c++) { int tk = tok[t][c]; bool ok = tk < N;
            lx[t][c] = !ok ? -32768 : XF8 ? xf8_exp(xs, (size_t)tk * KB + b, sxs[t][c]) : xl[(size_t)tk * KB + b];
            nxs[t][c] = ok && !XF8 ? xn[(size_t)tk * KB + b] : 0; }
        for (int h = 0; h < 2; h++) { bool ok = !F8 && row[h] < M;
            lw[h] = F8 ? ses[b] : ok ? wl[(size_t)row[h] * KB + b] : -32768; nws[h] = ok ? wn[(size_t)row[h] * KB + b] : 0; }
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                long long lo = 0, hi = 0;   // S = hi 2^35 + lo: |lo|, |hi| < 2^50 (at most 5 digit pairs a shift, each sum < 2^19)
#pragma unroll
                for (int s = 0; s < NS; s++) { if (s < 5) lo += (long long)P[s][t][i] << (7 * s); else hi += (long long)P[s][t][i] << (7 * (s - 5)); P[s][t][i] = 0; }
                int l = lx[t][i & 1], w = lw[i >> 1]; if (l == -32768 || w == -32768) continue;   // a zero block: exactly 0, no bound
                int e = w + l - lsb[t][i], nx = nxs[t][i & 1], nw = nws[i >> 1]; unsigned long long bound;
                if (XF8) {   // S = lo exactly (5 shifts); e <= 62 - C < 0: floor((S s sig) 2^e), one truncation
                    if (!lo) continue;
                    __int128 T = (__int128)(lo * ss) * sxs[t][i & 1]; tail[t][i]++;
                    acc[t][i] += -e < 127 ? (long long)(T >> -e) : (T < 0 ? -1 : 0); continue; }
                if (F8) { lo *= ss; hi *= ss; bound = ((unsigned long long)ss * nx) << 18; }   // |v| < 2^18 times the dropped parts, each < 1
                else bound = ((unsigned long long)nx << (7 * DW - 1)) + ((unsigned long long)nw << (7 * DX - 1)) + min(nw, nx);
                // floor(T 2^e), T = hi 2^35 + lo: e <= 62 - C, so it fits; e >= 0 only where C <= 62 (BF16 p 8: T < 2^54)
                if (!(hi | lo));
                else if (e >= 0) acc[t][i] += ((hi << 35) + lo) << e;
                else { tail[t][i]++;
                    if (e >= -35) acc[t][i] += (hi << (35 + e)) + (lo >> -e);
                    else { long long q = hi + (lo >> 35); acc[t][i] += e > -98 ? q >> (-e - 35) : (q < 0 ? -1 : 0); } }
                if (bound) tail[t][i] += e >= 0 ? bound << e : (e > -64 ? bound >> -e : 0) + 1;
            }
    }
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int tk = tok[t][i & 1], r = row[i >> 1]; if (tk >= N || r >= M) continue;
            float o; if (!cert_round(acc[t][i], tail[t][i], lsb[t][i], 1, 0, p, o)) { atomicAdd(nbad, 1); o = __int_as_float(0x7FC00000); }
            Y[(size_t)tk * M + r] = o;
        }
}
template <int DW, int DX, bool F8, int BK, int ST, bool XF8 = false>   // grid: token tiles (fastest: a weight tile is read by all of them together) x rows
static void tgemm_go(const int8_t *wd, const uint16_t *tsc, const int16_t *wl, const uint8_t *wn, const int8_t *xd, const int16_t *xl,
                     const uint8_t *xn, float *Y, int M, int N, int K, int p, int *nbad, const float *xs = nullptr) {
    constexpr int SM = tg::smem<DW, DX, BK, ST>(); static bool raised = false;
    if (!raised) { CK(cudaFuncSetAttribute(tgemm<DW, DX, F8, BK, ST, XF8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM)); raised = true; }
    tgemm<DW, DX, F8, BK, ST, XF8><<<dim3((N + tg::BN - 1) / tg::BN, (M + tg::BM - 1) / tg::BM), 256, SM>>>(wd, tsc, wl, wn, xd, xl, xn, Y, M, N, K, p, nbad, xs);
}

// up to two FP8 maps that share their input dimension Ko, stacked along m (an expert's gate and up)
struct F8Cat { const uint8_t *w[2]; const uint16_t *sc[2]; int Mo[2], n, Ko; };
__device__ __host__ inline long long f8t_term(const F8Cat &C, const uint8_t *const *w, const uint16_t *const *sc, int r, int m, uint16_t xu, int &E) {
    int s = 0; while (s + 1 < C.n && m >= C.Mo[s]) { m -= C.Mo[s]; s++; }   // Wt[r][m] x = pr 2^E, exact (|pr| < 2^34)
    if (m >= C.Mo[s] || !(xu & 0x7FFF)) return 0;
    uint8_t b = w[s][(size_t)m * C.Ko + r]; int e = (b >> 3) & 15, f = b & 7, v = e ? (8 + f) << (e - 1) : f; if (b & 0x80) v = -v;
    uint16_t u = sc[s][(size_t)(m / 128) * (C.Ko / 128) + r / 128]; int x = (u >> 7) & 255, ss = x ? 128 + (u & 127) : u & 127;
    int xe = (xu >> 7) & 255, mx = xe ? 128 + (xu & 127) : xu & 127; if (xu >> 15) mx = -mx;
    E = (x ? x - 134 : -133) - 9 + (xe ? xe - 134 : -133); return (long long)v * ss * mx;
}
// the outputs the F8 product left open, one warp each: every product exact, a 128-bit window at the largest product's top bit
// (as bf16_fixup); what it cannot certify stays NaN for the host
__global__ void f8t_fixup(F8Cat C, const uint16_t *xb, int K, const unsigned *list, const unsigned *cnt, unsigned cap, float *Y, int M, int p, int *nbad) {
    unsigned nl = min(*cnt, cap), lane = threadIdx.x & 31;
    for (unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) / 32; j < nl; j += gridDim.x * blockDim.x / 32) {
        size_t i = list[j], tk = i / M; int r = (int)(i % M); const uint16_t *xr = xb + tk * K;
        int hp = -100000;
        for (int m = lane; m < K; m += 32) { int E; long long pr = f8t_term(C, C.w, C.sc, r, m, xr[m], E); if (pr) hp = max(hp, E + 64 - __clzll(pr < 0 ? -pr : pr)); }
        for (int o = 16; o; o >>= 1) hp = max(hp, __shfl_xor_sync(~0u, hp, o));
        if (hp == -100000) { if (!lane) Y[i] = 0.f; continue; }
        int lsb = hp + 16 - 125; __int128 acc = 0; int tail = 0;
        for (int m = lane; m < K; m += 32) {
            int E; long long pr = f8t_term(C, C.w, C.sc, r, m, xr[m], E); if (!pr) continue; int sh = E - lsb;
            if (sh >= 0) acc += (__int128)pr << sh; else { tail++; acc += sh > -63 ? (__int128)(pr >> -sh) : (__int128)(pr < 0 ? -1 : 0); }
        }
        for (int o = 16; o; o >>= 1) {
            unsigned long long lo = (unsigned long long)acc, hi = (unsigned long long)((unsigned __int128)acc >> 64);
            lo = __shfl_xor_sync(~0u, lo, o); hi = __shfl_xor_sync(~0u, hi, o);
            acc += (__int128)(((unsigned __int128)hi << 64) | lo); tail += __shfl_xor_sync(~0u, tail, o);
        }
        if (!lane) { float rr; if (!cert_round(acc, tail, lsb, 1, 0, p, rr)) { atomicAdd(nbad, 1); rr = __int_as_float(0x7FC00000); } Y[i] = rr; }
    }
}
// the outputs the XF8 product left open, one warp each, a lane per block of 128: fp8b_gemm's sum for that output alone (each
// block sum exact in int64, |S| < 2^43; the term S s sig into a 128-bit window at the largest block's exponent, the same lsb),
// so the same output, certified or NaN for the host GEMV
__global__ void f8x_fixup(const uint8_t *w, const uint16_t *ws, const uint8_t *xq, const float *xs, int K, const unsigned *list,
                          const unsigned *cnt, unsigned cap, float *Y, int M, int p, int *nbad) {
    unsigned nl = min(*cnt, cap), lane = threadIdx.x & 31; int KB = K / 128;
    auto v8 = [](uint8_t b) { int e = (b >> 3) & 15, f = b & 7, v = e ? (8 + f) << (e - 1) : f; return b & 0x80 ? -v : v; };
    auto wexp = [&](int r, int b, int &sig) { uint16_t u = ws[(size_t)(r / 128) * KB + b]; int x = (u >> 7) & 255; sig = x ? 128 + (u & 127) : u & 127; return x ? x - 134 : -133; };
    auto xexp = [&](size_t tk, int b, int &sig) { int x; sig = (int)ldexpf(frexpf(xs[tk * KB + b], &x), 24); return x - 24; };
    for (unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) / 32; j < nl; j += gridDim.x * blockDim.x / 32) {
        size_t i = list[j], tk = i / M; int r = (int)(i % M); const uint8_t *wr = w + (size_t)r * K, *xr = xq + tk * K;
        int hi = -100000, s1, s2;
        for (int b = lane; b < KB; b += 32) hi = max(hi, wexp(r, b, s1) + xexp(tk, b, s2));
        for (int o = 16; o; o >>= 1) hi = max(hi, __shfl_xor_sync(~0u, hi, o));
        int lsb = hi - 18 + 81 - 125, tail = 0; __int128 acc = 0;   // as fp8b_gemm: |sum| < 2^(hi - 18 + 75 + 6)
        for (int b = lane; b < KB; b += 32) {
            long long S = 0; for (int k = b * 128; k < b * 128 + 128; k++) S += (long long)v8(wr[k]) * v8(xr[k]);   // |v v'| < 2^36: int64
            if (!S) continue;
            int sw, sx, e = wexp(r, b, sw) + xexp(tk, b, sx) - 18 - lsb; __int128 t = (__int128)S * ((long long)sw * sx);
            if (e >= 0) acc += t << e; else { tail++; acc += e > -127 ? t >> -e : (t < 0 ? -1 : 0); }
        }
        for (int o = 16; o; o >>= 1) {
            unsigned long long lo = (unsigned long long)acc, h = (unsigned long long)((unsigned __int128)acc >> 64);
            lo = __shfl_xor_sync(~0u, lo, o); h = __shfl_xor_sync(~0u, h, o);
            acc += (__int128)(((unsigned __int128)h << 64) | lo); tail += __shfl_xor_sync(~0u, tail, o);
        }
        if (!lane) { float rr; if (!cert_round(acc, tail, lsb, 1, 0, p, rr)) { atomicAdd(nbad, 1); rr = __int_as_float(0x7FC00000); } Y[i] = rr; }
    }
}
