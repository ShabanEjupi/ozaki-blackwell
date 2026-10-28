// fp8gemm.cu -- the FP8 block-scaled products of Jais-3's routed experts for many tokens at once on INT8 tensor cores,
// correctly rounded, so equal bit for bit to the per-token GEMV of ultra.cu (F_E4M3B, act_fp8g): the batched evaluation
// of a benchmark (2500 questions x ~200 tokens) needs a GEMM, and correct rounding makes batching invisible.
// W = E4M3 times a BF16 scale per 128x128 block; X = E4M3 times an FP32 scale per token and group of 128.
// Why it is exact. An E4M3 value times 2^9 is an integer v, |v| <= 448 * 512 < 2^18: normal (8 + m) 2^(e-1), subnormal m.
// Three balanced base-128 digits in [-64, 63] hold it (v = d0 + 128 d1 + 16384 d2), so a block sum S_b = sum_k vw va over
// the 128 k of a block is 9 digit-pair sums, each a chain of 8 m16n8k16 INT8 MMAs (|d d'| <= 2^12, 128 terms < 2^19 in
// int32), recombined as sum P_pq << 7(p+q) in int64 (|S_b| <= 128 * 229376^2 < 2^43). The block term S_b sig(ws) sig(xs)
// 2^(e(ws) + e(xs) - 18) is below 2^75 times its power of two; the K/128 terms go into a 128-bit window anchored at the
// largest block exponent, bits below it counted as tail and the rounding certified (cert_round of ultra.cu).
// nvcc -O3 -arch=sm_120a -std=c++17 -o out_fp8gemm llm/fp8gemm.cu
// out_fp8gemm <model dir> <tensor without .weight> <M> <K> [N tokens, default 4096]: the batched product against the
// per-token GEMV on the first 256 tokens (BF16 and FP32 outputs), then the time of the batched product alone.
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
        if (!cert_round(acc[i], tail[i], lsb[i & 1], 1, 0, p, r)) atomicAdd(nbad, 1);
        Y[(size_t)col * M + row] = r;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: out_fp8gemm <model dir> <tensor> <M> <K> [N]\n"); return 1; }
    DIR = argv[1]; std::string T = argv[2]; int M = atoi(argv[3]), K = atoi(argv[4]), N = argc > 5 ? atoi(argv[5]) : 4096, NREF = 256;
    read_index();
    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)M * K)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, K * 4)); CK(cudaMalloc(&dxe, K * 2)); CK(cudaMalloc(&dy, M * 4)); CK(cudaMalloc(&dbad, 4));
    DevW W = load_w(T + ".weight", F_E4M3B, M, K); to_dev(T + ".weight_scale", dS); W.sw = (uint8_t *)dS;
    // tokens: BF16 values, each token with its own scale and a few outliers (as hidden states have)
    std::mt19937_64 rng(28102026); std::normal_distribution<double> nd; std::vector<float> x((size_t)N * K);
    for (int n = 0; n < N; n++) { double sc = exp2((double)(rng() % 24) - 12);
        for (int k = 0; k < K; k++) x[(size_t)n * K + k] = bf(nd(rng) * sc * (rng() % 97 == 0 ? 64 : 1)); }
    int8_t *wd, *xd; uint8_t *xq; float *dx, *xsc, *Y; int *nb;
    CK(cudaMalloc(&wd, (size_t)3 * M * K)); CK(cudaMalloc(&xd, (size_t)3 * N * K)); CK(cudaMalloc(&xq, (size_t)N * K));
    CK(cudaMalloc(&dx, (size_t)N * K * 4)); CK(cudaMalloc(&xsc, (size_t)N * K / 128 * 4)); CK(cudaMalloc(&Y, (size_t)N * M * 4)); CK(cudaMalloc(&nb, 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    fp8_digits<<<1024, 256>>>((const uint8_t *)W.w, wd, (size_t)M * K);
    auto run = [&](int p) {
        quant_fp8g<<<1024, 256>>>(dx, xq, xsc, (size_t)N * K / 128);
        fp8_digits<<<1024, 256>>>(xq, xd, (size_t)N * K);
        fp8b_gemm<<<dim3(M / 128, N / 8), 256>>>(wd, (const uint16_t *)dS, xd, xsc, Y, M, N, K, p, nb);
    };
    for (int p : {8, 24}) {
        CK(cudaMemset(nb, 0, 4)); run(p); CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t)N * M); CK(cudaMemcpy(y.data(), Y, y.size() * 4, cudaMemcpyDeviceToHost));
        int hb; CK(cudaMemcpy(&hb, nb, 4, cudaMemcpyDeviceToHost));
        long long diff = 0; st_bad = 0;
        for (int n = 0; n < NREF; n++) {
            std::vector<float> xn(x.begin() + (size_t)n * K, x.begin() + (size_t)(n + 1) * K);
            auto r = gemv(W, act_fp8g(xn), p, 0);
            for (int m = 0; m < M; m++) diff += r[m] != y[(size_t)n * M + m];
        }
        printf("%s %d x %d, %d tokens, p %d: batched vs per-token GEMV on %d tokens: %lld of %lld differ; uncertified batched %d, GEMV %lld\n",
               T.c_str(), M, K, N, p, NREF, diff, (long long)NREF * M, hb, st_bad);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int r = 0; r < 5; r++) run(8);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    printf("batched: %.2f ms for %d tokens (quantize + digits + product) = %.2f TOPS of the FP8 product\n", ms, N, 2.0 * M * K * N / (ms * 1e9));
    return 0;
}
