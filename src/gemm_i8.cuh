// gemm_i8.cuh -- C[M,N] (int32) = A[M,K] (int8, row-major) * B[K,N], with B given as Bt[N,K] (K contiguous).
// The engine of the FP64 emulation: every Ozaki-II modulus is one call of this.
// Tile 128x128x64, 8 warps (2 x 4), each warp 64x32 = 4 x 4 mma.sync.m16n8k32. Three-stage cp.async pipeline.
// Shared tiles are 64-byte rows; the 16-byte chunk c of row r lives at c ^ ((r >> 1) & 3), so the 8 rows one ldmatrix
// reads land in 8 different 16-byte bank groups (no conflicts).
// Needs M, N % 128 == 0 and K % 64 == 0 (the caller pads).
#pragma once
#include <cstdint>

namespace oz {

constexpr int BM = 128, BN = 128, BK = 64, THREADS = 256, STAGES = 3;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int r, int c) { return r * BK + ((c ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp16(uint32_t dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}

// one ROWSx64 tile of a K-contiguous matrix into shared memory: ROWS * 4 chunks of 16 bytes, ROWS / 64 per thread
template <int ROWS = 128>
__device__ __forceinline__ void load_tile(int8_t *s, const int8_t *g, int ld, int tid) {
#pragma unroll
    for (int i = 0; i < ROWS / 64; i++) {
        int idx = tid + i * THREADS, r = idx >> 2, c = idx & 3;
        cp16(smem_u32(s + swz(r, c)), g + (size_t)r * ld + c * 16);
    }
}

// the block's 128 x (4 WN) tile of A Bt; epi(row, col, c00, c01, c10, c11) gets (row, col..col+1) and (row + 8, col..col+1).
// WN = 32 (default): warps 64x32, 48 KB static shared memory. WN = 64: warps 64x64, a third fewer shared-memory bytes per
// MAC and a quarter fewer L2 bytes (energy: the GPU is power-capped); ST stages x 24 KB of DYNAMIC shared memory, which
// the kernel must be granted (cudaFuncAttributeMaxDynamicSharedMemorySize) and launched with: gemm_smem<64, ST>().
template <int WN = 32, int ST = STAGES> constexpr int gemm_smem() { return WN == 32 ? 0 : ST * (BM + 4 * WN) * BK; }
template <int WN = 32, int ST = STAGES, class Epi>
__device__ __forceinline__ void gemm_i8_tile(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt, int K, Epi epi,
                                             int lda = 0, int ldb = 0, int group = 0) {   // row strides; 0 = K (sub-blocks of a bigger matrix: Strassen)
    // group > 0: blocks walk the grid in bands of `group` tile rows, column by column, so a band's A rows and the B columns
    // in use stay in L2 (row-major order re-reads all of B per tile row: n = 32768 fell to 86 TOPS from ~130 at 4096)
    constexpr int BNW = 4 * WN, NJ = WN / 8;   // block tile width, n8 tiles per warp
    int bx = blockIdx.x, by = blockIdx.y;
    if (group) {
        const int pid = by * gridDim.x + bx, span = group * gridDim.x, first = pid / span * group;
        const int gs = min((int)gridDim.y - first, group);
        by = first + (pid % span) % gs; bx = (pid % span) / gs;
    }
    if (!lda) lda = K;
    if (!ldb) ldb = K;
    __shared__ __align__(128) int8_t sst[WN == 32 ? ST * (BM + BNW) * BK : 16];   // WN = 32: 3 x 16 KB = the 48 KB static limit
    extern __shared__ __align__(128) int8_t sdyn[];
    int8_t *const sbase = WN == 32 ? sst : sdyn;
    auto sA = [&](int s) { return sbase + s * BM * BK; };
    auto sB = [&](int s) { return sbase + ST * BM * BK + s * BNW * BK; };
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * WN;   // this warp's corner inside the block tile
    const int8_t *gA = A + (size_t)by * BM * lda, *gB = Bt + (size_t)bx * BNW * ldb;

    int32_t acc[4][NJ][4] = {};   // [m16 tile][n8 tile][fragment]
    const int kt = K / BK;
    // three stages: tiles t+1 and t+2 are in flight while tile t is multiplied (15-10: two stages gave 57 % of peak).
    // One group is committed per tile, empty ones too, so "wait_group 1" always means "tile t has landed".
#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < kt) { load_tile(sA(s), gA + s * BK, lda, tid); load_tile<BNW>(sB(s), gB + s * BK, ldb, tid); }
        asm volatile("cp.async.commit_group;");
    }

    for (int t = 0; t < kt; t++) {
        const int cur = t % ST;
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 2));
        __syncthreads();   // tile t visible to all, and everybody is done with tile t-1, whose buffer is refilled now
        if (t + ST - 1 < kt) {
            const int nxt = (t + ST - 1) % ST;
            load_tile(sA(nxt), gA + (t + ST - 1) * BK, lda, tid); load_tile<BNW>(sB(nxt), gB + (t + ST - 1) * BK, ldb, tid);
        }
        asm volatile("cp.async.commit_group;");

#pragma unroll
        for (int ks = 0; ks < 2; ks++) {   // two k32 steps per 64-wide tile
            uint32_t a[4][4], b[NJ][2];
#pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                // x4: (rows 0-7, k 0-15) (rows 8-15, k 0-15) (rows 0-7, k 16-31) (rows 8-15, k 16-31) = a0..a3
                int r = wm + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, c = ks * 2 + (lane >> 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])
                             : "r"(smem_u32(sA(cur) + swz(r, c))));
            }
#pragma unroll
            for (int nj = 0; nj < NJ; nj += 2) {
                // x4 over two n8 tiles: (n 0-7, k 0-15) (n 0-7, k 16-31) (n 8-15, k 0-15) (n 8-15, k 16-31)
                int r = wn + nj * 8 + (lane & 7) + (lane >> 4) * 8, c = ks * 2 + ((lane >> 3) & 1);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[nj][0]), "=r"(b[nj][1]), "=r"(b[nj + 1][0]), "=r"(b[nj + 1][1])
                             : "r"(smem_u32(sB(cur) + swz(r, c))));
            }
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+r"(acc[mi][nj][0]), "+r"(acc[mi][nj][1]), "+r"(acc[mi][nj][2]), "+r"(acc[mi][nj][3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[nj][0]), "r"(b[nj][1]));
        }
    }

    // c0,c1: row g, cols 2t,2t+1; c2,c3: row g+8
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            int row = by * BM + wm + mi * 16 + g, col = bx * BNW + wn + nj * 8 + q * 2;
            epi(row, col, acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]);
        }
}

__global__ void __launch_bounds__(THREADS) gemm_i8(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt,
                                                    int32_t *__restrict__ C, int M, int N, int K) {
    A += (size_t)blockIdx.z * M * K; Bt += (size_t)blockIdx.z * N * K; C += (size_t)blockIdx.z * M * N;   // z = modulus: one launch for all planes
    gemm_i8_tile(A, Bt, K, [&](int row, int col, int c00, int c01, int c10, int c11) {
        *(int2 *)&C[(size_t)row * N + col] = make_int2(c00, c01);
        *(int2 *)&C[(size_t)(row + 8) * N + col] = make_int2(c10, c11);
    });
}

}  // namespace oz
