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

// one 128x64 tile of a K-contiguous matrix into shared memory: 512 chunks of 16 bytes, 2 per thread
__device__ __forceinline__ void load_tile(int8_t *s, const int8_t *g, int ld, int tid) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid + i * THREADS, r = idx >> 2, c = idx & 3;
        cp16(smem_u32(s + swz(r, c)), g + (size_t)r * ld + c * 16);
    }
}

__global__ void __launch_bounds__(THREADS) gemm_i8(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt,
                                                    int32_t *__restrict__ C, int M, int N, int K) {
    A += (size_t)blockIdx.z * M * K; Bt += (size_t)blockIdx.z * N * K; C += (size_t)blockIdx.z * M * N;   // z = modulus: one launch for all planes
    __shared__ __align__(128) int8_t sA[STAGES][BM * BK], sB[STAGES][BN * BK];   // 3 x 16 KB = the 48 KB static limit
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;   // this warp's corner inside the block tile
    const int8_t *gA = A + (size_t)blockIdx.y * BM * K, *gB = Bt + (size_t)blockIdx.x * BN * K;

    int32_t acc[4][4][4] = {};   // [m16 tile][n8 tile][fragment]
    const int kt = K / BK;
    // three stages: tiles t+1 and t+2 are in flight while tile t is multiplied (15-10: two stages gave 57 % of peak).
    // One group is committed per tile, empty ones too, so "wait_group 1" always means "tile t has landed".
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < kt) { load_tile(sA[s], gA + s * BK, K, tid); load_tile(sB[s], gB + s * BK, K, tid); }
        asm volatile("cp.async.commit_group;");
    }

    for (int t = 0; t < kt; t++) {
        const int cur = t % STAGES;
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();   // tile t visible to all, and everybody is done with tile t-1, whose buffer is refilled now
        if (t + STAGES - 1 < kt) {
            const int nxt = (t + STAGES - 1) % STAGES;
            load_tile(sA[nxt], gA + (t + STAGES - 1) * BK, K, tid); load_tile(sB[nxt], gB + (t + STAGES - 1) * BK, K, tid);
        }
        asm volatile("cp.async.commit_group;");

#pragma unroll
        for (int ks = 0; ks < 2; ks++) {   // two k32 steps per 64-wide tile
            uint32_t a[4][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                // x4: (rows 0-7, k 0-15) (rows 8-15, k 0-15) (rows 0-7, k 16-31) (rows 8-15, k 16-31) = a0..a3
                int r = wm + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, c = ks * 2 + (lane >> 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])
                             : "r"(smem_u32(sA[cur] + swz(r, c))));
            }
#pragma unroll
            for (int nj = 0; nj < 4; nj += 2) {
                // x4 over two n8 tiles: (n 0-7, k 0-15) (n 0-7, k 16-31) (n 8-15, k 0-15) (n 8-15, k 16-31)
                int r = wn + nj * 8 + (lane & 7) + (lane >> 4) * 8, c = ks * 2 + ((lane >> 3) & 1);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[nj][0]), "=r"(b[nj][1]), "=r"(b[nj + 1][0]), "=r"(b[nj + 1][1])
                             : "r"(smem_u32(sB[cur] + swz(r, c))));
            }
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int nj = 0; nj < 4; nj++)
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
        for (int nj = 0; nj < 4; nj++) {
            int row = blockIdx.y * BM + wm + mi * 16 + g, col = blockIdx.x * BN + wn + nj * 8 + q * 2;
            *(int2 *)&C[(size_t)row * N + col] = make_int2(acc[mi][nj][0], acc[mi][nj][1]);
            *(int2 *)&C[(size_t)(row + 8) * N + col] = make_int2(acc[mi][nj][2], acc[mi][nj][3]);
        }
}

}  // namespace oz
