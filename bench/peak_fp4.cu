// peak_fp4.cu -- register-only peak of the two instructions behind llm/nvexact.cu: mma.m16n8k16 INT8 (one NVFP4 block of
// 16 per instruction, the exact path) and mma.m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X (the native NVFP4 path).
// nvcc -O3 -arch=sm_120a -o out_peak_fp4 bench/peak_fp4.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
constexpr int ITER = 4096, CH = 8;   // 8 independent accumulator chains per warp
__global__ void k_i8(int *out) {
    uint32_t a0 = threadIdx.x, a1 = a0 * 3, b0 = a0 * 5; int d[CH][4] = {};
    for (int it = 0; it < ITER; it++)
#pragma unroll
        for (int c = 0; c < CH; c++)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+r"(d[c][0]), "+r"(d[c][1]), "+r"(d[c][2]), "+r"(d[c][3]) : "r"(a0), "r"(a1), "r"(b0));
    int s = 0; for (int c = 0; c < CH; c++) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void k_fp4(float *out) {
    uint32_t a0 = threadIdx.x & 0x77777777u, a1 = a0 ^ 0x11111111u, a2 = a0 ^ 0x22222222u, a3 = a0 ^ 0x33333333u;
    uint32_t b0 = a0 ^ 0x44444444u, b1 = a0 ^ 0x55555555u, sf = 0x38383838u;   // E4M3 1.0 in every byte
    float d[CH][4] = {};
    for (int it = 0; it < ITER; it++)
#pragma unroll
        for (int c = 0; c < CH; c++)
            asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};\n"
                         : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sf), "h"((uint16_t)0), "h"((uint16_t)0));
    float s = 0; for (int c = 0; c < CH; c++) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
int main() {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sms * 8, threads = 128; void *o; cudaMalloc(&o, blocks * threads * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
    for (int rep = 0; rep < 2; rep++) {
        k_i8<<<blocks, threads>>>((int *)o); cudaEventRecord(e0); k_i8<<<blocks, threads>>>((int *)o); cudaEventRecord(e1);
        cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        double ops_i8 = 2.0 * 16 * 8 * 16 * (double)ITER * CH * blocks * threads / 32;
        printf("mma.m16n8k16 s8 (exact path, one NVFP4 block per instruction): %.0f TOPS\n", ops_i8 / ms / 1e9);
        k_fp4<<<blocks, threads>>>((float *)o); cudaEventRecord(e0); k_fp4<<<blocks, threads>>>((float *)o); cudaEventRecord(e1);
        cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        double ops_f4 = 2.0 * 16 * 8 * 64 * (double)ITER * CH * blocks * threads / 32;
        printf("mma.m16n8k64 mxf4nvf4 block_scale (native NVFP4): %.0f TOPS  (%s)\n", ops_f4 / ms / 1e9, cudaGetErrorString(cudaGetLastError()));
    }
}
