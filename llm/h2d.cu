// h2d.cu -- host-to-GPU bandwidth from pinned memory (the second link of an out-of-core model: SSD -> RAM -> GPU)
// nvcc -O3 -arch=sm_120 -o out_h2d llm/h2d.cu
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    const size_t n = (size_t)1 << 30;
    void *h, *d; cudaMallocHost(&h, n); cudaMalloc(&d, n);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (size_t c : {(size_t)18 << 20, (size_t)256 << 20, n}) {   // one MXFP4 expert matrix set is ~17.5 MB
        cudaMemcpy(d, h, c, cudaMemcpyHostToDevice);
        cudaEventRecord(a);
        for (int r = 0; r < 5; r++) cudaMemcpy(d, h, c, cudaMemcpyHostToDevice);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("pinned host -> GPU, %zu MB chunks: %.1f GB/s\n", c >> 20, 5.0 * c / ms / 1e6);
    }
    return 0;
}
