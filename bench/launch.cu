// launch.cu -- what one launch costs here (WSL2): empty kernels, 2-D device copies, and the same captured in a CUDA graph.
// nvcc -O3 -arch=sm_120 -o out_launch bench/launch.cu
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k_empty() {}
#include <cooperative_groups.h>
__global__ void k_sync(int it, double *x) {   // it grid syncs, with one global round trip each (as the leaf has)
    auto g = cooperative_groups::this_grid();
    for (int i = 0; i < it; i++) { if (!threadIdx.x) x[blockIdx.x] = i; g.sync(); if (!threadIdx.x) x[blockIdx.x] += x[(blockIdx.x + 1) % gridDim.x]; }
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    const int R = 20000;
    double *a, *b; cudaMalloc(&a, 1 << 24); cudaMalloc(&b, 1 << 24);
    cudaStream_t s; cudaStreamCreate(&s);
    auto run = [&](const char *w, auto f) {
        f(); cudaStreamSynchronize(s);
        double t = now(); for (int r = 0; r < R; r++) f(); cudaStreamSynchronize(s);
        printf("%-28s %.2f us each\n", w, (now() - t) / R * 1e6);
    };
    run("empty kernel", [&] { k_empty<<<1, 32, 0, s>>>(); });
    run("cudaMemcpy2DAsync 32x32 D2D", [&] { cudaMemcpy2DAsync(a, 256, b, 4096, 256, 32, cudaMemcpyDeviceToDevice, s); });
    run("cudaMemsetAsync 64 KB", [&] { cudaMemsetAsync(a, 0, 65536, s); });
    for (int B : {8, 16, 46, 92}) for (int T : {64, 256}) {
        int it = 2000; void *args[] = {&it, &a};
        cudaLaunchCooperativeKernel((void *)k_sync, B, T, args); cudaDeviceSynchronize();
        double t = now(); cudaLaunchCooperativeKernel((void *)k_sync, B, T, args); cudaDeviceSynchronize();
        printf("grid.sync %3d x %3d          %.2f us each\n", B, T, (now() - t) / it * 1e6);
    }
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int r = 0; r < 1000; r++) { k_empty<<<1, 32, 0, s>>>(); cudaMemcpy2DAsync(a, 256, b, 4096, 256, 32, cudaMemcpyDeviceToDevice, s); }
    cudaStreamEndCapture(s, &g);
    double t = now(); cudaGraphInstantiate(&ge, g, 0); printf("instantiate 2000 nodes        %.2f ms\n", (now() - t) * 1e3);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    t = now(); for (int r = 0; r < 10; r++) cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    printf("graph: per node              %.2f us\n", (now() - t) / 10 / 2000 * 1e6);
}
