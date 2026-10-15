// phases.cu -- where an Ozaki-II DGEMM spends its time (n=4096)
#include <cstdio>
#include "../src/ozaki2.cuh"
template <int NM> void run(double *, double *, double *);
int main() {
    const int n = 4096; double *A, *B, *C;
    cudaMalloc(&A, (size_t)n * n * 8); cudaMalloc(&B, (size_t)n * n * 8); cudaMalloc(&C, (size_t)n * n * 8);
    {   // random data: tensor-core power (and so the clock) depends on the bits
        double *h = new double[(size_t)n * n]; unsigned long long s = 88172645463325252ull;
        for (size_t i = 0; i < (size_t)n * n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; h[i] = (double)(long long)s / 9.2e18; }
        cudaMemcpy(A, h, (size_t)n * n * 8, cudaMemcpyHostToDevice); cudaMemcpy(B, h, (size_t)n * n * 8, cudaMemcpyHostToDevice); delete[] h;
    }
    run<12>(A, B, C); run<16>(A, B, C);
}
template <int NM> void run(double *A, double *B, double *C) {
    const int n = 4096, nm = NM; {
        oz::Plan p = oz::plan(n, n, n, nm);
        oz::dgemm(p, A, B, C);
        cudaEvent_t e[6]; for (auto &x : e) cudaEventCreate(&x);
        cudaEventRecord(e[0]);
        oz::scales<<<p.M, 256>>>(A, p.M, p.K, p.K, 1, p.bits, p.sa); oz::scales<<<p.N, 256>>>(B, p.N, p.K, 1, p.N, p.bits, p.sb);
        cudaEventRecord(e[1]);
        oz::residues<NM><<<dim3((p.K + 255) / 256, p.M), 256>>>(A, p.M, p.K, p.K, 1, p.sa, p.Ar, p.Kp, (size_t)p.Mp * p.Kp);
        cudaEventRecord(e[2]);
        oz::residues<NM><<<dim3((p.K + 255) / 256, p.N), 256>>>(B, p.N, p.K, 1, p.N, p.sb, p.Br, p.Kp, (size_t)p.Np * p.Kp);
        cudaEventRecord(e[3]);
        for (int l = 0; l < nm; l++)
            oz::gemm_i8<<<dim3(p.Np / oz::BN, p.Mp / oz::BM), oz::THREADS>>>(p.Ar + (size_t)l * p.Mp * p.Kp, p.Br + (size_t)l * p.Np * p.Kp, p.Cl + (size_t)l * p.Mp * p.Np, p.Mp, p.Np, p.Kp);
        cudaEventRecord(e[4]);
        oz::crt<NM><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, p.sa, p.sb, C);
        cudaEventRecord(e[5]); cudaEventSynchronize(e[5]);
        const char *nm_[] = {"scales", "residues A", "residues B", "INT8 GEMMs", "CRT"};
        printf("nm=%d:", nm);
        for (int i = 0; i < 5; i++) { float ms; cudaEventElapsedTime(&ms, e[i], e[i + 1]); printf("  %s %.2f ms", nm_[i], ms); }
        printf("\n"); oz::release(p);
    }
}
