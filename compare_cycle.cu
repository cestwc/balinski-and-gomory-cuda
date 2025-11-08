#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define IDX2C(i,j,n) ((j)*(n)+(i))

// =====================================================
// (1) Original sequential while-loop kernel
// =====================================================
__global__ void process_cycle_serial(float* B, float* V, int* d_X,
                                     const int* d_R, const int* d_Q,
                                     int n, int* k, int* l) {
    int k_ = *k;
    int l_ = *l;
    while (true) {
        d_X[IDX2C(k_, l_, n)] = 1;
        l_ = d_R[k_];
        d_X[IDX2C(k_, l_, n)] = 0;
        k_ = d_Q[l_];
        if (k_ == *k && l_ == *l) break;
    }
    V[*l] += B[IDX2C(*k, *l, n)];
}

// =====================================================
// (2) Simplified cooperative kernel (only pointer jump)
// =====================================================
__global__ void cooperative_cycle_detect(const int* R, const int* Q, int n, int* fR, int* fQ) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        int r = R[i];
        if (r >= n) fR[i] = i;
        else {
            int q = Q[r];
            fR[i] = (q >= n) ? i : q;
        }
        int q0 = Q[i];
        if (q0 >= n) fQ[i] = i;
        else {
            int r2 = R[q0];
            fQ[i] = (r2 >= n) ? i : r2;
        }
    }
    grid.sync();

    // Pointer jumping
    int rounds = 0; for (int t = n; t > 1; t >>= 1) ++rounds;
    for (int r = 0; r < rounds; ++r) {
        for (int i = tid; i < n; i += stride) {
            fR[i] = fR[fR[i]];
            fQ[i] = fQ[fQ[i]];
        }
        grid.sync();
    }
}

// =====================================================
// Host benchmark
// =====================================================
void benchmark(int n) {
    printf("\n========== Benchmark for n = %d ==========\n", n);
    int *d_R, *d_Q, *d_X, *d_fR, *d_fQ, *d_k, *d_l;
    float *d_B, *d_V;

    int *h_R = new int[n];
    int *h_Q = new int[n];
    for (int i = 0; i < n; ++i) {
        h_R[i] = (i + 1) % n;
        h_Q[i] = (i + n - 1) % n;
    }

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
    cudaMalloc(&d_X, n*n*sizeof(int));
    cudaMalloc(&d_B, n*n*sizeof(float));
    cudaMalloc(&d_V, n*sizeof(float));
    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_l, sizeof(int));

    int h_k = 0, h_l = 1;
    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, &h_l, sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(1);

    // ---- Serial kernel timing ----
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    process_cycle_serial<<<1,1>>>(d_B, d_V, d_X, d_R, d_Q, n, d_k, d_l);
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float time_serial = 0;
    cudaEventElapsedTime(&time_serial, start1, stop1);
    printf("Serial kernel time: %.4f ms\n", time_serial);

    // ---- Cooperative kernel timing ----
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    void* args[] = {(void*)&d_R, (void*)&d_Q, (void*)&n, (void*)&d_fR, (void*)&d_fQ};

    cudaEventRecord(start2);
    cudaLaunchCooperativeKernel((void*)cooperative_cycle_detect, grid, block, args);
    cudaEventRecord(stop2);
    cudaDeviceSynchronize();

    float time_coop = 0;
    cudaEventElapsedTime(&time_coop, start2, stop2);
    printf("Cooperative kernel time: %.4f ms\n", time_coop);

    printf("Speedup: %.2fx faster\n", time_serial / time_coop);

    cudaFree(d_R); cudaFree(d_Q); cudaFree(d_fR); cudaFree(d_fQ);
    cudaFree(d_X); cudaFree(d_B); cudaFree(d_V);
    cudaFree(d_k); cudaFree(d_l);
    delete[] h_R; delete[] h_Q;
}

int main() {
    for (int n : {128, 1024, 8192, 65536})
        benchmark(n);
    return 0;
}
