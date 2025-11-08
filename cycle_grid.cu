// cooperative_cycle_ident_verbose.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define IDX2C(i,j,n) ((j)*(n)+(i))

// -----------------------------------------------------------
// Device helpers
// -----------------------------------------------------------
__device__ void build_composed_maps_dev(const int* R, const int* Q,
                                        int n, int tid, int stride,
                                        int* fR, int* fQ)
{
    for (int x = tid; x < n; x += stride) {
        int r = R[x];
        if (r >= n) fR[x] = x;
        else {
            int q = Q[r];
            fR[x] = (q >= n ? x : q);
        }

        int q0 = Q[x];
        if (q0 >= n) fQ[x] = x;
        else {
            int r2 = R[q0];
            fQ[x] = (r2 >= n ? x : r2);
        }
    }
}

__device__ void power_double_dev(const int* src, int* dst, int n, int tid, int stride)
{
    for (int x = tid; x < n; x += stride) {
        int y = src[x];
        dst[x] = src[y];
    }
}

// -----------------------------------------------------------
// Cooperative kernel: identifies cycles fully on GPU
// -----------------------------------------------------------
__global__ void identify_cycles_coop(const int* R, const int* Q,
                                     int n, int k, int l,
                                     int* fR_A, int* fR_B,
                                     int* fQ_A, int* fQ_B,
                                     unsigned char* hasPredR,
                                     unsigned char* hasPredQ,
                                     unsigned char* cycR,
                                     unsigned char* cycQ)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // 1. Build composed maps
    build_composed_maps_dev(R, Q, n, tid, stride, fR_A, fQ_A);
    grid.sync();

    // 2. Pointer jumping (power doubling)
    int rounds = 0; for (int t = n; t > 1; t >>= 1) ++rounds;
    int *fR_src = fR_A, *fR_dst = fR_B;
    int *fQ_src = fQ_A, *fQ_dst = fQ_B;
    for (int r = 0; r < rounds; ++r) {
        power_double_dev(fR_src, fR_dst, n, tid, stride);
        power_double_dev(fQ_src, fQ_dst, n, tid, stride);
        grid.sync();
        int* tmp;
        tmp = fR_src; fR_src = fR_dst; fR_dst = tmp;
        tmp = fQ_src; fQ_src = fQ_dst; fQ_dst = tmp;
    }
    grid.sync();

    // 3. Representatives and broadcast
    __shared__ int repR_shared, repQ_shared;
    if (grid.thread_rank() == 0) {
        repR_shared = fR_src[k];
        repQ_shared = fQ_src[l];
        printf("Representative of (k=%d,l=%d): (repR=%d, repQ=%d)\n", k, l, repR_shared, repQ_shared);
    }
    grid.sync();
    int repR = repR_shared;
    int repQ = repQ_shared;
    grid.sync();

    // 4. Mark candidates
    for (int x = tid; x < n; x += stride) {
        cycR[x] = (fR_src[x] == repR) ? 1 : 0;
        cycQ[x] = (fQ_src[x] == repQ) ? 1 : 0;
        hasPredR[x] = 0;
        hasPredQ[x] = 0;
    }
    grid.sync();

    // 5. Flag predecessors
    for (int u = tid; u < n; u += stride) {
        if (cycR[u]) hasPredR[fR_src[u]] = 1;
        if (cycQ[u]) hasPredQ[fQ_src[u]] = 1;
    }
    grid.sync();

    // 6. Finalize cycle flags
    for (int x = tid; x < n; x += stride) {
        cycR[x] = (cycR[x] && hasPredR[x]) ? 1 : 0;
        cycQ[x] = (cycQ[x] && hasPredQ[x]) ? 1 : 0;
    }
    grid.sync();

    // Optional: Print from thread 0
    if (grid.thread_rank() == 0) {
        printf("Cycle detection complete (device-only)\n");
    }
}

// -----------------------------------------------------------
// Host driver with printouts
// -----------------------------------------------------------
int main()
{
    const int n = 4, k = 0, l = 1;
    int h_R[n] = {0, 1, 4, 4};
    int h_Q[n] = {1, 0, 4, 4};

    int *d_R, *d_Q, *d_fR_A, *d_fR_B, *d_fQ_A, *d_fQ_B;
    unsigned char *d_hasPredR, *d_hasPredQ, *d_cycR, *d_cycQ;

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_fR_A, n*sizeof(int));
    cudaMalloc(&d_fR_B, n*sizeof(int));
    cudaMalloc(&d_fQ_A, n*sizeof(int));
    cudaMalloc(&d_fQ_B, n*sizeof(int));
    cudaMalloc(&d_hasPredR, n);
    cudaMalloc(&d_hasPredQ, n);
    cudaMalloc(&d_cycR, n);
    cudaMalloc(&d_cycQ, n);

    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(1);

    void* args[] = {
        (void*)&d_R, (void*)&d_Q,
        (void*)&n, (void*)&k, (void*)&l,
        (void*)&d_fR_A, (void*)&d_fR_B,
        (void*)&d_fQ_A, (void*)&d_fQ_B,
        (void*)&d_hasPredR, (void*)&d_hasPredQ,
        (void*)&d_cycR, (void*)&d_cycQ
    };

    printf("Launching cooperative kernel...\n");
    cudaLaunchCooperativeKernel((void*)identify_cycles_coop, grid, block, args);
    cudaDeviceSynchronize();

    unsigned char h_cycR[n], h_cycQ[n];
    cudaMemcpy(h_cycR, d_cycR, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycQ, d_cycQ, n, cudaMemcpyDeviceToHost);

    printf("row-cycle indices: ");
    for (int i=0;i<n;i++) if(h_cycR[i]) printf("%d ", i);
    printf("\ncol-cycle indices: ");
    for (int j=0;j<n;j++) if(h_cycQ[j]) printf("%d ", j);
    printf("\n");

    // Optional: reconstruct pair cycle on host
    std::vector<std::pair<int,int>> pair_cycle;
    int i = k, j = l;
    do {
        pair_cycle.emplace_back(i,j);
        int j_next = (h_R[i] < n) ? h_R[i] : j;
        int i_next = (h_Q[j] < n) ? h_Q[j] : i;
        i = i_next; j = j_next;
    } while (!(i == k && j == l) && pair_cycle.size() <= 2*n);

    printf("pair cycle from (k=%d,l=%d): ", k, l);
    for (auto &p : pair_cycle) printf("(%d,%d) ", p.first, p.second);
    printf("\n");

    cudaFree(d_R); cudaFree(d_Q);
    cudaFree(d_fR_A); cudaFree(d_fR_B);
    cudaFree(d_fQ_A); cudaFree(d_fQ_B);
    cudaFree(d_hasPredR); cudaFree(d_hasPredQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);
    return 0;
}
