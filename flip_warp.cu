// cycle_warp.cu
#include <cuda_runtime.h>
#include <cstdio>

#define IDX2C(i,j,n) ((j)*(n)+(i))
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Build composed maps:
//   fR[i] = Q[R[i]]  (row -> col -> row), with dead-end (>=n) -> self
//   fQ[j] = R[Q[j]]  (col -> row -> col), with dead-end (>=n) -> self
__device__ inline void build_composed_maps_warp(const int* __restrict__ R,
                                                const int* __restrict__ Q,
                                                int n, int lane, int* __restrict__ fR,
                                                int* __restrict__ fQ)
{
    for (int x = lane; x < n; x += WARP_SIZE) {
        int r = R[x];
        fR[x] = (r >= n) ? x : ((Q[r] >= n) ? x : Q[r]);

        int q0 = Q[x];
        fQ[x] = (q0 >= n) ? x : ((R[q0] >= n) ? x : R[q0]);
    }
}

// One pointer-jumping (power-doubling) round using ping-pong buffers:
//   dst[x] = src[src[x]]
__device__ inline void power_double_round_warp(const int* __restrict__ src,
                                               int* __restrict__ dst,
                                               int n, int lane)
{
    for (int x = lane; x < n; x += WARP_SIZE) {
        int y = src[x];
        dst[x] = src[y];
    }
}

// Main kernel: warp-synchronous cycle identification + flipping.
// Requirements: launch with <<<1, 32>>>.
__global__ void identify_and_flip_warp(const int* __restrict__ R,
                                       const int* __restrict__ Q,
                                       int* __restrict__ X,
                                       int n,
                                       const int* __restrict__ k,
                                       const int* __restrict__ l,
                                       // ping-pong buffers for fR/fQ
                                       int* __restrict__ fR_A,
                                       int* __restrict__ fR_B,
                                       int* __restrict__ fQ_A,
                                       int* __restrict__ fQ_B,
                                       // temp + outputs
                                       unsigned char* __restrict__ hasPredR,
                                       unsigned char* __restrict__ hasPredQ,
                                       unsigned char* __restrict__ cycR,
                                       unsigned char* __restrict__ cycQ)
{
    const unsigned mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x; // 0..31

    // Dereference start indices (device pointers), broadcast to warp
    int k_ = (lane == 0) ? *k : 0;
    int l_ = (lane == 0) ? *l : 0;
    k_ = __shfl_sync(mask, k_, 0);
    l_ = __shfl_sync(mask, l_, 0);

    // 1) Build composed maps into A-buffers
    build_composed_maps_warp(R, Q, n, lane, fR_A, fQ_A);

    // 2) Pointer jumping: log2(n) rounds with ping-pong buffers
    int rounds = 0; for (int t = n; t > 1; t >>= 1) ++rounds;
    const int* fR_src = fR_A; int* fR_dst = fR_B;
    const int* fQ_src = fQ_A; int* fQ_dst = fQ_B;

    for (int r = 0; r < rounds; ++r) {
        power_double_round_warp(fR_src, fR_dst, n, lane);
        power_double_round_warp(fQ_src, fQ_dst, n, lane);
        // swap local pointers; warp lockstep keeps all lanes consistent
        const int* tmpR = fR_src; fR_src = fR_dst; fR_dst = (int*)tmpR;
        const int* tmpQ = fQ_src; fQ_src = fQ_dst; fQ_dst = (int*)tmpQ;
        // (no block/barrier needed: single warp, ping-pong ensures no RAW hazards)
    }

    // Representatives from fR_src[k], fQ_src[l]; lane0 reads, then warp-broadcast
    int repR = (lane == 0) ? fR_src[k_] : 0;
    int repQ = (lane == 0) ? fQ_src[l_] : 0;
    repR = __shfl_sync(mask, repR, 0);
    repQ = __shfl_sync(mask, repQ, 0);

    // 3) Mark candidates and clear predecessor flags
    for (int x = lane; x < n; x += WARP_SIZE) {
        cycR[x] = (fR_src[x] == repR);
        cycQ[x] = (fQ_src[x] == repQ);
        hasPredR[x] = 0;
        hasPredQ[x] = 0;
    }

    // 4) For each candidate u, flag hasPred[f[u]] = 1
    for (int u = lane; u < n; u += WARP_SIZE) {
        if (cycR[u]) hasPredR[fR_src[u]] = 1;
        if (cycQ[u]) hasPredQ[fQ_src[u]] = 1;
    }

    // 5) Final in-cycle flags: candidate & hasPred
    for (int x = lane; x < n; x += WARP_SIZE) {
        cycR[x] = (cycR[x] && hasPredR[x]) ? 1 : 0;
        cycQ[x] = (cycQ[x] && hasPredQ[x]) ? 1 : 0;
    }

    // 6) Flip along the (k,l) pair-cycle (small; do it with lane 0)
    if (lane == 0) {
        int i = k_, j = l_;
        do {
            X[IDX2C(i, j, n)] = 1 - X[IDX2C(i, j, n)];
            int j_next = (R[i] < n) ? R[i] : j;
            int i_next = (Q[j] < n) ? Q[j] : i;
            i = i_next; j = j_next;
        } while (!(i == k_ && j == l_));
    }
}

// -------------------- Minimal host driver (example) --------------------
int main() {
    const int n = 4;
    int h_R[n] = {0, 1, 4, 4};
    int h_Q[n] = {1, 0, 4, 4};
    int h_X[n*n] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    int h_k = 0, h_l = 1;

    // Device allocations
    int *d_R, *d_Q, *d_X, *d_fR_A, *d_fR_B, *d_fQ_A, *d_fQ_B, *d_k, *d_l;
    unsigned char *d_hasPredR, *d_hasPredQ, *d_cycR, *d_cycQ;
    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_X, n*n*sizeof(int));
    cudaMalloc(&d_fR_A, n*sizeof(int));
    cudaMalloc(&d_fR_B, n*sizeof(int));
    cudaMalloc(&d_fQ_A, n*sizeof(int));
    cudaMalloc(&d_fQ_B, n*sizeof(int));
    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_l, sizeof(int));
    cudaMalloc(&d_hasPredR, n);
    cudaMalloc(&d_hasPredQ, n);
    cudaMalloc(&d_cycR, n);
    cudaMalloc(&d_cycQ, n);

    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, &h_l, sizeof(int), cudaMemcpyHostToDevice);

    // Launch with a single warp
    identify_and_flip_warp<<<1, WARP_SIZE>>>(
        d_R, d_Q, d_X, n, d_k, d_l,
        d_fR_A, d_fR_B, d_fQ_A, d_fQ_B,
        d_hasPredR, d_hasPredQ, d_cycR, d_cycQ
    );
    cudaDeviceSynchronize();

    // (Optional) read back X if you want to verify
    cudaMemcpy(h_X, d_X, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) printf("%d ", h_X[IDX2C(row,col,n)]);
        printf("\n");
    }

    cudaFree(d_R); cudaFree(d_Q); cudaFree(d_X);
    cudaFree(d_fR_A); cudaFree(d_fR_B);
    cudaFree(d_fQ_A); cudaFree(d_fQ_B);
    cudaFree(d_k); cudaFree(d_l);
    cudaFree(d_hasPredR); cudaFree(d_hasPredQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);
    return 0;
}
