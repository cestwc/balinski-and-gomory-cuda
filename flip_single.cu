#include <cuda_runtime.h>
#include <cstdio>

#define IDX2C(i,j,n) ((j)*(n)+(i))

// ============================================================================
// Device helper: build composed mappings
// fR[i] = Q[R[i]]   (row → column → row)
// fQ[j] = R[Q[j]]   (column → row → column)
// Dead-end (index ≥ n) maps to itself.
// ============================================================================
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

// ============================================================================
// Device helper: pointer-jumping (path-doubling)
// Repeatedly compresses mappings until convergence.
// ============================================================================
__device__ void power_double_dev(int* fR, int* fQ, int n, int tid, int stride)
{
    for (int x = tid; x < n; x += stride) {
        fR[x] = fR[fR[x]];
        fQ[x] = fQ[fQ[x]];
    }
}

// ============================================================================
// Main kernel: identify the (*k,*l) cycle and flip entries in X along it.
//  - Parallelized within a single block using __syncthreads()
//  - Works fully on device, no host intervention.
// ============================================================================
__global__ void identify_and_flip_singleblock(const int* R, const int* Q,
                                              int* X, int n,
                                              const int* k, const int* l,
                                              int* fR, int* fQ,
                                              unsigned char* hasPredR,
                                              unsigned char* hasPredQ,
                                              unsigned char* cycR,
                                              unsigned char* cycQ)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Dereference k and l (device pointers)
    int k_ = *k;
    int l_ = *l;

    // Step 1. Build composed maps fR, fQ
    build_composed_maps_dev(R, Q, n, tid, stride, fR, fQ);
    __syncthreads();

    // Step 2. Pointer-jumping: log(n) doubling steps
    int rounds = 0; for (int t = n; t > 1; t >>= 1) ++rounds;
    for (int r = 0; r < rounds; ++r) {
        power_double_dev(fR, fQ, n, tid, stride);
        __syncthreads();
    }

    // Step 3. Shared representative broadcast
    __shared__ int repR, repQ;
    if (tid == 0) {
        repR = fR[k_];
        repQ = fQ[l_];
    }
    __syncthreads();

    // Step 4. Mark candidate cycle members and reset predecessor flags
    for (int x = tid; x < n; x += stride) {
        cycR[x] = (fR[x] == repR);
        cycQ[x] = (fQ[x] == repQ);
        hasPredR[x] = 0;
        hasPredQ[x] = 0;
    }
    __syncthreads();

    // Step 5. Mark nodes that have predecessors within their candidate set
    for (int u = tid; u < n; u += stride) {
        if (cycR[u]) hasPredR[fR[u]] = 1;
        if (cycQ[u]) hasPredQ[fQ[u]] = 1;
    }
    __syncthreads();

    // Step 6. Finalize true cycle membership
    for (int x = tid; x < n; x += stride) {
        cycR[x] = (cycR[x] && hasPredR[x]);
        cycQ[x] = (cycQ[x] && hasPredQ[x]);
    }
    __syncthreads();

    // Step 7. Flip all entries in X that lie on the (k,l) cycle
    if (tid == 0) {
        int i = k_, j = l_;
        do {
            int idx = IDX2C(i, j, n);
            X[idx] = 1 - X[idx];
            int j_next = (R[i] < n) ? R[i] : j;
            int i_next = (Q[j] < n) ? Q[j] : i;
            i = i_next;
            j = j_next;
        } while (!(i == k_ && j == l_));
    }
}

// ============================================================================
// Minimal host driver
// ============================================================================
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

    // Device memory
    int *d_R, *d_Q, *d_X, *d_fR, *d_fQ, *d_k, *d_l;
    unsigned char *d_hasPredR, *d_hasPredQ, *d_cycR, *d_cycQ;

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_X, n*n*sizeof(int));
    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
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

    // Launch kernel (single block, many threads)
    identify_and_flip_singleblock<<<1, 128>>>(d_R, d_Q, d_X, n,
                                              d_k, d_l,
                                              d_fR, d_fQ,
                                              d_hasPredR, d_hasPredQ,
                                              d_cycR, d_cycQ);
    cudaDeviceSynchronize();

    // Copy back result (for verification)
    cudaMemcpy(h_X, d_X, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col)
            printf("%d ", h_X[IDX2C(row,col,n)]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_R); cudaFree(d_Q); cudaFree(d_X);
    cudaFree(d_fR); cudaFree(d_fQ);
    cudaFree(d_k); cudaFree(d_l);
    cudaFree(d_hasPredR); cudaFree(d_hasPredQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);
}
