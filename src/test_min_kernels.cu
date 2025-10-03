#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>      // for INFINITY
#include <cuda_runtime.h>
#include <math_constants.h>   // for CUDART_INF_F

// =====================================================
// Device globals
// =====================================================
__device__ int   d_found;
__device__ float d_minVal;

// =====================================================
// AtomicMin for floats (CAS based, generic)
// =====================================================
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        if (f_old <= value) break; // already smaller
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

// =====================================================
// Kernel 1: Find most negative element with coords
// =====================================================
__global__ void find_most_negative(const float* __restrict__ d_B,
                                   int n,
                                   int* d_out_i, int* d_out_j) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    extern __shared__ float s_vals[];
    __shared__ int s_rows[256];
    __shared__ int s_cols[256];

    float val = INFINITY;
    int myRow = -1, myCol = -1;

    if (row < n && col < n) {
        float tmp = d_B[row * n + col];
        if (tmp < 0.0f) {
            val = tmp;
            myRow = row;
            myCol = col;
        }
    }

    s_vals[tid] = val;
    s_rows[tid] = myRow;
    s_cols[tid] = myCol;
    __syncthreads();

    for (int stride = threads_per_block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] < s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_rows[tid] = s_rows[tid + stride];
                s_cols[tid] = s_cols[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && s_vals[0] < INFINITY) {
        d_found = 1;
        float oldMin = atomicMinFloat(&d_minVal, s_vals[0]);
        if (s_vals[0] < oldMin) {
            *d_out_i = s_rows[0];
            *d_out_j = s_cols[0];
        }
    }
}

// =====================================================
// Kernel 2: Find min valid (example based on your style)
// =====================================================
__global__ void find_min_valid_atomic2d(const float* __restrict__ d_B,
                                        const int*   __restrict__ d_R,
                                        const int*   __restrict__ d_Q,
                                        int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    extern __shared__ float sdata[];

    float val = INFINITY;
    if (row < n && col < n) {
        if (d_R[row] != n && d_Q[col] == n) {
            float tmp = d_B[row * n + col];
            if (tmp >= 0.0f) val = tmp;
        }
    }

    sdata[tid] = val;
    __syncthreads();

    for (int stride = threads_per_block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float block_min = sdata[0];
        if (block_min < INFINITY) {
            atomicMinFloat(&d_minVal, block_min);
            d_found = 1;
        }
    }
}

// =====================================================
// CPU reference helpers
// =====================================================
void cpu_find_most_negative(const float* B, int n,
                            bool& found, float& val, int& ri, int& cj) {
    found = false;
    val = INFINITY;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float x = B[i*n + j];
            if (x < 0.0f) {
                if (!found || x < val) {
                    found = true;
                    val = x;
                    ri = i;
                    cj = j;
                }
            }
        }
    }
}

void cpu_find_min_valid(const float* B, const int* R, const int* Q, int n,
                        bool& found, float& val) {
    found = false;
    val = INFINITY;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (R[i] != n && Q[j] == n) {
                float x = B[i*n + j];
                if (x >= 0.0f) {
                    if (!found || x < val) {
                        found = true;
                        val = x;
                    }
                }
            }
        }
    }
}

// =====================================================
// Helper: Fill matrix with ints in [lo,hi]
// =====================================================
void fillMatrix(float* M, int n, int lo, int hi) {
    for (int i = 0; i < n*n; i++) {
        int r = lo + (rand() % (hi - lo + 1));
        M[i] = static_cast<float>(r);
    }
}

// =====================================================
// Main
// =====================================================
int main() {
    srand((unsigned)time(NULL));
    int n = 8; // small for readability

    // Scenarios to test
    struct Scenario { int lo, hi; const char* name; };
    Scenario scenarios[] = {
        {-10, 10, "Full span [-10,10]"},
        {  2,  9, "Positive only [2,9] (no negatives)"},
        { -2,  9, "Mostly positive [-2,9] (few negatives)"},
        { -9, -1, "All negative [-9,-1]"}
    };

    for (auto sc : scenarios) {
        printf("\n=== Scenario: %s ===\n", sc.name);

        // Host matrix
        float *h_B = new float[n*n];
        fillMatrix(h_B, n, sc.lo, sc.hi);

        // Print matrix
        printf("Matrix B (%dx%d):\n", n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%4.0f ", h_B[i*n + j]);
            }
            printf("\n");
        }
        printf("\n");

        // Device memory
        float *d_B;
        cudaMalloc(&d_B, n*n*sizeof(float));
        cudaMemcpy(d_B, h_B, n*n*sizeof(float), cudaMemcpyHostToDevice);

        int *h_R = new int[n];
        int *h_Q = new int[n];
        for (int i = 0; i < n; i++) {
            h_R[i] = 0;  // allow all rows
            h_Q[i] = n;  // allow all cols
        }
        int *d_R, *d_Q;
        cudaMalloc(&d_R, n*sizeof(int));
        cudaMalloc(&d_Q, n*sizeof(int));
        cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);

        int *d_i, *d_j;
        cudaMalloc(&d_i, sizeof(int));
        cudaMalloc(&d_j, sizeof(int));

        dim3 block(16,16);
        dim3 grid((n+15)/16, (n+15)/16);
        size_t shmem = block.x * block.y * sizeof(float);

        // -------------------------------
        // Kernel 1 test
        // -------------------------------
        {
            int zero = 0; float inf = INFINITY;
            cudaMemcpyToSymbol(d_found, &zero, sizeof(int));
            cudaMemcpyToSymbol(d_minVal, &inf, sizeof(float));

            find_most_negative<<<grid, block, shmem>>>(d_B, n, d_i, d_j);
            cudaDeviceSynchronize();

            int h_found, h_i, h_j;
            float h_val;
            cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int));
            cudaMemcpyFromSymbol(&h_val, d_minVal, sizeof(float));
            cudaMemcpy(&h_i, d_i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_j, d_j, sizeof(int), cudaMemcpyDeviceToHost);

            bool ref_found; float ref_val; int ref_i, ref_j;
            cpu_find_most_negative(h_B, n, ref_found, ref_val, ref_i, ref_j);

            printf("[find_most_negative]\n");
            if (h_found)
                printf("  GPU: %f at (%d,%d)\n", h_val, h_i, h_j);
            else
                printf("  GPU: no negatives found\n");

            if (ref_found)
                printf("  CPU: %f at (%d,%d)\n", ref_val, ref_i, ref_j);
            else
                printf("  CPU: no negatives found\n");
        }

        // -------------------------------
        // Kernel 2 test
        // -------------------------------
        {
            int zero = 0; float inf = INFINITY;
            cudaMemcpyToSymbol(d_found, &zero, sizeof(int));
            cudaMemcpyToSymbol(d_minVal, &inf, sizeof(float));

            find_min_valid_atomic2d<<<grid, block, shmem>>>(d_B, d_R, d_Q, n);
            cudaDeviceSynchronize();

            int h_found; float h_val;
            cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int));
            cudaMemcpyFromSymbol(&h_val, d_minVal, sizeof(float));

            bool ref_found; float ref_val;
            cpu_find_min_valid(h_B, h_R, h_Q, n, ref_found, ref_val);

            printf("\n[find_min_valid_atomic2d]\n");
            if (h_found)
                printf("  GPU: %f\n", h_val);
            else
                printf("  GPU: no valid values found\n");

            if (ref_found)
                printf("  CPU: %f\n", ref_val);
            else
                printf("  CPU: no valid values found\n");
        }

        // Cleanup for this scenario
        cudaFree(d_B); cudaFree(d_R); cudaFree(d_Q);
        cudaFree(d_i); cudaFree(d_j);
        delete[] h_B; delete[] h_R; delete[] h_Q;
    }

    return 0;
}
