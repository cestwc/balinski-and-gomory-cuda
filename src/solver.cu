#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

#define IDX2C(i,j,n) ((j)*(n)+(i))
#include <math_constants.h>   // for CUDART_INF_F

__global__ void compute_B(const float* C, const float* U, const float* V, float* B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
    }
}

__global__ void find_argmin(const float* B, int* out_idx, float* out_val, int n) {
    __shared__ float min_val[256];
    __shared__ int min_idx[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int total = n * n;

    float val = (index < total) ? B[index] : FLT_MAX;
    min_val[tid] = val;
    min_idx[tid] = index;
    __syncthreads();

    // Parallel reduction to find min
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && min_val[tid + s] < min_val[tid]) {
            min_val[tid] = min_val[tid + s];
            min_idx[tid] = min_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_val[blockIdx.x] = min_val[0];
        out_idx[blockIdx.x] = min_idx[0];
    }
}

__global__ void solve_1bc_kernel(
    int n,
    // const int* col_to_row,
    const int* X,
    int k,
    int l,
    const float* B,
    int* R,
    int* Q,
    bool* changed
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // rows
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columns

    if (i >= n || j >= n) return;

    if (Q[j] != n && X[IDX2C(i, j, n)] == 1){

        if (atomicCAS(&R[i], n, j) == n) {
            *changed = true;
        }

    }

    // Step (c): one thread per (i, j)
    if (i != k && R[i] != n && Q[j] == n) {
        float b_val = B[IDX2C(i, j, n)];
        if (b_val == 0.0f) {
            if (atomicMin(&Q[j], i) > i) {
                *changed = true;
            }
        }
    }
}




__global__ void set_array_value(int* arr, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = value;
}

__global__ void update_duals(int* R, int* Q, float* U, float* V, float* epsilon, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (R[i] != n) U[i] += *epsilon;
        if (Q[i] != n) V[i] -= *epsilon;
    }
}


__global__ void find_negative(const float* d_B, int n, int* d_found, int* d_i, int* d_j) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    float val = d_B[IDX2C(i, j, n)]; // row-major
    if (val < 0.0f) {
        if (atomicExch(d_found, 1) == 0) {
            *d_i = i;
            *d_j = j;
        }

    }
}

// Use integer atomicMin for non-negative floats
__device__ inline void atomicMinFloatNonNeg(float* addr, float val) {
    // Reinterpret as unsigned to preserve ordering for non-negative floats
    atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
}

__global__ void init_minval(float* d_minval) {
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    *d_minval = CUDART_INF_F;
}


__global__ void find_min_valid_atomic2d(const float* __restrict__ d_B,
                                      const int*   __restrict__ d_R,
                                      const int*   __restrict__ d_Q,
                                      int n,
                                      float* d_minval)
{
    // 2D thread coords
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Flattened thread id within block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    // Shared memory buffer (size = blockDim.x * blockDim.y floats)
    extern __shared__ float sdata[];

    // Compute this thread's candidate
    float val = CUDART_INF_F;  // default = +∞ (neutral for min)
    if (row < n && col < n) {
        if (d_R[row] != n && d_Q[col] == n) {
            float tmp = d_B[IDX2C(row, col, n)];  // row-major
            if (tmp >= 0.0f) val = tmp;
        }
    }

    // Write to shared memory
    sdata[tid] = val;
    __syncthreads();

    // Shared-memory parallel reduction to find block minimum
    // (simple, readable version)
    for (int stride = threads_per_block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 of the block updates the global result
    if (tid == 0) {
        // Only do the atomic if the block found something < +∞
        float block_min = sdata[0];
        if (block_min < CUDART_INF_F) {
            atomicMinFloatNonNeg(d_minval, block_min);
        }
    }
}

__global__ void process_cycle(float* B, float* V, int* d_X,
                              const int* d_R,
                              const int* d_Q,
                              int n,
                              int k, int l)
{
    int k_ = k;
    int l_ = l;

    while (true) {
        // X[k_, l_] = 1
        d_X[IDX2C(k_, l_, n)] = 1;

        // l_ = R[k_]
        l_ = d_R[k_];

        // X[k_, l_] = 0
        d_X[IDX2C(k_, l_, n)] = 0;

        // k_ = Q[l_]
        k_ = d_Q[l_];

        // stop if cycle closed
        if (k_ == k && l_ == l)
            break;
    }

    V[l] += B[IDX2C(k, l, n)];
}

__global__ void finalize_epsilon(const float* d_min,
                                 const float* d_B,
                                 int n,
                                 int k, int l,
                                 float* d_epsilon)
{
    if (isinf(*d_min)) {
        *d_epsilon = -d_B[IDX2C(k, l, n)];
    } else {
        *d_epsilon = *d_min;
    }
}



bool solve_from_kl(
    int n,
    float* d_C, int* d_X, int* d_R,
    int* d_Q, int* d_col_to_row, int* k, int* l, bool* d_changed, int* d_waiting, float* d_epsilon,
    float* d_U, float* d_V, float* d_B, int* d_found, int* d_i, int* d_j, float* d_min
) {

    

    // Q[*l] = *k
    cudaMemcpy(&d_Q[*l], k, sizeof(int), cudaMemcpyHostToDevice);

    // Step 1: Solve 1BC
    dim3 threads(16, 16);  // 1024 threads/block
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);

    // compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    cudaDeviceSynchronize();

    // dim3 threads(16, 16);
    // dim3 blocks((n + 15) / 16, (n + 15) / 16);

    bool h_changed;

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // NOTE: kernel takes k by value — pass *k
        solve_1bc_kernel<<<blocks, threads>>>(
            n, d_X, *k, *l, d_B, d_R, d_Q, d_changed
        );

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    } while (h_changed);

    // Step 2: Check if R[*k] != n and R[*k] != *l
    int h_Rk;
    cudaMemcpy(&h_Rk, &d_R[*k], sizeof(int), cudaMemcpyDeviceToHost);

    if (h_Rk != n && h_Rk != *l) {
        // say k and l are already defined on host
        process_cycle<<<1,1>>>(d_B, d_V, d_X, d_R, d_Q, n, *k, *l);

        // sync *once* if you need results on host right away
        cudaDeviceSynchronize();

        *k = n;
        *l = n;
        set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);

        compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

        return true;
    }

    init_minval<<<1, 1>>>(d_min);
    
    find_min_valid_atomic2d<<<blocks, threads>>>(d_B, d_R, d_Q, n, d_min);

    finalize_epsilon<<<1, 1>>>(d_min, d_B, n, *k, *l, d_epsilon);




    update_duals<<<(n + 255) / 256, 256>>>(d_R, d_Q, d_U, d_V, d_epsilon, n);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
    cudaDeviceSynchronize();

    // Check B[*k,*l]
    float b_kl_check;
    cudaMemcpy(&b_kl_check, &d_B[IDX2C(*k, *l, n)], sizeof(float), cudaMemcpyDeviceToHost);

    if (b_kl_check < 0) {
        return true;
    }

    int h_found = 0;

    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

    find_negative<<<blocks, threads>>>(d_B, n, d_found, d_i, d_j);
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
        int row, col;
        cudaMemcpy(&row, d_i, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&col, d_j, sizeof(int), cudaMemcpyDeviceToHost);
        *k = row;
        *l = col;
        set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
        return true;
    } else {
        return false;
    }

}

__global__ void initVars(int* k, int* l, int n) {
    *k = n;
    *l = n;
}


void solve(float* d_C, int* d_X, float* d_U, float* d_V, int n) {
    size_t sizeMat = n * n * sizeof(float);

    // Allocate B
    float* d_B;
    cudaMalloc(&d_B, sizeMat);

    // Allocate buffers for argmin
    int* d_idx; float* d_val;
    cudaMalloc(&d_idx, sizeof(int));
    cudaMalloc(&d_val, sizeof(float));

    int *d_found, *d_i, *d_j;
    // int *k, *l;
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_i, sizeof(int));
    cudaMalloc(&d_j, sizeof(int));
    // cudaMalloc(&k, sizeof(int));
    // cudaMalloc(&l, sizeof(int));

    float* d_min;
    cudaMalloc(&d_min, sizeof(float));



    // Compute initial B
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

    const size_t num_items = static_cast<size_t>(n) * static_cast<size_t>(n);
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceReduce::ArgMin(nullptr, temp_bytes, d_B, d_val, d_idx, num_items);
    cudaMalloc(&d_temp, temp_bytes);

    int* d_R; int* d_Q;
    cudaMalloc(&d_R, n * sizeof(int));
    cudaMalloc(&d_Q, n * sizeof(int));

    set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);

    int* d_col_to_row;
    cudaMalloc(&d_col_to_row, n * sizeof(int));

    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    float* d_epsilon;
    cudaMalloc(&d_epsilon, sizeof(float));

    int* d_waiting;
    cudaMalloc(&d_waiting, sizeof(int));

    int steps = 0;
    int k = n;
    int l = n;
    while (true) {
        if (k == n){
            cub::DeviceReduce::ArgMin(d_temp, temp_bytes, d_B, d_val, d_idx, num_items);

            // Find argmin(B)
            // int totalThreads = n * n;
            // int blockSize = 256;
            // int gridSize = (totalThreads + blockSize - 1) / blockSize;
            // find_argmin<<<gridSize, blockSize>>>(d_B, d_idx, d_val, n);

            int h_idx;
            cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);

            k = h_idx % n;
            l = h_idx / n;
        }

        std::cout << "Step " << steps << ": argmin at B[" << k << "][" << l << "] \n";

        // Call solve_from_kl, which returns false if we should stop
        bool should_continue = solve_from_kl(n, d_C, d_X, d_R, d_Q, d_col_to_row, &k, &l, d_changed, d_waiting, d_epsilon, d_U, d_V, d_B, d_found, d_i, d_j, d_min);
        steps++;

        if (!should_continue) {
            std::cout << "Solver has converged after " << steps << " steps.\n";
            break;
        }
    }

    // Cleanup
    cudaFree(d_B);
    cudaFree(d_temp);
    cudaFree(d_idx);
    cudaFree(d_val);

    cudaFree(d_R);
    cudaFree(d_Q);
    cudaFree(d_col_to_row);
    cudaFree(d_changed);
    cudaFree(d_epsilon);
    cudaFree(d_waiting);

    cudaFree(d_found);
    cudaFree(d_i);
    cudaFree(d_j);
    // cudaFree(k);
    // cudaFree(l);
    cudaFree(d_min);
}


__global__ void check_feasible_condition(const float* C, const int* X, const float* U, const float* V, int* out, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        if (X[IDX2C(i, j, n)] == 1) {
            float diff = C[IDX2C(i, j, n)] - U[i] - V[j];
            if (fabsf(diff) > 1e-4f) {
                atomicExch(out, 1); // feasible violated
            }
        }
    }
}

__global__ void check_slack_condition(const float* C, const float* U, const float* V, int* out, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        float diff = C[IDX2C(i, j, n)] - U[i] - V[j];
        if (diff < -1e-4f) {
            atomicExch(out, 1); // slack violated
        }
    }
}

void verify_solution(float* d_C, int* d_X, float* d_U, float* d_V, int n) {
    int h_feasible = 0, h_slack = 0;
    int *d_feasible, *d_slack;
    cudaMalloc(&d_feasible, sizeof(int));
    cudaMalloc(&d_slack, sizeof(int));
    cudaMemset(d_feasible, 0, sizeof(int));
    cudaMemset(d_slack, 0, sizeof(int));

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    check_feasible_condition<<<blocks, threads>>>(d_C, d_X, d_U, d_V, d_feasible, n);
    check_slack_condition<<<blocks, threads>>>(d_C, d_U, d_V, d_slack, n);

    cudaMemcpy(&h_feasible, d_feasible, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_slack, d_slack, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_feasible);
    cudaFree(d_slack);

    bool feasible_ok = (h_feasible == 0);
    bool slack_ok = (h_slack == 0);

    std::cout << "\n=== Post-Solution Verification ===\n";
    std::cout << "feasible condition: " << (feasible_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "slack condition: " << (slack_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "Overall check: " << ((feasible_ok && slack_ok) ? "✓ Passed" : "✗ Failed") << "\n\n";
}