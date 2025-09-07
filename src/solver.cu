#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

#define IDX2C(i,j,n) ((j)*(n)+(i))
// #define IDX2C(i,j,n) ((i)*(n)+(j))

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

__global__ void compute_col_to_row(int n, const int* X, int* col_to_row) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    if (X[IDX2C(i, j, n)] == 1) {
        col_to_row[j] = i;
    }
}


__global__ void solve_1bc_kernel(
    int n,
    const int* col_to_row,
    int k,
    const float* B,
    int* R,
    int* Q,
    bool* changed
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // rows
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columns

    if (i >= n || j >= n) return;

    // Step (b): still only run one thread per column
    if (i == 0 && Q[j] != n) {
        int row = col_to_row[j];
        if (R[row] == n) {
            R[row] = j;
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


// void solve_1bc(
//     int n,
//     int* d_col_to_row,
//     int* k,            // now pointer
//     int* l,            // now pointer (not used here, but kept for symmetry)
//     bool* d_changed,
//     float* d_B,
//     int* d_R,
//     int* d_Q
// ){
//     dim3 threads(16, 16);
//     dim3 blocks((n + 15) / 16, (n + 15) / 16);

//     bool h_changed;

//     do {
//         h_changed = false;
//         cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

//         // NOTE: kernel takes k by value — pass *k
//         solve_1bc_kernel<<<blocks, threads>>>(
//             n, d_col_to_row, *k, d_B, d_R, d_Q, d_changed
//         );

//         cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
//         cudaDeviceSynchronize();
//     } while (h_changed);

// }

__device__ int d_outstanding;
__device__ int d_progress;
__device__ int d_stop_flag;
__device__ int d_moving;
__device__ int done;

__global__ void reset_globals() {
    d_outstanding = 0;
    d_progress    = 0;
    d_stop_flag   = 0;
    d_moving = 1;
    done = 0;
}

__global__ void reset_done(
    int n,
    int k,
    const float* B
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n){
        if (B[IDX2C(i, j, n)] != 0.0f ) {
            atomicAdd(&done, 1);
        }
    }    
}

__global__ void solve_1bc_kernel_full(
    int n,
    const int* col_to_row,
    int k,
    const float* B,
    int* R,
    int* Q
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    if (B[IDX2C(i, j, n)] != 0.0f) return;


    int contributing = 1;
    int step = 0;
    int local_move = 0;
    int doing = 0;
    int iteration;
    int accumulate;
    int expected = n * n - done;
    int old_p;
    int lastout;

    do {
        
        
        if (Q[j] != n && i == col_to_row[j] && R[i] == n) {
            R[i] = j;
            doing = 1;
        } else if (i != k && R[i] != n && Q[j] == n) {
            Q[j] = i;
            doing = 1;
        } else {
            doing = 0;
        }
        


        old_p = atomicAdd(&d_progress, contributing);
        if (contributing == 1 && doing == 0){
            atomicAdd(&d_outstanding, 1);
        }
        __threadfence();

        iteration = (old_p + contributing) / (expected);
        accumulate = (old_p + contributing) % (expected);

        if (iteration > step && d_moving > local_move){
            contributing = 1;
            step++;
            local_move++;
            if (accumulate == 0){
                lastout = atomicExch(&d_outstanding, 0);
                __threadfence();
                if (lastout == expected){
                    atomicAdd(&d_stop_flag, 1);
                }
                atomicAdd(&d_moving, 1);
            }
        } else {
            contributing = 0;
        }


        

    } while (d_stop_flag == 0);

}





__global__ void set_array_value(int* arr, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = value;
}

__global__ void update_duals(int* R, int* Q, float* U, float* V, float epsilon, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (R[i] != n) U[i] += epsilon;
        if (Q[i] != n) V[i] -= epsilon;
    }
}

__global__ void updateVals(float* B, float* V, int k, int l, int n) {
    int idx = IDX2C(k, l, n);   // compute column-major offset
    float b_kl = B[idx];
    float epsilon = -b_kl;

    V[l] -= epsilon;
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

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;

    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    }
    return __int_as_float(old);
}

__global__ void find_min_valid_atomic2d(const float* d_B,
                                        const int* d_R,
                                        const int* d_Q,
                                        int n,
                                        float* d_minval) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    if (d_R[row] != n && d_Q[col] == n) {
        float val = d_B[IDX2C(row, col, n)]; // row-major
        if (val >= 0.0f) {
            atomicMinFloat(d_minval, val);
        }
    }
}


bool solve_from_kl(
    int n,
    float* d_C, int* d_X, int* d_R,
    int* d_Q, int* d_col_to_row, int* k, int* l, int* d_changed, int* d_waiting,
    float* d_U, float* d_V, float* d_B, int* d_found, int* d_i, int* d_j, float* d_min
) {

    set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);

    // Q[*l] = *k
    cudaMemcpy(&d_Q[*l], k, sizeof(int), cudaMemcpyHostToDevice);

    // Step 1: Solve 1BC
    dim3 threads(16, 16);  // 1024 threads/block
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);

    compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    cudaDeviceSynchronize();

    // for (int s = 0; s < n; ++s) {
    // solve_1bc(n, d_col_to_row, k, l, d_changed, d_B, d_R, d_Q);
    // cudaMemset(row_done, 0, n * sizeof(int));

    reset_globals<<<1,1>>>();
    reset_done<<<blocks, threads>>>(n, *k, d_B);

    solve_1bc_kernel_full<<<blocks, threads>>>(n, d_col_to_row, *k, d_B, d_R, d_Q);
    // void* args[] = { &n, &d_col_to_row, &k, &d_B, &d_R, &d_Q };
    // cudaLaunchCooperativeKernel(
    //     (void*)solve_1bc_kernel_full,
    //     blocks, threads,
    //     args
    // );
    // void* kernelArgs[] = {
    //     (void*)&n,
    //     (void*)&d_col_to_row,
    //     (void*)&k,
    //     (void*)&d_B,
    //     (void*)&d_R,
    //     (void*)&d_Q,
    //     (void*)&d_changed
    // };
    
    // cudaLaunchCooperativeKernel(
    //     (void*)solve_1bc_kernel_full,
    //     blocks,
    //     threads,
    //     kernelArgs
    // );

    // }

    // Step 2: Check if R[*k] != n and R[*k] != *l
    int h_Rk;
    cudaMemcpy(&h_Rk, &d_R[*k], sizeof(int), cudaMemcpyDeviceToHost);

    if (h_Rk != n && h_Rk != *l) {
        int k_ = *k;
        int l_ = *l;

        int h_R, h_Q;

        while (true) {
            // X[k_, l_] = 1
            int one = 1;
            int idx_on = IDX2C(k_, l_, n);
            cudaMemcpy(&d_X[idx_on], &one, sizeof(int), cudaMemcpyHostToDevice);

            // l_ = R[k_]
            cudaMemcpy(&h_R, &d_R[k_], sizeof(int), cudaMemcpyDeviceToHost);
            l_ = h_R;

            // X[k_, l_] = 0
            int zero = 0;
            int idx_off = IDX2C(k_, l_, n);
            cudaMemcpy(&d_X[idx_off], &zero, sizeof(int), cudaMemcpyHostToDevice);

            // k_ = Q[l_]
            cudaMemcpy(&h_Q, &d_Q[l_], sizeof(int), cudaMemcpyDeviceToHost);
            k_ = h_Q;

            if (k_ == *k && l_ == *l)
                break;
        }

        updateVals<<<1,1>>>(d_B, d_V, *k, *l, n);


        // Recompute B = C - U.unsqueeze(1) - V
        // dim3 threads(16, 16);
        // dim3 blocks((n + 15) / 16, (n + 15) / 16);
        compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

        *k = n;
        *l = n;

        return true;
    }

    float h_min = INFINITY;
    
    cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);

    // configure launch
    // dim3 threads(16, 16);
    // dim3 blocks((n + 15) / 16, (n + 15) / 16);

    find_min_valid_atomic2d<<<blocks, threads>>>(d_B, d_R, d_Q, n, d_min);

    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    

    // finalize epsilon
    float epsilon;
    if (h_min == INFINITY) {
        float b_kl;
        cudaMemcpy(&b_kl, &d_B[IDX2C(*k, *l, n)], sizeof(float), cudaMemcpyDeviceToHost);
        epsilon = -b_kl;
    } else {
        epsilon = h_min;
    }



    update_duals<<<(n + 255) / 256, 256>>>(d_R, d_Q, d_U, d_V, epsilon, n);
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
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_i, sizeof(int));
    cudaMalloc(&d_j, sizeof(int));

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

    int* d_col_to_row;
    cudaMalloc(&d_col_to_row, n * sizeof(int));

    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

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

        // std::cout << "Step " << steps << ": argmin at B[" << k << "][" << l << "] \n";

        // Call solve_from_kl, which returns false if we should stop
        bool should_continue = solve_from_kl(n, d_C, d_X, d_R, d_Q, d_col_to_row, &k, &l, d_changed, d_waiting, d_U, d_V, d_B, d_found, d_i, d_j, d_min);
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
    cudaFree(d_waiting);

    cudaFree(d_found);
    cudaFree(d_i);
    cudaFree(d_j);
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