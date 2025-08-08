#include <cuda_runtime.h>
// #include <iostream>
// #include <cstdlib>
// #include <ctime>

// CUDA kernel for matrix multiplication
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Forward declaration
// void solve(float* d_C, int* d_X, float* d_U, float* d_V, int n);
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <float.h>

#define IDX2C(i,j,n) ((j)*(n)+(i))
// #define IDX2C(i,j,n) ((i)*(n)+(j))

// Device kernels
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    for (int i = 0; i < n; ++i) {
        if (X[IDX2C(i, j, n)] == 1) {
            col_to_row[j] = i;
            return;
        }
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
            // if (atomicCAS(&Q[j], n, i) == n) {
            //     *changed = true;
            // }
            // if (Q[j] == n) {
            //     Q[j] = i;
            //     *changed = true;
            // }
        }
    }
}


void solve_1bc(
    int n,
    int* d_col_to_row,
    int k,
    int l,
    float* d_B,
    int* d_R,
    int* d_Q
){

    dim3 threadsPerBlock(16, 16); // 16x16 = 256 threads per block
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16); // ceil(n / 16) in each dimension

    bool h_changed;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        solve_1bc_kernel<<<numBlocks, threadsPerBlock>>>(
            n, d_col_to_row, k, d_B, d_R, d_Q, d_changed
        );
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    } while (h_changed);

    cudaFree(d_changed);
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

bool solve_from_kl(
    int n,
    float* d_C, int* d_X, int k, int l,
    float* d_U, float* d_V, float* d_B
) {
    // Allocate and initialize R and Q
    int* d_R; int* d_Q;
    cudaMalloc(&d_R, n * sizeof(int));
    cudaMalloc(&d_Q, n * sizeof(int));

    set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);

    // Q[l] = k
    cudaMemcpy(&d_Q[l], &k, sizeof(int), cudaMemcpyHostToDevice);

    // Step 1: Solve 1BC
    int* d_col_to_row;
    cudaMalloc(&d_col_to_row, n * sizeof(int));
    compute_col_to_row<<<(n + 255) / 256, 256>>>(n, d_X, d_col_to_row);
    cudaDeviceSynchronize();

    for (int s = 0; s < n; ++s) {
        solve_1bc(n, d_col_to_row, k, l, d_B, d_R, d_Q);
    }
    cudaFree(d_col_to_row);

    // Step 2: Check if R[k] != n and R[k] != l
    int h_Rk;
    cudaMemcpy(&h_Rk, &d_R[k], sizeof(int), cudaMemcpyDeviceToHost);

    if (h_Rk != n && h_Rk != l) {
        int k_ = k;
        int l_ = l;

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

            if (k_ == k && l_ == l)
                break;
        }
        

        float b_kl;
        cudaMemcpy(&b_kl, &d_B[IDX2C(k, l, n)], sizeof(float), cudaMemcpyDeviceToHost);
        float epsilon = -b_kl;

        float v_l;
        cudaMemcpy(&v_l, &d_V[l], sizeof(float), cudaMemcpyDeviceToHost);
        v_l -= epsilon;
        cudaMemcpy(&d_V[l], &v_l, sizeof(float), cudaMemcpyHostToDevice);

        // Recompute B = C - U.unsqueeze(1) - V
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (n + 15) / 16);
        compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

        float* h_B = new float[n * n];
        cudaMemcpy(h_B, d_B, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

        int min_idx = 0;
        float min_val = h_B[0];
        for (int idx = 1; idx < n * n; ++idx) {
            if (h_B[idx] < min_val) {
                min_val = h_B[idx];
                min_idx = idx;
            }
        }
        delete[] h_B;

        // Update k, l in-place
        k = min_idx % n;
        l = min_idx / n;

        // cudaFree(d_R);
        // cudaFree(d_Q);
        // return false;
        cudaFree(d_R);
        cudaFree(d_Q);
        return solve_from_kl(n, d_C, d_X, k, l, d_U, d_V, d_B);  // recursion

    }

    // Branch B: find epsilon satisfying mask
    float epsilon = -1;
    bool found = false;
    // int i_found = -1, j_found = -1;

    for (int i = 0; i < n; ++i) {
        int r_val;
        cudaMemcpy(&r_val, &d_R[i], sizeof(int), cudaMemcpyDeviceToHost);
        if (r_val == n) continue;

        for (int j = 0; j < n; ++j) {
            int q_val;
            cudaMemcpy(&q_val, &d_Q[j], sizeof(int), cudaMemcpyDeviceToHost);
            if (q_val != n) continue;

            float b_ij;
            cudaMemcpy(&b_ij, &d_B[IDX2C(i, j, n)], sizeof(float), cudaMemcpyDeviceToHost);
            if (b_ij >= 0 && (!found || b_ij < epsilon)) {
                epsilon = b_ij;
                // i_found = i;
                // j_found = j;
                found = true;
            }
        }
    }

    if (!found) {
        float b_kl;
        cudaMemcpy(&b_kl, &d_B[IDX2C(k, l, n)], sizeof(float), cudaMemcpyDeviceToHost);
        epsilon = -b_kl;
    }

    // Update duals
    update_duals<<<(n + 255) / 256, 256>>>(d_R, d_Q, d_U, d_V, epsilon, n);

    // Recompute B = C - U.unsqueeze(1) - V
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
    cudaDeviceSynchronize();

    // Check B[k,l]
    float b_kl_check;
    cudaMemcpy(&b_kl_check, &d_B[IDX2C(k, l, n)], sizeof(float), cudaMemcpyDeviceToHost);

    if (b_kl_check < 0) {
        // cudaFree(d_R);
        // cudaFree(d_Q);
        // return true;
        cudaFree(d_R);
        cudaFree(d_Q);
        return solve_from_kl(n, d_C, d_X, k, l, d_U, d_V, d_B);  // recursion

    }

    // Check if any B[i,j] < 0
    bool any_negative = false;
    for (int i = 0; i < n && !any_negative; ++i) {
        for (int j = 0; j < n; ++j) {
            float b_ij;
            cudaMemcpy(&b_ij, &d_B[IDX2C(i, j, n)], sizeof(float), cudaMemcpyDeviceToHost);
            if (b_ij < 0) {
                any_negative = true;
                // break;
                cudaFree(d_R);
                cudaFree(d_Q);
                return solve_from_kl(n, d_C, d_X, i, j, d_U, d_V, d_B);  // recursion
            }
        }
    }

    cudaFree(d_R);
    cudaFree(d_Q);
    return any_negative;
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

    // Compute initial B
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

    int steps = 0;
    while (true) {
        // Find argmin(B)
        int totalThreads = n * n;
        int blockSize = 256;
        int gridSize = (totalThreads + blockSize - 1) / blockSize;
        find_argmin<<<gridSize, blockSize>>>(d_B, d_idx, d_val, n);

        int h_idx;
        float h_val;
        cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost);

        int k = h_idx / n;
        int l = h_idx % n;

        // std::cout << "Step " << steps << ": argmin at B[" << k << "][" << l << "] = " << h_val << "\n";

        // Call solve_from_kl, which returns false if we should stop
        bool should_continue = solve_from_kl(n, d_C, d_X, k, l, d_U, d_V, d_B);
        steps++;

        if (!should_continue) {
            std::cout << "Solver has converged after " << steps << " steps.\n";
            break;
        }
    }

    // Cleanup
    cudaFree(d_B);
    cudaFree(d_idx);
    cudaFree(d_val);
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