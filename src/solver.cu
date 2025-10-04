#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>
#include <string>
#include <cstdio>
#include <type_traits>
#include <math_constants.h>

template <typename T>
void printDeviceVar(const char* name, const T& symbol) {
    T host_val;
    cudaMemcpyFromSymbol(&host_val, symbol, sizeof(T), 0, cudaMemcpyDeviceToHost);
    std::cout << name << " = " << host_val << std::endl;
}

template<typename T>
void printDeviceMatrix(const char* name, const T* d_M, int n) {
    T* h_M = new T[n*n];
    cudaMemcpy(h_M, d_M, n*n*sizeof(T), cudaMemcpyDeviceToHost);
    printf("%s (%dx%d):\n", name, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = j*n + i;
            if constexpr (std::is_floating_point<T>::value)
                printf("%6.2f ", static_cast<double>(h_M[idx]));
            else
                printf("%4d ", static_cast<int>(h_M[idx]));
        }
        printf("\n");
    }
    printf("\n");
    delete[] h_M;
}

template<typename T>
void printDeviceVector(const char* name, const T* d_V, int n) {
    T* h_V = new T[n];
    cudaMemcpy(h_V, d_V, n*sizeof(T), cudaMemcpyDeviceToHost);
    printf("%s (len=%d): ", name, n);
    for (int i = 0; i < n; i++) {
        if constexpr (std::is_floating_point<T>::value)
            printf("%6.2f ", static_cast<double>(h_V[i]));
        else
            printf("%d ", static_cast<int>(h_V[i]));
    }
    printf("\n\n");
    delete[] h_V;
}

template<typename T>
void printDeviceScalar(const char* name, const T* d_val) {
    T h_val;
    cudaMemcpy(&h_val, d_val, sizeof(T), cudaMemcpyDeviceToHost);
    if constexpr (std::is_floating_point<T>::value)
        printf("%s = %f\n\n", name, static_cast<double>(h_val));
    else
        printf("%s = %d\n\n", name, static_cast<int>(h_val));
}

#define IDX2C(i,j,n) ((j)*(n)+(i))

__global__ void compute_B(const float* C, const float* U, const float* V, float* B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
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

__device__  float d_min;
__device__  int d_changed;
__device__  float d_epsilon;
__device__  int d_found;
__device__  int d_flag;
__device__  int d_b_kl_neg;

__global__ void solve_1bc_kernel(int n, const int* X, int* k, int* l, const float* B, int* R, int* Q) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    if (Q[j] != n && X[IDX2C(i, j, n)] == 1) {
        if (atomicCAS(&R[i], n, j) == n) d_changed = 1;
    }
    if (i != *k && R[i] != n && Q[j] == n) {
        float b_val = B[IDX2C(i, j, n)];
        if (b_val == 0.0f) {
            if (atomicMin(&Q[j], i) > i) d_changed = 1;
        }
    }
}

__global__ void set_array_value(int* arr, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = value;
}

__global__ void update_duals(int* R, int* Q, float* U, float* V, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (R[i] != n) U[i] += d_epsilon;
        if (Q[i] != n) V[i] -= d_epsilon;
    }
}

__device__ inline void atomicMinFloatNonNeg(float* addr, float val) {
    atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
}

__global__ void init_minval() {
    d_min = CUDART_INF_F;
}

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        if (f_old <= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void find_most_negative(const float* __restrict__ d_B, int n, int* d_out_i, int* d_out_j) {
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
        float tmp = d_B[IDX2C(row, col, n)];
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
        float oldMin = atomicMinFloat(&d_min, s_vals[0]);
        if (s_vals[0] < oldMin) {
            *d_out_i = s_rows[0];
            *d_out_j = s_cols[0];
        }
    }
}

__global__ void find_min_valid_atomic2d(const float* __restrict__ d_B, const int* __restrict__ d_R, const int* __restrict__ d_Q, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    extern __shared__ float sdata[];
    float val = CUDART_INF_F;
    if (row < n && col < n) {
        if (d_R[row] != n && d_Q[col] == n) {
            float tmp = d_B[IDX2C(row, col, n)];
            if (tmp >= 0.0f) val = tmp;
        }
    }
    sdata[tid] = val;
    __syncthreads();
    for (int stride = threads_per_block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) {
        float block_min = sdata[0];
        if (block_min < CUDART_INF_F) atomicMinFloatNonNeg(&d_min, block_min);
    }
}

__global__ void process_cycle(float* B, float* V, int* d_X, const int* d_R, const int* d_Q, int n, int* k, int* l) {
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

__global__ void finalize_epsilon(const float* d_B, int n, int* k, int* l) {
    if (isinf(d_min)) d_epsilon = -d_B[IDX2C(*k, *l, n)];
    else d_epsilon = d_min;
}

__global__ void update_Q(int* d_Q, const int* k, const int* l) {
    d_Q[*l] = *k;
}

__global__ void reset_d_changed() {
    d_changed = 0;
}

__global__ void check_Rk(const int* d_R, const int* k, const int* l, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int Rk = d_R[*k];
        d_flag = (Rk != n && Rk != *l);
    }
}

__global__ void check_bkl(const float* d_B, const int* k, const int* l, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int row = *k;
        int col = *l;
        float b_kl = d_B[IDX2C(row, col, n)];
        d_b_kl_neg = (b_kl < 0.0f);
    }
}

__global__ void reset_d_found() {
    if (threadIdx.x == 0 && blockIdx.x == 0) d_found = 0;
}

bool solve_from_kl(float* d_C, int* d_X, float* d_U, float* d_V, int n, float* d_B, int* d_R, int* d_Q, int* k, int* l) {
    update_Q<<<1,1>>>(d_Q, k, l);
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    
    int h_changed;
    do {
        reset_d_changed<<<1,1>>>();
        
        solve_1bc_kernel<<<blocks, threads>>>(n, d_X, k, l, d_B, d_R, d_Q);
        
        cudaMemcpyFromSymbol(&h_changed, d_changed, sizeof(int), 0, cudaMemcpyDeviceToHost);

    } while (h_changed == 1);
    check_Rk<<<1,1>>>(d_R, k, l, n);
    

    int h_flag;
    cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (h_flag == 1) {
        process_cycle<<<1,1>>>(d_B, d_V, d_X, d_R, d_Q, n, k, l);
        
        find_most_negative<<<blocks, threads>>>(d_B, n, k, l);
        set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
        compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
        return true;
    }
    init_minval<<<1, 1>>>();
    
    find_min_valid_atomic2d<<<blocks, threads>>>(d_B, d_R, d_Q, n);
    finalize_epsilon<<<1, 1>>>(d_B, n, k, l);
    update_duals<<<(n + 255) / 256, 256>>>(d_R, d_Q, d_U, d_V, n);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
    
    check_bkl<<<1,1>>>(d_B, k, l, n);
    

    int h_b_kl_neg;
    cudaMemcpyFromSymbol(&h_b_kl_neg, d_b_kl_neg, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (h_b_kl_neg == 1) return true;
    reset_d_found<<<1,1>>>();
    
    find_most_negative<<<blocks, threads>>>(d_B, n, k, l);

    int h_found;
    cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int), 0, cudaMemcpyDeviceToHost);

    
    if (h_found) {
        set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
        return true;
    } else {
        return false;
    }
}

void solve(float* d_C, int* d_X, float* d_U, float* d_V, int n) {
    float* d_B;
    cudaMalloc(&d_B, n * n * sizeof(float));
    int *k, *l;
    cudaMalloc(&k, sizeof(int));
    cudaMalloc(&l, sizeof(int));
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
    int* d_R; int* d_Q;
    cudaMalloc(&d_R, n * sizeof(int));
    cudaMalloc(&d_Q, n * sizeof(int));
    set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
    find_most_negative<<<blocks, threads>>>(d_B, n, k, l);
    int steps = 0;
    while (true) {
        // std::cout << "Step " << steps << " \n";
        bool should_continue = solve_from_kl(d_C, d_X, d_U, d_V, n, d_B, d_R, d_Q, k, l);
        steps++;
        if (!should_continue) {
            // std::cout << "Solver has converged after " << steps << " steps.\n";
            break;
        }
    }
    cudaFree(d_B);
    cudaFree(d_R);
    cudaFree(d_Q);
    cudaFree(k);
    cudaFree(l);
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