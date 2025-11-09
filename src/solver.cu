#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>
#include <string>
#include <cstdio>
#include <type_traits>
#include <math_constants.h>

#include <cub/cub.cuh>
#include <cstdio>

// ---------------------------------------------
// 1. Predicate functor: marks entries that are 0
// ---------------------------------------------
struct IsZero {
    const float* B;
    __host__ __device__ bool operator()(const int idx) const {
        return B[idx] == 0.0f;
    }
};

// void setup_cub_buffers(
//     int n, 
//     float* d_B,                  // device matrix (n*n elements)
//     int*& d_indices,             // [output] 0..n*n-1 indices
//     int*& d_selected,            // [output] selected indices
//     int*& d_num_selected,        // [output] count of selected
//     void*& d_temp_storage,       // [output] CUB temp storage
//     size_t& temp_bytes)          // [output] CUB temp size
// {
//     int N = n * n;

//     // Allocate and initialize input index array (0..N-1)
//     cudaMalloc(&d_indices, N * sizeof(int));
//     int threads = 256;
//     int blocks  = (N + threads - 1) / threads;
//     // simple sequence initialization
//     auto init_sequence = [] __global__ (int* data, int N) {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if (tid < N) data[tid] = tid;
//     };
//     init_sequence<<<blocks, threads>>>(d_indices, N);

//     // Allocate outputs
//     cudaMalloc(&d_selected, N * sizeof(int));  // maximum possible output
//     cudaMalloc(&d_num_selected, sizeof(int));

//     // -------------------------------------
//     // Query CUB temporary storage (once)
//     // -------------------------------------
//     d_temp_storage = nullptr;
//     temp_bytes = 0;

//     // cub::DeviceSelect::If(
//     //     d_temp_storage, temp_bytes,
//     //     d_indices, d_selected, d_num_selected,
//     //     IsZero{d_B}, N
//     // );

//     cub::DeviceSelect::If(
//         d_temp_storage,
//         temp_bytes,
//         d_indices,
//         d_selected,
//         d_num_selected,
//         IsZero{d_B},
//         N
//     );

//     // Allocate the required temp storage once
//     cudaMalloc(&d_temp_storage, temp_bytes);

//     printf("[setup] Allocated %zu bytes for CUB temp storage\n", temp_bytes);
// }

// ---------------------------------------------
// 3. Per-iteration function
// ---------------------------------------------
// void select_zeros_with_cub(
//     int n,
//     float* d_B,                    // updated matrix data
//     int* d_indices,
//     int* d_selected,
//     int* d_num_selected,
//     void* d_temp_storage,
//     size_t temp_bytes)
// {
//     const int N = n * n;

//     // Reset output counter to zero
//     cudaMemset(d_num_selected, 0, sizeof(int));

//     // Run the CUB selection (no reallocation)
//     cub::DeviceSelect::If(
//         d_temp_storage, temp_bytes,
//         d_indices, d_selected, d_num_selected,
//         IsZero{d_B}, N
//     );

//     // (Optional) Check results
//     int h_num_selected;
//     cudaMemcpy(&h_num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
//     printf("[iteration] Found %d zero elements in B\n", h_num_selected);
// }



__device__ int blockPrefixSum(int* s_data, int tid, int n)
{
    for (int offset = 1; offset < n; offset <<= 1) {
        int val = 0;
        if (tid >= offset) val = s_data[tid - offset];
        __syncthreads();
        s_data[tid] += val;
        __syncthreads();
    }
    return s_data[tid];
}

// ---------------------------------------------------------------
// CUDA kernel for selecting indices where B[i] == 0
// ---------------------------------------------------------------
__global__ void selectZerosEfficient(const float* __restrict__ B,
                                     int* __restrict__ out_indices,
                                     int* __restrict__ global_count,
                                     int total)
{
    extern __shared__ int s_flags[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int flag = (gid < total && B[gid] == 0.0f);
    s_flags[tid] = flag;
    __syncthreads();

    int prefix = blockPrefixSum(s_flags, tid, blockDim.x);

    int block_total = s_flags[blockDim.x - 1];
    if (threadIdx.x == blockDim.x - 1 && gid < total)
        block_total = prefix;

    __shared__ int block_offset;
    if (tid == 0)
        block_offset = atomicAdd(global_count, block_total);
    __syncthreads();

    if (flag) {
        int pos = block_offset + prefix - 1;
        out_indices[pos] = gid;
    }
}

// ---------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------
void selectZerosOptimized(const float* d_B, int* d_out, int* d_count, int total)
{
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    cudaMemset(d_count, 0, sizeof(int));
    size_t sharedBytes = blockSize * sizeof(int);
    selectZerosEfficient<<<numBlocks, blockSize, sharedBytes>>>(d_B, d_out, d_count, total);
    cudaDeviceSynchronize();
}



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

// __global__ void find_most_negative(const float* __restrict__ d_B,
//                                    int n,
//                                    int* d_i, int* d_j) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.y * blockDim.x + threadIdx.x;
//     int threads_per_block = blockDim.x * blockDim.y;

//     extern __shared__ float s_vals[];
//     __shared__ int s_rows[1024];  // enough for <=1024 threads
//     __shared__ int s_cols[1024];

//     // Candidate
//     float val = CUDART_INF_F;
//     int myRow = -1, myCol = -1;

//     if (row < n && col < n) {
//         // float tmp = d_B[row * n + col]; // row-major
//         float tmp = d_B[IDX2C(row, col, n)]; // correct column-major indexing
//         if (tmp < 0.0f) {
//             val = tmp;
//             myRow = row;
//             myCol = col;
//         }
//     }

//     s_vals[tid] = val;
//     s_rows[tid] = myRow;
//     s_cols[tid] = myCol;
//     __syncthreads();

//     // Block reduction
//     for (int stride = threads_per_block >> 1; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             if (s_vals[tid + stride] < s_vals[tid]) {
//                 s_vals[tid] = s_vals[tid + stride];
//                 s_rows[tid] = s_rows[tid + stride];
//                 s_cols[tid] = s_cols[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     // Each block’s winner updates global minimum
//     if (tid == 0 && s_vals[0] < CUDART_INF_F) {
//         d_found = 1;

//         int newBits = __float_as_int(s_vals[0]);
//         int oldBits = atomicMin((int*)&d_min, newBits);

//         if (s_vals[0] < __int_as_float(oldBits)) {
//             *d_i = s_rows[0];
//             *d_j = s_cols[0];
//         }
//     }
// }

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

__global__ void update_V(float* B, float* V, int* d_X, const int* d_R, const int* d_Q, int n, int* k, int* l) {
    // int k_ = *k;
    // int l_ = *l;
    // while (true) {
    //     d_X[IDX2C(k_, l_, n)] = 1;
    //     l_ = d_R[k_];
    //     d_X[IDX2C(k_, l_, n)] = 0;
    //     k_ = d_Q[l_];
    //     if (k_ == *k && l_ == *l) break;
    // }
    V[*l] += B[IDX2C(*k, *l, n)];
}

__global__ void init_mapping(const int* R, const int* Q,
                             int* next_i, int* next_j,
                             int* parity, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;

    int idx = i * n + j;

    // successor pair
    next_i[idx] = Q[j];
    next_j[idx] = R[i];

    // parity bit: row→col = 1 (odd)
    parity[idx] = 1;
}

__global__ void pointer_jump_cycle(int* next_i, int* next_j,
                                   int* parity, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;

    int idx = i * n + j;
    int rounds = ceilf(log2f((float)n));

    for (int r = 0; r < rounds; ++r) {
        int ni = next_i[idx];
        int nj = next_j[idx];
        int next_idx = ni * n + nj;

        int ni2 = next_i[next_idx];
        int nj2 = next_j[next_idx];

        // jump twice as far
        next_i[idx] = ni2;
        next_j[idx] = nj2;

        // combine parity with successor’s parity
        parity[idx] ^= parity[next_idx];

        __syncthreads(); // optional safety if small grids reuse data
    }
}


__global__ void update_dX(const int* next_i, const int* next_j,
                          const int* parity, int* d_X, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;

    int idx = i * n + j;

    // Check if this (i,j) is part of a cycle
    bool in_cycle = (next_i[idx] == i && next_j[idx] == j);
    if (!in_cycle) return;

    // Update based on parity
    d_X[idx] = parity[idx];  // 1 for row→col, 0 for col→row
}


__global__ void finalize_epsilon(const float* d_B, int n, int* k, int* l) {
    if (isinf(d_min)) d_epsilon = -d_B[IDX2C(*k, *l, n)];
    else d_epsilon = d_min;
}

__global__ void update_Q(int* d_Q, const int* k, const int* l) {
    d_Q[*l] = *k;
}

__global__ void update_RQ(int* d_R, int* d_Q, const int* k, const int* l, int* d_col_to_row) {
    d_Q[*l] = *k;
    // printf("my *k, %d\n", *k);
    // printf("my *l, %d\n", *l);
    // printf("my d_col_to_row[*l], %d\n", d_col_to_row[*l]);
    d_R[d_col_to_row[*l]] = *l;
    // printf("my d_col_to_row[*l], %d\n", d_col_to_row[*l]);

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

// __global__ void solve_1bc_persistent(
//     int n, const int* X, int* k, int* l,
//     const float* B, int* R, int* Q)
// {
//     // We'll use shared memory for per-block "changed" flag
//     __shared__ int block_changed;
//     // Persistent device-side loop
//     while (true) {
//         // Step 1: reset block-local flag
//         if (threadIdx.x == 0 && threadIdx.y == 0)
//             block_changed = 0;
//         __syncthreads();
//         // Step 2: perform the same logic as solve_1bc_kernel
//         int i = blockIdx.y * blockDim.y + threadIdx.y;
//         int j = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i < n && j < n) {
//             if (Q[j] != n && X[IDX2C(i, j, n)] == 1) {
//                 if (atomicCAS(&R[i], n, j) == n)
//                     block_changed = 1;
//             }
//             if (i != *k && R[i] != n && Q[j] == n) {
//                 float b_val = B[IDX2C(i, j, n)];
//                 if (b_val == 0.0f) {
//                     if (atomicMin(&Q[j], i) > i)
//                         block_changed = 1;
//                 }
//             }
//         }
//         __syncthreads();
//         // Step 3: reduce to a single global flag
//         if (threadIdx.x == 0 && threadIdx.y == 0 && block_changed)
//             atomicExch(&d_changed, 1);
//         // Step 4: global sync — use cooperative groups
//         cooperative_groups::grid_group grid = cooperative_groups::this_grid();
//         grid.sync();
//         // Step 5: one thread decides whether to continue
//         if (grid.thread_rank() == 0) {
//             if (d_changed == 0) {
//                 d_changed = -1;  // signal exit
//             } else {
//                 d_changed = 0;   // reset for next iteration
//             }
//         }
//         grid.sync();
//         // Step 6: exit condition
//         if (d_changed == -1)
//             break;
//         // then loop again
//     }
// }

// ==== Device flags ====
__device__ int d_outstanding;  // threads idle in current iteration
__device__ int d_progress;     // global progress counter
__device__ int d_stop;         // global stop condition
__device__ int d_done;         // total number of nonzero elements in B
__device__ int d_moving;

// ==== Resetters ====
__global__ void reset_device_flags() {
    d_outstanding = 0;
    d_progress    = 0;
    d_stop        = 0;
    d_done        = 0;
    d_moving      = 1;    
}

__global__ void count_done_entries(int n, const float* B) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && B[IDX2C(i, j, n)] != 0.0f)
        atomicAdd(&d_done, 1);
}

__global__ void solve_1bc_kernel_full(
    int n,
    const int* X,
    const int *k,
    const float* B,
    int* R,
    int* Q
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    if (B[IDX2C(i, j, n)] != 0.0f) return;

    // ---- Local state ----
    int contributing = 1;
    int step = 0;
    int local_move = 0;
    int doing = 0;
    int iteration;
    int accumulate;
    int expected = n * n - d_done;  // depends on precomputed d_done
    int old_p;
    int last_out;

    // ---- Main loop ----
    do {
        // compute local work
        if (Q[j] != n && X[IDX2C(i, j, n)] == 1 && R[i] == n) {
            R[i] = j;
            doing = 1;
        } else if (i != *k && R[i] != n && Q[j] == n) {
            Q[j] = i;
            doing = 1;
        } else {
            doing = 0;
        }
        __threadfence_system();

        old_p = atomicAdd(&d_progress, contributing);
        if (contributing == 1 && doing == 0){
            atomicAdd(&d_outstanding, 1);
        }
        __threadfence();

        iteration  = (old_p + contributing) / expected;
        accumulate = (old_p + contributing) % expected;

        if (iteration > step && d_moving > local_move) {
            contributing = 1;
            step++;
            local_move++;
            if (accumulate == 0) {
                last_out = atomicExch(&d_outstanding, 0);
                __threadfence();
                if (last_out == expected)
                    {atomicAdd(&d_stop, 1);}
                atomicAdd(&d_moving, 1);
                __threadfence_system();
            }
        } else {
            contributing = 0;
        }
    } while (d_stop == 0);
}

__global__ void collectZeroIndicesSingleKernel(const float* B, int n,
                                               int* zero_indices,
                                               int* row_start,
                                               int* row_count,
                                               int* global_counter)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start_pos = atomicAdd(global_counter, n);
    int count = 0;
    for (int j = 0; j < n; ++j)
        if (B[IDX2C(row, j, n)] == 0.0f)
            zero_indices[start_pos + count++] = j;

    row_start[row] = start_pos;
    row_count[row] = count;
}

// __global__ void compute_col_to_row(int n, const int* X, int* col_to_row) {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (j >= n) return;

//     for (int i = 0; i < n; ++i) {
//         if (X[IDX2C(i, j, n)] == 1) {
//             col_to_row[j] = i;
//             return;
//         }
//     }
// }

__global__ void compute_col_to_row(int n, const int* X, int* col_to_row) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    if (X[IDX2C(i, j, n)] == 1) {
        // atomic because multiple threads can write to the same col_to_row[j]
        col_to_row[j] = i;
    }
}



__global__ void solve_1bc_rowseq_async_parallel_nb(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ zero_indices,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    const int *k,
    int* __restrict__ R,
    int* __restrict__ Q,
    int* __restrict__ visited
) {
    const int i = blockIdx.x;
    if (i >= n) return;

    // Shared memory for block coordination
    __shared__ int block_doing; // whether any thread in this block did work this epoch

    // One thread per block will handle epoch counters
    int step = 0, local_move = 0;
    int contributing = 1;
    const int expected = n; // total row-workers
    int iteration, accumulate;
    int old_p, last_out;

    do {
        if (threadIdx.x == 0) block_doing = 0;
        // __syncthreads();

        // WORK PHASE: all threads in the block collaborate
        if (R[i] != n && visited[i] == 0) {
            if (threadIdx.x == 0)
                visited[i] = 1;

            // __syncthreads();

            if (i != *k) {
                const int base = row_start[i];
                const int nz   = row_count[i];

                // Divide the zero_indices work among threads
                for (int t = threadIdx.x; t < nz; t += blockDim.x) {
                    const int j = zero_indices[base + t];

                    if (atomicCAS(&Q[j], n, i) == n) {
                        const int r2 = d_col_to_row[j];
                        R[r2] = j;
                        // atomicExch(&block_doing, 1);
                        block_doing = 1;
                    }
                }
            }
        }

        // __syncthreads();
        // __threadfence_system();

        // One thread per block performs epoch accounting
        if (threadIdx.x == 0) {
            int doing = block_doing;

            old_p = atomicAdd(&d_progress, contributing);
            if (contributing == 1 && doing == 0) {
                atomicAdd(&d_outstanding, 1);
            }
            // __threadfence();

            iteration  = (old_p + contributing) / expected;
            accumulate = (old_p + contributing) % expected;

            if (iteration > step && d_moving > local_move) {
                contributing = 1;
                step++;
                local_move++;

                if (accumulate == 0) {
                    last_out = atomicExch(&d_outstanding, 0);
                    // __threadfence();
                    if (last_out == expected) {
                        atomicAdd(&d_stop, 1);
                    }
                    atomicAdd(&d_moving, 1);
                    // __threadfence_system();
                    __threadfence();
                }
            } else {
                contributing = 0;
            }
        }

        // __syncthreads();

    } while (d_stop == 0);

    if (threadIdx.x == 0)
        visited[i] = 0;
}

__global__ void solve_1bc_rowseq_async_parallel_sb(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ zero_indices,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    const int* k,
    int* __restrict__ R,
    int* __restrict__ Q
) {
    // This version runs on a single block with many threads.
    // The "queue" concept is implicit — we repeatedly scan until convergence.

    // extern __shared__ int shmem[];
    __shared__ int changed;

    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    // Only launch one block
    if (blockIdx.x > 0) return;

    do {
        if (tid == 0) changed = 0;
        __syncthreads();

        // Each thread processes multiple rows
        for (int i = tid; i < n; i += T) {
            if (R[i] != n && i != *k) {
                int base = row_start[i];
                int nz   = row_count[i];
                for (int t = 0; t < nz; ++t) {
                    int j = zero_indices[base + t];

                    // Try to label this column j with row i
                    if (atomicCAS(&Q[j], n, i) == n) {
                        int r2 = d_col_to_row[j];
                        R[r2] = j;
                        changed = 1;  // mark that we updated something
                    }
                }
            }
        }

        __syncthreads();

        // Continue until no changes were made in this pass
    } while (changed);
}




__global__ void solve_1bc_rowseq_async_parallel_sba(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ zero_indices,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    const int* k,
    int* __restrict__ R,
    int* __restrict__ Q
) {
    // Single block, many threads. Asynchronous propagation by repeated sweeps.
    if (blockIdx.x > 0) return;

    __shared__ int changed;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    do {
        if (tid == 0) changed = 0;
        __syncthreads();

        // Each thread processes multiple rows
        for (int i = tid; i < n; i += T) {
            // Process only rows that are currently labeled and not the special row *k
            if (i != *k && R[i] != n) {
                const int base = row_start[i];
                const int nz   = row_count[i];

                // Walk this row's zeros
                for (int t = 0; t < nz; ++t) {
                    const int j = zero_indices[base + t];

                    // Try to claim column j for row i
                    if (atomicCAS(&Q[j], n, i) == n) {
                        const int r2 = d_col_to_row[j];

                        // First-writer-wins on R as well to avoid races
                        if (atomicCAS(&R[r2], n, j) == n) {
                            // Mark that the sweep made progress
                            atomicExch(&changed, 1);
                        }
                    }
                }
            }
        }

        // Ensure all global writes to R/Q are visible to the whole block
        __threadfence_block();
        __syncthreads();

        // Continue until no changes were made in this sweep
    } while (changed);
}

// Drop-in replacement: same signature, same inputs.
// Requires cooperative launch support (cg::this_grid()).

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Single device-side flag for grid-wide "did anything change?" tracking.
// This does NOT alter/repurpose any user input arrays.
// __device__ int d_changed; # reuse

__global__ void solve_1bc_rowseq_async_parallel_cg(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ zero_indices,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    const int* k,
    int* __restrict__ R,
    int* __restrict__ Q
) {
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int T   = blockDim.x;
    const int B   = gridDim.x;
    const int bid = blockIdx.x;

    while (true) {
        // Reset change flag once per grid
        if (bid == 0 && tid == 0)
            d_changed = 0;

        // A single sync is enough here
        grid.sync();

        // Each block handles one or more rows
        for (int i = bid; i < n; i += B) {
            if (i != *k && R[i] != n) {
                const int base = row_start[i];
                const int nz   = row_count[i];

                // Each thread covers several zeros
                for (int t = tid; t < nz; t += T) {
                    const int j = zero_indices[base + t];
                    // First-writer-wins on Q
                    if (atomicCAS(&Q[j], n, i) == n) {
                        const int r2 = d_col_to_row[j];
                        // First-writer-wins on R
                        if (atomicCAS(&R[r2], n, j) == n)
                            d_changed = 1; // relaxed write, no atomicExch needed
                    }
                }
            }
        }

        // Ensure global writes visible before read
        __threadfence();

        // Grid-wide sync before check
        grid.sync();

        // Read global flag (cached in a register)
        if (!d_changed)
            break;

        // Only need one sync before next iteration
        grid.sync();
    }
}





__device__ __forceinline__
unsigned warp_exclusive_scan(unsigned v) {
    unsigned lane = threadIdx.x & 31;
    unsigned n = v;
    // inclusive scan in 'n'
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffff, n, offset);
        if (lane >= offset) n += y;
    }
    // convert to exclusive
    return n - v;
}

// ---------------------------------------------
// Block exclusive scan over the first n_active threads.
// - Each thread provides 'flag' (0/1) and receives 'excl' (exclusive prefix).
// - Returns tile_sum (sum of flags over n_active threads) via shared memory.
// - Works for arbitrary n_active <= blockDim.x.
// ---------------------------------------------
__device__ __forceinline__
unsigned block_exclusive_scan_0_1(unsigned flag,
                                  int n_active,
                                  unsigned &excl) {
    // Per-warp exclusive scan
    unsigned lane   = threadIdx.x & 31;
    unsigned warpId = threadIdx.x >> 5;

    unsigned excl_warp = warp_exclusive_scan(flag);
    unsigned incl_warp = excl_warp + flag;

    // Number of warps that have active threads
    int numWarps = (n_active + 31) >> 5;

    // Shared arrays to propagate warp sums
    __shared__ unsigned warp_sums[32];    // inclusive sum of each warp
    __shared__ unsigned warp_offsets[32]; // exclusive prefix sum for warps

    // Initialize to zero to avoid stale values for inactive warps
    if (lane == 0) warp_sums[warpId] = 0;
    __syncthreads();

    // Identify last active lane in this warp
    int warp_base = warpId * 32;
    int last_lane = min(31, max(0, n_active - warp_base - 1));
    bool is_last_in_warp = (lane == last_lane) && (warpId < (unsigned)numWarps);

    // The last active lane in each warp publishes its INCLUSIVE sum
    if (is_last_in_warp) {
        warp_sums[warpId] = incl_warp;
    }
    __syncthreads();

    // First warp scans the warp sums (numWarps entries)
    unsigned warp_excl = 0;
    if (warpId == 0) {
        unsigned wval = (lane < (unsigned)numWarps) ? warp_sums[lane] : 0;
        unsigned wexcl = warp_exclusive_scan(wval);
        if (lane < (unsigned)numWarps) warp_offsets[lane] = wexcl;
    }
    __syncthreads();

    // Each thread's block-exclusive prefix = its warp-exclusive + offset of previous warps
    unsigned block_offset = (warpId < (unsigned)numWarps) ? warp_offsets[warpId] : 0;
    excl = (threadIdx.x < n_active) ? (excl_warp + block_offset) : 0;

    // Tile sum is the last warp's inclusive sum + offset of previous warps.
    // Let the last active thread in the block compute it:
    __shared__ unsigned tile_sum;
    if (threadIdx.x == n_active - 1) {
        unsigned last_warp = (n_active - 1) >> 5;
        tile_sum = warp_sums[last_warp] + warp_offsets[last_warp];
    }
    __syncthreads();

    return tile_sum; // valid for all threads after the sync
}

// --------------------------------------------------------
// Kernel: One block per row, shared-memory compaction.
// Each row writes into a fixed segment [row*n, row*n + count).
// No atomics. Coalesced reads & writes.
// Supports n > blockDim.x via tiling over columns.
// --------------------------------------------------------
// __global__ void collectZeroIndicesSharedMem(const float* __restrict__ B,
//                                             int n,
//                                             int* __restrict__ zero_indices,
//                                             int* __restrict__ row_start,
//                                             int* __restrict__ row_count) {
//     int row = blockIdx.x;
//     if (row >= n) return;

//     // Our fixed segment start for this row:
//     const int base_out = row * n;
//     if (threadIdx.x == 0) {
//         row_start[row] = base_out; // deterministic segment per row
//     }
//     __syncthreads();

//     unsigned running = 0; // total zeros written so far in this row

//     // Tile over columns by blockDim.x
//     for (int base = 0; base < n; base += blockDim.x) {
//         int tid   = threadIdx.x;
//         int col   = base + tid;
//         int active = min(blockDim.x, n - base);

//         // Load flag: is zero?
//         unsigned flag = 0;
//         if (tid < active) {
//             flag = (B[IDX2C(row, col, n)] == 0) ? 1u : 0u;
//         }

//         // Block-wide exclusive scan of flags over 'active' threads
//         unsigned excl = 0;
//         unsigned tile_sum = block_exclusive_scan_0_1(flag, active, excl);

//         // Threads with flag=1 write their column index at compacted position
//         if (tid < active && flag) {
//             zero_indices[base_out + running + excl] = col;
//         }

//         // Advance running total
//         if (tid == 0) running += tile_sum;
//         __syncthreads();
//     }

//     if (threadIdx.x == 0) {
//         row_count[row] = running;
//     }
// }

__global__ void collectZeroIndicesSharedMem(const float* __restrict__ B,
                                            int n,
                                            int* __restrict__ zero_indices,
                                            int* __restrict__ row_start,
                                            int* __restrict__ row_count)
{
    // Each block handles multiple rows if gridDim.x < n
    for (int row = blockIdx.x; row < n; row += gridDim.x) {

        const int base_out = row * n;

        if (threadIdx.x == 0)
            row_start[row] = base_out;
        __syncthreads();

        // ---------------------------------------
        // Pass 1: Each thread counts its own zeros
        // ---------------------------------------
        unsigned local_count = 0;
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            if (B[IDX2C(row, col, n)] == 0)
                local_count++;
        }

        // Shared counters
        __shared__ unsigned prefix_counter;
        __shared__ unsigned total_zeros;

        if (threadIdx.x == 0)
            prefix_counter = 0;
        __syncthreads();

        // Each thread reserves its segment
        unsigned thread_base = atomicAdd(&prefix_counter, local_count);
        __syncthreads();

        if (threadIdx.x == 0)
            total_zeros = prefix_counter;
        __syncthreads();

        // ---------------------------------------
        // Pass 2: Each thread writes its zero indices
        // ---------------------------------------
        unsigned offset = 0;
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            if (B[IDX2C(row, col, n)] == 0) {
                zero_indices[base_out + thread_base + offset] = col;
                ++offset;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
            row_count[row] = total_zeros;
        __syncthreads();
    }
}



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


// global convergence flag (device-side)
// __device__ volatile int d_changed;

// __global__ void solve_1bc_persistent(
//     int n,
//     const int* __restrict__ X,
//     const int* __restrict__ k,
//     const float* __restrict__ B,
//     int* __restrict__ R,
//     int* __restrict__ Q)
// {
//     cg::grid_group grid = cg::this_grid();
//     __shared__ int block_changed;

//     // Each block handles one or more rows: i = blockIdx.x, i += gridDim.x
//     while (true) {
//         if (threadIdx.x == 0) block_changed = 0;
//         __syncthreads();

//         int k_val = *k;

//         for (int i = blockIdx.x; i < n; i += gridDim.x) {
//             for (int j = threadIdx.x; j < n; j += blockDim.x) {

//                 int Ri = R[i];
//                 if (Ri == n) {
//                     int Qj = Q[j];
//                     if (Qj != n && X[IDX2C(i, j, n)] == 1) {
//                         if (atomicCAS(&R[i], n, j) == n)
//                             block_changed = 1;
//                     }
//                 }

//                 int Ri_now = R[i];
//                 if (i != k_val && Ri_now != n) {
//                     int Qj_now = Q[j];
//                     if (Qj_now == n && B[IDX2C(i, j, n)] == 0.0f) {
//                         if (atomicMin(&Q[j], i) > i)
//                             block_changed = 1;
//                     }
//                 }
//             }
//             __syncthreads();
//         }

//         __syncthreads();
//         __threadfence();

//         if (threadIdx.x == 0 && block_changed)
//             atomicExch(&d_changed, 1);

//         grid.sync();

//         if (grid.thread_rank() == 0) {
//             if (d_changed == 0)
//                 d_changed = -1;
//             else
//                 d_changed = 0;
//         }

//         grid.sync();
//         if (d_changed == -1) break;
//     }
// }

// __global__ void solve_1bc_persistent(
//     int n, const int* X, int* k, int* l,
//     const float* B, int* R, int* Q)
// {
//     __shared__ int block_changed;

//     while (true) {
//         // Step 1: reset block-local flag
//         if (threadIdx.x == 0 && threadIdx.y == 0)
//             block_changed = 0;
//         __syncthreads();

//         // Step 2: perform the same logic as solve_1bc_kernel
//         int i = blockIdx.y * blockDim.y + threadIdx.y;
//         int j = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i < n && j < n) {
//             if (Q[j] != n && X[IDX2C(i, j, n)] == 1) {
//                 if (atomicCAS(&R[i], n, j) == n)
//                     block_changed = 1;
//             }
//             if (i != *k && R[i] != n && Q[j] == n) {
//                 float b_val = B[IDX2C(i, j, n)];
//                 if (b_val == 0.0f) {
//                     if (atomicMin(&Q[j], i) > i)
//                         block_changed = 1;
//                 }
//             }
//         }

//         __syncthreads();

//         // Step 3: reduce to a single global flag
//         if (threadIdx.x == 0 && threadIdx.y == 0 && block_changed)
//             atomicExch(&d_changed, 1);

//         // Step 4: global sync — use cooperative groups
//         cooperative_groups::grid_group grid = cooperative_groups::this_grid();
//         grid.sync();

//         // Step 5: one thread decides whether to continue
//         if (grid.thread_rank() == 0) {
//             if (d_changed == 0) {
//                 d_changed = -1;  // signal exit
//             } else {
//                 d_changed = 0;   // reset for next iteration
//             }
//         }

//         grid.sync();

//         // Step 6: exit condition
//         if (d_changed == -1)
//             break;

//         // then loop again
//     }
// }

__global__ void solve_1bc_persistent_global(
    int n, const int* d_col_to_row, int* k, int* l,
    const float* B, int* R, int* Q)
{
    __shared__ int block_changed;

    // Create cooperative grid group
    cg::grid_group grid = cg::this_grid();

    while (true) {
        // Step 1: reset block-local flag
        if (threadIdx.x == 0 && threadIdx.y == 0)
            block_changed = 0;
        __syncthreads();

        // Step 2: iterate over all rows/cols with grid-stride loops
        for (int i = blockIdx.y * blockDim.y + threadIdx.y;
             i < n;
             i += gridDim.y * blockDim.y)
        {
            for (int j = blockIdx.x * blockDim.x + threadIdx.x;
                 j < n;
                 j += gridDim.x * blockDim.x)
            {

                if (i != *k && R[i] != n && Q[j] == n) {
                    float b_val = B[IDX2C(i, j, n)];
                    if (b_val == 0.0f) {
                        if (atomicMin(&Q[j], i) == n){
                            block_changed = 1;
                            R[d_col_to_row[j]] = j;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Step 3: reduce to a single global flag
        if (threadIdx.x == 0 && threadIdx.y == 0 && block_changed)
            atomicExch(&d_changed, 1);

        // Step 4: global sync — wait for all blocks
        grid.sync();

        // Step 5: one thread decides whether to continue
        if (grid.thread_rank() == 0) {
            if (d_changed == 0) {
                d_changed = -1;  // signal exit
            } else {
                d_changed = 0;   // reset for next iteration
            }
        }

        grid.sync();

        // Step 6: exit condition
        if (d_changed == -1)
            break;
    }
}


__device__ int  d_active_count;   // number of active rows
__device__ int  d_next_count;     // number of next rows

// -------------------------------------------------------------------
// Kernel: 1D active-row labeling, cooperative persistent grid
// -------------------------------------------------------------------
__global__ void solve_1bc_persistent_3xslow(
    int n,
    const int* __restrict__ d_col_to_row,
    int* __restrict__ k,
    int* __restrict__ l,
    const float* __restrict__ B,
    int* __restrict__ R,
    int* __restrict__ Q,
    int* __restrict__ d_active_rows,
    int* __restrict__ d_next_rows)
{
    __shared__ int block_changed;
    cg::grid_group grid = cg::this_grid();

    // Step 0: initialize active rows once
    // if (grid.thread_rank() == 0) {
    if (grid.thread_rank() == 0)
    d_active_count = 0;  // make sure count starts clean
grid.sync();

// Each thread checks rows in parallel
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n; i += gridDim.x * blockDim.x)
{
    if (R[i] != n && i != *k) {
        int pos = atomicAdd(&d_active_count, 1);
        d_active_rows[pos] = i;
    }
}
grid.sync();

// Now all initially labeled rows are in active list
if (grid.thread_rank() == 0) {
    d_next_count = 0;
    d_changed = 0;
}
// grid.sync();
    // }
    // printf("Before %d\n", d_active_count);
    grid.sync();
    // printf("Before2 %d\n", d_active_count);

    while (true) {
        if (threadIdx.x == 0)
            block_changed = 0;
        __syncthreads();

        if (grid.thread_rank() == 0)
            d_changed = 0;   // reset global flag
        grid.sync();

        int active = d_active_count;
        if (active == 0) {
            if (grid.thread_rank() == 0)
                d_changed = -1;
            grid.sync();
            if (d_changed == -1)
                break;
            continue;
        }

        const long long nt  = 1LL * gridDim.x * blockDim.x;
        const long long tid = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
        const long long total = 1LL * active * n;
        const int row_k = *k;

        // printf("Before active %d, nt %d\n, tid %d, block_changed %d\n", active, nt, tid, block_changed);


        for (long long t = tid; t < total; t += nt) {
            // printf("t %d\n\n", t);
            int ar = (int)(t / n);
            int j  = (int)(t % n);
            int i  = d_active_rows[ar];

            // printf("i %d, j %d\n \n", i, j);

            if (i == row_k || R[i] == n || Q[j] != n)
                continue;

            float b = B[IDX2C(i, j, n)];

            
            if (b == 0.0f) {
                if (atomicMin(&Q[j], i) == n) {
                    int rj = d_col_to_row[j];
                    if (atomicCAS(&R[rj], n, j) == n) {
                        int pos = atomicAdd(&d_next_count, 1);
                        d_next_rows[pos] = rj;
                        block_changed = 1;
                    }
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && block_changed)
            atomicExch(&d_changed, 1);

        grid.sync();

        if (grid.thread_rank() == 0) {
            if (d_changed == 0) {
                d_changed = -1;  // no change -> stop
            } else {
                // rotate lists
                int* tmp = d_active_rows;  d_active_rows = d_next_rows;  d_next_rows = tmp;
                d_active_count = d_next_count;
                d_next_count = 0;
                d_changed = 0;
            }
        }

        grid.sync();
        if (d_changed == -1)
            break;
    }
}

__global__ void select_zeros(
    const float* __restrict__ B,
    int N,
    int* __restrict__ selected,
    int* __restrict__ count)
{
    extern __shared__ int s_flags[]; // per-thread flags
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;

    // Step 1: flag zeros in this block
    int flag = 0;
    if (global < N && B[global] == 0.0f)
        flag = 1;
    s_flags[tid] = flag;
    __syncthreads();

    // Step 2: exclusive prefix sum (scan) within block
    // simple Hillis–Steele scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = 0;
        if (tid >= offset) val = s_flags[tid - offset];
        __syncthreads();
        s_flags[tid] += val;
        __syncthreads();
    }

    int block_total = s_flags[blockDim.x - 1] + (flag ? 1 : 0);
    __shared__ int block_base;
    if (tid == 0)
        block_base = atomicAdd(count, block_total);
    __syncthreads();

    // Step 3: write results to global output
    if (flag) {
        int out_idx = block_base + s_flags[tid] - 1;
        selected[out_idx] = global;
    }
}

__global__ void solve_1bc_sparse_indices_1D(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ d_indices,   // flattened zero indices
    const int* __restrict__ d_count,     // number of zero indices
    int* __restrict__ k,
    int* __restrict__ l,
    const float* __restrict__ B,
    int* __restrict__ R,
    int* __restrict__ Q)
{
    __shared__ int block_changed;
    cg::grid_group grid = cg::this_grid();

    int num_indices = *d_count;
    int total_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (true) {
        if (threadIdx.x == 0)
            block_changed = 0;
        __syncthreads();

        // Grid-stride loop over known zero indices
        for (int idx = tid; idx < num_indices; idx += total_threads) {
            int flat = d_indices[idx];

            // column-major indexing
            int j = flat / n;
            int i = flat % n;

            // Perform labeling logic
            if (i != *k && R[i] != n && Q[j] == n) {
                // No need to check B[i,j] == 0
                if (atomicMin(&Q[j], i) == n) {
                    block_changed = 1;
                    R[d_col_to_row[j]] = j;
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && block_changed)
            atomicExch(&d_changed, 1);

        grid.sync();

        if (grid.thread_rank() == 0) {
            if (d_changed == 0)
                d_changed = -1;
            else
                d_changed = 0;
        }

        grid.sync();

        if (d_changed == -1)
            break;
    }
}

__global__ void solve_1bc_sparse_single_block(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ d_indices,
    const int* __restrict__ d_count,
    int* __restrict__ k,
    int* __restrict__ l,
    const float* __restrict__ B,
    int* __restrict__ R,
    int* __restrict__ Q)
{
    __shared__ int block_changed;
    int num_indices = *d_count;

    while (true) {
        if (threadIdx.x == 0)
            block_changed = 0;
        __syncthreads();

        for (int idx = threadIdx.x; idx < num_indices; idx += blockDim.x) {
            int flat = d_indices[idx];
            int j = flat / n;
            int i = flat % n;

            if (i != *k && R[i] != n && Q[j] == n) {
                if (atomicMin(&Q[j], i) == n) {
                    block_changed = 1;
                    R[d_col_to_row[j]] = j;
                }
            }
        }

        __syncthreads();
        if (block_changed == 0)
            break;  // no change, exit
    }
}


bool solve_from_kl(float* d_C, int* d_X, float* d_U, float* d_V, int n, float* d_B, int* d_R, int* d_Q, int* k, int* l,int* d_col_to_row, int* d_indices, int* d_row_start, int* d_row_count, int* d_counter, int* d_row_visited, int* d_next_i, int* d_next_j, int* d_parity, int* d_fR, int* d_fQ, unsigned char* d_hasPredR, unsigned char* d_hasPredQ, unsigned char* d_cycR, unsigned char* d_cycQ, int* d_active_rows, int* d_next_rows, int* d_count, int* d_selected, int* d_num_selected, void* d_temp_storage, size_t temp_bytes) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    // update_Q<<<1,1>>>(d_Q, k, l);
    update_RQ<<<1,1>>>(d_R, d_Q, k, l, d_col_to_row);
    reset_d_changed<<<1,1>>>();         



    // reset_device_flags<<<1,1>>>();
    // count_done_entries<<<blocks, threads>>>(n, d_B);

    

    // printDeviceVar("d_done", d_done);

    // cudaDeviceSynchronize();
    // printDeviceMatrix("Before d_B", d_B, n);
    // cudaMemset(d_counter, 0, sizeof(int));
    // collectZeroIndicesSingleKernel<<< (n + 256 - 1) / 256, 256>>>(d_B, n, d_indices, d_row_start, d_row_count, d_counter);
    // collectZeroIndicesSharedMem<<<n, 512>>>(          d_B, n, d_indices, d_row_start, d_row_count);
    // printDeviceVector("d_col_to_row", d_col_to_row, n);
    // printDeviceVector("d_row_start", d_row_start, n);
    // printDeviceVector("d_row_count", d_row_count, n);

    // solve_1bc_kernel_full<<<blocks, threads>>>(n, d_X, k, d_B, d_R, d_Q);

    // std::vector<int> h_queue(n, -1);
    // h_queue[0] = seed_row;
    // int h_head = 0, h_tail = 1;
    // cudaMemcpy(d_queue, h_queue.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_q_head, &h_head, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_q_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice);
    // init_from_kl<<<1, 1>>>(n, k, l, d_col_to_row,
    //                    d_R, d_Q, d_queue, d_q_head, d_q_tail);

    // int blockSize = 256, gridSize = (n + blockSize - 1) / blockSize;
    
    // cudaDeviceSynchronize();

    // solve_1bc_rowseq_async_parallel_nb<<<n, 128>>>( n, d_col_to_row, d_indices, d_row_start, d_row_count, k, d_R, d_Q, d_row_visited);
    // solve_1bc_rowseq_async_parallel_sb<<<1, 1024>>>( n, d_col_to_row, d_indices, d_row_start, d_row_count, k, d_R, d_Q);
    // solve_1bc_rowseq_async_parallel_sba<<<1, 1024>>>( n, d_col_to_row, d_indices, d_row_start, d_row_count, k, d_R, d_Q);
    // void* args[] = { &n, &d_col_to_row, &d_indices, &d_row_start, &d_row_count, &k, &d_R, &d_Q };
    // cudaLaunchCooperativeKernel(
    //     (void*)solve_1bc_rowseq_async_parallel_cg, 352, 256, args
    // );

    // size_t shmem = 256 * sizeof(int);
    // select_zeros<<<2, 2, shmem>>>(d_B, n * n, d_selected, d_count);
    // select_zeros_with_cub(n, d_B, d_indices, d_selected, d_num_selected, d_temp_storage, temp_bytes);
    selectZerosOptimized(d_B, d_indices, d_count, n * n);
    // printDeviceVector("d_indices", d_indices, n*n);

    if (n <= 1200) {
        //------------------------------------------------------
        // Very sparse case — single block kernel
        //------------------------------------------------------
        int blockSize = 1024;
        solve_1bc_sparse_single_block<<<1, blockSize>>>(
            n, d_col_to_row, d_indices, d_count, k, l, d_B, d_R, d_Q);
        cudaDeviceSynchronize();
    } else {
        int blockSize = 256;
    int numBlocks = 256;  // tune as needed per GPU and problem size

    void* args[] = {
        &n, &d_col_to_row, &d_indices, &d_count,
        &k, &l, &d_B, &d_R, &d_Q
    };

    cudaLaunchCooperativeKernel(
        (void*)solve_1bc_sparse_indices_1D,
        numBlocks, blockSize, args);
    }





    // void* args[] = { &n, &d_col_to_row, &k, &l, &d_B, &d_R, &d_Q };
    // cudaLaunchCooperativeKernel((void*)solve_1bc_persistent_global,  dim3(22,22),  dim3(16,16), args);
    // cudaLaunchCooperativeKernel((void*)solve_1bc_persistent,  352,  256, args);
    // void* args[] = {
    //     &n, &d_col_to_row, &k, &l,
    //     &d_B, &d_R, &d_Q,
    //     &d_active_rows, &d_next_rows
    // };
    // cudaLaunchCooperativeKernel(
    //     (void*)solve_1bc_persistent_3xslow,
    //     352, 256,
    //     args, 0, nullptr);

    cudaDeviceSynchronize();

    




    // printDeviceVector("d_R", d_R, n);
    // printDeviceVector("d_Q", d_Q, n);

    // printDeviceMatrix("after d_B", d_B, n);
    // printDeviceVector("d_row_visited", d_row_visited, n);

    // dim3 block2(256);
    // dim3 grid2((n + block2.x - 1) / block2.x);

    // solve_1bc_queue_kernel<<<grid2, block2>>>(
    //     n, d_col_to_row,
    //     d_indices, d_row_start, d_row_count,
    //     d_R, d_Q,
    //     d_queue, d_q_head, d_q_tail);

    // cudaDeviceSynchronize();
        
    // } else {
    //     void* args[] = { &n, &d_X, &k, &l, &d_B, &d_R, &d_Q };
    // cudaLaunchCooperativeKernel(
    //     (void*)solve_1bc_persistent, blocks, threads, args);
    // }

    check_Rk<<<1,1>>>(d_R, k, l, n);

    // printDeviceVar("d_flag", d_flag);
    

    int h_flag;
    cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (h_flag == 1) {

        dim3 blockDim(16, 16);
        dim3 gridDim((n + 15) / 16, (n + 15) / 16);

        // init_mapping<<<gridDim, blockDim>>>(d_R, d_Q, d_next_i, d_next_j, d_parity, n);
        // pointer_jump_cycle<<<gridDim, blockDim>>>(d_next_i, d_next_j, d_parity, n);
        // update_dX<<<gridDim, blockDim>>>(d_next_i, d_next_j, d_parity, d_X, n);

        
        // printf("here\n");
        // printDeviceMatrix("d_B", d_B, n);
        // printDeviceMatrix("d_X", d_X, n);

        // printDeviceVector("d_V", d_V, n);

        // printDeviceVector("d_R", d_R, n);
        // printDeviceVector("d_Q", d_Q, n);

        // printDeviceScalar("k", k);
        // printDeviceScalar("l", l);

        // printDeviceMatrix("d_X before", d_X, n);
        // identify_and_flip_singleblock<<<1, 256>>>(d_R, d_Q, d_X, n, k, l,
        //                                       d_fR, d_fQ,
        //                                       d_hasPredR, d_hasPredQ,
        //                                       d_cycR, d_cycQ);

        // cudaDeviceSynchronize();
        // update_V<<<1,1>>>(d_B, d_V, d_X, d_R, d_Q, n, k, l);
        // printDeviceMatrix("d_X after", d_X, n);

        // printDeviceMatrix("d_X and then", d_X, n);
        
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
    
    int* d_col_to_row;
    cudaMalloc(&d_col_to_row, n * sizeof(int));
    
    
    int *d_indices, *d_row_start, *d_row_count, *d_counter;
    cudaMalloc(&d_indices, n*n*sizeof(int));
    cudaMalloc(&d_row_start, n*sizeof(int));
    cudaMalloc(&d_row_count, n*sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));

    int *d_next_i, *d_next_j, *d_parity;

    cudaMalloc(&d_next_i, n*sizeof(int));
    cudaMalloc(&d_next_j, n*sizeof(int));
    cudaMalloc(&d_parity, n*sizeof(int));
    

    // weird
    // int *d_queue, *d_q_head, *d_q_tail;
    // cudaMalloc(&d_queue, n * sizeof(int));
    // cudaMalloc(&d_q_head, sizeof(int));
    // cudaMalloc(&d_q_tail, sizeof(int));

    int *d_active_rows, *d_next_rows;
    cudaMalloc(&d_active_rows, n * sizeof(int));
    cudaMalloc(&d_next_rows,   n * sizeof(int));



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

    int *d_fR, *d_fQ;
    // unsigned char *d_cycR, *d_cycQ;
    unsigned char *d_hasPredR, *d_hasPredQ, *d_cycR, *d_cycQ;

    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
    cudaMalloc(&d_hasPredR, n);
    cudaMalloc(&d_hasPredQ, n);
    cudaMalloc(&d_cycR, n);
    cudaMalloc(&d_cycQ, n);

    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // int* d_selected;
    // cudaMalloc(&d_selected, n * n * sizeof(int));

    int *d_selected, *d_num_selected;//*d_indices, 
    void* d_temp_storage;
    size_t temp_bytes;

    // setup_cub_buffers(n, d_B, d_indices, d_selected, d_num_selected, d_temp_storage, temp_bytes);

    // int repR, repQ;

    int* d_row_visited;
    cudaMalloc(&d_row_visited, n * sizeof(int));
    set_array_value<<<(n + 255)/256, 256>>>(d_row_visited, 0, n);

    find_most_negative<<<blocks, threads>>>(d_B, n, k, l);
    int steps = 0;
    while (true) {
        // std::cout << "Step " << steps << " \n";
        bool should_continue = solve_from_kl(d_C, d_X, d_U, d_V, n, d_B, d_R, d_Q, k, l, d_col_to_row, d_indices, d_row_start, d_row_count, d_counter, d_row_visited,  d_next_i, d_next_j, d_parity, d_fR, d_fQ, d_hasPredR, d_hasPredQ, d_cycR, d_cycQ, d_active_rows, d_next_rows, d_count, d_selected, d_num_selected, d_temp_storage, temp_bytes);
        steps++;
        if (!should_continue) {
            // std::cout << "Solver has converged after " << steps << " steps.\n";
            break;
        }
    }
    cudaFree(d_B);
    cudaFree(d_R);
    cudaFree(d_Q);
    cudaFree(d_row_visited);

    cudaFree(k);
    cudaFree(l);

    cudaFree(d_col_to_row);

    cudaFree(d_indices);
    cudaFree(d_row_start);
    cudaFree(d_row_count);
    cudaFree(d_counter);

    // cudaFree(d_queue);
    // cudaFree(d_q_head);
    // cudaFree(d_q_tail);

    cudaFree(d_next_i);
    cudaFree(d_next_j);
    cudaFree(d_parity);

    cudaFree(d_fR); cudaFree(d_fQ);
    cudaFree(d_hasPredR); cudaFree(d_hasPredQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);

    cudaFree(d_active_rows);
    cudaFree(d_next_rows);

    cudaFree(d_count);
    // cudaFree(d_selected);

    cudaFree(d_B);
    cudaFree(d_indices);
    cudaFree(d_selected);
    cudaFree(d_num_selected);
    cudaFree(d_temp_storage);
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