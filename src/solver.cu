#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <string>
#include <cstdio>
#include <math_constants.h>


#include "cuda_debug_utils.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


#define IDX2C(i,j,n) ((j)*(n)+(i))


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

// // ---------------------------------------------------------------
// // Host launcher
// // ---------------------------------------------------------------





__device__  float d_min;
__device__  int d_changed;
__device__  float d_epsilon;
__device__  int d_found;
__device__  int d_flag;
__device__  int d_b_kl_neg;

// __global__ void set_array_value(int* arr, int value, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) arr[idx] = value;
// }

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

// __global__ void process_cycle(float* B, float* V, int* d_X, const int* d_R, const int* d_Q, int n, int* k, int* l) {
//     int k_ = *k;
//     int l_ = *l;
//     while (true) {
//         d_X[IDX2C(k_, l_, n)] = 1;
//         l_ = d_R[k_];
//         d_X[IDX2C(k_, l_, n)] = 0;
//         k_ = d_Q[l_];
//         if (k_ == *k && l_ == *l) break;
//     }
//     V[*l] += B[IDX2C(*k, *l, n)];
// }

__global__ void finalize_epsilon(const float* d_B, int n, int* k, int* l) {
    if (isinf(d_min)) d_epsilon = -d_B[IDX2C(*k, *l, n)];
    else d_epsilon = d_min;
}

__global__ void update_Q(int* d_Q, const int* k, const int* l) {
    d_Q[*l] = *k;
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


__global__ void compute_col_to_row(int n, const int* X, int* col_to_row) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    if (X[IDX2C(i, j, n)] == 1) {
        col_to_row[j] = i;
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


// void selectZerosOptimized(const float* d_B, int* d_out, int* d_count, int total)
// {
    
// }

// __global__ void update_RQ(int* d_R, int* d_Q, const int* k, const int* l, int* d_col_to_row) {
//     d_Q[*l] = *k;
//     d_R[d_col_to_row[*l]] = *l;
// }

__global__ void step_2_init(int* d_R, int* d_Q, const int* k, const int* l, int* d_col_to_row, int* d_count) {
    d_changed = 0;
    // d_count = 0;
    d_Q[*l] = *k;
    d_R[d_col_to_row[*l]] = *l;
}

__global__ void compute_B(const float* C, const float* U, const float* V, float* B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
}

__global__ void reset_RQ(int* R, int* Q, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        R[idx] = n;
        Q[idx] = n;
    }
}


// __global__ void compute_B_and_find_most_negative(const float* C, const float* U, const float* V, float* __restrict__ d_B, int n, int* d_out_i, int* d_out_j) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.y * blockDim.x + threadIdx.x;
//     int threads_per_block = blockDim.x * blockDim.y;
//     extern __shared__ float s_vals[];
//     __shared__ int s_rows[256];
//     __shared__ int s_cols[256];
//     float val = INFINITY;
//     int myRow = -1, myCol = -1;
//     if (row < n && col < n) {
//         // B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
//         float tmp = C[IDX2C(row, col, n)] - U[row] - V[col];
//         if (tmp < 0.0f) {
//             val = tmp;
//             myRow = row;
//             myCol = col;
//         }
//         d_B[IDX2C(row, col, n)] = tmp;
//     }
//     s_vals[tid] = val;
//     s_rows[tid] = myRow;
//     s_cols[tid] = myCol;
//     __syncthreads();
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
//     if (tid == 0 && s_vals[0] < INFINITY) {
//         d_found = 1;
//         float oldMin = atomicMinFloat(&d_min, s_vals[0]);
//         if (s_vals[0] < oldMin) {
//             *d_out_i = s_rows[0];
//             *d_out_j = s_cols[0];
//         }
//     }
// }

// __global__ void compute_B_and_find_most_negative_and_reset_RQ(const float* C, const float* U, const float* V, float* __restrict__ d_B, int n, int* d_out_i, int* d_out_j, int* R, int* Q) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.y * blockDim.x + threadIdx.x;
//     int threads_per_block = blockDim.x * blockDim.y;
//     extern __shared__ float s_vals[];
//     __shared__ int s_rows[256];
//     __shared__ int s_cols[256];
//     float val = INFINITY;
//     int myRow = -1, myCol = -1;
//     int idx = IDX2C(row, col, n);
//     if (row < n && col < n) {
//         // B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
//         float tmp = C[idx] - U[row] - V[col];
//         if (tmp < 0.0f) {
//             val = tmp;
//             myRow = row;
//             myCol = col;
//         }
//         d_B[idx] = tmp;
//         if (idx < n) {
//             R[idx] = n;
//             Q[idx] = n;
//         }
//     }
//     s_vals[tid] = val;
//     s_rows[tid] = myRow;
//     s_cols[tid] = myCol;
//     __syncthreads();
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
//     if (tid == 0 && s_vals[0] < INFINITY) {
//         d_found = 1;
//         float oldMin = atomicMinFloat(&d_min, s_vals[0]);
//         if (s_vals[0] < oldMin) {
//             *d_out_i = s_rows[0];
//             *d_out_j = s_cols[0];
//         }
//     }
// }


__global__ void step_3a(const float* C, const float* U, float* V, float* __restrict__ d_B, int n, int* d_out_i, int* d_out_j, int* R, int* Q, int* d_X, int* k, int* l) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row == 0 && col == 0) {
        int k_ = *k;
        int l_ = *l;
        while (true) {
            d_X[IDX2C(k_, l_, n)] = 1;
            l_ = R[k_];
            d_X[IDX2C(k_, l_, n)] = 0;
            k_ = Q[l_];
            if (k_ == *k && l_ == *l) break;
        }
        V[*l] += d_B[IDX2C(*k, *l, n)];
    } 
    __syncthreads();
    __threadfence_system();

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    extern __shared__ float s_vals[];
    __shared__ int s_rows[256];
    __shared__ int s_cols[256];
    float val = INFINITY;
    int myRow = -1, myCol = -1;
    int idx = IDX2C(row, col, n);
    if (row < n && col < n) {
        // B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
        float tmp = C[idx] - U[row] - V[col];
        if (tmp < 0.0f) {
            val = tmp;
            myRow = row;
            myCol = col;
        }
        d_B[idx] = tmp;
        if (idx < n) {
            R[idx] = n;
            Q[idx] = n;
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


bool solve_from_kl(float* d_C, int* d_X, float* d_U, float* d_V, int n, float* d_B, int* d_R, int* d_Q, int* k, int* l,int* d_col_to_row, int* d_indices, int* d_count) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    // selectZerosOptimized(d_B, d_indices, d_count, n * n);
    
    compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    // update_Q<<<1,1>>>(d_Q, k, l);
    step_2_init<<<1,1>>>(d_R, d_Q, k, l, d_col_to_row, d_count);
    // reset_d_changed<<<1,1>>>();   
       
    
    int blockSize = 1024;
    int numBlocks = (n * n + blockSize - 1) / blockSize;
    cudaMemset(d_count, 0, sizeof(int));
    size_t sharedBytes = blockSize * sizeof(int);
    selectZerosEfficient<<<numBlocks, blockSize, sharedBytes>>>(d_B, d_indices, d_count, n * n);
    cudaDeviceSynchronize();

    // printDeviceVector("d_indices", d_indices, n*n);

    if (n <= 1200) {
        //------------------------------------------------------
        // Very sparse case — single block kernel
        //------------------------------------------------------
        int blockSize = 1024;
        solve_1bc_sparse_single_block<<<1, blockSize>>>(
            n, d_col_to_row, d_indices, d_count, k, l, d_B, d_R, d_Q);
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

    cudaDeviceSynchronize();


    check_Rk<<<1,1>>>(d_R, k, l, n);

   

    int h_flag;
    cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (h_flag == 1) {
       
        // process_cycle<<<1,1>>>(d_B, d_V, d_X, d_R, d_Q, n, k, l);
        
        // compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);
        // find_most_negative<<<blocks, threads>>>(d_B, n, k, l);

        // set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        // set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
        // compute_B_and_find_most_negative<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n, k, l);
        // reset_RQ<<<(n + 255)/256, 256>>>(d_R, d_Q, n);
        // compute_B_and_find_most_negative_and_reset_RQ<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n, k, l, d_R, d_Q);
        step_3a<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n, k, l, d_R, d_Q, d_X, k, l);
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
        // set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
        // set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
        reset_RQ<<<(n + 255)/256, 256>>>(d_R, d_Q, n);
        return true;
    } else {
        return false;
    }
}

void solve(float* d_C, int* d_X, float* d_U, float* d_V, int n) {
    float* d_B;
    
    int* d_col_to_row;
    cudaMalloc(&d_col_to_row, n * sizeof(int));
    
    int* d_indices;
    cudaMalloc(&d_indices, n * n * sizeof(int));

    cudaMalloc(&d_B, n * n * sizeof(float));
    int *k, *l;
    cudaMalloc(&k, sizeof(int));
    cudaMalloc(&l, sizeof(int));

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    compute_B<<<blocks, threads>>>(d_C, d_U, d_V, d_B, n);

    int *d_R, *d_Q;
    cudaMalloc(&d_R, n * sizeof(int));
    cudaMalloc(&d_Q, n * sizeof(int));
    // set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    // set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);
    reset_RQ<<<(n + 255)/256, 256>>>(d_R, d_Q, n);


    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    find_most_negative<<<blocks, threads>>>(d_B, n, k, l);

    int steps = 0;
    while (true) {
        // std::cout << "Step " << steps << " \n";
        bool should_continue = solve_from_kl(
            d_C, d_X, d_U, d_V, n, d_B, d_R, d_Q,
            k, l, d_col_to_row, d_indices, d_count);
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
    cudaFree(d_col_to_row);
    cudaFree(d_indices);
    cudaFree(d_count);
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