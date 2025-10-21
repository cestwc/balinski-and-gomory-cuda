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


// __global__ void init_from_kl(
//     int n,
//     const int* __restrict__ d_k,
//     const int* __restrict__ d_l,
//     const int* __restrict__ d_col_to_row,
//     int* __restrict__ d_R,
//     int* __restrict__ d_Q,
//     int* __restrict__ d_queue,
//     int* __restrict__ d_q_head,
//     int* __restrict__ d_q_tail)
// {
//     // Thread 0 does all initialization
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         int k = *d_k;
//         int l = *d_l;

//         // 1. Label column l with row k
//         d_Q[l] = k;

//         // 2. Find which row corresponds to column l
//         int r = d_col_to_row[l];

//         // 3. Label that row
//         d_R[r] = l;

//         // 4. Initialize the queue: all n → sentinel, then put r at front
//         for (int i = 0; i < n; ++i)
//             d_queue[i] = n;
//         d_queue[0] = r;

//         // 5. Initialize queue pointers
//         *d_q_head = 0;
//         *d_q_tail = 1;
//     }
// }




// __global__ void solve_1bc_queue_kernel(
//     int n,
//     const int* __restrict__ d_col_to_row,
//     const int* __restrict__ zero_indices,
//     const int* __restrict__ row_start,
//     const int* __restrict__ row_count,
//     int* __restrict__ R,
//     int* __restrict__ Q,
//     int* __restrict__ queue,
//     int* __restrict__ q_head,
//     int* __restrict__ q_tail)
// {
//     // multiple threads cooperate to pop rows and process them
//     while (true) {
//         int my_idx = atomicAdd(q_head, 1);
//         int cur_tail = atomicAdd(q_tail, 0);
//         if (my_idx >= cur_tail)
//             break; // no more work at this moment

//         int row = queue[my_idx];

//         if (R[row] == n) continue; // skip unlabeled rows

//         int base = row_start[row];
//         int nz   = row_count[row];

//         for (int k = 0; k < nz; ++k) {
//             int j = zero_indices[base + k];
//             if (atomicCAS(&Q[j], n, row) == n) {
//                 int r2 = d_col_to_row[j];
//                 if (r2 >= 0 && r2 < n) {
//                     if (atomicCAS(&R[r2], n, j) == n) {
//                         int pos = atomicAdd(q_tail, 1);
//                         if (pos < n) queue[pos] = r2;
//                     }
//                 }
//             }
//         }
//     }
// }


// Assumptions:
// - One block per row, blockDim.x == 1
// - row i's zero columns live at zero_indices[row_start[i] .. row_start[i] + row_count[i]-1]
// - R[i] == n means row i unlabeled; Q[j] == n means column j unlabeled
// - d_col_to_row[j] gives the row reached from column j

__global__ void solve_1bc_rowseq_async(
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
    // if (i == *k) return;

    // Local epoch state (mirrors your older kernel’s pattern)
    int contributing = 1;      // this worker will contribute to the epoch counter
    int step = 0;              // worker's local epoch number
    int local_move = 0;        // seen value of d_moving
    int doing = 0;             // whether this worker did useful work in this epoch
    int iteration, accumulate; // epoch arithmetic
    int old_p, last_out;

    // Number of contributors required to “close” an epoch:
    const int expected = n;    // one contributor per row-worker

    // Main epoch loop (barrier-free global sync via counters)
    do {
        // ---------- WORK PHASE: try to expand from this row if labeled ----------
        // doing = 0;

        // printf("Blind row: %d, label %d\n", i, R[i]);

        if (R[i] != n && visited[i] == 0) {
            visited[i] = 1;
            // printf("row: %d, label %d\n", i, R[i]);
            doing = 1;  // we made progress this epoch

            // printf("this is i, %d", i);

            if (i != *k){           

                const int base = row_start[i];
                const int nz   = row_count[i];

                // printf("base, %d\n", base);

                // printf("nz, %d\n", nz);


                // Sequential per-row visit over pre-collected zeros
                // printf("Row %d has %d zero columns:\n", i, nz);
                for (int index = 0; index < nz; ++index) {
                    const int j = zero_indices[base + index];
                    // printf("column j, %d\n", j);
                    

                    // Column labeling: claim Q[j] if still unlabeled
                    if (atomicCAS(&Q[j], n, i) == n) {
                        // If we labeled column j, immediately label the mapped row
                        const int r2 = d_col_to_row[j];
                        // printf("Second row: %d, label %d\n", r2, R[r2]);

                        R[r2] = j;
                        // printf("After Second row: %d, label %d\n", r2, R[r2]);
                        // if (r2 >= 0 && r2 < n) {
                            // if (atomicCAS(&R[r2], n, j) == n) {
                            // }
                        // }
                    }
                }
            }
        } else {
            doing = 0;
        }

        // Make writes visible globally (labels R/Q) before we account progress
        __threadfence_system();

        // ---------- EPOCH ACCOUNTING (same pattern as your old code) ----------
        // Contribute to global progress and possibly to "idle" (outstanding) count.
        old_p = atomicAdd(&d_progress, contributing);
        if (contributing == 1 && doing == 0) {
            atomicAdd(&d_outstanding, 1);  // this worker had no work in this epoch
        }
        __threadfence(); // order the counter updates

        // Figure out our epoch index and position within the epoch
        iteration  = (old_p + contributing) / expected; // which epoch we’re now in
        accumulate = (old_p + contributing) % expected; // how many contributions so far in this epoch

        // printf("iteration, %d\n", iteration);
        // printf("accumulate, %d\n", accumulate);

        // If the global epoch (d_moving) has advanced beyond what we’ve seen,
        // this worker should re-enable its contribution for the next epoch.
        if (iteration > step && d_moving > local_move) {
            contributing = 1;
            step++;
            local_move++;

            // If we were the last contributor in the epoch (accumulate == 0),
            // close the epoch: test if everyone was idle; if yes -> stop.
            if (accumulate == 0) {
                last_out = atomicExch(&d_outstanding, 0);
                __threadfence();
                if (last_out == expected) {
                    atomicAdd(&d_stop, 1); // all idle in this epoch → stop
                }
                // Start next epoch
                atomicAdd(&d_moving, 1);
                __threadfence_system();

                // Reset d_progress to 0 for the next epoch (one thread can do it safely here)
                // Optional if you want progress per-epoch; not mandatory if you rely on arithmetic.
                // atomicExch(&d_progress, 0);
            }
        } else {
            // Already contributed this epoch; don’t double-count
            contributing = 0;
        }

        // Optional tiny backoff to reduce contention
        // __nanosleep(64);

    } while (d_stop == 0);
    visited[i] = 0;
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
__global__ void collectZeroIndicesSharedMem(const float* __restrict__ B,
                                            int n,
                                            int* __restrict__ zero_indices,
                                            int* __restrict__ row_start,
                                            int* __restrict__ row_count) {
    int row = blockIdx.x;
    if (row >= n) return;

    // Our fixed segment start for this row:
    const int base_out = row * n;
    if (threadIdx.x == 0) {
        row_start[row] = base_out; // deterministic segment per row
    }
    __syncthreads();

    unsigned running = 0; // total zeros written so far in this row

    // Tile over columns by blockDim.x
    for (int base = 0; base < n; base += blockDim.x) {
        int tid   = threadIdx.x;
        int col   = base + tid;
        int active = min(blockDim.x, n - base);

        // Load flag: is zero?
        unsigned flag = 0;
        if (tid < active) {
            flag = (B[IDX2C(row, col, n)] == 0) ? 1u : 0u;
        }

        // Block-wide exclusive scan of flags over 'active' threads
        unsigned excl = 0;
        unsigned tile_sum = block_exclusive_scan_0_1(flag, active, excl);

        // Threads with flag=1 write their column index at compacted position
        if (tid < active && flag) {
            zero_indices[base_out + running + excl] = col;
        }

        // Advance running total
        if (tid == 0) running += tile_sum;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_count[row] = running;
    }
}

bool solve_from_kl(float* d_C, int* d_X, float* d_U, float* d_V, int n, float* d_B, int* d_R, int* d_Q, int* k, int* l,int* d_col_to_row, int* d_indices, int* d_row_start, int* d_row_count, int* d_counter, int* d_row_visited) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    // update_Q<<<1,1>>>(d_Q, k, l);
    update_RQ<<<1,1>>>(d_R, d_Q, k, l, d_col_to_row);

    // if (n > 350){
    
        // int h_changed;
        // do {
        //     reset_d_changed<<<1,1>>>();
            
        //     solve_1bc_kernel<<<blocks, threads>>>(n, d_X, k, l, d_B, d_R, d_Q);
            
        //     cudaMemcpyFromSymbol(&h_changed, d_changed, sizeof(int), 0, cudaMemcpyDeviceToHost);

        // } while (h_changed == 1);

    cudaDeviceSynchronize();


    reset_device_flags<<<1,1>>>();
    count_done_entries<<<blocks, threads>>>(n, d_B);

    

    // printDeviceVar("d_done", d_done);

    // cudaDeviceSynchronize();
    // printDeviceMatrix("Before d_B", d_B, n);
    cudaMemset(d_counter, 0, sizeof(int));
    // collectZeroIndicesSingleKernel<<< (n + 256 - 1) / 256, 256>>>(d_B, n, d_indices, d_row_start, d_row_count, d_counter);
    collectZeroIndicesSharedMem<<<n, 512>>>(          d_B, n, d_indices, d_row_start, d_row_count);
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
    
    cudaDeviceSynchronize();

    solve_1bc_rowseq_async<<<n, 1>>>( n, d_col_to_row, d_indices, d_row_start, d_row_count, k, d_R, d_Q, d_row_visited);

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

    cudaDeviceSynchronize();
        
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
    

    // weird
    // int *d_queue, *d_q_head, *d_q_tail;
    // cudaMalloc(&d_queue, n * sizeof(int));
    // cudaMalloc(&d_q_head, sizeof(int));
    // cudaMalloc(&d_q_tail, sizeof(int));



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

    int* d_row_visited;
    cudaMalloc(&d_row_visited, n * sizeof(int));
    set_array_value<<<(n + 255)/256, 256>>>(d_row_visited, 0, n);

    find_most_negative<<<blocks, threads>>>(d_B, n, k, l);
    int steps = 0;
    while (true) {
        // std::cout << "Step " << steps << " \n";
        bool should_continue = solve_from_kl(d_C, d_X, d_U, d_V, n, d_B, d_R, d_Q, k, l, d_col_to_row, d_indices, d_row_start, d_row_count, d_counter, d_row_visited);
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