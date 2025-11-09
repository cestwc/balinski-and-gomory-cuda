#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <string>
#include <cstdio>
#include <math_constants.h>


#include "cuda_debug_utils.cuh"

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
void selectZerosOptimized(const float* d_B, int* d_out, int* d_count, int total)
{
    int blockSize = 1024;
    int numBlocks = (total + blockSize - 1) / blockSize;
    cudaMemset(d_count, 0, sizeof(int));
    size_t sharedBytes = blockSize * sizeof(int);
    selectZerosEfficient<<<numBlocks, blockSize, sharedBytes>>>(d_B, d_out, d_count, total);
    cudaDeviceSynchronize();
}



__global__ void compute_B(const float* C, const float* U, const float* V, float* B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) B[IDX2C(i, j, n)] = C[IDX2C(i, j, n)] - U[i] - V[j];
}

__device__  float d_min;
__device__  int d_changed;
__device__  float d_epsilon;
__device__  int d_found;
__device__  int d_flag;
__device__  int d_b_kl_neg;

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

__global__ void update_RQ(int* d_R, int* d_Q, const int* k, const int* l, int* d_col_to_row) {
    d_Q[*l] = *k;
    d_R[d_col_to_row[*l]] = *l;
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


__global__ void compute_col_to_row(int n, const int* X, int* col_to_row) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    if (X[IDX2C(i, j, n)] == 1) {
        col_to_row[j] = i;
    }
}


// __global__ void solve_1bc_rowseq_async_parallel_sb(
//     int n,
//     const int* __restrict__ d_col_to_row,
//     const int* __restrict__ zero_indices,
//     const int* __restrict__ row_start,
//     const int* __restrict__ row_count,
//     const int* k,
//     int* __restrict__ R,
//     int* __restrict__ Q
// ) {
//     // This version runs on a single block with many threads.
//     // The "queue" concept is implicit — we repeatedly scan until convergence.

//     // extern __shared__ int shmem[];
//     __shared__ int changed;

//     const int tid = threadIdx.x;
//     const int T   = blockDim.x;

//     // Only launch one block
//     if (blockIdx.x > 0) return;

//     do {
//         if (tid == 0) changed = 0;
//         __syncthreads();

//         // Each thread processes multiple rows
//         for (int i = tid; i < n; i += T) {
//             if (R[i] != n && i != *k) {
//                 int base = row_start[i];
//                 int nz   = row_count[i];
//                 for (int t = 0; t < nz; ++t) {
//                     int j = zero_indices[base + t];

//                     // Try to label this column j with row i
//                     if (atomicCAS(&Q[j], n, i) == n) {
//                         int r2 = d_col_to_row[j];
//                         R[r2] = j;
//                         changed = 1;  // mark that we updated something
//                     }
//                 }
//             }
//         }

//         __syncthreads();

//         // Continue until no changes were made in this pass
//     } while (changed);
// }




// __global__ void solve_1bc_rowseq_async_parallel_sba(
//     int n,
//     const int* __restrict__ d_col_to_row,
//     const int* __restrict__ zero_indices,
//     const int* __restrict__ row_start,
//     const int* __restrict__ row_count,
//     const int* k,
//     int* __restrict__ R,
//     int* __restrict__ Q
// ) {
//     // Single block, many threads. Asynchronous propagation by repeated sweeps.
//     if (blockIdx.x > 0) return;

//     __shared__ int changed;
//     const int tid = threadIdx.x;
//     const int T   = blockDim.x;

//     do {
//         if (tid == 0) changed = 0;
//         __syncthreads();

//         // Each thread processes multiple rows
//         for (int i = tid; i < n; i += T) {
//             // Process only rows that are currently labeled and not the special row *k
//             if (i != *k && R[i] != n) {
//                 const int base = row_start[i];
//                 const int nz   = row_count[i];

//                 // Walk this row's zeros
//                 for (int t = 0; t < nz; ++t) {
//                     const int j = zero_indices[base + t];

//                     // Try to claim column j for row i
//                     if (atomicCAS(&Q[j], n, i) == n) {
//                         const int r2 = d_col_to_row[j];

//                         // First-writer-wins on R as well to avoid races
//                         if (atomicCAS(&R[r2], n, j) == n) {
//                             // Mark that the sweep made progress
//                             atomicExch(&changed, 1);
//                         }
//                     }
//                 }
//             }
//         }

//         // Ensure all global writes to R/Q are visible to the whole block
//         __threadfence_block();
//         __syncthreads();

//         // Continue until no changes were made in this sweep
//     } while (changed);
// }

// Drop-in replacement: same signature, same inputs.
// Requires cooperative launch support (cg::this_grid()).

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Single device-side flag for grid-wide "did anything change?" tracking.
// This does NOT alter/repurpose any user input arrays.
// __device__ int d_changed; # reuse

// __global__ void solve_1bc_rowseq_async_parallel_cg(
//     int n,
//     const int* __restrict__ d_col_to_row,
//     const int* __restrict__ zero_indices,
//     const int* __restrict__ row_start,
//     const int* __restrict__ row_count,
//     const int* k,
//     int* __restrict__ R,
//     int* __restrict__ Q
// ) {
//     cg::grid_group grid = cg::this_grid();

//     const int tid = threadIdx.x;
//     const int T   = blockDim.x;
//     const int B   = gridDim.x;
//     const int bid = blockIdx.x;

//     while (true) {
//         // Reset change flag once per grid
//         if (bid == 0 && tid == 0)
//             d_changed = 0;

//         // A single sync is enough here
//         grid.sync();

//         // Each block handles one or more rows
//         for (int i = bid; i < n; i += B) {
//             if (i != *k && R[i] != n) {
//                 const int base = row_start[i];
//                 const int nz   = row_count[i];

//                 // Each thread covers several zeros
//                 for (int t = tid; t < nz; t += T) {
//                     const int j = zero_indices[base + t];
//                     // First-writer-wins on Q
//                     if (atomicCAS(&Q[j], n, i) == n) {
//                         const int r2 = d_col_to_row[j];
//                         // First-writer-wins on R
//                         if (atomicCAS(&R[r2], n, j) == n)
//                             d_changed = 1; // relaxed write, no atomicExch needed
//                     }
//                 }
//             }
//         }

//         // Ensure global writes visible before read
//         __threadfence();

//         // Grid-wide sync before check
//         grid.sync();

//         // Read global flag (cached in a register)
//         if (!d_changed)
//             break;

//         // Only need one sync before next iteration
//         grid.sync();
//     }
// }





// __device__ __forceinline__
// unsigned warp_exclusive_scan(unsigned v) {
//     unsigned lane = threadIdx.x & 31;
//     unsigned n = v;
//     // inclusive scan in 'n'
//     for (int offset = 1; offset < 32; offset <<= 1) {
//         unsigned y = __shfl_up_sync(0xffffffff, n, offset);
//         if (lane >= offset) n += y;
//     }
//     // convert to exclusive
//     return n - v;
// }

// ---------------------------------------------
// Block exclusive scan over the first n_active threads.
// - Each thread provides 'flag' (0/1) and receives 'excl' (exclusive prefix).
// - Returns tile_sum (sum of flags over n_active threads) via shared memory.
// - Works for arbitrary n_active <= blockDim.x.
// ---------------------------------------------
// __device__ __forceinline__
// unsigned block_exclusive_scan_0_1(unsigned flag,
//                                   int n_active,
//                                   unsigned &excl) {
//     // Per-warp exclusive scan
//     unsigned lane   = threadIdx.x & 31;
//     unsigned warpId = threadIdx.x >> 5;

//     unsigned excl_warp = warp_exclusive_scan(flag);
//     unsigned incl_warp = excl_warp + flag;

//     // Number of warps that have active threads
//     int numWarps = (n_active + 31) >> 5;

//     // Shared arrays to propagate warp sums
//     __shared__ unsigned warp_sums[32];    // inclusive sum of each warp
//     __shared__ unsigned warp_offsets[32]; // exclusive prefix sum for warps

//     // Initialize to zero to avoid stale values for inactive warps
//     if (lane == 0) warp_sums[warpId] = 0;
//     __syncthreads();

//     // Identify last active lane in this warp
//     int warp_base = warpId * 32;
//     int last_lane = min(31, max(0, n_active - warp_base - 1));
//     bool is_last_in_warp = (lane == last_lane) && (warpId < (unsigned)numWarps);

//     // The last active lane in each warp publishes its INCLUSIVE sum
//     if (is_last_in_warp) {
//         warp_sums[warpId] = incl_warp;
//     }
//     __syncthreads();

//     // First warp scans the warp sums (numWarps entries)
//     unsigned warp_excl = 0;
//     if (warpId == 0) {
//         unsigned wval = (lane < (unsigned)numWarps) ? warp_sums[lane] : 0;
//         unsigned wexcl = warp_exclusive_scan(wval);
//         if (lane < (unsigned)numWarps) warp_offsets[lane] = wexcl;
//     }
//     __syncthreads();

//     // Each thread's block-exclusive prefix = its warp-exclusive + offset of previous warps
//     unsigned block_offset = (warpId < (unsigned)numWarps) ? warp_offsets[warpId] : 0;
//     excl = (threadIdx.x < n_active) ? (excl_warp + block_offset) : 0;

//     // Tile sum is the last warp's inclusive sum + offset of previous warps.
//     // Let the last active thread in the block compute it:
//     __shared__ unsigned tile_sum;
//     if (threadIdx.x == n_active - 1) {
//         unsigned last_warp = (n_active - 1) >> 5;
//         tile_sum = warp_sums[last_warp] + warp_offsets[last_warp];
//     }
//     __syncthreads();

//     return tile_sum; // valid for all threads after the sync
// }

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

// __global__ void collectZeroIndicesSharedMem(const float* __restrict__ B,
//                                             int n,
//                                             int* __restrict__ zero_indices,
//                                             int* __restrict__ row_start,
//                                             int* __restrict__ row_count)
// {
//     // Each block handles multiple rows if gridDim.x < n
//     for (int row = blockIdx.x; row < n; row += gridDim.x) {

//         const int base_out = row * n;

//         if (threadIdx.x == 0)
//             row_start[row] = base_out;
//         __syncthreads();

//         // ---------------------------------------
//         // Pass 1: Each thread counts its own zeros
//         // ---------------------------------------
//         unsigned local_count = 0;
//         for (int col = threadIdx.x; col < n; col += blockDim.x) {
//             if (B[IDX2C(row, col, n)] == 0)
//                 local_count++;
//         }

//         // Shared counters
//         __shared__ unsigned prefix_counter;
//         __shared__ unsigned total_zeros;

//         if (threadIdx.x == 0)
//             prefix_counter = 0;
//         __syncthreads();

//         // Each thread reserves its segment
//         unsigned thread_base = atomicAdd(&prefix_counter, local_count);
//         __syncthreads();

//         if (threadIdx.x == 0)
//             total_zeros = prefix_counter;
//         __syncthreads();

//         // ---------------------------------------
//         // Pass 2: Each thread writes its zero indices
//         // ---------------------------------------
//         unsigned offset = 0;
//         for (int col = threadIdx.x; col < n; col += blockDim.x) {
//             if (B[IDX2C(row, col, n)] == 0) {
//                 zero_indices[base_out + thread_base + offset] = col;
//                 ++offset;
//             }
//         }
//         __syncthreads();

//         if (threadIdx.x == 0)
//             row_count[row] = total_zeros;
//         __syncthreads();
//     }
// }



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


bool solve_from_kl(float* d_C, int* d_X, float* d_U, float* d_V, int n, float* d_B, int* d_R, int* d_Q, int* k, int* l,int* d_col_to_row, int* d_indices, int* d_count) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    compute_col_to_row<<<blocks, threads>>>(n, d_X, d_col_to_row);
    // update_Q<<<1,1>>>(d_Q, k, l);
    update_RQ<<<1,1>>>(d_R, d_Q, k, l, d_col_to_row);
    reset_d_changed<<<1,1>>>();         

    selectZerosOptimized(d_B, d_indices, d_count, n * n);
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
    set_array_value<<<(n + 255)/256, 256>>>(d_R, n, n);
    set_array_value<<<(n + 255)/256, 256>>>(d_Q, n, n);

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