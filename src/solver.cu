#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     // shared memory flag for this block
//     __shared__ bool block_changed;

//     do {
//         if (threadIdx.x == 0 && threadIdx.y == 0)
//             block_changed = false;
//         __syncthreads();

//         int i = blockIdx.y * blockDim.y + threadIdx.y;
//         int j = blockIdx.x * blockDim.x + threadIdx.x;

//         if (i < n && j < n) {
//             // Step (b): one thread per column
//             if (i == 0 && Q[j] != n) {
//                 int row = col_to_row[j];
//                 if (R[row] == n) {
//                     R[row] = j;
//                     block_changed = true;
//                 }
//             }

//             // Step (c): one thread per (i, j)
//             if (i != k && R[i] != n && Q[j] == n) {
//                 float b_val = B[IDX2C(i, j, n)];
//                 if (b_val == 0.0f) {
//                     if (atomicMin(&Q[j], i) > i) {
//                         block_changed = true;
//                     }
//                 }
//             }
//         }

//         __syncthreads();

//         // If any thread in the grid changed, repeat
//         // Grid-wide reduction: use atomic OR into global flag
//         __shared__ bool any_changed;
//         if (threadIdx.x == 0 && threadIdx.y == 0) {
//             if (block_changed) {
//                 atomicExch((int*)&any_changed, 1);
//             }
//         }
//         __syncthreads();

//         if (!any_changed) break;

//         // Reset any_changed for next loop
//         if (threadIdx.x == 0 && threadIdx.y == 0)
//             any_changed = false;
//         __syncthreads();

//     } while (true);
// }

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q,
//     int* d_changed,
//     int* d_waiting
// ) {
    
//     if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//     *d_changed = 0;
//     *d_waiting = 0;
//     }
//     __syncthreads(); 

//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= n || j >= n) return;

//     // printf("Thread %d: arr[%d] = %d\n", *d_waiting, *d_changed, n * n);
//     int local_change = 0;
//     int local_step = 0;

//     while (true) {

//         // Step (b): one thread per column
//         float b_val = B[IDX2C(i, j, n)];

//         bool cond1 = Q[j] != n && i == col_to_row[j] && R[i] == n;
//         bool cond2 = i != k && R[i] != n && Q[j] == n && b_val == 0.0f;
//         // printf("Thread %d: arr[%d] = %d\n",i, j, n * n);

//         // printf("d waiting %d: d change[%d] = %d, and localstep %d\n", *d_waiting, *d_changed, n * n, local_step);


//         if (!cond1 && !cond2){
//             if (local_change != 2){                
//                 atomicAdd(d_waiting, 1);
//             }
//             local_change = 2;
//             if (*d_waiting + *d_changed == n * n){
//                 local_step++;
//             }
//             if (local_step == 3){
//                 break;
//             }
//         } else {
//             if (local_change == 2){
//                 atomicAdd(d_waiting, -1);
//                 __syncthreads();
//             }
//             if (cond1){
//                 R[i] = j;
//             }
//             if (cond2){
//                 Q[j] = i;
//             }
//             local_change = 1;
//             atomicAdd(d_changed, 1);
//             __syncthreads();
//             break;
//         }
        

//     }
// }
// __device__ int d_round_done;    // how many blocks finished
// __device__ int d_continue_flag; // 1 = keep going, 0 = stop

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q,
//     int* d_changed
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     while (true) {
//         // --- step 1: reset counters ---
//         if (blockIdx.x == 0 && blockIdx.y == 0 &&
//             threadIdx.x == 0 && threadIdx.y == 0) {
//             *d_changed   = 0;
//             d_round_done = 0;
//             d_continue_flag = 1; // assume continue
//         }
//         __syncthreads(); // per-block barrier

//         // --- step 2: local work ---
//         if (i < n && j < n) {
//             float b_val = B[IDX2C(i, j, n)];

//             bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//             bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//             if (cond1) {
//                 R[i] = j;
//                 atomicAdd(d_changed, 1);
//             } else if (cond2) {
//                 Q[j] = i;
//                 atomicAdd(d_changed, 1);
//             }
//         }
//         __syncthreads();

//         // --- step 3: mark block as finished ---
//         if (threadIdx.x == 0 && threadIdx.y == 0) {
//             atomicAdd(&d_round_done, 1);
//         }
//         __syncthreads();

//         // --- step 4: global decision by one thread ---
//         if (blockIdx.x == 0 && blockIdx.y == 0 &&
//             threadIdx.x == 0 && threadIdx.y == 0) {
//             // spin until all blocks finished
//             while (d_round_done < gridDim.x * gridDim.y) { }

//             if (*d_changed == 0) {
//                 d_continue_flag = 0; // converged
//             }
//         }
//         __syncthreads();

//         // --- step 5: check flag ---
//         if (d_continue_flag == 0) break;
//     }
// }

// __device__ int d_continue_flag; // 1 = keep going, 0 = stop

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q,
//     int* d_changed
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     while (true) {
//         // --- step 1: reset convergence counter ---
//         if (blockIdx.x == 0 && blockIdx.y == 0 &&
//             threadIdx.x == 0 && threadIdx.y == 0) {
//             *d_changed     = 0;
//             d_continue_flag = 1; // assume we need another round
//             __threadfence();     // make visible
//         }
//         __syncthreads();

//         // --- step 2: local work (one thread handles one (i,j)) ---
//         if (i < n && j < n) {
//             float b_val = B[IDX2C(i, j, n)];

//             bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//             bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//             if (cond1) {
//                 R[i] = j;
//                 atomicAdd(d_changed, 1);
//             } else if (cond2) {
//                 Q[j] = i;
//                 atomicAdd(d_changed, 1);
//             }
//         }
//         __syncthreads();

//         // --- step 3: one block decides convergence ---
//         if (blockIdx.x == 0 && blockIdx.y == 0 &&
//             threadIdx.x == 0 && threadIdx.y == 0) {
//             if (*d_changed == 0) {
//                 d_continue_flag = 0;
//                 __threadfence();
//             }
//         }
//         __syncthreads();

//         // --- step 4: check flag ---
//         if (d_continue_flag == 0) break;
//     }
// }


// #define IDX2C(i,j,n) ((i) + (j)*(n))

// __device__ int d_changed;   // global "did anything change" flag

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,   // row assignments
//     int* Q    // column assignments
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= n || j >= n) return;

//     int stable_passes = 0;

//     while (true) {
//         // --- reset global flag once per pass ---
//         if (i == 0 && j == 0) {
//             d_changed = 0;
//             __threadfence();  // ensure visibility
//         }
//         __syncthreads();

//         // --- local work ---
//         float b_val = B[IDX2C(i, j, n)];

//         bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//         bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//         if (cond1) {
//             int old = atomicCAS(&R[i], n, j);
//             if (old == n) atomicExch(&d_changed, 1);
//         }
//         else if (cond2) {
//             int old = atomicCAS(&Q[j], n, i);
//             if (old == n) atomicExch(&d_changed, 1);
//         }

//         __syncthreads();

//         // --- convergence check with delay ---
//         if (d_changed == 0) {
//             stable_passes++;
//         } else {
//             stable_passes = 0;
//         }

//         if (stable_passes >= 2) {
//             // two consecutive passes with no updates
//             break;
//         }

//         // otherwise, go again
//     }
// }


// __device__ int d_changed;   // global flag for this round

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     cg::grid_group grid = cg::this_grid();

//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     while (true) {
//         // --- reset flag once per round ---
//         if (i == 0 && j == 0) {
//             d_changed = 0;
//         }
//         grid.sync();  // all threads see reset

//         // --- local work ---
//         if (i < n && j < n) {
//             float b_val = B[IDX2C(i, j, n)];

//             bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//             bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//             if (cond1) {
//                 int old = atomicCAS(&R[i], n, j);
//                 if (old == n) atomicExch(&d_changed, 1);
//             }
//             else if (cond2) {
//                 int old = atomicCAS(&Q[j], n, i);
//                 if (old == n) atomicExch(&d_changed, 1);
//             }
//         }

//         grid.sync();  // wait for all updates

//         // --- stop condition ---
//         if (d_changed == 0) {
//             break;  // no changes this round → converge
//         }

//         // otherwise, repeat
//         grid.sync();
//     }
// }


// __device__ int d_changed;   // global flag for each round

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     cg::grid_group grid = cg::this_grid();

//     int i = blockIdx.y * blockDim.y + threadIdx.y; // rows
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // columns

//     while (true) {
//         // reset change flag once per round
//         if (i == 0 && j == 0) {
//             d_changed = 0;
//         }
//         grid.sync();

//         // ---- Step (b): one thread per column ----
//         if (i == 0 && j < n && Q[j] != n) {
//             int row = col_to_row[j];
//             if (R[row] == n) {
//                 R[row] = j;
//                 atomicExch(&d_changed, 1);
//             }
//         }

//         grid.sync();  // make sure step (b) updates are visible

//         // ---- Step (c): one thread per (i, j) ----
//         if (i < n && j < n) {
//             if (i != k && R[i] != n && Q[j] == n) {
//                 float b_val = B[IDX2C(i, j, n)];
//                 if (b_val == 0.0f) {
//                     if (atomicMin(&Q[j], i) > i) {
//                         atomicExch(&d_changed, 1);
//                     }
//                 }
//             }
//         }

//         grid.sync();

//         // ---- global convergence check ----
//         if (d_changed == 0) {
//             break;  // nothing changed in this round → stop
//         }

//         grid.sync(); // prepare for next round
//     }
// }


// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= n || j >= n) return;

//     // --- Structural pruning (masking stage) ---

//     // cond1 possible? (column-driven)
//     bool cond1_possible = false;
//     if (i == col_to_row[j]) {
//         for (int ii = 0; ii < n; ii++) {
//             if (ii != k && B[IDX2C(ii,j,n)] == 0.0f) {
//                 cond1_possible = true;
//                 break;
//             }
//         }
//     }

//     // cond2 possible? (row-driven)
//     bool cond2_possible = false;
//     if (i != k && B[IDX2C(i,j,n)] == 0.0f) {
//         for (int jj = 0; jj < n; jj++) {
//             if (B[IDX2C(i,jj,n)] == 0.0f) {
//                 cond2_possible = true;
//                 break;
//             }
//         }
//     }

//     if (!(cond1_possible || cond2_possible)) {
//         // impossible forever → exit immediately
//         return;
//     }

//     // --- Runtime activation loop ---
//     while (true) {
//         // cond1: column j assigns its row
//         if (i == col_to_row[j] && Q[j] != n && R[i] == n) {
//             int old = atomicCAS(&R[i], n, j);
//             if (old == n) {
//                 // success
//             }
//             return; // cond1 is one-shot
//         }

//         // cond2: row i assigns to column j
//         if (i != k && R[i] != n && Q[j] == n && B[IDX2C(i,j,n)] == 0.0f) {
//             int old = atomicMin(&Q[j], i);
//             if (old > i) {
//                 // success
//             }
//             return; // cond2 is one-shot
//         }

//         // exit if permanently impossible
//         if (Q[j] != n && i != col_to_row[j]) return;  // column taken, not my row
//         if (R[i] == n && i == k) return;              // excluded row

//         // otherwise, prerequisites not ready yet → spin until they are
//     }
// }

// #define IDX2C(i,j,n) ((i) + (j)*(n))

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // row
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // col

//     if (i >= n || j >= n) return;

//     // -----------------------
//     // Step 1: structural pruning
//     // -----------------------

//     bool cond1_possible = false;
//     if (i == col_to_row[j]) {
//         // column j viable? → must have at least one zero (≠ k)
//         for (int ii = 0; ii < n; ii++) {
//             if (ii != k && B[IDX2C(ii, j, n)] == 0.0f) {
//                 cond1_possible = true;
//                 break;
//             }
//         }
//     }

//     bool cond2_possible = false;
//     if (i != k && B[IDX2C(i, j, n)] == 0.0f) {
//         // row i must have at least TWO zeros to make Step (c) useful
//         int zeroCount = 0;
//         for (int jj = 0; jj < n; jj++) {
//             if (B[IDX2C(i, jj, n)] == 0.0f) {
//                 zeroCount++;
//                 if (zeroCount >= 2) break;
//             }
//         }
//         if (zeroCount >= 2) {
//             cond2_possible = true;
//         }
//     }


//     if (!(cond1_possible || cond2_possible)) {
//         // impossible forever → exit immediately
//         return;
//     }

//     // -----------------------
//     // Step 2: runtime loop
//     // -----------------------

//     while (true) {
//         // --- cond1: column j assigns its row ---
//         if (i == col_to_row[j] && Q[j] != n && R[i] == n) {
//             // int old = atomicCAS(&R[i], n, j);
//             // if (old == n) {
//             //     // success: row i claimed by column j
//             // }
//             R[i] = j;
//             return; // cond1 is one-shot
//         }

//         // --- cond2: row i assigns itself to column j ---
//         if (i != k && R[i] != n && Q[j] == n && B[IDX2C(i, j, n)] == 0.0f) {
//             // int old = atomicMin(&Q[j], i);
//             // if (old > i) {
//             //     // success: row i claimed column j
//             // }
//             Q[j] = i;
//             return; // cond2 is one-shot
//         }

//         // --- exit if permanently impossible ---
//         if (Q[j] != n && i != col_to_row[j]) {
//             // column j already taken, and I’m not its designated row
//             return;
//         }
//         if (i == k) {
//             // excluded row never participates
//             return;
//         }

//         if (R[i] != n) {
//             // excluded row never participates
//             return;
//         }

//         // else: prerequisites not ready yet → wait
//         // __nanosleep(50); // backoff to reduce contention
//     }
// }

// #define IDX2C(i,j,n) ((i) + (j)*(n))

// row_done[i] = 0 → row still active
// row_done[i] = 1 → row has already succeeded once in Step (c)
// __device__ int* row_done;

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q,
//     // int* row_done // allocated length n, zero-initialized before launch
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // row
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // col

//     if (i >= n || j >= n) return;

//     // -----------------------
//     // Step 1: structural pruning
//     // -----------------------

//     bool cond1_possible = false;
//     if (i == col_to_row[j]) {
//         // column j viable? must have ≥ 1 zero (ignoring row k)
//         for (int ii = 0; ii < n; ii++) {
//             if (ii != k && B[IDX2C(ii, j, n)] == 0.0f) {
//                 cond1_possible = true;
//                 break;
//             }
//         }
//     }

//     bool cond2_possible = false;
//     if (i != k && B[IDX2C(i, j, n)] == 0.0f) {
//         // row i viable? must have ≥ 2 zeros
//         int zeroCount = 0;
//         for (int jj = 0; jj < n; jj++) {
//             if (B[IDX2C(i, jj, n)] == 0.0f) {
//                 zeroCount++;
//                 if (zeroCount >= 2) break;
//             }
//         }
//         if (zeroCount >= 2) {
//             cond2_possible = true;
//         }
//     }

//     if (!(cond1_possible || cond2_possible)) {
//         // impossible forever → exit immediately
//         return;
//     }

//     // -----------------------
//     // Step 2: runtime loop
//     // -----------------------

//     while (true) {
//         // --- cond1: column j assigns its row ---
//         if (i == col_to_row[j] && Q[j] != n && R[i] == n) {
//             int old = atomicCAS(&R[i], n, j);
//             if (old == n) {
//                 // success: row i claimed by column j
//             }
//             return; // cond1 is one-shot
//         }

//         // --- cond2: row i assigns itself to column j ---
//         if (i != k && R[i] != n && Q[j] == n && B[IDX2C(i, j, n)] == 0.0f) {
//             if (atomicMin(&Q[j], i) > i) {
//                 // success: row i claimed at least one column
//                 atomicExch(&row_done[i], 1);
//             }
//             return; // this thread’s job is finished
//         }

//         // --- early exit if row already succeeded elsewhere ---
//         if (row_done[i] == 1) {
//             return; // another (i,j') thread succeeded → exit
//         }

//         // --- exit if permanently impossible ---
//         if (Q[j] != n && i != col_to_row[j]) {
//             return; // column j already taken, and I’m not its designated row
//         }
//         if (i == k) {
//             return; // excluded row never participates
//         }

//         // else: prerequisites not ready yet → wait
//         __nanosleep(50); // backoff to reduce contention
//     }
// }

//efficient impossibility mask
// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // row
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // col

//     if (i >= n || j >= n) return;

//     // -----------------------
//     // Step 1: static pruning
//     // -----------------------

//     if (i == 0 && j == 0) {
//         for (int w = 0; w < n; w++) {
//             printf("R[%d] = %d\n", w, R[w]);
//         }
//         for (int w = 0; w < n; w++) {
//             printf("Q[%d] = %d\n", w, Q[w]);
//         }
//     }
//     __syncthreads();

//     if (!B[IDX2C(i, j, n)] == 0.0f || i == k) return;
//     __syncthreads();
//     // bool cond1_possible = false;
//     // if (i == col_to_row[j]) {
//     //     // column j viable? at least one row (≠ k) with B[ii,j] == 0
//     //     for (int ii = 0; ii < n; ii++) {
//     //         if (ii != k && B[IDX2C(ii, j, n)] == 0.0f) {
//     //             cond1_possible = true;
//     //             break;
//     //         }
//     //     }
//     // }

//     // bool cond2_possible = false;
//     // if (i != k && B[IDX2C(i, j, n)] == 0.0f) {
//     //     // row i viable? must have ≥ 2 zeros
//     //     int zeroCount = 0;
//     //     for (int jj = 0; jj < n; jj++) {
//     //         if (B[IDX2C(i, jj, n)] == 0.0f) {
//     //             zeroCount++;
//     //             if (zeroCount >= 2) break;
//     //         }
//     //     }
//     //     if (zeroCount >= 2) {
//     //         cond2_possible = true;
//     //     }
//     // }

//     // if (!(cond1_possible || cond2_possible)) {
//     //     // impossible forever → exit immediately
//     //     return;
//     // }

//     // -----------------------
//     // Step 2: runtime loop
//     // -----------------------
    
//     while (true) {
        
//         // --- recompute conditions ---
//         __syncthreads();
//         if (Q[j] != n || R[i] != n){
//             printf("Available i = %d, j = %d\n", i, j);
//             if (i == col_to_row[j] && R[i] == n){
//                 // atomicCAS(&R[i], n, j);
//                 R[i] = j;
//                 printf("labelling R[%d] = %d\n", i, R[i]);
//             }
//             if (Q[j] == n){
//                 // atomicMin(&Q[j], i);
//                 Q[j] = i;
//                 printf("labelling Q[%d] = %d\n", j, Q[j]);
//             }

            

//             // if (threadIdx.x == 0 && blockIdx.x == 0) {
//             //     for (int w = 0; w < n; w++) {
//             //         printf("R[%d] = %d\n", w, R[w]);
//             //     }
//             //     for (int w = 0; w < n; w++) {
//             //         printf("Q[%d] = %d\n", w, Q[w]);
//             //     }
//             // }
            
//             return;
//         } else {
//             int zero_in_col = 0;
//             int zero_in_row = 0;
//             for (int jj = 0; jj < n; jj++) {
//                 if (B[IDX2C(i, jj, n)] == 0.0f) {
//                     zero_in_col++;
//                     if (zero_in_col >= 2) break;
//                 }
//             }

//             for (int ii = 0; ii < n; ii++) {
//                 if (B[IDX2C(ii, j, n)] == 0.0f) {
//                     zero_in_row++;
//                     if (zero_in_row >= 2) break;
//                 }
//             }

//             if (zero_in_row == 1 && zero_in_col == 1) return;
//         }
//         // bool can_do_cond1 = (i == col_to_row[j] && Q[j] != n && R[i] == n);
//         // bool can_do_cond2 = (i != k && R[i] != n && Q[j] == n && 
//         //                      B[IDX2C(i, j, n)] == 0.0f);

//         // // --- perform action if possible ---
//         // if (can_do_cond1) {
//         //     atomicCAS(&R[i], n, j);  // claim row i for col j
//         //     return; // cond1 is one-shot
//         // }

//         // if (can_do_cond2) {
//         //     atomicMin(&Q[j], i);     // try to claim column j
//         //     return; // cond2 is one-shot
//         // }

//         // // --- check if permanently impossible now ---
//         // if (R[i] != n || Q[j] != n){
//         //     return;
//         // }
//         // if (Q[j] != n){
//         //     cond2_possible = false;
//         // }
//         // bool cond1_forever_false = (i != col_to_row[j]) || (Q[j] == n);
//         // bool cond2_forever_false = (i == k) || (B[IDX2C(i, j, n)] != 0.0f) 
//         //                            || (Q[j] != n);

//         // if (cond1_forever_false && cond2_forever_false) {
//         //     return; // no hope left
//         // }

//         // // otherwise, still possible → wait and retry
//         // __nanosleep(50);
//     }
// }

// global flags
// d_flags[0] = changed count
// d_flags[1] = waiting count
// d_flags[2] = blocks finished
// d_flags[3] = continue flag
// __device__ int d_active_workers = 0;
// __device__ int d_stop_flag = 0;


// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // row
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // col

//     if (i >= n || j >= n) return;

//     while (atomicAdd(&d_stop_flag, 0) == 0) {   // check stop flag

//         float b_val = B[IDX2C(i, j, n)];
//         bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//         bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//         if (cond1 || cond2) {
//             atomicAdd(&d_active_workers, 1);

//             if (cond1) {
//                 R[i] = j;
//             } else if (cond2) {
//                 atomicMin(&Q[j], i);
//             }

//             atomicSub(&d_active_workers, 1);
//         }

//         __syncthreads();

//         // if (threadIdx.x == 0 && threadIdx.y == 0 &&
//         //     blockIdx.x == 0 && blockIdx.y == 0) {

//             if (d_active_workers == 0) {
//                 d_stop_flag = 1;
//             }
//         // }

//         __syncthreads();
//     }
// }

// __device__ int d_active_workers = 0;
// __device__ int d_progress = 0;
// __device__ int d_stop_flag = 0;

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n || j >= n) return;

//     while (atomicAdd(&d_stop_flag, 0) == 0) {
//         float b_val = B[IDX2C(i, j, n)];
//         bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//         bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//         if (cond1 || cond2) {
//             // Mark active
//             atomicAdd(&d_active_workers, 1);
//             atomicExch(&d_progress, 1);

//             if (cond1) {
//                 R[i] = j;
//             } else if (cond2) {
//                 atomicMin(&Q[j], i);
//             }

//             // Done
//             atomicSub(&d_active_workers, 1);
//         }

//         // A single "watchdog" thread decides on stop
//         if (i == 0 && j == 0) {
//             if (d_active_workers == 0 && d_progress == 0) {
//                 d_stop_flag = 1;
//             }
//             d_progress = 0; // reset heartbeat
//         }
//     }
// }

// __device__ int d_stop_flag = 0;
// __device__ int d_token = 0;   // token value

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;

//     bool active = false;

//     while (atomicAdd(&d_stop_flag, 0) == 0) {
//         // do some local work
//         float b_val = B[IDX2C(tid, tid % n, n)];
//         if (R[tid] == n && b_val != 0.0f) {
//             R[tid] = tid; // fake work
//             active = true;
//         }

//         // token passing (simplified linear chain)
//         if (tid == d_token % (gridDim.x * blockDim.x)) {
//             if (active) {
//                 // mark that some work happened
//                 active = false;
//                 d_token++;
//             } else {
//                 // if token made full round with no work -> stop
//                 if (d_token >= gridDim.x * blockDim.x) {
//                     d_stop_flag = 1;
//                 } else {
//                     d_token++;
//                 }
//             }
//         }
//     }
// }

// __device__ int d_outstanding = 0;
// __device__ int d_stop_flag   = 0;

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n || j >= n) return;

//     if (i == 0 && j == 0) {
//         for (int w = 0; w < n; w++) {
//             printf("R[%d] = %d\n", w, R[w]);
//         }
//         for (int w = 0; w < n; w++) {
//             printf("Q[%d] = %d\n", w, Q[w]);
//         }
//     }

//     while (atomicAdd(&d_stop_flag, 0) == 0) {
//         float b_val = B[IDX2C(i, j, n)];
//         bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//         bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//         if (cond1 || cond2) {
//             //
//             // 1. Claim responsibility for this unit of work
//             //
//             atomicAdd(&d_outstanding, 1);

//             //
//             // 2. Do the work
//             //
//             if (cond1) {
//                 R[i] = j;
//             } else if (cond2) {
//                 // Publishing new work safely:
//                 //   increment has already been done above!
//                 atomicMin(&Q[j], i);
//             }

//             //
//             // 3. Finished the work (including publishing any new work)
//             //
//             atomicSub(&d_outstanding, 1);
//         }

//         //
//         // 4. Global quiescence check (one watchdog thread)
//         //
//         if (i == 0 && j == 0) {
//             if (d_outstanding == 0) {
//                 d_stop_flag = 1;
//             }
//         }
//     }
// }
__device__ int d_outstanding = 0;
__device__ int d_progress    = 0;
__device__ int d_stop_flag   = 0;
__device__ int d_moving = 1;

__global__ void reset_globals() {
    d_outstanding = 0;
    d_progress    = 0;
    d_stop_flag   = 0;
    d_moving = 1;
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

    // d_outstanding = 0;
    // d_progress    = 0;
    // d_stop_flag   = 0;
    // d_moving = 1;

    int contributing = 1;
    int step = 0;
    int doing = 0;
    int iteration;
    int accumulate;
    int local_move = 0;

    do {
        // if (i == 0 && j == 0) {
        //     for (int w = 0; w < n; w++) {
        //         printf("R[%d] = %d\n", w, R[w]);
        //     }
        //     for (int w = 0; w < n; w++) {
        //         printf("Q[%d] = %d\n", w, Q[w]);
        //     }
        // }
        __threadfence();
        
        float b_val = B[IDX2C(i, j, n)];
        bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
        bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

        if (cond1 || cond2) {
            // Claim unit of work
            // atomicAdd(&d_outstanding, 1);
            // atomicExch(&d_progress, 1);

            if (cond1) {
                R[i] = j;
            } else if (cond2) {
                atomicMin(&Q[j], i);
            }

            doing = 1;

            // Done
            // atomicSub(&d_outstanding, 1);

        } else {
            doing = 0;
        }

        // printf("Contributing? (%d) Doing? (%d)\n", contributing, doing * contributing);

        int old_p = atomicAdd(&d_progress, contributing);
        int old_o = atomicAdd(&d_outstanding, (1 - doing) * contributing);
        __threadfence();

        iteration = (old_p + contributing) / (n * n);
        accumulate = (old_p + contributing) % (n * n);
        // printf("All d_progress? (%d) \n", old_p);
        // printf("accumulate? (%d) \n", accumulate);
        // printf("Total iteration? (%d) This accumulate? (%d)\n", iteration, accumulate);
        // printf("My iteration state? (%d) \n", step);

        // printf("My moving state? (%d) Should move on? (%d)\n", local_move, atomicAdd(&d_moving, 0));
        if (iteration > step && d_moving > local_move){
            contributing = 1;
            step++;
            local_move++;
            // printf("I am here! (%d) \n", step);
        } else {
            contributing = 0;
        }


        if (accumulate == 0){
            int lastout = atomicExch(&d_outstanding, 0);
            // printf("I am there! (%d) \n", lastout);
            __threadfence();
            if (lastout == n * n){
                atomicAdd(&d_stop_flag, 1);
            }
            atomicAdd(&d_moving, 1);
        }

    } while (atomicAdd(&d_stop_flag, 0) == 0);

}


// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n || j >= n) return;

//     while (atomicAdd(&d_stop_flag, 0) == 0) {
//         float b_val = B[IDX2C(i, j, n)];
//         bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//         bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//         if (cond1 || cond2) {
//             // Claim unit of work
//             atomicAdd(&d_outstanding, 1);
//             atomicExch(&d_progress, 1);

//             if (cond1) {
//                 R[i] = j;
//             } else if (cond2) {
//                 atomicMin(&Q[j], i);
//             }

//             // Done
//             atomicSub(&d_outstanding, 1);
//         }

//         // One watchdog thread checks
//         if (i == 0 && j == 0) {
//             if (d_outstanding == 0) {
//                 if (atomicAdd(&d_progress, 0) == 0) {
//                     d_stop_flag = 1;   // stable quiescence
//                 }
//                 atomicExch(&d_progress, 0);   // reset heartbeat
//             }
//         }
//     }
// }



// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q
// ) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // rows
//     int j = blockIdx.x * blockDim.x + threadIdx.x; // columns

//     if (i >= n || j >= n) return;

//     // ---- Step (b): one thread per column ----
//     if (i == 0) {
//         if (Q[j] != n) {
//             int row = col_to_row[j];
//             if (R[row] == n) {
//                 // claim the row for this column
//                 atomicCAS(&R[row], n, j);
//             }
//         }
//         return; // column threads are one-shot
//     }

//     // ---- Step (c): one thread per (i, j) ----
//     // prune threads that can never contribute
//     if (i == k || B[IDX2C(i,j,n)] != 0.0f) {
//         return; // impossible to activate
//     }

//     // dependency-based loop
//     while (true) {
//         if (Q[j] != n) {
//             // column already claimed → this (i,j) can't do anything
//             return;
//         }
//         if (R[i] != n) {
//             // row is assigned → try to claim column j
//             int old = atomicMin(&Q[j], i);
//             // if old > i, this thread wins, otherwise another smaller i wins
//             return; // after one attempt, done
//         }
//         // else: row[i] is still unassigned, keep waiting
//     }
// }
// #define IDX2C(i,j,n) ((i) + (j)*(n))

// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* col_to_row,
//     int k,
//     const float* B,
//     int* R,
//     int* Q,
//     int* d_changed
// ) {
//     // thread coordinates
//     int i0 = blockIdx.y * blockDim.y + threadIdx.y;
//     int j0 = blockIdx.x * blockDim.x + threadIdx.x;

//     // strides (how far to jump in each loop step)
//     int stride_i = blockDim.y * gridDim.y;
//     int stride_j = blockDim.x * gridDim.x;

//     // per-block continue flag
//     __shared__ int local_continue;
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         local_continue = 1;
//     }
//     __syncthreads();

//     while (local_continue) {
//         // reset global counter
//         if (i0 == 0 && j0 == 0) {
//             *d_changed = 0;
//         }
//         __syncthreads();

//         // grid-stride over all (i,j)
//         for (int i = i0; i < n; i += stride_i) {
//             for (int j = j0; j < n; j += stride_j) {
//                 float b_val = B[IDX2C(i, j, n)];

//                 bool cond1 = (Q[j] != n && i == col_to_row[j] && R[i] == n);
//                 bool cond2 = (i != k && R[i] != n && Q[j] == n && b_val == 0.0f);

//                 if (cond1) {
//                     R[i] = j;
//                     atomicAdd(d_changed, 1);
//                 } else if (cond2) {
//                     Q[j] = i;
//                     atomicAdd(d_changed, 1);
//                 }
//             }
//         }

//         __syncthreads();

//         // decide if another iteration is needed
//         if (threadIdx.x == 0 && threadIdx.y == 0) {
//             local_continue = (*d_changed > 0);
//         }
//         __syncthreads();
//     }
// }




// Cooperative kernel for solve_1bc
// __global__ void solve_1bc_kernel_full(
//     int n,
//     const int* __restrict__ col_to_row,
//     int k,
//     const float* __restrict__ B,
//     int* R,
//     int* Q,
//     int* d_changed
// ) {
//     // Create a grid-wide group
//     cg::grid_group grid = cg::this_grid();

//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     while (true) {
//         // reset the global flag ONCE per iteration
//         if (grid.thread_rank() == 0) {
//             *d_changed = 0;
//         }
//         grid.sync(); // make sure all threads see reset

//         bool local_change = false;

//         if (i < n && j < n) {
//             // --- Step (b): one thread per column ---
//             if (i == 0 && Q[j] != n) {
//                 int r = col_to_row[col];
//                 if (R[r] == n && r == row) {
//                     R[r] = col;
//                     local_change = true;
//                 }
//             }

//             // --- Step (c): one thread per (i,j) ---
//             if (row != k && R[row] != n && Q[col] == n) {
//                 float b_val = B[IDX2C(row, col, n)];
//                 if (b_val == 0.0f) {
//                     if (atomicMin(&Q[col], row) > row) {
//                         local_change = true;
//                     }
//                 }
//             }
//         }

//         // Any thread can set the global change flag
//         if (local_change) {
//             atomicExch(d_changed, 1);
//         }

//         grid.sync(); // wait for all threads

//         // Exit loop if no changes anywhere
//         if (*d_changed == 0) {
//             break;
//         }

//         grid.sync(); // ensure all threads agree before next iteration
//     }
// }



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

        std::cout << "Step " << steps << ": argmin at B[" << k << "][" << l << "] \n";

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