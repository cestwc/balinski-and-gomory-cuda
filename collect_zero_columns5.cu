#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define CUDA_OK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while (0)

// -------------------------
// Kernel 1: Collect zero indices per row
// -------------------------
__global__ void collectZeroIndicesSingleKernel(
    const int* B, int n,
    int* zero_indices, int* row_start, int* row_count, int* global_counter)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start_pos = atomicAdd(global_counter, n); // reserve n slots per row
    int count = 0;
    for (int j = 0; j < n; ++j)
        if (B[row * n + j] == 0)
            zero_indices[start_pos + count++] = j;

    row_start[row] = start_pos;
    row_count[row] = count;
}

// -------------------------
// Kernel 2: GPU-only BFS labeling using a global atomic queue
// -------------------------
__global__ void solve_1bc_queue_kernel(
    int n,
    const int* __restrict__ d_col_to_row,
    const int* __restrict__ zero_indices,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    int* __restrict__ R,
    int* __restrict__ Q,
    int* __restrict__ queue,
    int* __restrict__ q_head,
    int* __restrict__ q_tail)
{
    // multiple threads cooperate to pop rows and process them
    while (true) {
        int my_idx = atomicAdd(q_head, 1);
        int cur_tail = atomicAdd(q_tail, 0);
        if (my_idx >= cur_tail)
            break; // no more work at this moment

        int row = queue[my_idx];

        if (R[row] == n) continue; // skip unlabeled rows

        int base = row_start[row];
        int nz   = row_count[row];

        for (int k = 0; k < nz; ++k) {
            int j = zero_indices[base + k];
            if (atomicCAS(&Q[j], n, row) == n) {
                int r2 = d_col_to_row[j];
                if (r2 >= 0 && r2 < n) {
                    if (atomicCAS(&R[r2], n, j) == n) {
                        int pos = atomicAdd(q_tail, 1);
                        if (pos < n) queue[pos] = r2;
                    }
                }
            }
        }
    }
}

// -------------------------
// Main program
// -------------------------
int main(int argc, char** argv)
{
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float zero_prob = 0.1f;
    int seed_row = 0;

    std::cout << "Matrix size: " << n << "x" << n << " (seed row " << seed_row << ")\n";

    // --- host memory
    std::srand(0);
    std::vector<int> h_B(n * n);
    for (int i = 0; i < n * n; ++i)
        h_B[i] = ((std::rand() / (float)RAND_MAX) < zero_prob) ? 0 : 1;

    std::vector<int> h_col_to_row(n);
    for (int j = 0; j < n; ++j)
        h_col_to_row[j] = j; // identity mapping (you can customize)

    // --- device allocations
    int *d_B, *d_col_to_row, *d_zero_indices, *d_row_start, *d_row_count, *d_global_counter;
    int *d_R, *d_Q, *d_queue, *d_q_head, *d_q_tail;

    CUDA_OK(cudaMalloc(&d_B, n * n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_col_to_row, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_zero_indices, n * n * sizeof(int))); // upper bound
    CUDA_OK(cudaMalloc(&d_row_start, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_row_count, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_global_counter, sizeof(int)));
    CUDA_OK(cudaMalloc(&d_R, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_Q, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_queue, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_q_head, sizeof(int)));
    CUDA_OK(cudaMalloc(&d_q_tail, sizeof(int)));

    CUDA_OK(cudaMemcpy(d_B, h_B.data(), n*n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_col_to_row, h_col_to_row.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_global_counter, 0, sizeof(int)));

    // --- build zero indices fully on GPU
    int block = 128;
    int grid  = (n + block - 1) / block;
    collectZeroIndicesSingleKernel<<<grid, block>>>(d_B, n, d_zero_indices, d_row_start, d_row_count, d_global_counter);
    CUDA_OK(cudaDeviceSynchronize());

    // --- initialize labels and queue
    std::vector<int> h_R(n, n), h_Q(n, n);
    h_R[seed_row] = -1;
    CUDA_OK(cudaMemcpy(d_R, h_R.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Q, h_Q.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> h_queue(n, -1);
    h_queue[0] = seed_row;
    int h_head = 0, h_tail = 1;
    CUDA_OK(cudaMemcpy(d_queue, h_queue.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_q_head, &h_head, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_q_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice));

    // --- launch BFS-like queue kernel (no host loop)
    dim3 block2(256);
    dim3 grid2((n + block2.x - 1) / block2.x);

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start));

    solve_1bc_queue_kernel<<<grid2, block2>>>(
        n, d_col_to_row,
        d_zero_indices, d_row_start, d_row_count,
        d_R, d_Q,
        d_queue, d_q_head, d_q_tail);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "GPU processing time: " << ms << " ms\n";

    // --- copy results back
    CUDA_OK(cudaMemcpy(h_R.data(), d_R, n*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_Q.data(), d_Q, n*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&h_tail, d_q_tail, sizeof(int), cudaMemcpyDeviceToHost));

    if (n < 100) {
        std::cout << "\nMatrix B:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                std::cout << h_B[i*n + j] << " ";
            std::cout << "\n";
        }

        std::cout << "\nRow labels (R): ";
        for (int i = 0; i < n; ++i) std::cout << h_R[i] << " ";
        std::cout << "\nCol labels (Q): ";
        for (int j = 0; j < n; ++j) std::cout << h_Q[j] << " ";
        std::cout << "\nQueue tail = " << h_tail << "\n";
    } else {
        int labeled_rows = 0, labeled_cols = 0;
        for (int i = 0; i < n; ++i) if (h_R[i] != n) ++labeled_rows;
        for (int j = 0; j < n; ++j) if (h_Q[j] != n) ++labeled_cols;
        std::cout << "Labeled rows: " << labeled_rows << " / " << n
                  << " | Labeled cols: " << labeled_cols << " / " << n << "\n";
    }

    // --- cleanup
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    CUDA_OK(cudaFree(d_B));
    CUDA_OK(cudaFree(d_col_to_row));
    CUDA_OK(cudaFree(d_zero_indices));
    CUDA_OK(cudaFree(d_row_start));
    CUDA_OK(cudaFree(d_row_count));
    CUDA_OK(cudaFree(d_global_counter));
    CUDA_OK(cudaFree(d_R));
    CUDA_OK(cudaFree(d_Q));
    CUDA_OK(cudaFree(d_queue));
    CUDA_OK(cudaFree(d_q_head));
    CUDA_OK(cudaFree(d_q_tail));
    return 0;
}
