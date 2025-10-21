// find_zeros_sharedmem.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

// ---------------------------------------------
// Warp exclusive scan (sum) using shuffles
// Returns EXCLUSIVE prefix sum of 'v' within the warp.
// ---------------------------------------------
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
__global__ void collectZeroIndicesSharedMem(const int* __restrict__ B,
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
            flag = (B[row * n + col] == 0) ? 1u : 0u;
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

// --------------------------------------------------------
// Helper to optionally print results for small n
// --------------------------------------------------------
void print_small_result(int n,
                        const std::vector<int>& h_B,
                        const std::vector<int>& h_row_count,
                        const std::vector<int>& h_zero_indices) {
    std::cout << "Matrix B:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) std::cout << h_B[i*n + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nZero column indices per row:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << " (count=" << h_row_count[i] << "): ";
        for (int k = 0; k < h_row_count[i]; ++k) {
            std::cout << h_zero_indices[i * n + k] << " ";
        }
        std::cout << "\n";
    }
}

// --------------------------------------------------------
// MAIN: generates random matrix, runs kernel, times it,
// prints result if n < 10.
// --------------------------------------------------------
int main(int argc, char** argv) {
    // You can pass n on the command line. Default: 4096.
    int n = (argc >= 2) ? std::atoi(argv[1]) : 4096;
    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    // Host matrix with ~10% zeros
    std::vector<int> h_B(n * n);
    std::srand(0);
    for (int i = 0; i < n * n; ++i)
        h_B[i] = (std::rand() % 10 == 0) ? 0 : (std::rand() % 100);

    // Device allocations
    int *d_B = nullptr, *d_zero_indices = nullptr, *d_row_start = nullptr, *d_row_count = nullptr;
    CUDA_OK(cudaMalloc(&d_B, n * n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_zero_indices, n * n * sizeof(int))); // worst-case reserve
    CUDA_OK(cudaMalloc(&d_row_start, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_row_count, n * sizeof(int)));

    CUDA_OK(cudaMemcpy(d_B, h_B.data(), n * n * sizeof(int), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start));

    // Launch: one block per row; choose block size for good occupancy.
    // 256 or 512 is usually a good start; you can try 1024 if n is big.
    int blockSize = 512;
    int gridSize  = n;

    collectZeroIndicesSharedMem<<<gridSize, blockSize>>>(
        d_B, n, d_zero_indices, d_row_start, d_row_count);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Total GPU processing time: " << ms << " ms\n";

    // If n < 10, fetch & print results
    if (n < 10) {
        std::vector<int> h_row_start(n), h_row_count(n), h_zero_indices(n * n);
        CUDA_OK(cudaMemcpy(h_row_start.data(), d_row_start, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_row_count.data(), d_row_count, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_zero_indices.data(), d_zero_indices, n * n * sizeof(int), cudaMemcpyDeviceToHost));

        print_small_result(n, h_B, h_row_count, h_zero_indices);
    }

    // Cleanup
    CUDA_OK(cudaFree(d_B));
    CUDA_OK(cudaFree(d_zero_indices));
    CUDA_OK(cudaFree(d_row_start));
    CUDA_OK(cudaFree(d_row_count));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));

    return 0;
}
