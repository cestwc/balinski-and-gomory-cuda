#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

__global__ void collectZeroIndicesSingleKernel(const int* B, int n,
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
        if (B[row * n + j] == 0)
            zero_indices[start_pos + count++] = j;

    row_start[row] = start_pos;
    row_count[row] = count;
}

int main() {
    int n = 4096;
    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    std::vector<int> h_B(n * n);
    std::srand((unsigned)time(0));
    for (int i = 0; i < n * n; ++i)
        h_B[i] = (std::rand() % 10 == 0) ? 0 : std::rand() % 100;

    int *d_B, *d_indices, *d_row_start, *d_row_count, *d_counter;
    cudaMalloc(&d_B, n*n*sizeof(int));
    cudaMalloc(&d_indices, n*n*sizeof(int));
    cudaMalloc(&d_row_start, n*sizeof(int));
    cudaMalloc(&d_row_count, n*sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_B, h_B.data(), n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_counter, 0, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256, gridSize = (n + blockSize - 1) / blockSize;
    collectZeroIndicesSingleKernel<<<gridSize, blockSize>>>(
        d_B, n, d_indices, d_row_start, d_row_count, d_counter);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Total GPU processing time: " << ms << " ms" << std::endl;

    // optional print for small n
    if (n < 10) {
        std::vector<int> h_indices(n*n);
        std::vector<int> h_row_start(n), h_row_count(n);
        cudaMemcpy(h_indices.data(), d_indices, n*n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_start.data(), d_row_start, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_count.data(), d_row_count, n*sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Matrix B:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) std::cout << h_B[i*n + j] << " ";
            std::cout << "\n";
        }
        std::cout << "\nZero column indices per row:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "Row " << i << " (count=" << h_row_count[i] << "): ";
            for (int k = 0; k < h_row_count[i]; ++k)
                std::cout << h_indices[h_row_start[i] + k] << " ";
            std::cout << "\n";
        }
    }

    cudaFree(d_B);
    cudaFree(d_indices);
    cudaFree(d_row_start);
    cudaFree(d_row_count);
    cudaFree(d_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
