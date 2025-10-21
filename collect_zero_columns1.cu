#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

__global__ void countZerosPerRow(const int* B, int* zero_counts, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int count = 0;
    for (int j = 0; j < n; ++j)
        if (B[row * n + j] == 0) count++;
    zero_counts[row] = count;
}

__global__ void collectZeroIndices(const int* B, const int* row_offsets,
                                   int* zero_indices, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int start = row_offsets[row];
    int count = 0;
    for (int j = 0; j < n; ++j)
        if (B[row * n + j] == 0)
            zero_indices[start + count++] = j;
}

void prefixSum(const int* counts, int* offsets, int n) {
    offsets[0] = 0;
    for (int i = 1; i < n; ++i)
        offsets[i] = offsets[i-1] + counts[i-1];
}

int main() {
    int n = 4096; // change for testing
    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    std::vector<int> h_B(n * n);
    std::srand((unsigned)time(0));
    for (int i = 0; i < n * n; ++i)
        h_B[i] = (std::rand() % 10 == 0) ? 0 : std::rand() % 100;

    int *d_B, *d_counts, *d_offsets, *d_indices;
    cudaMalloc(&d_B, n*n*sizeof(int));
    cudaMalloc(&d_counts, n*sizeof(int));
    cudaMemcpy(d_B, h_B.data(), n*n*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256, gridSize = (n + blockSize - 1) / blockSize;
    countZerosPerRow<<<gridSize, blockSize>>>(d_B, d_counts, n);
    cudaDeviceSynchronize();

    std::vector<int> h_counts(n), h_offsets(n);
    cudaMemcpy(h_counts.data(), d_counts, n*sizeof(int), cudaMemcpyDeviceToHost);
    prefixSum(h_counts.data(), h_offsets.data(), n);
    int totalZeros = h_offsets[n-1] + h_counts[n-1];

    cudaMalloc(&d_offsets, n*sizeof(int));
    cudaMalloc(&d_indices, totalZeros*sizeof(int));
    cudaMemcpy(d_offsets, h_offsets.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    collectZeroIndices<<<gridSize, blockSize>>>(d_B, d_offsets, d_indices, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Total GPU processing time: " << ms << " ms" << std::endl;

    // optional print for small n
    if (n < 10) {
        std::vector<int> h_indices(totalZeros);
        cudaMemcpy(h_indices.data(), d_indices, totalZeros*sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Matrix B:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) std::cout << h_B[i*n + j] << " ";
            std::cout << "\n";
        }
        std::cout << "\nZero column indices per row:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "Row " << i << " (count=" << h_counts[i] << "): ";
            for (int k = 0; k < h_counts[i]; ++k)
                std::cout << h_indices[h_offsets[i] + k] << " ";
            std::cout << "\n";
        }
    }

    cudaFree(d_B);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_indices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
