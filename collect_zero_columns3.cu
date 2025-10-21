#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// One thread per element version
__global__ void collectZeroIndicesElementWise(const int* B, int n,
                                              int* zero_indices,
                                              int* row_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int row = idx / n;
    int col = idx % n;

    if (B[idx] == 0) {
        int pos = atomicAdd(&row_count[row], 1);
        zero_indices[row * n + pos] = col;
    }
}


int main() {
    int n = 4096; // or large like 4096
    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    std::vector<int> h_B(n * n);
    srand(0);
    for (int i = 0; i < n * n; ++i)
        h_B[i] = (rand() % 10 == 0) ? 0 : rand() % 100; // ~10% zeros

    int *d_B, *d_indices, *d_row_count;
    cudaMalloc(&d_B, n * n * sizeof(int));
    cudaMalloc(&d_indices, n * n * sizeof(int));
    cudaMalloc(&d_row_count, n * sizeof(int));
    cudaMemcpy(d_B, h_B.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_row_count, 0, n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int totalThreads = n * n;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    collectZeroIndicesElementWise<<<gridSize, blockSize>>>(d_B, n, d_indices, d_row_count);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU processing time: " << ms << " ms\n";

    if (n < 10) {
        std::vector<int> h_indices(n * n);
        std::vector<int> h_row_count(n);
        cudaMemcpy(h_indices.data(), d_indices, n * n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_count.data(), d_row_count, n * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "\nMatrix B:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                std::cout << h_B[i * n + j] << " ";
            std::cout << "\n";
        }

        std::cout << "\nZero column indices per row:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "Row " << i << " (count=" << h_row_count[i] << "): ";
            for (int k = 0; k < h_row_count[i]; ++k)
                std::cout << h_indices[i * n + k] << " ";
            std::cout << "\n";
        }
    }

    cudaFree(d_B);
    cudaFree(d_indices);
    cudaFree(d_row_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
