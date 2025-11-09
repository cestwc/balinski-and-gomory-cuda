#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------
// Block prefix sum using shared memory (exclusive scan)
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Main test
// ---------------------------------------------------------------
int main()
{
    const int rows = 8;
    const int cols = 10;
    const int total = rows * cols;

    float *h_B = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; ++i) {
        float r = (float)(rand() % 10);
        h_B[i] = (r < 1.0f) ? 0.0f : r;
    }

    printf("Input Matrix B (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%4.0f ", h_B[i * cols + j]);
        printf("\n");
    }

    float *d_B;
    int *d_indices, *d_count;
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_indices, total * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    selectZerosOptimized(d_B, d_indices, d_count, total);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_indices = (int*)malloc(h_count * sizeof(int));
    cudaMemcpy(h_indices, d_indices, h_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nZero indices (flattened):\n");
    for (int i = 0; i < h_count; ++i)
        printf("%d ", h_indices[i]);
    printf("\n");

    free(h_B);
    free(h_indices);
    cudaFree(d_B);
    cudaFree(d_indices);
    cudaFree(d_count);
    return 0;
}
