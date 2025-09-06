#include <iostream>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "solver.h"

#define IDX2C(i,j,n) ((j)*(n)+(i))

// extern void solve(const float*, const float*, float*, int, int, int);
// extern void solve(float* d_C, int* d_X, float* d_U, float* d_V, int n);


void fill_random(float* matrix, int n) {
    for (int i = 0; i < n * n; ++i)
        matrix[i] = static_cast<float>(rand() % 100); // 0–9
}

void initialize_identity_mask(int* X, int n) {
    for (int i = 0; i < n * n; ++i)
        X[i] = 0;
    for (int i = 0; i < n; ++i)
        X[IDX2C(i, i, n)] = 1;
}

void compute_V_from_C_and_X(const float* C, const int* X, float* V, int n) {
    for (int j = 0; j < n; ++j) {
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += C[IDX2C(i, j, n)] * X[IDX2C(i, j, n)];
        }
        V[j] = sum;
    }
}

template <typename T>
void print_matrix(const T* matrix, int n, const char* name) {
    std::cout << name << " (" << n << "x" << n << "):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << matrix[IDX2C(i, j, n)] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}


void print_vector(const float* vec, int n, const char* name) {
    std::cout << name << " (" << n << "): ";
    for (int i = 0; i < n; ++i)
        std::cout << vec[i] << " ";
    std::cout << "\n\n";
}

int main(int argc, char* argv[]) {
    int n = 100; // default

    if (argc >= 2) {
        if (std::string(argv[1]) == "-n") {
            if (argc >= 3) {
                n = std::atoi(argv[2]);
            } else {
                std::cerr << "Error: -n requires a value.\n";
                return 1;
            }
        } else {
            n = std::atoi(argv[1]); // simple style
        }
    }

    // std::cout << "Matrix size n = " << n << std::endl;
    std::cout << "Matrix size n = " << n << std::endl;

    size_t matSize = n * n * sizeof(float);
    size_t maskSize = n * n * sizeof(int);
    size_t vecSize = n * sizeof(float);

    // Host allocations
    float* h_C = new float[n * n];
    int* h_X = new int[n * n];
    float* h_U = new float[n];
    float* h_V = new float[n];

    // Fill values
    srand(static_cast<unsigned>(time(0)));
    fill_random(h_C, n);
    // float vals[4][4] = {
    //     {2, 3, 1, 1},
    //     {5, 8, 3, 2},
    //     {4, 9, 5, 1},
    //     {8, 7, 8, 4}
    // };

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         h_C[i * n + j] = vals[i][j];  // row-major
    //     }
    // }

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         // h_C[i + j * n] = vals[i][j];  // column-major
    //         h_C[IDX2C(i,j,n)] = vals[i][j];
    //     }
    // }
    initialize_identity_mask(h_X, n);

    for (int i = 0; i < n; ++i) h_U[i] = 0; // U = 0
    compute_V_from_C_and_X(h_C, h_X, h_V, n);

    // Print initialized values
    print_matrix(h_C, n, "Matrix C");
    // print_matrix(reinterpret_cast<float*>(h_X), n, "Mask X");
    // print_matrix(h_X, n, "Mask X");
    // print_vector(h_U, n, "Vector U");
    // print_vector(h_V, n, "Vector V");

    // Device allocations
    float *d_C, *d_U, *d_V;
    int* d_X;
    cudaMalloc(&d_C, matSize);
    cudaMalloc(&d_X, maskSize);
    cudaMalloc(&d_U, vecSize);
    cudaMalloc(&d_V, vecSize);

    // Copy to device
    cudaMemcpy(d_C, h_C, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, maskSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vecSize, cudaMemcpyHostToDevice);

    // Call solver
    // solve(d_C, d_X, d_U, d_V, n);
    // verify_solution(d_C, d_X, d_U, d_V, n);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_ms = 0.0f;

    // ---- Time solve() ----
    cudaEventRecord(start);
    solve(d_C, d_X, d_U, d_V, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "solve() time: " << elapsed_ms << " ms" << std::endl;

    // ---- Time verify_solution() ----
    cudaEventRecord(start);
    verify_solution(d_C, d_X, d_U, d_V, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "verify_solution() time: " << elapsed_ms << " ms" << std::endl;

    // Destroy events after use
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_X, d_X, maskSize, cudaMemcpyDeviceToHost);
    // print_matrix(reinterpret_cast<float*>(h_X), n, "Mask X");
    print_matrix(h_X, n, "Mask X");
    // print_vector(h_U, n, "Vector U");
    // print_vector(h_V, n, "Vector V");


    // Cleanup
    cudaFree(d_C);
    cudaFree(d_X);
    cudaFree(d_U);
    cudaFree(d_V);
    delete[] h_C;
    delete[] h_X;
    delete[] h_U;
    delete[] h_V;

    return 0;
}
