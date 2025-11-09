#include "cuda_debug_utils.cuh"
#include <cstdio>

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

// Explicit template instantiations (so linker knows what to build)
template void printDeviceVar<int>(const char*, const int&);
template void printDeviceVar<float>(const char*, const float&);
template void printDeviceVar<double>(const char*, const double&);

template void printDeviceMatrix<float>(const char*, const float*, int);
template void printDeviceMatrix<double>(const char*, const double*, int);
template void printDeviceVector<float>(const char*, const float*, int);
template void printDeviceVector<double>(const char*, const double*, int);
template void printDeviceScalar<float>(const char*, const float*);
template void printDeviceScalar<double>(const char*, const double*);
