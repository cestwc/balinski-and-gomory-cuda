#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

template <typename T>
void printDeviceVar(const char* name, const T& symbol);

template<typename T>
void printDeviceMatrix(const char* name, const T* d_M, int n);

template<typename T>
void printDeviceVector(const char* name, const T* d_V, int n);

template<typename T>
void printDeviceScalar(const char* name, const T* d_val);
