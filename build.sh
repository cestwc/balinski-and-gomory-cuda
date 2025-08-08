#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Run cmake configuration and build inside build/
cmake -S . -B build
cmake --build build
