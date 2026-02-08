#ifndef MOCK_HIP_RUNTIME_H
#define MOCK_HIP_RUNTIME_H

#include <stdio.h>

// Mock defines to satisfy VS Code IntelliSense on Mac
#define __global__
#define __device__
#define __host__

struct uint3 { unsigned int x, y, z; };
struct dim3 { 
    unsigned int x, y, z; 
    dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {}
};

// Mock Built-in Variables
extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;

// Mock HIP Functions
int hipDeviceSynchronize();
void* malloc(size_t);
void free(void*);

// Mock Kernel Launch Macro
#define hipLaunchKernelGGL(kernel, grid, block, mem, stream, ...) kernel(__VA_ARGS__)

// Mock Intrinsics
int __shfl(int var, int lane);
int __shfl_sync(unsigned mask, int var, int lane);

#endif
