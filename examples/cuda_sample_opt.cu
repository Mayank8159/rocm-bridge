#include <hip/hip_runtime.h>
#include <stdio.h>

// GOOD: Optimized for AMD CDNA Architecture (Wavefront 64)
#define WAVEFRONT_SIZE 64 

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    
    // GOOD: Logic aligns with AMD hardware execution width
    if (blockDim.x == WAVEFRONT_SIZE) {
        // Optimized wavefront logic
    }
    
    // GOOD: Portable HIP Intrinsic (Works on both AMD & NVIDIA)
    int val = __shfl(tx, 0); 
}

int main() {
    // GOOD: Launching with 64x16 block aligns perfectly with AMD Compute Units
    dim3 block(64, 16); 
    hipLaunchKernelGGL(matrixMul, dim3(1), block, 0, 0, NULL, NULL, NULL, 1024);
    return 0;
}