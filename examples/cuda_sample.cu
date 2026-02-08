#include <hip/hip_runtime.h>
#include <stdio.h>

// ------------------------------------------------------------------
// ROCm Bridge - Demo Test Case
// This file contains intentional "Anti-Patterns" for AMD architectures.
// ------------------------------------------------------------------

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    // 1. Thread Index Calculation
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // [ISSUE 1] Hardcoded Warp Size Assumption
    // NVIDIA Warps are 32 threads. AMD Wavefronts are 64.
    // This logic will fail or cause divergence on CDNA/RDNA architectures.
    if (blockDim.x == 32) {
        // ... Warp-specific optimization logic ...
    }

    // [ISSUE 2] NVIDIA-Specific Intrinsic
    // __shfl_sync is not portable to HIP. It must be replaced with __shfl().
    int val = __shfl(tx, 0);

    // Standard Matrix Multiplication Logic (Safe)
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024;
    
    // [ISSUE 3] Suboptimal Block Dimensions for AMD
    // 32x32 = 1024 threads. 
    // AMD Compute Units prefer Wavefronts of 64 (e.g., 64x16 or 256x1).
    // Using 32 in X-dimension reinforces the Warp-32 dependency.
    dim3 block(32, 32); 
    dim3 grid(N/32, N/32);

    printf("Launching MatrixMul Kernel with NVIDIA-optimized configuration...\n");
    hipLaunchKernelGGL(matrixMul, grid, block, 0, 0, NULL, NULL, NULL, N);
    
    hipDeviceSynchronize();
    return 0;
}