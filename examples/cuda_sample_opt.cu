#include <hip/hip_runtime.h>
#include <stdio.h>

// ------------------------------------------------------------------
// ROCm Bridge - Optimized Demo Test Case
// This file is already optimized for AMD CDNA/RDNA architectures.
// ------------------------------------------------------------------

#define WAVEFRONT_SIZE 64

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // GOOD: Wavefront-aware logic (64 threads for CDNA)
    if (blockDim.x == WAVEFRONT_SIZE) {
        int lane = tx % WAVEFRONT_SIZE;
    }

    // GOOD: Portable HIP intrinsic
    int val = __shfl(tx, 0);

    // GOOD: Padded shared memory to avoid bank conflicts
    __shared__ float sharedData[32][33];  // 33 instead of 32
    sharedData[ty][tx] = A[by * 32 + ty][bx * 32 + tx];
    __syncthreads();

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

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
    
    // GOOD: 64x16 block aligns with AMD Wavefront 64
    dim3 block(64, 16);
    dim3 grid(N/64, N/16);

    printf("Launching MatrixMul with AMD-optimized config...\n");
    
    hipLaunchKernelGGL(matrixMul, grid, block, 0, 0, nullptr, nullptr, nullptr, N);
    
    hipDeviceSynchronize();
    return 0;
}