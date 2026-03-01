#include <hip/hip_runtime.h>
#include <stdio.h>

// ------------------------------------------------------------------
// ROCm Bridge - Already Ported HIP Sample
// This file demonstrates proper HIP code for AMD hardware.
// ------------------------------------------------------------------

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Host allocation
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Device allocation
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    
    // Copy to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    
    // Launch kernel (256 threads per block - good for both NVIDIA and AMD)
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // Copy back
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    // Verify
    printf("Vector Add Complete. First result: %f\n", h_C[0]);
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}