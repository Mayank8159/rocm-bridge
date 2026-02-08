#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    
    // BAD: Hardcoded 32 assumes NVIDIA Warp
    if (blockDim.x == 32) {
        // Warp-specific logic
    }
    
    // BAD: NVIDIA Intrinsic
    int val = __shfl_sync(0xffffffff, tx, 0);
}

int main() {
    // BAD: Launching with 32 threads
    dim3 block(32, 32); 
    matrixMul<<<1, block>>>(NULL, NULL, NULL, 1024);
    return 0;
}
