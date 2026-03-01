#include <hip/hip_runtime.h>
#include <stdio.h>

// Expected output profile:
// - Static analyzer: fewer/highly reduced issues
// - Profiler simulation: high-performance behavior (filename contains "fix")

#define WAVEFRONT_SIZE 64

__global__ void wave64_fix_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Wave64-aware launch assumption for AMD GPUs.
    if (blockDim.x == WAVEFRONT_SIZE && idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    dim3 block(64, 1, 1);
    dim3 grid(16, 1, 1);
    hipLaunchKernelGGL(wave64_fix_kernel, grid, block, 0, 0, nullptr, nullptr, nullptr, 1024);
    hipDeviceSynchronize();
    printf("wave64-fix sample executed\n");
    return 0;
}
