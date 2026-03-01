#include <hip/hip_runtime.h>
#include <stdio.h>

// Expected output profile:
// - Static analyzer: multiple issues (ROCM_001, ROCM_002, ROCM_003)
// - Profiler simulation: poor utilization (critical-like)

__global__ void critical_kernel(float* data, int n) {
    int tx = threadIdx.x;

    // ROCM_001: hardcoded 32 assumes NVIDIA warp width.
    if (blockDim.x == 32) {
        data[tx] = data[tx] * 2.0f;
    }

    // ROCM_002: NVIDIA-specific intrinsic.
    int lane0 = __shfl_sync(0xffffffff, tx, 0);

    // ROCM_003: likely LDS bank conflict pattern.
    __shared__ float scratch[32][32];
    if (tx < 32) {
        scratch[tx][tx] = (float)lane0;
    }

    if (tx < n) {
        data[tx] += scratch[tx % 32][tx % 32];
    }
}

int main() {
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    hipLaunchKernelGGL(critical_kernel, grid, block, 0, 0, nullptr, 1024);
    hipDeviceSynchronize();
    printf("critical sample executed\n");
    return 0;
}
