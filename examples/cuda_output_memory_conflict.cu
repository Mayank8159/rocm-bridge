#include <hip/hip_runtime.h>
#include <stdio.h>

// Expected output profile:
// - Static analyzer: shared-memory bank conflict warning (ROCM_003)
// - Profiler simulation: memory-bound style behavior

__global__ void memory_conflict_kernel(float* in, float* out, int n) {
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + tx;

    if (idx < n && ty < 32 && tx < 32) {
        // Access pattern intentionally aligned with 32x32 tile to provoke warnings.
        tile[ty][tx] = in[idx];
        __syncthreads();
        out[idx] = tile[tx][ty];
    }
}

int main() {
    dim3 block(32, 32, 1);
    dim3 grid(1, 1, 1);
    hipLaunchKernelGGL(memory_conflict_kernel, grid, block, 0, 0, nullptr, nullptr, 1024);
    hipDeviceSynchronize();
    printf("memory-conflict sample executed\n");
    return 0;
}
