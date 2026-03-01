#include <hip/hip_runtime.h>
#include <stdio.h>

// Expected output profile:
// - Static analyzer: portability-focused failures (ROCM_002)
// - Profiler simulation: likely poor/medium depending on run

__global__ void portability_fail_kernel(int* out, int n) {
    int tx = threadIdx.x;

    // NVIDIA-only intrinsics that should be ported for ROCm/HIP.
    unsigned mask = __activemask();
    int vote = __all_sync(mask, tx < n);
    int shuffled = __shfl_down_sync(mask, tx, 1);

    if (tx < n) {
        out[tx] = vote + shuffled;
    }
}

int main() {
    dim3 block(128, 1, 1);
    dim3 grid(1, 1, 1);
    hipLaunchKernelGGL(portability_fail_kernel, grid, block, 0, 0, nullptr, 1024);
    hipDeviceSynchronize();
    printf("portability-fail sample executed\n");
    return 0;
}
