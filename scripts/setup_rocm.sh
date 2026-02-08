#!/bin/bash
# Run this to set up environment variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=$ROCM_PATH/bin:$HIP_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

echo "✅ ROCm Environment Variables Set"
