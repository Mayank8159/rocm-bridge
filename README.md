# The ROCm Bridge 🌉
### Automated CUDA-to-HIP Optimization Engine

**Team 7SENSITIVE** | AMD Slingshot Hackathon 2026

## Overview
A hybrid static-dynamic analysis tool that detects NVIDIA-specific optimizations (like Warp Size 32) and auto-corrects them for AMD ROCm architectures (Wavefront 64).

## Structure
- **analyzer/**: Static AST parsing using LibClang.
- **profiler/**: Hardware telemetry using rocprof.
- **engine/**: Correlation logic to map warnings to fixes.
- **app/**: Streamlit dashboard.

## Quick Start
1. `./scripts/setup_rocm.sh`
2. `pip install -r requirements.txt`
3. `streamlit run app/main.py`
