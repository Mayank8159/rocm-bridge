"""
ROCm Bridge Profiling Subsystem
===============================
The hardware telemetry & execution layer for ROCm Bridge.

This package provides a unified interface to:
1. Compile HIP kernels via `hipcc`.
2. Execute kernels under `rocprof` to capture hardware counters.
3. Parse and analyze the resulting telemetry CSVs.

The subsystem is designed to degrade gracefully into "Simulation Mode"
if ROCm hardware is not detected, ensuring stability for demos.

Usage Example:
    >>> from profiler import profile_kernel
    >>> report = profile_kernel("matrix_mul.cu", output_dir="./data")
    >>> print(f"Kernel Health: {report['health_score']}/100")

Metadata:
    Version: 0.1.0
    Author: Team 7SENSITIVE
    License: MIT
"""

import logging
import os
from typing import Dict, Any, Optional

# --- LOGGING CONFIGURATION ---
# Ensure a default handler exists to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

# --- LAZY IMPORTS & FAÇADE ---
# We import the core classes safely to allow inspection even if dependencies fail

try:
    from .runner import ProfilingRunner, runner
    from .collector import ProfileCollector, collector
except ImportError as e:
    import sys
    sys.stderr.write(f"[ROCm Bridge] Warning: Failed to import profiling subsystem: {e}\n")
    ProfilingRunner = None
    ProfileCollector = None
    runner = None
    collector = None

# --- CONVENIENCE PIPELINE ---

def profile_kernel(source_file: str, output_dir: str = "profiles/") -> Dict[str, Any]:
    """
    Orchestrates the full profiling pipeline: Compile -> Run -> Collect.
    
    Args:
        source_file (str): Path to the .cu or .cpp HIP kernel file.
        output_dir (str): Directory to store binary and CSV artifacts.
        
    Returns:
        Dict[str, Any]: A structured report containing execution status,
                        paths to artifacts, and parsed telemetry metrics.
    """
    if not runner or not collector:
        return {
            "status": "ERROR",
            "message": "Profiling subsystem not initialized (ImportError)."
        }

    # Step 1: Execute (Compile + Run + Rocprof)
    # This handles both real hardware execution AND simulation mode.
    exec_report = runner.run_profile(source_file, output_dir)
    
    if exec_report.get("status") not in ["SUCCESS", "SIMULATED"]:
        return {
            "status": "EXECUTION_FAILED",
            "error": exec_report.get("message", "Unknown error"),
            "logs": exec_report.get("logs", "")
        }

    # Step 2: Collect (Parse CSV)
    csv_path = exec_report.get("profile_csv")
    telemetry = collector.collect(csv_path)
    
    # Step 3: Merge & Return
    final_report = {
        "status": "SUCCESS",
        "mode": exec_report.get("status"),  # 'SUCCESS' (Real) or 'SIMULATED' (Demo)
        "execution_time": exec_report.get("execution_time"),
        "artifacts": {
            "binary": exec_report.get("binary"),
            "csv": csv_path
        },
        "telemetry": telemetry  # Contains health_score, bottleneck, metrics
    }
    
    return final_report

# --- PUBLIC API EXPORTS ---
# Explicitly define what is available to the outside world
__all__ = [
    "profile_kernel",
    "ProfilingRunner",
    "ProfileCollector",
    "runner",
    "collector"
]