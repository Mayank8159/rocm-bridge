"""
ROCm Bridge - Profiling Execution Runner
----------------------------------------
The execution engine that compiles HIP kernels and runs hardware profiling.

Key Capabilities:
1. Compilation: meaningful wrappers around `hipcc`.
2. Profiling: Execution of `rocprof` to capture hardware counters.
3. Simulation: Automatic fallback for non-ROCm environments (e.g., MacBook).

Author: Team 7SENSITIVE
Hackathon Track: AMD Slingshot 2026
"""

import subprocess
import logging
import shutil
import os
import time
import csv
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfilingRunner:
    """
    Manages the compilation and profiling lifecycle of a HIP kernel.
    """
    
    def __init__(self):
        self.hipcc_path = shutil.which("hipcc")
        self.rocprof_path = shutil.which("rocprof")
        self.is_rocm_available = self.hipcc_path is not None and self.rocprof_path is not None
        
        if not self.is_rocm_available:
            logger.warning("⚠️  ROCm toolchain not found. Switching to SIMULATION MODE.")

    def _compile_kernel(self, source_file: str, output_binary: str) -> bool:
        """
        Compiles the source file using hipcc.
        """
        cmd = [
            "hipcc",
            source_file,
            "-o", output_binary,
            "-std=c++17",
            "--genco",  # Generate code object
            "-D__HIP_PLATFORM_AMD__"
        ]
        
        logger.info(f"Compiling: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation Failed:\n{e.stderr}")
            return False

    def _execute_profiler(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Runs the binary under rocprof and captures metrics.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        output_csv = os.path.join(output_dir, "results.csv")
        
        # We use a stats-only run for speed in this demo
        cmd = [
            "rocprof",
            "--stats", 
            "--basenames", "on",
            "-o", output_csv,
            binary_path
        ]
        
        logger.info(f"Profiling: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=output_dir # Run inside output dir to keep generated files together
            )
            duration = time.time() - start_time
            
            return {
                "status": "SUCCESS",
                "execution_time": duration,
                "profile_csv": output_csv,
                "logs": result.stdout
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Profiling Failed:\n{e.stderr}")
            return {
                "status": "FAILED",
                "error": e.stderr
            }

    def _simulate_profile(self, source_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Generates realistic dummy data for hackathon demos on non-AMD hardware.
        """
        logger.info("⚡ Simulating profiling run (Demo Mode)...")
        time.sleep(1.5) # Fake work
        
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "results.csv")
        
        # Create a fake ROCm profile CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Standard rocprof header
            writer.writerow(['Index', 'KernelName', 'gpu-id', 'queue-id', 'queue-index', 'pid', 'tid', 'grd', 'wgr', 'lds', 'scr', 'vgpr', 'sgpr', 'fbar', 'sig', 'obj', 'VALUUtilization', 'WavefrontOccupancy', 'LDSBankConflict'])
            # Fake Data: Low utilization to trigger the "Bad Code" path
            writer.writerow([1, 'matrixMul_kernel', 0, 0, 0, 1234, 5678, 128, 64, 0, 0, 32, 16, 0, 0, 0, 38.5, 42.0, 150])

        return {
            "status": "SIMULATED",
            "execution_time": 0.42,
            "profile_csv": csv_path,
            "binary": "simulated_binary",
            "message": "Generated mock telemetry for MI300X"
        }

    def run_profile(self, source_file: str, output_dir: str = "profiles/") -> Dict[str, Any]:
        """
        Public API: Orchestrates the compile-run-profile loop.
        """
        file_path = Path(source_file)
        if not file_path.exists():
            return {"status": "ERROR", "message": "Source file not found"}

        binary_path = os.path.join(output_dir, file_path.stem)
        
        # 1. Fallback to Simulation if tools missing
        if not self.is_rocm_available:
            return self._simulate_profile(source_file, output_dir)
            
        # 2. Compile
        if not self._compile_kernel(source_file, binary_path):
            return {"status": "COMPILATION_ERROR", "message": "Check hipcc logs"}
            
        # 3. Profile
        return self._execute_profiler(binary_path, output_dir)

# --- CLI & EXPORT ---
runner = ProfilingRunner()

if __name__ == "__main__":
    # Test CLI
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m profiler.runner <file.cu>")
    else:
        print(runner.run_profile(sys.argv[1]))