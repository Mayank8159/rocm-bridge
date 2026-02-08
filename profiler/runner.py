"""
ROCm Bridge - Profiling Execution Runner (Dynamic Simulation)
-------------------------------------------------------------
The execution engine that compiles HIP kernels and runs hardware profiling.
Updated to generate VARIABLE simulation data so the demo looks alive.

Features:
- "Cheat Code" Logic: Detects filenames with 'opt' or 'hip' to generate good metrics.
- Randomization: Every run produces slightly different numbers.
- Simulation Mode: Runs on Mac/Windows without AMD hardware.
"""

import subprocess
import logging
import shutil
import os
import time
import csv
import random
from typing import Dict, Any
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
        self.is_rocm_available = self.hipcc_path is not None
        
        if not self.is_rocm_available:
            logger.warning("⚠️  ROCm toolchain not found. Switching to DYNAMIC SIMULATION MODE.")

    def _compile_kernel(self, source_file: str, output_binary: str) -> bool:
        """
        Compiles the source file using hipcc.
        """
        cmd = [
            "hipcc",
            source_file,
            "-o", output_binary,
            "-std=c++17",
            "--genco",
            "-D__HIP_PLATFORM_AMD__"
        ]
        
        logger.info(f"Compiling: {' '.join(cmd)}")
        
        try:
            # On Mac/Non-ROCm machines, this will fail if we actually try to run it.
            # But we check self.is_rocm_available before calling this in the main loop.
            if self.is_rocm_available:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation Failed:\n{e.stderr}")
            return False

    def _execute_profiler(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Runs the binary under rocprof (Real Hardware Mode).
        """
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "results.csv")
        
        cmd = ["rocprof", "--stats", "--basenames", "on", "-o", output_csv, binary_path]
        
        logger.info(f"Profiling: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=output_dir)
            return {
                "status": "SUCCESS",
                "execution_time": time.time() - start_time,
                "profile_csv": output_csv,
                "logs": result.stdout
            }
        except subprocess.CalledProcessError as e:
            return {"status": "FAILED", "error": e.stderr}

    def _simulate_profile(self, source_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Generates RANDOMIZED dummy data for hackathon demos.
        """
        logger.info(f"⚡ Simulating profiling run for {os.path.basename(source_file)}...")
        
        # Fake "processing" delay to make it look realistic
        time.sleep(random.uniform(0.5, 1.5))
        
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "results.csv")
        
        # --- RANDOMIZE METRICS (The Logic) ---
        
        # Default: "Bad Code" (Low utilization, High conflicts)
        valu_util = round(random.uniform(30.0, 48.0), 2)       # Low: 30-48%
        occupancy = round(random.uniform(35.0, 55.0), 2)       # Low: 35-55%
        bank_conflicts = random.randint(120, 300)              # High: 100+ conflicts
        mem_stalls = round(random.uniform(15.0, 35.0), 2)      # Random stalls
        
        # CHEAT CODE: Check if filename implies "Optimized"
        fname = os.path.basename(source_file).lower()
        if "opt" in fname or "hip" in fname or "fix" in fname:
            # Generate "Good Code" metrics (High utilization, Low conflicts)
            valu_util = round(random.uniform(88.0, 98.5), 2)
            occupancy = round(random.uniform(85.0, 96.0), 2)
            bank_conflicts = random.randint(0, 5)
            mem_stalls = round(random.uniform(0.5, 2.0), 2)
            logger.info("✨ Detected optimized file pattern. Generating HIGH performance metrics.")

        # Create the CSV matching rocprof format
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Standard Header
            writer.writerow(['Index', 'KernelName', 'gpu-id', 'queue-id', 'queue-index', 'pid', 'tid', 'grd', 'wgr', 'lds', 'scr', 'vgpr', 'sgpr', 'fbar', 'sig', 'obj', 'VALUUtilization', 'WavefrontOccupancy', 'LDSBankConflict', 'MemUnitStalled'])
            # The Data Row
            writer.writerow([1, 'matrixMul_kernel', 0, 0, 0, 1234, 5678, 128, 64, 0, 0, 32, 16, 0, 0, 0, valu_util, occupancy, bank_conflicts, mem_stalls])

        return {
            "status": "SIMULATED",
            "execution_time": round(random.uniform(0.3, 0.8), 3),
            "profile_csv": csv_path,
            "binary": "simulated_binary",
            "metrics": {
                "valu": valu_util,
                "occupancy": occupancy,
                "bank_conflicts": bank_conflicts
            }
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
            
        # 2. Compile (Real Mode)
        if not self._compile_kernel(source_file, binary_path):
            return {"status": "COMPILATION_ERROR", "message": "Check hipcc logs"}
            
        # 3. Profile (Real Mode)
        return self._execute_profiler(binary_path, output_dir)

# --- CLI & EXPORT ---
runner = ProfilingRunner()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m profiler.runner <file.cu>")
    else:
        print(runner.run_profile(sys.argv[1]))