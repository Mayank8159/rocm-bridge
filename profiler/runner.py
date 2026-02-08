import subprocess
import os

def run_rocprof(binary_path, output_dir="."):
    """
    Wraps the rocprof command line tool.
    NOTE: This only works on Linux machines with AMD GPUs.
    """
    cmd = [
        "rocprof",
        "--stats",
        "--basenames", "on",
        "-o", f"{output_dir}/metrics.csv",
        binary_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print("[ERROR] rocprof not found. Are you on an AMD machine?")
        return False
