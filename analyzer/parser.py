"""
ROCm Bridge - CUDA AST Parser & Analysis Frontend
-------------------------------------------------
A production-grade clang-based parser that builds a full Abstract Syntax Tree (AST)
from CUDA source code, identifies kernel boundaries, and dispatches the AST to
the static analysis rule engine.

Key Features:
- Full Clang AST traversal (not regex-based).
- CUDA-aware compilation flags (-x cuda, -D__CUDACC__).
- Kernel detection logic (__global__, __device__).
- Graceful degradation if libclang is missing.

Author: Team 7SENSITIVE
Hackathon Track: AMD Slingshot 2026
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LIBCLANG SETUP ---
try:
    import clang.cindex
    from clang.cindex import Index, CursorKind, TranslationUnit
    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    logger.warning("⚠️  libclang not found. Static analysis will be limited.")

# Import the Rule Engine
try:
    from .rules import engine as rule_engine
except ImportError:
    # Fallback for running as script
    from rules import engine as rule_engine


class CudaParser:
    """
    A specialized parser for CUDA C++ source files using libclang.
    It compiles the code into an AST and extracts kernel metadata.
    """

    def __init__(self, libclang_path: Optional[str] = None):
        self.index = None
        
        if LIBCLANG_AVAILABLE:
            # 1. Configuration: Set library path if provided (e.g., from ENV)
            if libclang_path:
                clang.cindex.Config.set_library_path(libclang_path)
            elif os.environ.get("LLVM_LIB_PATH"):
                clang.cindex.Config.set_library_path(os.environ["LLVM_LIB_PATH"])
            
            # 2. Initialize Clang Index
            try:
                self.index = Index.create()
            except Exception as e:
                logger.error(f"Failed to initialize Clang Index: {e}")
                self.index = None

    def _get_compilation_flags(self) -> List[str]:
        """
        Returns a list of compiler flags to ensure Clang treats the input as CUDA.
        """
        return [
            '-x', 'cuda',                  # Treat as CUDA C++
            '-std=c++17',                  # Modern C++ standard
            '-D__CUDACC__',                # Define CUDA compiler macro
            '--cuda-gpu-arch=sm_50',       # Target a generic NVIDIA architecture
            '-nocudainc',                  # Do not include standard CUDA headers (prevents noise)
            '-I/usr/local/cuda/include',   # Standard CUDA include path (Linux)
            '-I/opt/cuda/include'          # Arch Linux / Custom
        ]

    def _find_kernels(self, node) -> List[str]:
        """
        Recursively scans the AST to find functions marked as __global__ or __device__.
        Returns a list of detected kernel names.
        """
        kernels = []
        
        # In a real AST, __global__ is often an attribute or annotation.
        # Since standard Clang might not fully parse CUDA attributes without the toolkit,
        # we inspect tokens for keywords like "__global__" or "__device__".
        
        if node.kind == CursorKind.FUNCTION_DECL:
            # Heuristic: Scan tokens for CUDA qualifiers
            tokens = [t.spelling for t in node.get_tokens()]
            if "__global__" in tokens or "__device__" in tokens:
                kernels.append(node.spelling)
                
        # Recurse
        for child in node.get_children():
            kernels.extend(self._find_kernels(child))
            
        return list(set(kernels)) # Deduplicate

    def parse_file(self, file_path: str) -> Optional[Any]:
        """
        Parses a single file and returns the Translation Unit (AST Root).
        """
        if not self.index:
            logger.error("Clang Index not initialized.")
            return None

        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Parsing AST for: {file_path}")
        
        try:
            # The Magic: This parses the C++ code
            tu = self.index.parse(
                str(path),
                args=self._get_compilation_flags()
            )
            return tu
        except Exception as e:
            logger.error(f"Clang Parse Error: {e}")
            return None

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point:
        1. Parses the file.
        2. Detects kernels.
        3. Runs the Rule Engine.
        4. Returns a structured report.
        """
        result = {
            "file": file_path,
            "kernels_detected": [],
            "analysis": {
                "score": 0,
                "status": "FAIL",
                "issues": []
            }
        }

        # 1. Parse
        tu = self.parse_file(file_path)
        
        if not tu:
            result["analysis"]["issues"].append({
                "severity": "CRITICAL",
                "message": "Failed to parse AST. Check file path or Clang installation."
            })
            return result

        # 2. Extract Metadata (Kernels)
        # Note: In a full app, we pass the AST root (tu.cursor) to find kernels
        detected_kernels = self._find_kernels(tu.cursor)
        result["kernels_detected"] = detected_kernels
        logger.info(f"Kernels identified: {detected_kernels}")

        # 3. Run Rule Engine
        logger.info("Dispatching AST to Rule Engine...")
        engine_report = rule_engine.run_rules(tu.cursor)
        
        # 4. Merge Results
        result["analysis"] = engine_report
        
        # Add high-level summary
        logger.info(f"Analysis Complete. Score: {engine_report['score']}/100")
        
        return result

# --- CLI RUNNER (For Testing) ---
if __name__ == "__main__":
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python -m analyzer.parser <path_to_cuda_file>")
        sys.exit(1)
        
    target_file = sys.argv[1]
    
    # Initialize Parser
    parser = CudaParser()
    
    # Run Analysis
    report = parser.analyze(target_file)
    
    # Pretty Print
    print(json.dumps(report, indent=2))