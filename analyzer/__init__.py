"""
ROCm Bridge Analyzer
====================
The core static analysis engine for the ROCm Bridge portability tool.

This package provides the `CudaParser` and `RuleEngine` necessary to:
1. Parse CUDA C++ source code into an Abstract Syntax Tree (AST).
2. Detect NVIDIA-specific architectural patterns (Warp vs Wavefront).
3. Generate portability warnings and optimization hints for AMD ROCm.

Usage Example:
    >>> from analyzer import CudaParser
    >>> parser = CudaParser()
    >>> report = parser.analyze("kernel.cu")
    >>> print(f"Portability Score: {report['analysis']['score']}")

Metadata:
    Version: 0.1.0-alpha
    License: MIT
    Author: Team 7SENSITIVE (AMD Slingshot 2026)
"""

import logging

# --- PACKAGE METADATA ---
__version__ = "0.1.0"
__author__ = "Team 7SENSITIVE"
__license__ = "MIT"
__project__ = "ROCm Bridge"

# --- LOGGING CONFIGURATION ---
# Ensure a default handler exists to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

# --- PUBLIC API EXPORTS ---
# We expose the primary classes directly to the package namespace
# to support clean imports like: `from analyzer import CudaParser`

try:
    from .parser import CudaParser
    from .rules import RuleEngine, engine
except ImportError as e:
    # Graceful degradation if dependencies (like libclang) are missing during simple inspection
    # This prevents the entire app from crashing just because one module failed to load.
    import sys
    sys.stderr.write(f"[ROCm Bridge] Warning: Failed to import core analyzer components: {e}\n")
    
    # Define dummy classes to allow introspection tools to work
    CudaParser = None
    RuleEngine = None
    engine = None

# Define the explicit public API
__all__ = [
    "CudaParser",
    "RuleEngine",
    "engine",
    "__version__",
    "__author__"
]