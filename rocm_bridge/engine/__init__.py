"""
ROCm Bridge - Engine Module
===========================
Code transformation and recommendation engine.

This module provides:
- Two-pass transpilation (hipify-clang + AST optimization)
- Byte-offset safe code patching
- Correlation logic (AST + Metrics → Recommendations)
- Patch file generation for CI/CD

Usage:
    >>> from engine import TranspilationEngine
    >>> engine = TranspilationEngine()
    >>> result = engine.transpile("kernel.cu", output_dir="./output")
    >>> print(f"Generated {len(result['patches'])} patches")
"""

from .hipify_wrapper import HipifyWrapper
from .patcher import ByteOffsetPatcher
from .recommender import RecommendationEngine, Recommendation

__all__ = [
    "HipifyWrapper",
    "ByteOffsetPatcher",
    "RecommendationEngine",
    "Recommendation",
]

__version__ = "1.0.0"