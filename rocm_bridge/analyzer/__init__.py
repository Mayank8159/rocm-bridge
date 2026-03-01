"""
ROCm Bridge - Analyzer Module
=============================
Static analysis engine using libclang AST traversal.

This module provides:
- CUDA/HIP AST parsing with timeout protection
- Architecture-aware detection rules (no token matching)
- Deterministic performance metric calculation
- Thread-safe analysis operations

Usage:
    >>> from analyzer import CudaParser, RuleEngine
    >>> parser = CudaParser()
    >>> report = parser.analyze("kernel.cu")
    >>> print(f"Portability Score: {report['analysis']['score']}/100")
"""

from .parser import CudaParser, ParseResult
from .rules import RuleEngine, AnalysisIssue
from .metrics import PerformanceMetrics, RooflineModel

__all__ = [
    "CudaParser",
    "ParseResult",
    "RuleEngine",
    "AnalysisIssue",
    "PerformanceMetrics",
    "RooflineModel",
]

__version__ = "1.0.0"