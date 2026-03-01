"""
ROCm Bridge - Production CUDA-to-HIP Transpilation Engine
==========================================================

A mathematically deterministic, AST-driven transpilation daemon that converts
CUDA code to optimized HIP code for AMD CDNA/RDNA architectures.

Architecture:
- Layer 1: Core (HAL, State, Config, Logging)
- Layer 2: Discovery (Scanner, Registry)
- Layer 3: Analyzer (Parser, Rules, Metrics)
- Layer 4: Engine (Hipify, Patcher, Recommender)

Usage:
    >>> from rocm_bridge.core.hal import HardwareAbstractionLayer
    >>> from rocm_bridge.analyzer.parser import CudaParser
    >>> from rocm_bridge.engine.recommender import RecommendationEngine

Metadata:
    Version: 1.0.0
    Author: Team 7SENSITIVE
    License: MIT
    Python: 3.11+
"""

__version__ = "1.0.0"
__author__ = "Team 7SENSITIVE"
__license__ = "MIT"
__project__ = "ROCm Bridge"

# Package imports for convenience
from .core.hal import HardwareAbstractionLayer, HardwareProfile
from .core.state import StateManager, FileStatus
from .analyzer.parser import CudaParser, ParseResult
from .analyzer.metrics import PerformanceSimulator
from .engine.recommender import RecommendationEngine
from .engine.hipify_wrapper import HipifyWrapper
from .engine.patcher import ByteOffsetPatcher
from .discovery.scanner import ProjectScanner

# Explicit public API
__all__ = [
    # Version Info
    "__version__",
    "__author__",
    "__license__",
    
    # Layer 1: Core
    "HardwareAbstractionLayer",
    "HardwareProfile",
    "StateManager",
    "FileStatus",
    
    # Layer 2: Discovery
    "ProjectScanner",
    
    # Layer 3: Analyzer
    "CudaParser",
    "ParseResult",
    "PerformanceSimulator",
    
    # Layer 4: Engine
    "RecommendationEngine",
    "HipifyWrapper",
    "ByteOffsetPatcher",
]

# Initialize logging to prevent "No handler found" warnings
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())