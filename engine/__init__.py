"""
ROCm Bridge Optimization Engine
===============================
The intelligence layer for the ROCm Bridge portability tool.

This module is responsible for correlating static analysis findings with
dynamic hardware telemetry to generate high-confidence, AMD-native
optimization recommendations.

Usage Example:
    >>> from engine import engine
    >>> report = engine.generate(static_issues, profile_metrics)
    >>> print(f"Estimated Speedup: {report['summary']['estimated_speedup']}")

Metadata:
    Project: ROCm Bridge
    License: MIT
    Author: Team 7SENSITIVE
"""

import sys
import logging

# --- PACKAGE METADATA ---
__project__ = "ROCm Bridge"
__author__ = "Team 7SENSITIVE"
__license__ = "MIT"
__version__ = "0.1.0"

# --- LOGGING CONFIGURATION ---
# Prevent "No handler found" warnings if the app doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# --- PUBLIC API EXPORTS ---
# Safe import handling allows the package to be inspected even if dependencies fail
try:
    from .recommender import RecommendationEngine, Recommendation, engine
except ImportError as e:
    sys.stderr.write(f"[ROCm Bridge] Warning: Failed to import optimization engine: {e}\n")
    
    # Define dummy exports to prevent crash on import *
    RecommendationEngine = None
    Recommendation = None
    engine = None

# Explicitly define the public API surface
__all__ = [
    "RecommendationEngine",
    "Recommendation",
    "engine",
    "__version__",
    "__author__"
]