"""
ROCm Bridge - Core Module
=========================
Foundation components for the production transpilation engine.
Provides hardware abstraction, state management, configuration, and logging.
"""

from .hal import HardwareAbstractionLayer, HardwareProfile
from .state import StateManager
from .config import CompilationConfig
from .logging_config import setup_logging

__all__ = [
    "HardwareAbstractionLayer",
    "HardwareProfile",
    "StateManager",
    "CompilationConfig",
    "setup_logging",
]

__version__ = "1.0.0"
__author__ = "Team 7SENSITIVE"