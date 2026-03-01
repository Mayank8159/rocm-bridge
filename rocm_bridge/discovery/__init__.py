"""
ROCm Bridge - Discovery Module
==============================
File system scanning and registry management for the transpilation daemon.

This module provides:
- Bounded recursive directory scanning (respects .gitignore)
- File type detection and categorization
- Content hashing for change detection
- Thread-safe registry operations

Usage:
    >>> from discovery import ProjectScanner, FileRegistry
    >>> scanner = ProjectScanner("/path/to/project")
    >>> registry = scanner.scan()
    >>> print(f"Found {len(registry.files)} convertible files")
"""

from .scanner import ProjectScanner, FileEntry
from .registry import FileRegistry

__all__ = [
    "ProjectScanner",
    "FileEntry",
    "FileRegistry",
]

__version__ = "1.0.0"