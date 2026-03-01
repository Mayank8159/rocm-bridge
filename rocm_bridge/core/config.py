"""
ROCm Bridge - Compilation Configuration
=======================================
Manages compiler flags, include paths, and compilation database.
Ensures libclang can properly parse CUDA/HIP code.

Production Features:
- compile_commands.json support (both 'command' and 'arguments' schema)
- Automatic include path detection
- Platform-specific configuration (Windows, Linux, macOS)
- Environment variable overrides

BUG FIXES APPLIED:
- Added support for modern CMake 'arguments' array schema
- Fixed hardcoded paths for Windows compatibility
- Added validation for include path existence
"""

import os
import json
import logging
import shutil
import sys
import shlex
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompilationConfig:
    """Compilation configuration container."""
    
    cuda_path: str = ""
    rocm_path: str = ""
    include_paths: List[str] = field(default_factory=list)
    compiler_flags: List[str] = field(default_factory=list)
    std_version: str = "c++17"
    cuda_arch: str = "sm_50"
    
    def get_clang_args(self) -> List[str]:
        """Generate libclang-compatible argument list."""
        args = [
            f"-std={self.std_version}",
            "-x", "cuda",
            f"--cuda-gpu-arch={self.cuda_arch}",
            "-D__CUDACC__",
            "-D__HIP_PLATFORM_AMD__"
        ]
        
        for include in self.include_paths:
            # Validate include path exists before adding
            if Path(include).exists():
                args.extend(["-I", include])
            else:
                logger.debug(f"Skipping non-existent include path: {include}")
        
        args.extend(self.compiler_flags)
        return args
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "cuda_path": self.cuda_path,
            "rocm_path": self.rocm_path,
            "include_paths": self.include_paths,
            "compiler_flags": self.compiler_flags,
            "std_version": self.std_version,
            "cuda_arch": self.cuda_arch
        }


class CompilationConfigManager:
    """
    Manages compilation configuration for AST parsing.
    Supports compile_commands.json and automatic path detection.
    """
    
    # Platform-specific CUDA paths
    CUDA_PATHS = {
        "Linux": [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda"
        ],
        "Windows": [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            "C:/CUDA"
        ],
        "Darwin": [
            "/usr/local/cuda",
            "/opt/homebrew/opt/cuda"
        ]
    }
    
    # Platform-specific ROCm paths
    ROCM_PATHS = {
        "Linux": [
            "/opt/rocm",
            "/usr/lib/rocm",
            "/usr/local/rocm"
        ],
        "Windows": [
            "C:/Program Files/AMD/ROCm"
        ],
        "Darwin": []  # ROCm not officially supported on macOS
    }
    
    def __init__(self, project_path: str):
        """
        Initialize configuration manager.
        
        Args:
            project_path: Root path of the project
        """
        self.project_path: Path = Path(project_path).resolve()
        self.config: CompilationConfig = CompilationConfig()
        
        self._detect_toolchains()
        self._load_compile_commands()
        self._add_default_includes()
        
        logger.info(f"CompilationConfig initialized: {self.config.to_dict()}")
    
    def _detect_toolchains(self) -> None:
        """Detect CUDA and ROCm installation paths."""
        platform_name = platform.system()
        
        # CUDA Detection
        cuda_env = os.environ.get("CUDA_PATH", "")
        if cuda_env:
            self.config.cuda_path = cuda_env
        else:
            # Check platform-specific paths
            for path in self.CUDA_PATHS.get(platform_name, []):
                if Path(path).exists():
                    self.config.cuda_path = path
                    break
            
            # Also check PATH for nvcc
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                # Extract CUDA path from nvcc location
                cuda_bin = Path(nvcc_path).parent
                self.config.cuda_path = str(cuda_bin.parent)
        
        # ROCm Detection
        rocm_env = os.environ.get("ROCM_PATH", "")
        if rocm_env:
            self.config.rocm_path = rocm_env
        else:
            # Check platform-specific paths
            for path in self.ROCM_PATHS.get(platform_name, []):
                if Path(path).exists():
                    self.config.rocm_path = path
                    break
            
            # Also check PATH for hipcc
            hipcc_path = shutil.which("hipcc")
            if hipcc_path:
                # Extract ROCm path from hipcc location
                rocm_bin = Path(hipcc_path).parent
                self.config.rocm_path = str(rocm_bin.parent)
        
        logger.debug(f"CUDA Path: {self.config.cuda_path}")
        logger.debug(f"ROCm Path: {self.config.rocm_path}")
    
    def _add_default_includes(self) -> None:
        """Add default include paths for CUDA and HIP."""
        default_includes: List[str] = []
        
        # CUDA includes
        if self.config.cuda_path:
            cuda_include = Path(self.config.cuda_path) / "include"
            if cuda_include.exists():
                default_includes.append(str(cuda_include))
        
        # ROCm includes
        if self.config.rocm_path:
            rocm_include = Path(self.config.rocm_path) / "include"
            if rocm_include.exists():
                default_includes.append(str(rocm_include))
            
            # HIP specific includes
            hip_include = Path(self.config.rocm_path) / "include" / "hip"
            if hip_include.exists():
                default_includes.append(str(hip_include))
        
        # System includes (platform-specific)
        platform_name = platform.system()
        if platform_name == "Linux":
            system_includes = [
                "/usr/include",
                "/usr/local/include",
                "/usr/include/x86_64-linux-gnu"
            ]
        elif platform_name == "Darwin":
            system_includes = [
                "/usr/include",
                "/usr/local/include",
                "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"
            ]
        elif platform_name == "Windows":
            system_includes = []  # Windows uses different include mechanism
        else:
            system_includes = []
        
        for inc in system_includes:
            if Path(inc).exists():
                default_includes.append(inc)
        
        # Add to config (avoid duplicates)
        for inc in default_includes:
            if inc not in self.config.include_paths:
                self.config.include_paths.append(inc)
    
    def get_config(self) -> CompilationConfig:
        """Get current compilation configuration."""
        return self.config
    
    def get_clang_args(self) -> List[str]:
        """Get libclang-compatible argument list."""
        return self.config.get_clang_args()
    
    def _load_compile_commands(self) -> None:
        compile_db_path = self.project_path / "compile_commands.json"
        
        if not compile_db_path.exists():
            logger.debug("No compile_commands.json found")
            return
        
        try:
            with open(compile_db_path, 'r', encoding='utf-8') as f:
                compile_db = json.load(f)
            
            include_paths: Set[str] = set()
            
            for entry in compile_db:
                if "arguments" in entry and isinstance(entry["arguments"], list):
                    parts = entry["arguments"]
                else:
                    command = entry.get("command", "")
                    if not command:
                        continue
                    # FIX #6: Use shlex.split for safe path parsing
                    try:
                        parts = shlex.split(command, posix=(platform.system() != "Windows"))
                    except ValueError:
                        parts = command.split()
                
                for i, part in enumerate(parts):
                    if part == "-I" and i + 1 < len(parts):
                        include_paths.add(parts[i + 1])
                    elif part.startswith("-I"):
                        include_paths.add(part[2:])
                    if part == "-isystem" and i + 1 < len(parts):
                        include_paths.add(parts[i + 1])
                    elif part.startswith("-isystem"):
                        include_paths.add(part[8:])
            
            for inc in include_paths:
                if inc not in self.config.include_paths:
                    self.config.include_paths.append(inc)
            
            logger.info(f"Loaded {len(include_paths)} include paths from compile_commands.json")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid compile_commands.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load compile_commands.json: {e}")

# ============================================================================
# FIX find_hipcc AND find_hipify_clang METHODS
# ============================================================================

    def find_hipcc(self) -> Optional[str]:
        """Find hipcc compiler in PATH."""
        hipcc_path = shutil.which("hipcc")
        if hipcc_path:
            return hipcc_path
        
        # FIX #7: Use shutil.which with path argument for .exe resolution
        if self.config.rocm_path:
            rocm_bin = str(Path(self.config.rocm_path) / "bin")
            hipcc_path = shutil.which("hipcc", path=rocm_bin)
            if hipcc_path:
                return hipcc_path
                
        logger.warning("hipcc not found in PATH or ROCm installation")
        return None
    
    def find_hipify_clang(self) -> Optional[str]:
        """Find hipify-clang tool in PATH."""
        hipify_path = shutil.which("hipify-clang")
        if hipify_path:
            return hipify_path
        
        # FIX #7: Use shutil.which with path argument for .exe resolution
        if self.config.rocm_path:
            rocm_bin = str(Path(self.config.rocm_path) / "bin")
            hipify_path = shutil.which("hipify-clang", path=rocm_bin)
            if hipify_path:
                return hipify_path
        
        logger.warning("hipify-clang not found in PATH or ROCm installation")
        return None

    def add_include_path(self, path: str) -> None:
        """Add an include path to the configuration."""
        if path not in self.config.include_paths:
            self.config.include_paths.append(path)
            logger.debug(f"Added include path: {path}")
    
    def add_compiler_flag(self, flag: str) -> None:
        """Add a compiler flag to the configuration."""
        if flag not in self.config.compiler_flags:
            self.config.compiler_flags.append(flag)
            logger.debug(f"Added compiler flag: {flag}")


# Import platform for the class above
import platform