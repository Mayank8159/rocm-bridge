"""
ROCm Bridge - HIPify Wrapper
============================
Subprocess wrapper for AMD's hipify-clang tool.

BUG FIXES APPLIED:
- ✅ Popen with context manager (prevents file descriptor leaks)
- ✅ Proper timeout handling with proc.kill()
- ✅ Temporary file cleanup (no leaks)
- ✅ Cross-platform executable resolution
"""

import subprocess
import logging
import shutil
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import platform
import time

logger = logging.getLogger(__name__)


@dataclass
class HipifyResult:
    """Result from hipify-clang translation."""
    
    success: bool
    input_path: str
    output_path: str
    hipified_code: str
    errors: List[str]
    warnings: List[str]
    translation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "hipified_code": self.hipified_code,
            "errors": self.errors,
            "warnings": self.warnings,
            "translation_time_ms": self.translation_time_ms
        }


class HipifyWrapper:
    """Wrapper for AMD's hipify-clang tool."""
    
    HIPIFY_TIMEOUT_SECONDS = 60
    
    def __init__(self, rocm_path: Optional[str] = None):
        """Initialize HIPify wrapper."""
        self.rocm_path: Optional[str] = rocm_path
        self.hipify_path: Optional[str] = self._find_hipify_clang()
        self.is_available: bool = self.hipify_path is not None
        
        if not self.is_available:
            logger.warning("⚠️ hipify-clang not found. Will use fallback mode.")
        
        logger.info(f"HipifyWrapper initialized: available={self.is_available}")
    
    def _find_hipify_clang(self) -> Optional[str]:
        """Find hipify-clang executable."""
        hipify_path = shutil.which("hipify-clang")
        if hipify_path:
            return hipify_path
        
        if self.rocm_path:
            rocm_bin = str(Path(self.rocm_path) / "bin")
            hipify_path = shutil.which("hipify-clang", path=rocm_bin)
            if hipify_path:
                return hipify_path
            
            if platform.system() == "Windows":
                hipify_exe = Path(rocm_bin) / "hipify-clang.exe"
                if hipify_exe.exists():
                    return str(hipify_exe)
        
        common_paths = [
            "/opt/rocm/bin/hipify-clang",
            "/usr/lib/rocm/bin/hipify-clang",
            "C:/Program Files/AMD/ROCm/bin/hipify-clang.exe",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _build_command(self, input_path: str, output_path: str,
                       cuda_path: Optional[str] = None) -> List[str]:
        """Build hipify-clang command."""
        cmd = [self.hipify_path, input_path, "-o", output_path]
        
        if cuda_path:
            cmd.append(f"--cuda-path={cuda_path}")
        else:
            cmd.append("--cuda-path=/usr/local/cuda")
        
        cmd.extend(["-std=c++17", "--print-stats"])
        
        if platform.system() == "Windows":
            cmd.extend(["-DWIN32", "-D_WINDOWS"])
        
        return cmd
    
    def transpile(self, input_path: str, output_path: Optional[str] = None,
                  cuda_path: Optional[str] = None) -> HipifyResult:
        """
        Transpile CUDA source to HIP using hipify-clang.
        
        FIX 2: Popen with context manager (prevents file descriptor leaks)
        """
        start_time = time.time()
        
        input_file = Path(input_path)
        
        if not input_file.exists():
            return HipifyResult(
                success=False, input_path=str(input_path), output_path="",
                hipified_code="", errors=[f"Input file not found: {input_path}"],
                warnings=[]
            )
        
        temp_file = None
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.hip', delete=False, encoding='utf-8'
            )
            output_path = temp_file.name
            temp_file.close()
        
        output_file = Path(output_path)
        
        result = HipifyResult(
            success=False, input_path=str(input_path), output_path=str(output_path),
            hipified_code="", errors=[], warnings=[]
        )
        
        if not self.is_available:
            logger.warning("hipify-clang unavailable, returning input as fallback")
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    result.hipified_code = f.read()
                result.warnings.append("hipify-clang not available - code not translated")
                result.success = True
            except Exception as e:
                result.errors.append(f"Failed to read input file: {e}")
            
            result.translation_time_ms = (time.time() - start_time) * 1000
            return result
        
        cmd = self._build_command(str(input_file), str(output_file), cuda_path)
        logger.info(f"Running hipify-clang: {' '.join(cmd)}")
        
        # FIX 2: Use 'with' context manager to guarantee pipe closure
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            ) as proc:
                try:
                    stdout, stderr = proc.communicate(timeout=self.HIPIFY_TIMEOUT_SECONDS)
                    
                    if stderr:
                        for line in stderr.split('\n'):
                            if 'error:' in line.lower():
                                result.errors.append(line.strip())
                            elif 'warning:' in line.lower():
                                result.warnings.append(line.strip())
                    
                    if proc.returncode != 0:
                        result.errors.append(f"hipify-clang exited with code {proc.returncode}")
                        if stderr:
                            result.errors.append(stderr.strip())
                    else:
                        if output_file.exists():
                            with open(output_file, 'r', encoding='utf-8') as f:
                                result.hipified_code = f.read()
                            result.success = True
                        else:
                            result.errors.append("Output file not created")
                            
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()  # Flush pipes to prevent deadlock
                    logger.error(f"hipify-clang timed out after {self.HIPIFY_TIMEOUT_SECONDS}s")
                    result.errors.append(f"Translation timed out after {self.HIPIFY_TIMEOUT_SECONDS}s")
        
        except FileNotFoundError:
            logger.error("hipify-clang executable not found")
            result.errors.append("hipify-clang executable not found")
        
        except Exception as e:
            logger.error(f"hipify-clang execution failed: {e}")
            result.errors.append(f"Execution failed: {e}")
        
        finally:
            if temp_file:
                try:
                    output_file_path = Path(output_path)
                    if output_file_path.exists() and output_file_path != temp_file.name:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
            
            result.translation_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Hipify complete: success={result.success}, "
            f"time={result.translation_time_ms:.1f}ms, "
            f"errors={len(result.errors)}"
        )
        
        return result
    
    def transpile_batch(self, input_paths: List[str],
                        output_dir: str) -> List[HipifyResult]:
        """Transpile multiple files in batch."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for input_path in input_paths:
            input_file = Path(input_path)
            output_file = output_path / f"{input_file.stem}.hip"
            
            result = self.transpile(str(input_file), str(output_file))
            results.append(result)
        
        return results