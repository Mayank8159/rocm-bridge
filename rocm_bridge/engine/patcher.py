"""
ROCm Bridge - Byte Offset Patcher
=================================
Safe code rewriting using Clang byte offsets.

Production Features:
- Byte-offset precise patching (RAW BYTES, not strings)
- Encoding detection for diff generation
- Descending order patch application (prevents offset drift)
- Atomic file writes with backup verification
- Patch validation before application
- Overlap detection for conflicting patches
- Memory-safe diff generation

BUG FIXES APPLIED:
- ✅ RAW BYTE splicing (not string slicing - prevents UTF-8 corruption)
- ✅ Encoding passed to diff generator (prevents UnicodeDecodeError)
- ✅ Removed dead offset_delta code (descending sort handles it)
- ✅ Thread lock released on exception (try/finally)
- ✅ File size limit before diff (prevents memory spike)
- ✅ Backup file verification before overwrite
"""

import logging
import os
import shutil
import threading
import chardet
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """Represents a single code patch."""
    
    start_offset: int
    end_offset: int
    original_text: str
    replacement_text: str
    rule_id: str
    confidence: float = 1.0
    line_number: int = 0
    
    def __post_init__(self):
        """Validate patch on creation."""
        if self.start_offset < 0:
            raise ValueError(f"Invalid start_offset: {self.start_offset}")
        if self.end_offset < self.start_offset:
            raise ValueError(f"end_offset ({self.end_offset}) < start_offset ({self.start_offset})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "original_text": self.original_text,
            "replacement_text": self.replacement_text,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "line_number": self.line_number
        }


@dataclass
class PatchResult:
    """Result from patch application."""
    
    success: bool
    input_path: str
    output_path: str
    patches_applied: int
    backup_path: str
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "patches_applied": self.patches_applied,
            "backup_path": self.backup_path,
            "errors": self.errors
        }


class ByteOffsetPatcher:
    """
    Safe code rewriting using byte offsets from Clang AST.
    
    CRITICAL: Uses RAW BYTE splicing (not string slicing) to prevent
    UTF-8 corruption when files contain multi-byte characters.
    """
    
    MAX_DIFF_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, backup_dir: Optional[str] = None):
        """Initialize patcher."""
        self.backup_dir: Optional[str] = backup_dir
        self._lock: threading.Lock = threading.Lock()
        logger.info("ByteOffsetPatcher initialized")
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result.get('encoding', 'utf-8')
            if encoding:
                try:
                    ''.encode(encoding)
                    return encoding
                except (LookupError, UnicodeEncodeError):
                    pass
            return 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
            return 'utf-8'
    
    def _create_backup(self, file_path: str) -> str:
        """Create and verify backup of original file."""
        source = Path(file_path)
        
        if self.backup_dir:
            backup_path = Path(self.backup_dir) / f"{source.stem}.backup{source.suffix}"
        else:
            backup_path = source.parent / f"{source.stem}.backup{source.suffix}"
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(source, backup_path)
            
            if not backup_path.exists():
                raise IOError("Backup file was not created")
            
            if backup_path.stat().st_size != source.stat().st_size:
                raise IOError("Backup file size mismatch")
            
            logger.debug(f"Created and verified backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _validate_patches(self, patches: List[Patch], file_size: int) -> List[str]:
        """Validate patches before application."""
        errors = []
        
        for i, patch in enumerate(patches):
            if patch.start_offset < 0:
                errors.append(f"Patch {i}: Negative start_offset ({patch.start_offset})")
            
            if patch.end_offset > file_size:
                errors.append(f"Patch {i}: end_offset ({patch.end_offset}) exceeds file size ({file_size})")
            
            if patch.end_offset < patch.start_offset:
                errors.append(f"Patch {i}: end_offset < start_offset")
        
        # Check for overlapping patches
        sorted_patches = sorted(patches, key=lambda p: p.start_offset)
        for i in range(len(sorted_patches) - 1):
            if sorted_patches[i].end_offset > sorted_patches[i + 1].start_offset:
                errors.append(
                    f"Patches {i} and {i+1} overlap: "
                    f"[{sorted_patches[i].start_offset}-{sorted_patches[i].end_offset}] vs "
                    f"[{sorted_patches[i+1].start_offset}-{sorted_patches[i+1].end_offset}]"
                )
        
        return errors
    
    def apply_patches(self, file_path: str, patches: List[Patch],
                      output_path: Optional[str] = None) -> PatchResult:
        """
        Apply multiple patches to a source file.
        
        FIX 1: RAW BYTE splicing (not string slicing)
        FIX 4: Removed dead offset_delta code
        """
        lock_acquired = False
        try:
            lock_acquired = self._lock.acquire(timeout=30)
            
            if not lock_acquired:
                return PatchResult(
                    success=False, input_path=file_path, output_path="",
                    patches_applied=0, backup_path="", errors=["Lock timeout"]
                )
            
            source_path = Path(file_path)
            
            if not source_path.exists():
                return PatchResult(
                    success=False, input_path=file_path, output_path="",
                    patches_applied=0, backup_path="", errors=["Not found"]
                )
            
            if not patches:
                return PatchResult(
                    success=True, input_path=file_path,
                    output_path=output_path or file_path,
                    patches_applied=0, backup_path=""
                )
            
            # FIX 1: Detect encoding for diff generator, but read as RAW BYTES for patching
            encoding = self._detect_encoding(source_path)
            
            # FIX 1: Read as RAW BYTES (not string) to prevent UTF-8 corruption
            try:
                with open(source_path, 'rb') as f:
                    file_bytes = bytearray(f.read())
            except Exception as e:
                return PatchResult(
                    success=False, input_path=file_path, output_path="",
                    patches_applied=0, backup_path="", errors=[str(e)]
                )
            
            file_size = len(file_bytes)
            
            validation_errors = self._validate_patches(patches, file_size)
            if validation_errors:
                return PatchResult(
                    success=False, input_path=file_path, output_path="",
                    patches_applied=0, backup_path="", errors=validation_errors
                )
            
            backup_path = self._create_backup(file_path)
            
            # Sort descending to prevent offset drift for preceding code
            sorted_patches = sorted(patches, key=lambda p: p.start_offset, reverse=True)
            applied_count = 0
            
            for patch in sorted_patches:
                try:
                    start = patch.start_offset
                    end = patch.end_offset
                    
                    if start > len(file_bytes) or end > len(file_bytes):
                        continue
                    
                    # FIX 1: Splice RAW BYTES precisely using Clang's byte offsets
                    replacement_bytes = patch.replacement_text.encode(encoding)
                    file_bytes[start:end] = replacement_bytes
                    applied_count += 1
                    
                except Exception as e:
                    logger.warning(f"Patch failed: {e}")
            
            output_file = Path(output_path or file_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = output_file.with_suffix('.tmp')
            
            try:
                # FIX 1: Write as RAW BYTES (not string)
                with open(temp_file, 'wb') as f:
                    f.write(file_bytes)
                
                temp_file.replace(output_file)
                
                return PatchResult(
                    success=True, input_path=file_path,
                    output_path=str(output_file),
                    patches_applied=applied_count, backup_path=backup_path
                )
                
            except Exception as e:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                return PatchResult(
                    success=False, input_path=file_path, output_path="",
                    patches_applied=applied_count, backup_path=backup_path,
                    errors=[str(e)]
                )
        
        finally:
            if lock_acquired:
                self._lock.release()
    
    def generate_unified_diff(self, original_path: str, modified_path: str,
                              original_label: str = "a", modified_label: str = "b",
                              encoding: str = 'utf-8') -> str:
        """
        Generate unified diff between two files.
        
        FIX 3: Accepts dynamic encoding parameter
        """
        import difflib
        
        try:
            original_size = Path(original_path).stat().st_size
            modified_size = Path(modified_path).stat().st_size
            
            if original_size > self.MAX_DIFF_FILE_SIZE or modified_size > self.MAX_DIFF_FILE_SIZE:
                return f"// Diff skipped: File too large ({max(original_size, modified_size)} bytes)"
            
            # FIX 3: Use detected encoding (not hardcoded utf-8)
            with open(original_path, 'r', encoding=encoding) as f:
                original_lines = f.readlines()
            with open(modified_path, 'r', encoding=encoding) as f:
                modified_lines = f.readlines()
            
            # Fast bypass for identical files
            if original_lines == modified_lines:
                return ""
            
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"{original_label}/{Path(original_path).name}",
                tofile=f"{modified_label}/{Path(modified_path).name}",
                lineterm='',
                n=3
            )
            
            return ''.join(diff)
            
        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return f"// Error generating diff: {e}"
    
    def generate_patch_file(self, original_path: str, patches: List[Patch],
                            output_path: str) -> bool:
        """
        Generate a .patch file for CI/CD integration.
        
        FIX 1: RAW BYTE splicing
        FIX 3: Pass encoding to diff generator
        """
        try:
            source_path = Path(original_path)
            encoding = self._detect_encoding(source_path)
            
            # FIX 1: Read as RAW BYTES
            with open(source_path, 'rb') as f:
                file_bytes = bytearray(f.read())
            
            sorted_patches = sorted(patches, key=lambda p: p.start_offset, reverse=True)
            
            for patch in sorted_patches:
                start = patch.start_offset
                end = patch.end_offset
                
                if start > len(file_bytes) or end > len(file_bytes):
                    continue
                
                # FIX 1: Splice RAW BYTES
                file_bytes[start:end] = patch.replacement_text.encode(encoding)
            
            temp_modified = Path(output_path + ".modified")
            
            # FIX 1: Write as RAW BYTES
            with open(temp_modified, 'wb') as f:
                f.write(file_bytes)
            
            # FIX 3: Pass encoding to diff generator
            diff = self.generate_unified_diff(
                original_path, str(temp_modified),
                "original", "optimized", encoding
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(diff)
            
            if temp_modified.exists():
                temp_modified.unlink()
            
            logger.info(f"Generated patch file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate patch: {e}")
            return False