# ============================================================================
# COMPLETELY REPLACED core/state.py WITH ALL FIXES APPLIED
# ============================================================================

import json
import hashlib
import threading
import logging
import time  # FIX: Added for retry logic
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# FIX: Safe fcntl import for cross-platform compatibility
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = logging.getLogger(__name__)


class FileStatus(Enum):
    """Enumeration of file processing states."""
    UNANALYZED = "UNANALYZED"
    ANALYZING = "ANALYZING"
    CONVERTIBLE = "CONVERTIBLE"
    PARTIAL = "PARTIAL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    CONVERTED = "CONVERTED"
    EXCLUDED = "EXCLUDED"


@dataclass
class FileState:
    """Immutable file state record."""
    
    relative_path: str
    content_hash: str
    status: str
    confidence: float
    last_analyzed: str
    error_message: str = ""
    converted_path: str = ""
    patch_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        return cls(
            relative_path=data.get("relative_path", ""),
            content_hash=data.get("content_hash", ""),
            status=data.get("status", FileStatus.UNANALYZED.value),
            confidence=data.get("confidence", 0.0),
            last_analyzed=data.get("last_analyzed", ""),
            error_message=data.get("error_message", ""),
            converted_path=data.get("converted_path", ""),
            patch_path=data.get("patch_path", "")
        )
    # UPDATE THE validate() METHOD IN FileState CLASS

    def validate(self) -> bool:
        """Validate file state integrity."""
     # FIX #6: Allow empty hash for ERROR files (don't silently drop)
        if self.content_hash and len(self.content_hash) != 64:
            logger.warning(f"Invalid hash length for {self.relative_path}")
            return False
    
    # Validate status is known enum value
        try:
            FileStatus(self.status)
            return True
        except ValueError:
            logger.warning(f"Unknown status for {self.relative_path}: {self.status}")
            return False


class StateManager:
    """Thread-safe state persistence with cross-platform support."""
    
    def __init__(self, project_path: str):
        self.project_path: Path = Path(project_path).resolve()
        self.state_file: Path = self.project_path / ".rocm_bridge_state.json"
        
        # FIX #1: Single lock for ALL operations (no read/write lock split)
        self._lock: threading.Lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        
        self._load_state()
        logger.info(f"StateManager initialized for {self.project_path}")
    
    # FIX #2: Path normalization for cross-OS cache consistency
    def _normalize_path(self, path: str) -> str:
        """Force POSIX-style paths for consistent dictionary keys."""
        return Path(path).as_posix()
    
    def _load_state(self) -> None:
        with self._lock:
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        # FIX #3: Conditional fcntl usage
                        if HAS_FCNTL:
                            try:
                                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                                self._state = json.load(f)
                            finally:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        else:
                            self._state = json.load(f)
                    logger.info(f"Loaded existing state from {self.state_file}")
                except json.JSONDecodeError as e:
                    logger.error(f"State file corrupted: {e}. Rebuilding cache.")
                    self._state = self._create_empty_state()
                    self._save_state_unsafe()
                except Exception as e:
                    logger.error(f"Failed to load state file: {e}")
                    self._state = self._create_empty_state()
            else:
                self._state = self._create_empty_state()
                logger.info("Created new state file")
    
    def _create_empty_state(self) -> Dict[str, Any]:
        return {
            "project_path": str(self.project_path),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "files": {},
            "statistics": {
                "total_files": 0,
                "analyzed": 0,
                "converted": 0,
                "errors": 0
            }
        }
    
    def _save_state_unsafe(self) -> None:
        """Atomic write with Windows retry logic."""
        self._state["last_updated"] = datetime.now().isoformat()
        
        temp_file = self.state_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, default=str)
            
            # FIX #4: Windows PermissionError retry loop
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    temp_file.replace(self.state_file)
                    logger.debug(f"State saved to {self.state_file}")
                    break
                except PermissionError as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.05)  # 50ms backoff
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise
    
    def calculate_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha256.update(chunk)
            hash_result = sha256.hexdigest()
            if len(hash_result) != 64:
                logger.error(f"Invalid hash generated for {file_path}")
                return ""
            return hash_result
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def is_file_changed(self, relative_path: str, current_hash: str) -> bool:
        # FIX #5: Normalize path before lookup
        norm_path = self._normalize_path(relative_path)
        with self._lock:
            if norm_path not in self._state.get("files", {}):
                return True
            stored_hash = self._state["files"][norm_path].get("content_hash", "")
            return stored_hash != current_hash
    
    
    def _update_statistics_unsafe(self) -> None:
        files = self._state.get("files", {})
        self._state["statistics"] = {
            "total_files": len(files),
            "analyzed": sum(1 for f in files.values() 
                          if f.get("status") != FileStatus.UNANALYZED.value),
            "converted": sum(1 for f in files.values() 
                           if f.get("status") == FileStatus.CONVERTED.value),
            "errors": sum(1 for f in files.values() 
                        if f.get("status") == FileStatus.ERROR.value)
        }
    
    def get_file_state(self, relative_path: str) -> Optional[FileState]:
        norm_path = self._normalize_path(relative_path)
        with self._lock:
            file_data = self._state.get("files", {}).get(norm_path)
            if file_data:
                state = FileState.from_dict(file_data)
                if state.validate():
                    return state
                logger.warning(f"Invalid state data for {relative_path}")
            return None
    
    def get_all_files(self) -> List[FileState]:
        with self._lock:
            states = []
            for data in self._state.get("files", {}).values():
                state = FileState.from_dict(data)
                if state.validate():
                    states.append(state)
            return states
    
    def get_files_by_status(self, status: FileStatus) -> List[FileState]:
        with self._lock:
            return [
                FileState.from_dict(data)
                for data in self._state.get("files", {}).values()
                if data.get("status") == status.value
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.get("statistics", {}).copy()
    
    def clear_state(self) -> None:
        with self._lock:
            self._state = self._create_empty_state()
            self._save_state_unsafe()
            logger.info("State cache cleared")
    
    def remove_file(self, relative_path: str) -> None:
        norm_path = self._normalize_path(relative_path)
        with self._lock:
            if norm_path in self._state.get("files", {}):
                del self._state["files"][norm_path]
                self._update_statistics_unsafe()
                self._save_state_unsafe()

    # ADD THESE METHODS TO THE EXISTING StateManager CLASS

    def update_file_state(
    self, 
    relative_path: str, 
    status: FileStatus,
    confidence: float, 
    content_hash: str,
    error_message: str = "", 
    converted_path: str = "",
    patch_path: str = "",
    save_to_disk: bool = True  # FIX #2: Added parameter
    ) -> None:
        """
    Thread-safe update of file state.
    
    Args:
        relative_path: Relative path of file within project
        status: Current processing status
        confidence: Analysis confidence score (0.0 to 1.0)
        content_hash: SHA-256 hash of file content
        error_message: Error message if status is ERROR
        converted_path: Path to converted file if applicable
        patch_path: Path to patch file if applicable
        save_to_disk: Whether to immediately write to disk (default True)
        """
        with self._lock:
            if "files" not in self._state:
                self._state["files"] = {}
        
            self._state["files"][relative_path] = {
            "relative_path": relative_path,
            "content_hash": content_hash,
            "status": status.value,
            "confidence": confidence,
            "last_analyzed": datetime.now().isoformat(),
            "error_message": error_message,
            "converted_path": converted_path,
            "patch_path": patch_path
            }
        
        # Update statistics
            self._update_statistics_unsafe()
        
        # FIX #2: Only save to disk if requested
            if save_to_disk:
                self._save_state_unsafe()

    def save(self) -> None:
        """
    FIX #2: Public method to manually trigger a bulk save.
    Call this after batch updates to write to disk once.
        """
        with self._lock:
            self._save_state_unsafe()
    