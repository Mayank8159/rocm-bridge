"""
ROCm Bridge - File Registry
===========================
Thread-safe file metadata registry with state persistence.

Production Features:
- Thread-safe operations for concurrent access
- Integration with Core StateManager
- File change detection via hash comparison
- Atomic registry updates
- Query and filter operations
- BULK SAVE support to prevent write-amplification
- ZOMBIE FILE CLEANUP (deleted files removed from state)
- ERROR STATE PRESERVATION (scanner errors not overwritten)

BUG FIXES APPLIED:
- ✅ Zombie State Leak fixed (deleted files purged from state.json)
- ✅ Status Overwrite Trap fixed (ERROR states preserved)
- ✅ save_to_disk parameter added and passed through correctly
- ✅ Bulk save at end of batch operations (prevents I/O freeze)
- ✅ All operations under thread lock
- ✅ Path normalization for cross-OS consistency
"""

import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable

from .scanner import FileEntry, ProjectScanner
from rocm_bridge.core.state import StateManager, FileStatus

logger = logging.getLogger(__name__)


class FileRegistry:
    """
    Thread-safe file metadata registry with state persistence.
    
    This registry maintains the current state of all discovered files
    and integrates with the StateManager for persistence across sessions.
    """
    
    def __init__(self, project_path: str, state_manager: Optional[StateManager] = None):
        """
        Initialize file registry.
        
        Args:
            project_path: Root path of the project
            state_manager: Optional StateManager for persistence
        """
        self.project_path: Path = Path(project_path).resolve()
        self.state_manager: Optional[StateManager] = state_manager
        
        # Thread-safe registry storage
        self._lock: threading.Lock = threading.Lock()
        self._files: Dict[str, FileEntry] = {}
        
        # Change tracking
        self._changed_files: Set[str] = set()
        self._new_files: Set[str] = set()
        self._deleted_files: Set[str] = set()
        
        logger.info(f"FileRegistry initialized for {self.project_path}")
    
    def _normalize_path(self, path: str) -> str:
        """Force POSIX-style paths for consistent dictionary keys."""
        return Path(path).as_posix()
    
    def load_from_scanner(self, scanner: ProjectScanner) -> Dict[str, FileEntry]:
        """
        Load files from a ProjectScanner.
        
        FIX #1: Purge deleted files from persistent state (prevents zombie bloat).
        FIX #2: Respect ERROR states from scanner (prevents infinite crash loops).
        
        Args:
            scanner: ProjectScanner instance with completed scan
            
        Returns:
            Dictionary of loaded files
        """
        with self._lock:
            # Get current file paths for deletion detection
            old_paths = set(self._files.keys())
            
            # Load new scan results
            self._files = scanner.registry.copy()
            new_paths = set(self._files.keys())
            
            # Track changes
            self._new_files = new_paths - old_paths
            self._deleted_files = old_paths - new_paths
            self._changed_files = set()
            
            # Check for content changes
            if self.state_manager:
                state_changed = False
                
                # ================================================================
                # FIX #1: ZOMBIE STATE LEAK - Purge deleted files from persistent state
                # ================================================================
                for deleted_path in self._deleted_files:
                    self.state_manager.remove_file(deleted_path)
                    state_changed = True
                    logger.debug(f"Purged deleted file from state: {deleted_path}")
                
                # ================================================================
                # FIX #2: STATUS OVERWRITE TRAP - Respect scanner's initial state
                # ================================================================
                for path, entry in self._files.items():
                    if self.state_manager.is_file_changed(path, entry.content_hash):
                        self._changed_files.add(path)
                        
                        # FIX #2: Preserve ERROR status from scanner
                        # If scanner marked file as ERROR (e.g., hash failed), keep it
                        if entry.status == 'ERROR':
                            target_status = FileStatus.ERROR
                        else:
                            target_status = FileStatus.UNANALYZED
                        
                        # FIX #2: Pass error_message to StateManager
                        self.state_manager.update_file_state(
                            relative_path=path,
                            status=target_status,
                            confidence=0.0,
                            content_hash=entry.content_hash,
                            error_message=entry.error_message,  # ← CRITICAL FIX
                            save_to_disk=False  # ← Bulk save at end
                        )
                        state_changed = True
                
                # FIX: Write to disk exactly ONCE after the loop completes
                if state_changed:
                    self.state_manager.save()
            
            logger.info(
                f"Registry loaded: {len(self._files)} files, "
                f"{len(self._new_files)} new, {len(self._deleted_files)} deleted, "
                f"{len(self._changed_files)} changed"
            )
            
            return self._files.copy()
    
    def get_file(self, relative_path: str) -> Optional[FileEntry]:
        """
        Get file entry by relative path.
        
        Args:
            relative_path: Path relative to project root (POSIX-style)
            
        Returns:
            FileEntry or None if not found
        """
        norm_path = self._normalize_path(relative_path)
        
        with self._lock:
            return self._files.get(norm_path)
    
    def get_all_files(self) -> List[FileEntry]:
        """Get all file entries."""
        with self._lock:
            return list(self._files.values())
    
    def get_convertible_files(self) -> List[FileEntry]:
        """Get only convertible files."""
        with self._lock:
            return [f for f in self._files.values() if f.convertible]
    
    def get_files_by_status(self, status: str) -> List[FileEntry]:
        """Get files matching a specific status."""
        with self._lock:
            return [f for f in self._files.values() if f.status == status]
    
    def get_files_by_category(self, category: str) -> List[FileEntry]:
        """Get files matching a specific category."""
        with self._lock:
            return [f for f in self._files.values() if f.category == category]
    
    def get_changed_files(self) -> List[FileEntry]:
        """Get files that have changed since last analysis."""
        with self._lock:
            return [
                self._files[path] 
                for path in self._changed_files 
                if path in self._files
            ]
    
    def get_new_files(self) -> List[FileEntry]:
        """Get newly discovered files."""
        with self._lock:
            return [
                self._files[path]
                for path in self._new_files
                if path in self._files
            ]
    
    def update_file_status(
        self, 
        relative_path: str, 
        status: str, 
        confidence: float = 0.0, 
        error_message: str = "",
        save_to_disk: bool = True
    ) -> bool:
        """
        Update status for a single file.
        
        Args:
            relative_path: Path relative to project root
            status: New status value
            confidence: Analysis confidence score (0.0 to 1.0)
            error_message: Error message if status is ERROR
            save_to_disk: Whether to immediately write to disk (default True)
            
        Returns:
            True if update successful, False if file not found
        """
        norm_path = self._normalize_path(relative_path)
        
        with self._lock:
            if norm_path not in self._files:
                logger.warning(f"File not found in registry: {relative_path}")
                return False
            
            self._files[norm_path].status = status
            
            if error_message:
                self._files[norm_path].error_message = error_message
            
            # Update state manager if available
            if self.state_manager:
                self.state_manager.update_file_state(
                    relative_path=norm_path,
                    status=FileStatus(status),
                    confidence=confidence,
                    content_hash=self._files[norm_path].content_hash,
                    error_message=error_message,
                    save_to_disk=save_to_disk
                )
            
            return True
    
    def update_files_status(self, relative_paths: List[str], status: str) -> int:
        """
        Update status for multiple files.
        
        FIX: Batch updates with single disk write at end.
        
        Args:
            relative_paths: List of relative paths
            status: New status value
            
        Returns:
            Number of files successfully updated
        """
        updated = 0
        
        for path in relative_paths:
            # Set save_to_disk=False for the loop
            if self.update_file_status(path, status, save_to_disk=False):
                updated += 1
        
        # Trigger single bulk save at the end
        if updated > 0 and self.state_manager:
            self.state_manager.save()
        
        return updated
    
    def mark_analyzed(self, relative_path: str, convertible: bool = True) -> bool:
        """
        Mark a file as analyzed.
        
        Args:
            relative_path: Path relative to project root
            convertible: Whether the file is convertible
            
        Returns:
            True if update successful
        """
        status = "CONVERTIBLE" if convertible else "SKIP"
        return self.update_file_status(relative_path, status, confidence=1.0)
    
    def mark_error(self, relative_path: str, error_message: str) -> bool:
        """
        Mark a file as having an error.
        
        Args:
            relative_path: Path relative to project root
            error_message: Error description
            
        Returns:
            True if update successful
        """
        return self.update_file_status(
            relative_path, 
            "ERROR", 
            error_message=error_message
        )
    
    def remove_file(self, relative_path: str) -> bool:
        """
        Remove a file from the registry.
        
        Args:
            relative_path: Path relative to project root
            
        Returns:
            True if file was removed, False if not found
        """
        norm_path = self._normalize_path(relative_path)
        
        with self._lock:
            if norm_path in self._files:
                del self._files[norm_path]
                self._deleted_files.add(norm_path)
                
                # Update state manager
                if self.state_manager:
                    self.state_manager.remove_file(norm_path)
                
                logger.debug(f"Removed file from registry: {relative_path}")
                return True
            
            return False
    
    def clear_changes(self) -> None:
        """Clear change tracking sets."""
        with self._lock:
            self._changed_files.clear()
            self._new_files.clear()
            self._deleted_files.clear()
    
    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        with self._lock:
            stats = {
                'total_files': len(self._files),
                'convertible': 0,
                'unanalyzed': 0,
                'analyzed': 0,
                'errors': 0,
                'changed': len(self._changed_files),
                'new': len(self._new_files),
                'deleted': len(self._deleted_files),
                'by_category': {},
                'by_status': {}
            }
            
            for entry in self._files.values():
                if entry.convertible:
                    stats['convertible'] += 1
                
                # Status breakdown
                status = entry.status
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                
                if status == 'UNANALYZED':
                    stats['unanalyzed'] += 1
                elif status == 'ERROR':
                    stats['errors'] += 1
                else:
                    stats['analyzed'] += 1
                
                # Category breakdown
                cat = entry.category
                stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            return stats
    
    def filter(self, predicate: Callable[[FileEntry], bool]) -> List[FileEntry]:
        """
        Filter files using a custom predicate function.
        
        Args:
            predicate: Function that takes FileEntry and returns bool
            
        Returns:
            List of files matching the predicate
        """
        with self._lock:
            return [f for f in self._files.values() if predicate(f)]
    
    def clear(self) -> None:
        """Clear the registry."""
        with self._lock:
            self._files.clear()
            self.clear_changes()
            logger.debug("Registry cleared")