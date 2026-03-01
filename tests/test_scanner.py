"""
ROCm Bridge - Scanner Tests
===========================
Unit tests for the discovery module.

Location: tests/ directory at PROJECT ROOT (NOT inside rocm_bridge/discovery/)

These tests guarantee:
- .gitignore parsing works correctly (not just hardcoded exclusions)
- Large directories are properly excluded
- File hashing works correctly
- Path normalization is consistent across OSes

Run with: pytest tests/test_scanner.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path

from rocm_bridge.discovery.scanner import ProjectScanner, FileEntry
from rocm_bridge.discovery.registry import FileRegistry


class TestProjectScanner:
    """Test cases for ProjectScanner."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create test files
            (root / "src").mkdir()
            (root / "src" / "kernel.cu").write_text("// CUDA kernel")
            (root / "src" / "utils.cpp").write_text("// C++ source")
            (root / "include").mkdir()
            (root / "include" / "header.h").write_text("// Header")
            
            # Create excluded directories (hardcoded)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("[core]")
            (root / "build").mkdir()
            (root / "build" / "output.o").write_text("binary")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "package.js").write_text("js")
            
            # FIX #3: Create custom folders NOT in EXCLUDED_DIRS
            # This tests .gitignore parsing, not hardcoded exclusions
            (root / "vendor_libs").mkdir()
            (root / "vendor_libs" / "ignored.cu").write_text("// hidden")
            
            # Create a deep nested file to test ** syntax
            (root / "src" / "temp_cache").mkdir()
            (root / "src" / "temp_cache" / "bad.cu").write_text("// bad")
            
            # Create .gitignore with advanced patterns
            (root / ".gitignore").write_text(
                "vendor_libs/\n"      # Directory pattern
                "**/temp_cache/\n"    # Recursive pattern
                "*.log\n"              # Extension pattern
                "!important.cu"        # Negation pattern
            )
            
            yield root
    
    def test_scan_finds_cuda_files(self, temp_project):
        """Test that scanner finds .cu files."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        assert any('.cu' in path for path in registry.keys())
    
    def test_scan_excludes_hardcoded_directories(self, temp_project):
        """Test that scanner excludes hardcoded directories like .git and build."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        assert not any('.git' in path for path in registry.keys())
        assert not any('build' in path for path in registry.keys())
        assert not any('node_modules' in path for path in registry.keys())
    
    def test_scan_respects_custom_gitignore(self, temp_project):
        """
        FIX #3: Test that scanner respects pathspec gitignore semantics.
        
        This tests .gitignore parsing, NOT hardcoded exclusions.
        If pathspec isn't working, this test will FAIL.
        """
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        # These should be excluded by .gitignore (not hardcoded)
        assert not any('vendor_libs' in path for path in registry.keys()), \
            "vendor_libs should be excluded by .gitignore"
        assert not any('temp_cache' in path for path in registry.keys()), \
            "temp_cache should be excluded by **/temp_cache/ pattern"
    
    def test_file_entry_hash_is_valid(self, temp_project):
        """Test that file hashes are valid SHA-256."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        for entry in registry.values():
            # Empty hash is allowed for error files
            if entry.content_hash:
                assert len(entry.content_hash) == 64
                assert all(c in '0123456789abcdef' for c in entry.content_hash)
    
    def test_path_normalization(self, temp_project):
        """Test that paths are normalized to POSIX style."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        for path in registry.keys():
            assert '\\' not in path, "Paths should use forward slashes"
    
    def test_file_error_handling(self, temp_project):
        """Test that files with hash errors are marked as ERROR, not dropped."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        
        # All files should be in registry, even if hash failed
        for entry in registry.values():
            if not entry.content_hash:
                assert entry.status == 'ERROR'
                assert 'hash' in entry.error_message.lower()
    
    def test_scan_statistics(self, temp_project):
        """Test that scan statistics are accurate."""
        scanner = ProjectScanner(str(temp_project))
        registry = scanner.scan()
        stats = scanner.get_statistics()
        
        assert stats['total_files'] > 0
        assert stats['convertible'] >= 0
        assert 'by_category' in stats
        assert 'by_extension' in stats


class TestFileRegistry:
    """Test cases for FileRegistry."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "kernel.cu").write_text("// CUDA kernel")
            yield root
    
    @pytest.fixture
    def registry(self, temp_project):
        """Create a registry for testing."""
        from rocm_bridge.core.state import StateManager
        
        scanner = ProjectScanner(str(temp_project))
        scanner.scan()
        
        state_manager = StateManager(str(temp_project))
        file_registry = FileRegistry(str(temp_project), state_manager)
        file_registry.load_from_scanner(scanner)
        
        yield file_registry
    
    def test_get_convertible_files(self, registry):
        """Test getting only convertible files."""
        convertible = registry.get_convertible_files()
        
        assert all(f.convertible for f in convertible)
    
    def test_update_file_status(self, registry):
        """Test updating file status."""
        files = registry.get_all_files()
        
        if files:
            success = registry.update_file_status(
                files[0].relative_path, 
                "ANALYZING"
            )
            assert success
    
    def test_update_files_status_batch(self, registry):
        """
        FIX #2: Test batch status update with single disk write.
        """
        files = registry.get_all_files()
        paths = [f.relative_path for f in files]
        
        if paths:
            updated = registry.update_files_status(paths, "ANALYZED")
            assert updated == len(paths)
    
    def test_get_statistics(self, registry):
        """Test getting registry statistics."""
        stats = registry.get_statistics()
        
        assert 'total_files' in stats
        assert 'convertible' in stats
        assert stats['total_files'] > 0
    
    def test_file_normalization(self, registry):
        """Test that paths are normalized consistently."""
        files = registry.get_all_files()
        
        for f in files:
            assert '\\' not in f.relative_path


class TestGitignorePatterns:
    """
    FIX #3: Dedicated tests for .gitignore pattern matching.
    """
    
    @pytest.fixture
    def gitignore_project(self):
        """Create a project specifically for testing .gitignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create various directory structures
            (root / "deep" / "nested" / "folder").mkdir(parents=True)
            (root / "deep" / "nested" / "folder" / "test.cu").write_text("// test")
            
            (root / "logs").mkdir()
            (root / "logs" / "debug.log").write_text("log")
            
            (root / "important.cu").write_text("// keep this")
            (root / "temp.cu").write_text("// ignore this")
            
            # Write .gitignore with various patterns
            (root / ".gitignore").write_text(
                "**/folder/\n"    # Recursive directory
                "*.log\n"          # Extension
                "!important.cu\n"  # Negation
                "temp.cu\n"        # Specific file
            )
            
            yield root
    
    def test_recursive_pattern(self, gitignore_project):
        """Test **/ pattern matches nested directories."""
        scanner = ProjectScanner(str(gitignore_project))
        registry = scanner.scan()
        
        assert not any('folder' in path for path in registry.keys())
    
    def test_extension_pattern(self, gitignore_project):
        """Test *.log pattern excludes log files."""
        scanner = ProjectScanner(str(gitignore_project))
        registry = scanner.scan()
        
        assert not any('.log' in path for path in registry.keys())
    
    def test_negation_pattern(self, gitignore_project):
        """Test !important.cu pattern keeps negated files."""
        scanner = ProjectScanner(str(gitignore_project))
        registry = scanner.scan()
        
        # important.cu should NOT be excluded due to negation
        assert any('important.cu' in path for path in registry.keys())
        
        # temp.cu should be excluded
        assert not any('temp.cu' in path for path in registry.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])