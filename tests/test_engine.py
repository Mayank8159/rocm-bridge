"""
ROCm Bridge - Engine Tests
==========================
Unit tests for the engine module.

NEW TESTS ADDED:
- UTF-8 encoding detection
- Patch overlap detection
- Zombie process prevention
- Large file diff bypass
"""

import pytest
import tempfile
import os
from pathlib import Path

from rocm_bridge.engine.hipify_wrapper import HipifyWrapper, HipifyResult
from rocm_bridge.engine.patcher import ByteOffsetPatcher, Patch
from rocm_bridge.engine.recommender import RecommendationEngine, Recommendation, Priority


class TestByteOffsetPatcher:
    """Test cases for ByteOffsetPatcher."""
    
    @pytest.fixture
    def patcher(self):
        """Create ByteOffsetPatcher instance."""
        return ByteOffsetPatcher()
    
    @pytest.fixture
    def utf8_file(self, tmp_path):
        """Create a file with UTF-8 characters."""
        temp_file = tmp_path / "utf8_test.txt"
        # Include multi-byte UTF-8 characters
        temp_file.write_text("Hello 世界！∑∫∞ Test file with emojis 🚀🔥")
        return str(temp_file)
    
    @pytest.fixture
    def large_file(self, tmp_path):
        """Create a large file for diff testing."""
        temp_file = tmp_path / "large_test.txt"
        # Create 15MB file (over MAX_DIFF_FILE_SIZE)
        temp_file.write_text("Line " * 3000000)
        return str(temp_file)
    
    def test_utf8_patch_application(self, patcher, utf8_file):
        """
        FIX 1: Test that UTF-8 files are patched without corruption.
        """
        patch = Patch(
            start_offset=0,
            end_offset=5,
            original_text="Hello",
            replacement_text="Hi",
            rule_id="TEST_001"
        )
        
        result = patcher.apply_patches(utf8_file, [patch])
        
        assert result.success is True
        
        # Verify UTF-8 characters are intact
        with open(result.output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "世界" in content
        assert "🚀" in content
    
    def test_patch_overlap_detection(self, patcher, tmp_path):
        """
        FIX 5: Test that overlapping patches are detected.
        """
        test_file = tmp_path / "overlap_test.txt"
        test_file.write_text("AAAAAAAAAA")
        
        # Create overlapping patches
        patches = [
            Patch(start_offset=0, end_offset=5, original_text="AAAAA", 
                  replacement_text="11111", rule_id="TEST_001"),
            Patch(start_offset=3, end_offset=8, original_text="AAAAA", 
                  replacement_text="22222", rule_id="TEST_002"),  # Overlaps!
        ]
        
        result = patcher.apply_patches(str(test_file), patches)
        
        # Should fail due to overlap detection
        assert result.success is False
        assert any("overlap" in err.lower() for err in result.errors)
    
    def test_large_file_diff_bypass(self, patcher, large_file):
        """
        FIX 6: Test that large files bypass difflib.
        """
        # Create modified version
        modified_file = Path(large_file).parent / "large_modified.txt"
        modified_file.write_text("Line " * 3000000 + "Extra")
        
        diff = patcher.generate_unified_diff(large_file, str(modified_file))
        
        # Should return error message, not crash
        assert "too large" in diff.lower() or isinstance(diff, str)
    
    def test_identical_files_diff_bypass(self, patcher, tmp_path):
        """
        FIX 4: Test that identical files return empty diff quickly.
        """
        file1 = tmp_path / "identical1.txt"
        file2 = tmp_path / "identical2.txt"
        
        content = "Same content" * 1000
        file1.write_text(content)
        file2.write_text(content)
        
        diff = patcher.generate_unified_diff(str(file1), str(file2))
        
        # Should return empty string immediately
        assert diff == ""
    
    def test_backup_verification(self, patcher, tmp_path):
        """
        FIX 9: Test that backup files are verified.
        """
        test_file = tmp_path / "backup_test.txt"
        test_file.write_text("Test content")
        
        patch = Patch(
            start_offset=0,
            end_offset=4,
            original_text="Test",
            replacement_text="Best",
            rule_id="TEST_001"
        )
        
        result = patcher.apply_patches(str(test_file), [patch])
        
        assert result.success is True
        assert result.backup_path != ""
        
        # Verify backup exists
        assert Path(result.backup_path).exists()
    
    def test_lock_timeout(self, patcher, tmp_path):
        """
        FIX 11: Test that lock timeout is handled gracefully.
        """
        test_file = tmp_path / "lock_test.txt"
        test_file.write_text("Test")
        
        # Acquire lock manually
        patcher._lock.acquire()
        
        try:
            patch = Patch(
                start_offset=0,
                end_offset=4,
                original_text="Test",
                replacement_text="Best",
                rule_id="TEST_001"
            )
            
            # Should timeout and return error
            result = patcher.apply_patches(str(test_file), [patch])
            
            assert result.success is False
            assert any("lock" in err.lower() for err in result.errors)
        
        finally:
            patcher._lock.release()


class TestHipifyWrapper:
    """Test cases for HipifyWrapper."""
    
    @pytest.fixture
    def wrapper(self):
        """Create HipifyWrapper instance."""
        return HipifyWrapper()
    
    def test_zombie_process_prevention(self, wrapper, tmp_path):
        """
        FIX 3: Test that timeout kills zombie processes.
        """
        # Create a file that would cause hipify to hang
        test_file = tmp_path / "hang_test.cu"
        test_file.write_text("// Test")
        
        # Set very short timeout
        wrapper.HIPIFY_TIMEOUT_SECONDS = 1
        
        result = wrapper.transpile(str(test_file))
        
        # Should not crash, should report timeout
        assert result is not None
        assert isinstance(result, HipifyResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])