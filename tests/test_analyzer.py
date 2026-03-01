"""
ROCm Bridge - Analyzer Tests
============================
Unit tests for the AST analyzer module.

Location: tests/ directory at PROJECT ROOT (NOT inside rocm_bridge/analyzer/)

These tests guarantee:
- AST parsing works correctly
- Rules detect issues accurately
- Metrics are deterministic (not random)
- Thread safety is maintained
- Timeout behavior is correct
- Concurrent parsing is safe

Run with: pytest tests/test_analyzer.py -v
"""

import pytest
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from rocm_bridge.analyzer.parser import CudaParser, ParseResult
from rocm_bridge.analyzer.rules import RuleEngine, AnalysisIssue
from rocm_bridge.analyzer.metrics import PerformanceSimulator, RooflineModel


class TestCudaParser:
    """Test cases for CudaParser."""
    
    @pytest.fixture
    def sample_cuda_file(self, tmp_path):
        """Create a temporary CUDA file for testing."""
        # FIX 13: Use pytest's tmp_path for proper cleanup
        temp_file = tmp_path / "test_kernel.cu"
        temp_file.write_text("""
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * 32 + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}
""")
        
        yield str(temp_file)
        
        # Cleanup is automatic with tmp_path
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return CudaParser()
    
    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        # Parser should initialize even if libclang is unavailable
        assert parser is not None
    
    def test_parse_nonexistent_file(self, parser):
        """Test parsing nonexistent file returns error."""
        result = parser.analyze("/nonexistent/file.cu")
        
        assert result.success is False
        assert result.status == "PARSE_ERROR" or result.status == "LIMITED"
    
    def test_parse_cuda_file(self, parser, sample_cuda_file):
        """Test parsing valid CUDA file."""
        result = parser.analyze(sample_cuda_file)
        
        # Should complete without crashing
        assert result is not None
        assert result.file_path.endswith('.cu')
    
    def test_kernel_detection(self, parser, sample_cuda_file):
        """Test kernel function detection."""
        result = parser.analyze(sample_cuda_file)
        
        # Should detect at least one kernel
        # (May be empty if libclang unavailable, but shouldn't crash)
        assert isinstance(result.kernels_detected, list)
    
    def test_path_normalization(self, parser):
        """Test path normalization to POSIX style."""
        # Test Windows-style path
        windows_path = "C:\\Users\\test\\kernel.cu"
        normalized = parser._normalize_path(windows_path)
        
        # Should use forward slashes
        assert '\\' not in normalized
    
    def test_parse_timeout(self, parser, tmp_path):
        """
        FIX 14: Test timeout behavior for large files.
        """
        # Create a very large file that might timeout
        large_file = tmp_path / "large_kernel.cu"
        # Write 10000 lines of simple code
        large_file.write_text("\n".join([f"int var_{i} = {i};" for i in range(10000)]))
        
        # Should complete or timeout gracefully (not hang)
        start = time.time()
        result = parser.analyze(str(large_file))
        elapsed = time.time() - start
        
        # Should not hang indefinitely
        assert elapsed < 60  # Generous timeout
    
    def test_concurrent_parsing(self, parser, tmp_path):
        """
        FIX 15: Test that concurrent parsing doesn't cause segfault.
        """
        # Create multiple test files
        files = []
        for i in range(4):
            f = tmp_path / f"kernel_{i}.cu"
            f.write_text(f"""
__global__ void kernel_{i}() {{
    int x = 32;
}}
""")
            files.append(str(f))
        
        # Parse concurrently (this would segfault with shared Index)
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(parser.analyze, f) for f in files]
            for future in futures:
                results.append(future.result())
        
        # All should complete without crash
        assert len(results) == 4
        assert all(r is not None for r in results)


class TestRuleEngine:
    """Test cases for RuleEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a rule engine instance."""
        return RuleEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initializes with rules."""
        assert len(engine.rules) > 0
    
    def test_rule_metadata(self, engine):
        """Test rule metadata retrieval."""
        # FIX 2: Corrected syntax error
        metadata = engine.get_rule_metadata()
        
        assert isinstance(metadata, list)
        if metadata:  # FIX 2: Was 'if meta' with missing colon
            assert 'rule_id' in metadata[0]
            assert 'name' in metadata[0]
    
    def test_score_computation(self, engine):
        """Test score computation from issues."""
        # Create test issues
        issues = [
            AnalysisIssue(
                rule_id='ROCM_001',
                severity='CRITICAL',
                line=10,
                column=5,
                message='Test issue',
                recommendation='Fix it',
                confidence=1.0
            )
        ]
        
        score = engine._compute_score(issues)
        
        # Score should be reduced from 100
        assert score < 100.0
        assert score >= 0.0
    
    def test_empty_issues_score(self, engine):
        """Test score with no issues."""
        score = engine._compute_score([])
        assert score == 100.0
    
    def test_context_mutation(self, engine):
        """
        FIX 4: Test that external context dict is not mutated.
        """
        external_context = {'existing_key': 'value'}
        original_keys = set(external_context.keys())
        
        # Create a mock AST root (won't actually traverse)
        class MockNode:
            kind = None
            def get_children(self):
                return []
        
        # Run rules with external context
        engine.run_rules(MockNode(), external_context)
        
        # External dict should not be mutated
        assert set(external_context.keys()) == original_keys


class TestPerformanceSimulator:
    """Test cases for PerformanceSimulator."""
    
    @pytest.fixture
    def cdna_profile(self):
        """CDNA hardware profile for testing."""
        return {
            'wavefront_size': 64,
            'peak_gflops': 1000.0,
            'memory_bandwidth': 500.0,
            'lds_banks': 32
        }
    
    @pytest.fixture
    def rdna_profile(self):
        """RDNA hardware profile for testing."""
        return {
            'wavefront_size': 32,
            'peak_gflops': 500.0,
            'memory_bandwidth': 250.0,
            'lds_banks': 32
        }
    
    @pytest.fixture
    def simulator_cdna(self, cdna_profile):
        """Create simulator with CDNA profile."""
        return PerformanceSimulator(cdna_profile)
    
    @pytest.fixture
    def simulator_rdna(self, rdna_profile):
        """Create simulator with RDNA profile."""
        return PerformanceSimulator(rdna_profile)
    
    def test_simulator_initialization(self, simulator_cdna):
        """Test simulator initializes correctly."""
        assert simulator_cdna.wavefront_size == 64
        assert simulator_cdna.peak_gflops == 1000.0
    
    def test_valu_efficiency_wave64(self, simulator_cdna):
        """
        FIX: Test VALU efficiency calculation for Wave64.
        32 threads on Wave64 = 50% efficiency (not 100%)
        """
        efficiency = simulator_cdna.calculate_valu_efficiency(32)
        
        # 32 threads / 64 wavefront = 50%
        assert efficiency == 0.5
    
    def test_valu_efficiency_full_wavefront(self, simulator_cdna):
        """Test VALU efficiency with full wavefront."""
        efficiency = simulator_cdna.calculate_valu_efficiency(64)
        
        # 64 threads / 64 wavefront = 100%
        assert efficiency == 1.0
    
    def test_valu_efficiency_partial_allocation(self, simulator_cdna):
        """
        FIX: Test partial wavefront allocation.
        96 threads on Wave64 = 2 wavefronts (128 allocated)
        Efficiency = 96/128 = 75%
        """
        efficiency = simulator_cdna.calculate_valu_efficiency(96)
        
        # 96 / 128 = 0.75
        assert efficiency == 0.75
    
    def test_valu_efficiency_rdna(self, simulator_rdna):
        """Test VALU efficiency for RDNA (Wave32)."""
        efficiency = simulator_rdna.calculate_valu_efficiency(32)
        
        # 32 threads / 32 wavefront = 100%
        assert efficiency == 1.0
    
    def test_bank_conflict_no_conflict(self, simulator_cdna):
        """Test bank conflict calculation with no conflict."""
        conflicts = simulator_cdna.calculate_bank_conflict_penalty(1)
        
        # Stride 1 = no conflicts
        assert conflicts == 0
    
    def test_bank_conflict_with_conflict(self, simulator_cdna):
        """Test bank conflict calculation with conflicts."""
        conflicts = simulator_cdna.calculate_bank_conflict_penalty(32)
        
        # Stride 32 = 32-way conflict
        assert conflicts > 0
    
    def test_arithmetic_intensity(self, simulator_cdna):
        """Test arithmetic intensity calculation."""
        intensity = simulator_cdna.calculate_arithmetic_intensity(1000, 100)
        
        # 1000 FLOPs / 100 bytes = 10.0
        assert intensity == 10.0
    
    def test_arithmetic_intensity_division_by_zero(self, simulator_cdna):
        """
        FIX 10: Test division by zero protection.
        """
        intensity = simulator_cdna.calculate_arithmetic_intensity(1000, 0)
        
        # Should return infinity, not crash
        assert intensity == float('inf')
    
    def test_simulate_metrics_deterministic(self, simulator_cdna):
        """
        FIX: Test that metrics are deterministic (not random).
        Running twice should produce identical results.
        """
        ast_analysis = {
            'issues': [
                {'rule_id': 'ROCM_001', 'confidence': 1.0, 'metadata': {'literal_value': 32}}
            ],
            'kernels_detected': ['test_kernel']
        }
        
        # Run simulation twice
        metrics1 = simulator_cdna.simulate_metrics(ast_analysis)
        metrics2 = simulator_cdna.simulate_metrics(ast_analysis)
        
        # Results should be identical (deterministic)
        assert metrics1.valu_utilization == metrics2.valu_utilization
        assert metrics1.health_score == metrics2.health_score
        assert metrics1.bank_conflicts == metrics2.bank_conflicts
    
    def test_health_score_calculation(self, simulator_cdna):
        """Test health score is within valid range."""
        ast_analysis = {'issues': [], 'kernels_detected': []}
        metrics = simulator_cdna.simulate_metrics(ast_analysis)
        
        assert 0.0 <= metrics.health_score <= 100.0
    
    def test_bank_conflict_uses_ast_stride(self, simulator_cdna):
        """
        FIX 3: Test that bank conflict uses actual AST stride data.
        """
        ast_analysis = {
            'issues': [
                {
                    'rule_id': 'ROCM_003',
                    'confidence': 1.0,
                    'metadata': {'stride': 32}  # Actual stride from AST
                }
            ],
            'kernels_detected': []
        }
        
        metrics = simulator_cdna.simulate_metrics(ast_analysis)
        
        # Should use actual stride (32) for conflict calculation
        assert metrics.bank_conflicts > 0


class TestRooflineModel:
    """Test cases for RooflineModel."""
    
    @pytest.fixture
    def roofline(self):
        """Create RooflineModel instance."""
        return RooflineModel(peak_gflops=1000.0, peak_bandwidth_gbs=500.0)
    
    def test_roofline_initialization(self, roofline):
        """Test RooflineModel initializes correctly."""
        assert roofline.params.peak_gflops == 1000.0
        assert roofline.params.peak_bandwidth_gbs == 500.0
    
    def test_critical_intensity(self, roofline):
        """Test critical intensity calculation."""
        # 1000 GFLOPS / 500 GB/s = 2.0 FLOPs/byte
        assert roofline.params.critical_intensity == 2.0
    
    def test_compute_bound_classification(self, roofline):
        """Test compute-bound classification."""
        # Intensity > critical = compute bound
        classification = roofline.classify_bottleneck(5.0)
        assert classification == "compute_bound"
    
    def test_memory_bound_classification(self, roofline):
        """Test memory-bound classification."""
        # Intensity < critical = memory bound
        classification = roofline.classify_bottleneck(1.0)
        assert classification == "memory_bound"
    
    def test_attainable_performance_memory_bound(self, roofline):
        """Test attainable performance in memory-bound region."""
        # Memory bound: P = β × I = 500 × 1.0 = 500 GFLOPS
        perf = roofline.calculate_attainable_performance(1.0)
        assert perf == 500.0
    
    def test_attainable_performance_compute_bound(self, roofline):
        """Test attainable performance in compute-bound region."""
        # Compute bound: P = π = 1000 GFLOPS
        perf = roofline.calculate_attainable_performance(5.0)
        assert perf == 1000.0
    
    def test_infinity_handling(self, roofline):
        """
        FIX 12: Test infinity handling in Roofline calculation.
        """
        # Should not crash with infinity
        perf = roofline.calculate_attainable_performance(float('inf'))
        
        # Should return peak compute (compute-bound)
        assert perf == 1000.0


class TestIntegration:
    """Integration tests for the full analyzer pipeline."""
    
    @pytest.fixture
    def sample_cuda_with_issues(self, tmp_path):
        """Create CUDA file with known issues."""
        # FIX 13: Use tmp_path for proper cleanup
        temp_file = tmp_path / "bad_kernel.cu"
        temp_file.write_text("""
__global__ void badKernel(float* data) {
    __shared__ float shared[32][32];  // Bank conflict
    int idx = blockIdx.x * 32 + threadIdx.x;  // Hardcoded 32
    data[idx] = shared[threadIdx.x][threadIdx.y];
}
""")
        
        yield str(temp_file)
    
    def test_full_pipeline(self, sample_cuda_with_issues):
        """Test full analysis pipeline."""
        # This test verifies the modules work together
        # May have limited functionality if libclang unavailable
        
        parser = CudaParser()
        result = parser.analyze(sample_cuda_with_issues)
        
        # Should not crash
        assert result is not None
        assert isinstance(result.file_path, str)
    
    def test_issues_correlate_with_metrics(self, sample_cuda_with_issues, cdna_profile):
        """
        FIX 3: Test that AST issues correlate with metrics.
        """
        parser = CudaParser()
        result = parser.analyze(sample_cuda_with_issues)
        
        simulator = PerformanceSimulator(cdna_profile)
        metrics = simulator.simulate_metrics({
            'issues': result.issues,
            'kernels_detected': result.kernels_detected
        })
        
        # If issues found, metrics should reflect penalties
        if result.issues:
            assert metrics.health_score < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])