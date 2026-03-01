"""
ROCm Bridge - Performance Metrics & Roofline Model
==================================================
Deterministic performance simulation based on AST analysis results.

FIXES APPLIED:
- ✅ Max penalty instead of sum (hardware ceilings don't stack)
- ✅ ZeroDivision protection (enforce minimum values)
- ✅ Infinity check (Roofline calculation)
- ✅ Deterministic math (no random values)
"""

import logging
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Structured performance metrics container."""
    
    valu_utilization: float = 0.0
    wavefront_occupancy: float = 0.0
    memory_bandwidth_util: float = 0.0
    bank_conflicts: int = 0
    mem_stalls: float = 0.0
    branch_divergence: float = 0.0
    health_score: float = 0.0
    bottleneck: str = "unknown"
    arithmetic_intensity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RooflineParameters:
    """Roofline Model parameters for a specific GPU."""
    
    peak_gflops: float
    peak_bandwidth_gbs: float
    critical_intensity: float = 0.0
    
    def __post_init__(self):
        """Calculate critical intensity from peak values."""
        if self.peak_bandwidth_gbs > 0:
            self.critical_intensity = self.peak_gflops / self.peak_bandwidth_gbs
        else:
            self.critical_intensity = 0.0


class RooflineModel:
    """Roofline Performance Model implementation."""
    
    def __init__(self, peak_gflops: float, peak_bandwidth_gbs: float):
        self.params = RooflineParameters(
            peak_gflops=peak_gflops,
            peak_bandwidth_gbs=peak_bandwidth_gbs
        )
    
    def calculate_attainable_performance(self, arithmetic_intensity: float) -> float:
        """Calculate attainable performance based on arithmetic intensity."""
        pi = self.params.peak_gflops
        beta = self.params.peak_bandwidth_gbs
        
        # ✅ FIX: Handle infinity
        if not math.isfinite(arithmetic_intensity):
            return pi
        
        memory_bound_perf = beta * arithmetic_intensity
        attainable_perf = min(pi, memory_bound_perf)
        
        return attainable_perf
    
    def classify_bottleneck(self, arithmetic_intensity: float) -> str:
        """Classify kernel as compute-bound or memory-bound."""
        if not math.isfinite(arithmetic_intensity):
            return "compute_bound"
        
        if arithmetic_intensity >= self.params.critical_intensity:
            return "compute_bound"
        else:
            return "memory_bound"
    
    def get_efficiency(self, arithmetic_intensity: float, actual_gflops: float) -> float:
        """Calculate efficiency compared to Roofline bound."""
        attainable = self.calculate_attainable_performance(arithmetic_intensity)
        
        if attainable > 0:
            efficiency = (actual_gflops / attainable) * 100.0
            return min(100.0, max(0.0, efficiency))
        
        return 0.0


class PerformanceSimulator:
    """Deterministic performance simulator."""
    
    def __init__(self, hardware_profile: Dict[str, Any]):
        """
        Initialize simulator with hardware profile.
        
        ✅ FIX: Enforce hard floors to prevent ZeroDivisionErrors
        """
        # ✅ FIX 3: Enforce minimum values to prevent ZeroDivisionError
        raw_wavefront = hardware_profile.get('wavefront_size', 64)
        self.wavefront_size = max(1, int(raw_wavefront))
        
        self.peak_gflops = max(0.1, float(hardware_profile.get('peak_gflops', 1000.0)))
        self.memory_bandwidth = max(0.1, float(hardware_profile.get('memory_bandwidth', 500.0)))
        self.lds_banks = max(1, int(hardware_profile.get('lds_banks', 32)))
        
        self.roofline = RooflineModel(self.peak_gflops, self.memory_bandwidth)
        
        logger.info(
            f"PerformanceSimulator initialized: "
            f"Wavefront={self.wavefront_size}, Peak={self.peak_gflops}GFLOPS"
        )
    
    def calculate_valu_efficiency(self, launched_threads: int) -> float:
        """Calculate VALU efficiency based on thread count vs wavefront size."""
        if launched_threads <= 0:
            return 0.0
        
        allocated_wavefronts = math.ceil(launched_threads / self.wavefront_size)
        allocated_threads = allocated_wavefronts * self.wavefront_size
        
        efficiency = launched_threads / allocated_threads
        
        return round(efficiency, 4)
    
    def calculate_bank_conflict_penalty(self, array_stride: int, access_pattern: str = "linear") -> int:
        """Calculate bank conflict multiplier."""
        if array_stride <= 0:
            return 0
        
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
        
        common_factor = gcd(array_stride, self.lds_banks)
        
        if common_factor > 1:
            return common_factor
        
        return 0
    
    def calculate_arithmetic_intensity(self, flop_count: int, byte_count: int) -> float:
        """Calculate arithmetic intensity (FLOPs per byte)."""
        if byte_count <= 0:
            return float('inf')
        
        return flop_count / byte_count
    
    def simulate_metrics(self, ast_analysis: Dict[str, Any]) -> PerformanceMetrics:
        """
        Generate deterministic performance metrics from AST analysis.
        
        ✅ FIX: Use MAX instead of SUM for penalties (hardware ceilings)
        """
        metrics = PerformanceMetrics()
        
        base_valu = 100.0
        base_occupancy = 100.0
        
        issues = ast_analysis.get('issues', [])
        kernels = ast_analysis.get('kernels_detected', [])
        
        valu_penalties = []
        occupancy_penalties = []
        conflict_count = 0
        
        for issue in issues:
            rule_id = issue.get('rule_id', '')
            confidence = issue.get('confidence', 1.0)
            
            if rule_id == 'ROCM_001':
                literal_value = issue.get('metadata', {}).get('literal_value', 32)
                efficiency = self.calculate_valu_efficiency(literal_value)
                penalty = (1.0 - efficiency) * 100.0 * confidence
                valu_penalties.append(penalty)
            
            elif rule_id == 'ROCM_002':
                occupancy_penalties.append(10.0 * confidence)
            
            elif rule_id == 'ROCM_003':
                # ✅ FIX: Use actual mathematical function with AST stride data
                stride = issue.get('metadata', {}).get('stride', 32)
                penalty = self.calculate_bank_conflict_penalty(stride)
                conflict_count += int(penalty * confidence)
                valu_penalties.append(15.0 * confidence)
            
            elif rule_id == 'ROCM_004':
                occupancy_penalties.append(5.0 * confidence)
            
            elif rule_id == 'ROCM_005':
                occupancy_penalties.append(8.0 * confidence)
        
        # ✅ FIX 2: Apply penalties using MAX (hardware ceilings don't stack)
        max_valu_penalty = max(valu_penalties) if valu_penalties else 0.0
        max_occ_penalty = max(occupancy_penalties) if occupancy_penalties else 0.0
        
        metrics.valu_utilization = max(0.0, base_valu - max_valu_penalty)
        metrics.wavefront_occupancy = max(0.0, base_occupancy - max_occ_penalty)
        metrics.bank_conflicts = conflict_count
        
        metrics.mem_stalls = conflict_count * 0.5
        
        kernel_count = len(kernels) if kernels else 1
        estimated_flops = kernel_count * 100 * (len(issues) + 10)
        estimated_bytes = kernel_count * 32 * (len(issues) + 8)
        metrics.arithmetic_intensity = self.calculate_arithmetic_intensity(
            estimated_flops, estimated_bytes
        )
        
        metrics.bottleneck = self.roofline.classify_bottleneck(metrics.arithmetic_intensity)
        metrics.health_score = self._calculate_health_score(metrics)
        metrics.memory_bandwidth_util = min(100.0, metrics.arithmetic_intensity * 10.0)
        metrics.branch_divergence = min(100.0, len(issues) * 5.0)
        
        logger.info(
            f"Metrics simulated: VALU={metrics.valu_utilization:.1f}%, "
            f"Occupancy={metrics.wavefront_occupancy:.1f}%, "
            f"Health={metrics.health_score:.1f}"
        )
        
        return metrics
    
    def _calculate_health_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall health score from metrics."""
        score = (
            metrics.valu_utilization * 0.35 +
            metrics.wavefront_occupancy * 0.35 +
            (100.0 - min(100.0, metrics.bank_conflicts)) * 0.15 +
            (100.0 - metrics.branch_divergence) * 0.15
        )
        
        return round(max(0.0, min(100.0, score)), 2)