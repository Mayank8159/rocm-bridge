"""
ROCm Bridge - Intelligent Recommendation Engine
-----------------------------------------------
The core correlation layer that fuses Static Analysis (AST) with Dynamic Profiling (Hardware Counters)
to generate high-confidence, AMD-native optimization strategies.

Key Capabilities:
1. Issue Correlation: Verifies if static warnings actually degrade runtime performance.
2. Impact Estimation: Predicts speedup based on bottleneck severity.
3. Code Rewriting: Suggests ROCm-specific syntax (HIP) to replace CUDA intrinsics.

Author: Team 7SENSITIVE
Hackathon Track: AMD Slingshot 2026
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONSTANTS & THRESHOLDS ---
# AMD CDNA/RDNA architectures typically have 64-wide wavefronts.
# Occupancy below 60% often indicates register pressure or block size mismatches.
THRESHOLD_OCCUPANCY_CRITICAL = 45.0
THRESHOLD_OCCUPANCY_WARNING = 65.0
THRESHOLD_VALU_UTILIZATION_LOW = 40.0
THRESHOLD_LDS_CONFLICTS_HIGH = 100  # Arbitrary threshold for bank conflicts per wave

@dataclass
class Recommendation:
    """Standardized output structure for a single optimization suggestion."""
    id: str
    priority: str       # HIGH, MEDIUM, LOW
    confidence: float   # 0.0 to 1.0
    category: str       # Compute, Memory, Portability
    title: str
    description: str
    rationale: str      # The "Why" (e.g., "Confirmed by low VALU usage")
    fix_suggestion: str
    estimated_impact: str
    code_snippet_before: str = ""
    code_snippet_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RecommendationEngine:
    """
    The Intelligence Layer.
    Correlates AST warnings with Hardware Counters to prioritize fixes.
    """

    def __init__(self):
        # Optimization Knowledge Base
        # Maps Rule IDs to generic fix templates
        self.knowledge_base = {
            "ROCM_001": { # Hardcoded Warp Size
                "title": "Migrate Warp Logic to Wavefront Logic",
                "fix": "Replace fixed '32' assumptions with 'warpSize' or use Wave64-compatible logic.",
                "category": "Compute"
            },
            "ROCM_002": { # NVIDIA Intrinsics
                "title": "Port Legacy Intrinsics to HIP",
                "fix": "Replace __shfl_sync with __shfl (HIP) or use C++20 atomic_ref for cross-lane ops.",
                "category": "Portability"
            },
            "ROCM_003": { # LDS Bank Conflicts
                "title": "Optimize Shared Memory Access",
                "fix": "Pad shared memory arrays (e.g., [32][33]) to offset bank access patterns.",
                "category": "Memory"
            }
        }

    def _generate_snippet(self, source: str, fix_type: str) -> str:
        """
        Generates a naive 'After' code snippet for visualization.
        In a production compiler, this would use AST rewriting (LibTooling).
        """
        if fix_type == "ROCM_001":
            return source.replace("32", "64") + " // Optimized for AMD Wave64"
        elif fix_type == "ROCM_002":
            return source.replace("__shfl_sync", "__shfl") + " // Ported to HIP"
        elif fix_type == "ROCM_003":
            return "// Padding added to avoid Bank Conflicts\n__shared__ float data[32][33];"
        return "// See documentation for manual fix."

    def correlate(self, static_issues: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[Recommendation]:
        """
        The Core Logic: Merges Static Warnings with Dynamic Reality.
        """
        recommendations = []
        
        # Extract metrics safely (handle missing data for simulation mode)
        occupancy = metrics.get("wavefront_occupancy", 100.0)
        valu_util = metrics.get("valu_utilization", 100.0)
        lds_conflicts = metrics.get("lds_bank_conflicts", 0)

        for issue in static_issues:
            rule_id = issue.get("rule_id")
            line_content = issue.get("snippet", "")
            
            # --- LOGIC BRANCH 1: HARDCODED WARP SIZE (ROCM_001) ---
            if rule_id == "ROCM_001":
                confidence = 0.5 # Base confidence (it's definitely in the code)
                rationale = "Static analysis detected hardcoded '32'."
                priority = "MEDIUM"
                
                # Correlation: Is the GPU actually underutilized?
                if valu_util < THRESHOLD_VALU_UTILIZATION_LOW:
                    confidence = 0.95
                    priority = "HIGH"
                    rationale += f" CONFIRMED by low VALU utilization ({valu_util:.1f}%). The GPU is idle due to warp divergence."
                elif occupancy < THRESHOLD_OCCUPANCY_CRITICAL:
                    confidence = 0.85
                    priority = "HIGH"
                    rationale += f" CONFIRMED by critical occupancy drop ({occupancy:.1f}%)."

                rec = Recommendation(
                    id=rule_id,
                    priority=priority,
                    confidence=confidence,
                    category="Compute",
                    title="Optimize Kernel for Wave64",
                    description="Your kernel assumes 32 threads per warp. AMD architectures run 64-thread wavefronts, leaving half the vector units idle.",
                    rationale=rationale,
                    fix_suggestion="Update block dimensions to 64 and unroll loops.",
                    estimated_impact="20-40% Speedup",
                    code_snippet_before=line_content,
                    code_snippet_after=self._generate_snippet(line_content, rule_id)
                )
                recommendations.append(rec)

            # --- LOGIC BRANCH 2: NVIDIA INTRINSICS (ROCM_002) ---
            elif rule_id == "ROCM_002":
                # Portability issues are always high priority because code won't compile.
                rec = Recommendation(
                    id=rule_id,
                    priority="CRITICAL",
                    confidence=1.0,
                    category="Portability",
                    title="Replace Vendor-Specific Intrinsic",
                    description=f"The intrinsic '{line_content}' is NVIDIA-proprietary.",
                    rationale="Code will fail to compile on HIP/ROCm toolchain.",
                    fix_suggestion="Use HIP equivalents or portable atomics.",
                    estimated_impact="Enables Compilation",
                    code_snippet_before=line_content,
                    code_snippet_after=self._generate_snippet(line_content, rule_id)
                )
                recommendations.append(rec)

            # --- LOGIC BRANCH 3: LDS CONFLICTS (ROCM_003) ---
            elif rule_id == "ROCM_003":
                confidence = 0.4
                priority = "LOW"
                rationale = "Potential stride pattern detected."

                # Correlation: Do we see bank conflicts in the profiler?
                if lds_conflicts > THRESHOLD_LDS_CONFLICTS_HIGH:
                    confidence = 0.98
                    priority = "HIGH"
                    rationale = f"CONFIRMED: Profiler detected {lds_conflicts} bank conflicts per wave."
                
                rec = Recommendation(
                    id=rule_id,
                    priority=priority,
                    confidence=confidence,
                    category="Memory",
                    title="Resolve LDS Bank Conflicts",
                    description="Memory access stride aligns with bank width, serializing access.",
                    rationale=rationale,
                    fix_suggestion="Pad shared memory arrays (e.g. [32][33]).",
                    estimated_impact="10-15% Latency Reduction",
                    code_snippet_before=line_content,
                    code_snippet_after=self._generate_snippet(line_content, rule_id)
                )
                recommendations.append(rec)

        # Sort by Priority (Critical first)
        priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_map.get(x.priority, 99))
        
        return recommendations

    def generate(self, static_issues: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public API: Generates the full optimization report.
        """
        recs = self.correlate(static_issues, metrics)
        
        # Calculate summary stats
        confirmed = sum(1 for r in recs if r.confidence > 0.8)
        
        return {
            "summary": {
                "total_issues": len(static_issues),
                "confirmed_bottlenecks": confirmed,
                "potential_issues": len(recs) - confirmed,
                "estimated_speedup": "15-30%" if confirmed > 0 else "0-5%"
            },
            "recommendations": [r.to_dict() for r in recs]
        }

# --- EXPORT ---
engine = RecommendationEngine()