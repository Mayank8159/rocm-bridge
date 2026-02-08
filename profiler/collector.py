"""
ROCm Bridge - Telemetry Collector
---------------------------------
Parses raw ROCm Profiler (rocprof) CSV output and computes high-level
performance metrics for downstream optimization logic.

Key Capabilities:
1. Metric Extraction: Reads VALUUtilization, WavefrontOccupancy, etc.
2. Bottleneck Detection: Classifies kernels as Compute/Memory/Occupancy bound.
3. Health Scoring: Generates a 0-100 performance score.

Author: Team 7SENSITIVE
Hackathon Track: AMD Slingshot 2026
"""

import pandas as pd
import logging
import sys
import json
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PERFORMANCE THRESHOLDS ---
THRESHOLD_VALU_LOW = 40.0       # Below 40% indicates underutilized vector units
THRESHOLD_OCCUPANCY_LOW = 50.0  # Below 50% indicates poor latency hiding
THRESHOLD_LDS_CONFLICT = 10     # Any significant bank conflict is bad

class ProfileCollector:
    """
    Ingests raw hardware counters and synthesizes them into actionable insights.
    """

    def __init__(self):
        # Default metrics if columns are missing (Safety mechanism)
        self.default_metrics = {
            "VALUUtilization": 0.0,
            "WavefrontOccupancy": 0.0,
            "LDSBankConflict": 0.0,
            "MemUnitStalled": 0.0
        }

    def load_profile(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Reads the rocprof CSV file safely.
        """
        if not os.path.exists(csv_path):
            logger.error(f"Profile CSV not found at: {csv_path}")
            return None

        try:
            # Handle potential header mismatches or empty files
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning("Profile CSV is empty.")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            return None

    def extract_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extracts key columns from the first kernel execution in the log.
        In a full app, this would aggregate across all kernels.
        """
        metrics = self.default_metrics.copy()
        
        if df is None or df.empty:
            return metrics

        # We take the first row (assuming single kernel launch for this demo)
        row = df.iloc[0]

        # Safe extraction with fallback
        for key in metrics.keys():
            if key in df.columns:
                metrics[key] = float(row[key])
        
        return metrics

    def detect_bottleneck(self, metrics: Dict[str, float]) -> str:
        """
        Classifies the performance limiter based on hardware counters.
        """
        valu = metrics["VALUUtilization"]
        occupancy = metrics["WavefrontOccupancy"]
        lds_conflicts = metrics["LDSBankConflict"]
        
        # 1. Memory Bound (LDS Thrashing)
        if lds_conflicts > THRESHOLD_LDS_CONFLICT:
            return "memory_bound_lds"
            
        # 2. Occupancy Limited (Not enough waves to hide latency)
        if occupancy < THRESHOLD_OCCUPANCY_LOW:
            return "occupancy_limited"
            
        # 3. Compute Bound (But inefficiently used)
        if valu < THRESHOLD_VALU_LOW:
            # Low VALU usage often means we are stalled or diverging
            return "compute_inefficient"
            
        # 4. Balanced / Good
        return "balanced"

    def compute_health_score(self, metrics: Dict[str, float]) -> float:
        """
        Computes a normalized 0-100 score of kernel efficiency.
        """
        # Simple weighted formula
        # VALU (40%) + Occupancy (40%) - Penalties (20%)
        
        score = (metrics["VALUUtilization"] * 0.4) + \
                (metrics["WavefrontOccupancy"] * 0.4)
                
        # Penalize for conflicts
        if metrics["LDSBankConflict"] > 0:
            score -= min(20, metrics["LDSBankConflict"] * 0.5)
            
        return max(0.0, min(100.0, score))

    def collect(self, csv_path: str) -> Dict[str, Any]:
        """
        Public API: Main pipeline to process a profile.
        """
        logger.info(f"Collecting telemetry from: {csv_path}")
        
        df = self.load_profile(csv_path)
        raw_metrics = self.extract_metrics(df)
        
        bottleneck = self.detect_bottleneck(raw_metrics)
        health = self.compute_health_score(raw_metrics)
        
        # Construct final signal object
        result = {
            "kernel_name": df["KernelName"].iloc[0] if (df is not None and "KernelName" in df.columns) else "unknown_kernel",
            "metrics": raw_metrics,
            "health_score": round(health, 2),
            "bottleneck": bottleneck,
            "recommendation_signal": bottleneck.upper()
        }
        
        logger.info(f"Analysis Complete. Health: {result['health_score']}, Bottleneck: {bottleneck}")
        return result

# --- CLI EXECUTION ---
collector = ProfileCollector()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m profiler.collector <path_to_results.csv>")
        sys.exit(1)
        
    result = collector.collect(sys.argv[1])
    print(json.dumps(result, indent=2))