"""
ROCm Bridge - Static Analysis Rule Engine
-----------------------------------------
A modular, extensible rule engine designed to detect CUDA-specific anti-patterns
and portability bottlenecks in C++/CUDA source code.

Architecture:
- BaseRule: Abstract base class for all detection logic.
- Rule Registry: Automatically tracks available rules.
- AnalysisEngine: Orchestrates the scan and computes the "Portability Score".

Author: Team 7SENSITIVE
Hackathon Track: AMD Slingshot 2026
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Mocking clang.cindex types for development without full LLVM installed
# In production, these would be: from clang.cindex import Cursor, CursorKind
class MockCursor:
    pass

@dataclass
class AnalysisIssue:
    """Standardized output format for all detection rules."""
    rule_id: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    line: int
    column: int
    message: str
    recommendation: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "line": self.line,
            "message": self.message,
            "recommendation": self.recommendation,
            "snippet": self.snippet
        }

class BaseRule(ABC):
    """
    Abstract Base Class for all Static Analysis Rules.
    To add a new rule, inherit from this and implement detect().
    """
    rule_id: str = "GENERIC_000"
    name: str = "Generic Rule"
    severity: str = "INFO"
    
    @abstractmethod
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        """
        Analyzes a single AST node.
        Returns a list of AnalysisIssue objects if a violation is found.
        """
        pass

# --- CONCRETE RULE IMPLEMENTATIONS ---

class HardcodedWarpSizeRule(BaseRule):
    rule_id = "ROCM_001"
    name = "Hardcoded Warp Size Assumption"
    severity = "CRITICAL"

    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        # Check for Integer Literal '32'
        # In a real clang AST, node.kind == CursorKind.INTEGER_LITERAL
        # We assume 'node' has .kind and .spelling attributes provided by parser.py
        
        if getattr(node, 'kind', None) and 'INTEGER_LITERAL' in str(node.kind):
            tokens = list(node.get_tokens()) if hasattr(node, 'get_tokens') else []
            token_str = tokens[0].spelling if tokens else ""
            
            if token_str == '32':
                # Heuristic: verify if inside a kernel function (context check)
                if context.get('in_kernel', False):
                    issues.append(AnalysisIssue(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        line=node.location.line,
                        column=node.location.column,
                        message="Hardcoded value '32' detected inside a kernel.",
                        recommendation="AMD architectures (CDNA/RDNA) use 64-thread Wavefronts. "
                                       "Replace '32' with a wavefront-size agnostic constant or 'warpSize' (carefully).",
                        snippet=f"Found literal: {token_str}"
                    ))
        
        return issues


class NvidiaIntrinsicRule(BaseRule):
    rule_id = "ROCM_002"
    name = "NVIDIA-Specific Warp Intrinsic"
    severity = "HIGH"
    
    # List of known NVIDIA-only intrinsics
    TARGET_INTRINSICS = {
        "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", 
        "__ballot_sync", "__activemask", "__any_sync", "__all_sync"
    }

    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if getattr(node, 'kind', None) and 'CALL_EXPR' in str(node.kind):
            func_name = node.spelling
            if func_name in self.TARGET_INTRINSICS:
                issues.append(AnalysisIssue(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    line=node.location.line,
                    column=node.location.column,
                    message=f"NVIDIA-specific intrinsic '{func_name}' detected.",
                    recommendation=f"This will fail to compile on ROCm. Use the HIP equivalent (e.g., replace {func_name} with {func_name.replace('_sync', '')} or use portable C++20 atomic_ref if applicable).",
                    snippet=func_name
                ))
        
        return issues


class SharedMemoryBankConflictRule(BaseRule):
    rule_id = "ROCM_003"
    name = "Potential Shared Memory Bank Conflict"
    severity = "MEDIUM"

    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        # Heuristic: Look for __shared__ arrays with dimensions being multiples of 32
        # This is a simplification. A real AST pass would inspect VarDecl and ArraySubscriptExpr
        
        if getattr(node, 'kind', None) and 'VAR_DECL' in str(node.kind):
            # Check for __shared__ attribute (often represented in tokens)
            tokens = [t.spelling for t in node.get_tokens()]
            if "__shared__" in tokens:
                # Naive check for [32] or [64] in array declaration
                for t in tokens:
                    if t in ['32', '64', '128']:
                        issues.append(AnalysisIssue(
                            rule_id=self.rule_id,
                            severity=self.severity,
                            line=node.location.line,
                            column=node.location.column,
                            message=f"Shared memory array declared with dimension '{t}'.",
                            recommendation="Powers of 32 can cause bank conflicts on AMD LDS (Local Data Share). "
                                           "Consider padding the array (e.g., [32][33]) to offset access patterns.",
                            snippet=" ".join(tokens[:5]) + "..."
                        ))
                        break
        return issues


class CudaBuiltinRule(BaseRule):
    rule_id = "ROCM_004"
    name = "Legacy CUDA Builtin Usage"
    severity = "LOW"

    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        # Detect direct usage of 'warpSize' which might be assumed to be 32
        if getattr(node, 'kind', None) and 'DECL_REF_EXPR' in str(node.kind):
            if node.spelling == "warpSize":
                issues.append(AnalysisIssue(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    line=node.location.line,
                    column=node.location.column,
                    message="Usage of 'warpSize' builtin detected.",
                    recommendation="Ensure your logic handles AMD's 64-thread wavefront. "
                                   "If logic depends on warpSize==32, it will break on CDNA architectures.",
                    snippet="warpSize"
                ))
        return issues


# --- ENGINE ORCHESTRATOR ---

class RuleEngine:
    def __init__(self):
        # Register rules here
        self.rules: List[BaseRule] = [
            HardcodedWarpSizeRule(),
            NvidiaIntrinsicRule(),
            SharedMemoryBankConflictRule(),
            CudaBuiltinRule()
        ]
        
    def run_rules(self, ast_root: Any) -> Dict[str, Any]:
        """
        Traverses the AST and runs all registered rules.
        Returns a summary dict with 'score' and 'issues'.
        """
        all_issues = []
        context = {"in_kernel": False} # Context tracker

        def traverse(node):
            # Update context (simple heuristic for entering a kernel)
            # In parser.py, you'd check for __global__ attribute more robustly
            previous_context = context["in_kernel"]
            
            if getattr(node, 'kind', None) and 'FUNCTION_DECL' in str(node.kind):
                tokens = [t.spelling for t in node.get_tokens()]
                if "__global__" in tokens or "__device__" in tokens:
                    context["in_kernel"] = True

            # Run all rules on current node
            for rule in self.rules:
                try:
                    # Only run checks if we match specific node types to save perf
                    found = rule.detect(node, context)
                    all_issues.extend(found)
                except Exception as e:
                    print(f"[Engine Warning] Rule {rule.rule_id} failed on node: {e}")

            # Recurse
            for child in node.get_children():
                traverse(child)
            
            # Restore context
            context["in_kernel"] = previous_context

        # Start traversal
        traverse(ast_root)
        
        # Compute Portability Score (0-100)
        score = 100
        penalties = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 1}
        
        deductions = sum(penalties[i.severity] for i in all_issues)
        final_score = max(0, score - deductions)

        return {
            "score": final_score,
            "issues": [i.to_dict() for i in all_issues],
            "status": "PASS" if final_score > 70 else "FAIL"
        }

# --- EXPORT ---
# This instance will be imported by parser.py
engine = RuleEngine()