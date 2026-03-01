"""
ROCm Bridge - Static Analysis Rule Engine
=========================================
Modular, extensible rule engine for CUDA-specific anti-pattern detection.

FIXES APPLIED:
- ✅ Iterative AST traversal (prevents RecursionError)
- ✅ Context deep copy (prevents mutation bleeding)
- ✅ CursorKind filter before token extraction (performance)
- ✅ NoneType protection on node.location
- ✅ Mutable default argument fixed (metadata field)
"""

import logging
import copy
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from clang.cindex import Cursor, CursorKind, TypeKind, Type
    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    logger.warning("libclang not available - rules will use mock cursors")


class Severity(Enum):
    """Issue severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AnalysisIssue:
    """Standardized output format for all detection rules."""
    
    rule_id: str
    severity: str
    line: int
    column: int
    message: str
    recommendation: str
    snippet: str = ""
    confidence: float = 1.0
    # ✅ FIX: Use default_factory to prevent mutable default argument bug
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BaseRule(ABC):
    """Abstract Base Class for all Static Analysis Rules."""
    
    rule_id: str = "GENERIC_000"
    name: str = "Generic Rule"
    severity: str = "INFO"
    description: str = ""
    
    @abstractmethod
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        """Analyzes a single AST node."""
        pass
    
    def get_metadata(self) -> Dict[str, str]:
        """Get rule metadata for documentation."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'severity': self.severity,
            'description': self.description
        }


# ============================================================================
# CONCRETE RULE IMPLEMENTATIONS
# ============================================================================

class HardcodedWarpSizeRule(BaseRule):
    """ROCM_001: Detects hardcoded warp size assumptions (32 threads)."""
    
    rule_id = "ROCM_001"
    name = "Hardcoded Warp Size Assumption"
    severity = "CRITICAL"
    description = "Detects hardcoded '32' thread assumptions that waste 50% of AMD wavefront capacity"
    
    WARP_KEYWORDS = {
        'blockDim', 'threadIdx', 'blockIdx', 'gridDim',
        'threads', 'warps', 'block', 'grid'
    }
    
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if not LIBCLANG_AVAILABLE:
            return issues
        
        # ✅ FIX: Filter by CursorKind BEFORE expensive token extraction
        if node.kind != CursorKind.INTEGER_LITERAL:
            return issues
        
        try:
            literal_value = None
            
            if hasattr(node, 'get_tokens'):
                tokens = list(node.get_tokens())
                if tokens:
                    try:
                        literal_value = int(tokens[0].spelling)
                    except (ValueError, IndexError):
                        pass
            
            if literal_value == 32:
                if context.get('in_kernel', False):
                    parent_context = context.get('parent_keywords', set())
                    
                    confidence = 0.7
                    if parent_context & self.WARP_KEYWORDS:
                        confidence = 0.95
                    
                    # ✅ FIX: Safe location access with null checks
                    line = node.location.line if hasattr(node, 'location') and node.location else 0
                    column = node.location.column if hasattr(node, 'location') and node.location else 0
                    
                    issues.append(AnalysisIssue(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        line=line,
                        column=column,
                        message=f"Hardcoded value '32' detected in kernel context. "
                               f"AMD CDNA uses 64-thread wavefronts (50% VALU loss).",
                        recommendation="Replace with 64 for CDNA, or use dynamic warpSize. "
                                     "Adjust blockDim.x to be multiple of 64.",
                        snippet="32",
                        confidence=confidence,
                        metadata={'literal_value': literal_value}
                    ))
            
        except Exception as e:
            logger.debug(f"HardcodedWarpSizeRule error: {e}")
        
        return issues


class NvidiaIntrinsicRule(BaseRule):
    """ROCM_002: Detects NVIDIA-specific warp intrinsics."""
    
    rule_id = "ROCM_002"
    name = "NVIDIA-Specific Warp Intrinsic"
    severity = "HIGH"
    description = "Detects NVIDIA-only intrinsics that require HIP equivalents"
    
    TARGET_INTRINSICS = {
        '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync',
        '__ballot_sync', '__activemask', '__any_sync', '__all_sync',
        '__popc', '__clz', '__ffs', '__brev',
        '__syncthreads_and', '__syncthreads_or', '__syncthreads_count',
        '__threadfence', '__threadfence_block', '__threadfence_system',
    }
    
    HIP_EQUIVALENTS = {
        '__shfl_sync': '__shfl',
        '__shfl_up_sync': '__shfl_up',
        '__shfl_down_sync': '__shfl_down',
        '__shfl_xor_sync': '__shfl_xor',
        '__ballot_sync': '__ballot',
        '__activemask': '__activemask',
    }
    
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if not LIBCLANG_AVAILABLE:
            return issues
        
        # ✅ FIX: Filter by CursorKind BEFORE expensive operations
        if node.kind != CursorKind.CALL_EXPR:
            return issues
        
        try:
            func_name = node.spelling if hasattr(node, 'spelling') else ""
            
            for intrinsic in self.TARGET_INTRINSICS:
                if intrinsic in func_name:
                    hip_equiv = self.HIP_EQUIVALENTS.get(intrinsic, intrinsic)
                    
                    line = node.location.line if hasattr(node, 'location') and node.location else 0
                    column = node.location.column if hasattr(node, 'location') and node.location else 0
                    
                    issues.append(AnalysisIssue(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        line=line,
                        column=column,
                        message=f"NVIDIA-specific intrinsic '{func_name}' detected. "
                               f"Will fail to compile on ROCm.",
                        recommendation=f"Replace with HIP equivalent: {hip_equiv}. "
                                     f"Or use portable C++20 std::atomic_ref.",
                        snippet=func_name,
                        confidence=1.0,
                        metadata={'intrinsic': intrinsic, 'hip_equivalent': hip_equiv}
                    ))
                    break
            
        except Exception as e:
            logger.debug(f"NvidiaIntrinsicRule error: {e}")
        
        return issues


class SharedMemoryBankConflictRule(BaseRule):
    """ROCM_003: Detects potential LDS bank conflicts."""
    
    rule_id = "ROCM_003"
    name = "Potential Shared Memory Bank Conflict"
    severity = "MEDIUM"
    description = "Detects shared memory patterns that cause LDS bank conflicts"
    
    LDS_BANKS = 32
    LDS_BANK_WIDTH = 4
    
    def _get_array_dimensions(self, node: Any) -> List[int]:
        """Extract array dimensions from AST type."""
        dimensions = []
        
        if not LIBCLANG_AVAILABLE:
            return dimensions
        
        try:
            node_type = node.type if hasattr(node, 'type') else None
            
            if node_type and node_type.kind == TypeKind.CONSTANTARRAY:
                size = node_type.get_array_size()
                if size is not None:
                    dimensions.append(int(size))
                
                element_type = node_type.get_array_element_type()
                if element_type and element_type.kind == TypeKind.CONSTANTARRAY:
                    class MockNode:
                        def __init__(self, t):
                            self.type = t
                    dimensions.extend(self._get_array_dimensions(MockNode(element_type)))
        
        except Exception as e:
            logger.debug(f"Array dimension extraction error: {e}")
        
        return dimensions
    
    def _is_shared_memory(self, node: Any) -> bool:
        """Check if variable is shared memory."""
        if not LIBCLANG_AVAILABLE:
            return False
        
        try:
            if hasattr(node, 'get_tokens'):
                tokens = [t.spelling for t in node.get_tokens()]
                if '__shared__' in tokens or 'shared' in tokens:
                    return True
            
            if hasattr(node, 'type'):
                type_spelling = str(node.type)
                if 'shared' in type_spelling.lower():
                    return True
            
            if hasattr(node, 'get_tokens'):
                tokens = [t.spelling for t in node.get_tokens()]
                if '__shared__' in tokens or 'HIP_SHARED' in tokens:
                    return True
        
        except Exception:
            pass
        
        return False
    
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if not LIBCLANG_AVAILABLE:
            return issues
        
        # ✅ FIX: Filter by CursorKind BEFORE expensive operations
        if node.kind != CursorKind.VAR_DECL:
            return issues
        
        try:
            if not self._is_shared_memory(node):
                return issues
            
            dimensions = self._get_array_dimensions(node)
            
            if not dimensions:
                return issues
            
            for dim in dimensions:
                if dim > 0 and dim % self.LDS_BANKS == 0:
                    line = node.location.line if hasattr(node, 'location') and node.location else 0
                    column = node.location.column if hasattr(node, 'location') and node.location else 0
                    
                    issues.append(AnalysisIssue(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        line=line,
                        column=column,
                        message=f"Shared memory dimension {dim} is multiple of {self.LDS_BANKS}. "
                               f"Causes {dim // self.LDS_BANKS}-way bank conflict.",
                        recommendation=f"Pad array to avoid conflicts. "
                                     f"Example: [{dim}][{dim}] → [{dim}][{dim + 1}]",
                        snippet=f"dimension={dim}",
                        confidence=0.8,
                        # ✅ FIX: Pass actual stride for deterministic math
                        metadata={'stride': dim, 'banks': self.LDS_BANKS}
                    ))
                    break
            
        except Exception as e:
            logger.debug(f"SharedMemoryBankConflictRule error: {e}")
        
        return issues


class CudaBuiltinRule(BaseRule):
    """ROCM_004: Detects CUDA builtin usage that may not translate."""
    
    rule_id = "ROCM_004"
    name = "Legacy CUDA Builtin Usage"
    severity = "LOW"
    description = "Detects CUDA builtins that may have different behavior on AMD"
    
    WATCHED_BUILTINS = {
        'warpSize': 'AMD wavefront size is 64 on CDNA, 32 on RDNA',
        'clock64': 'Use hipClock64 for HIP compatibility',
        '__cuda_builtin_threadIdx': 'Use hipThreadIdx instead',
    }
    
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if not LIBCLANG_AVAILABLE:
            return issues
        
        # ✅ FIX: Filter by CursorKind BEFORE expensive operations
        if node.kind != CursorKind.DECL_REF_EXPR:
            return issues
        
        try:
            name = node.spelling if hasattr(node, 'spelling') else ""
            
            if name in self.WATCHED_BUILTINS:
                line = node.location.line if hasattr(node, 'location') and node.location else 0
                column = node.location.column if hasattr(node, 'location') and node.location else 0
                
                issues.append(AnalysisIssue(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    line=line,
                    column=column,
                    message=f"CUDA builtin '{name}' detected. {self.WATCHED_BUILTINS[name]}",
                    recommendation="Verify behavior on AMD hardware. "
                                 "Consider using HIP equivalents.",
                    snippet=name,
                    confidence=0.6,
                    metadata={'builtin': name}
                ))
        
        except Exception as e:
            logger.debug(f"CudaBuiltinRule error: {e}")
        
        return issues


class KernelLaunchConfigRule(BaseRule):
    """ROCM_005: Detects suboptimal kernel launch configurations."""
    
    rule_id = "ROCM_005"
    name = "Suboptimal Kernel Launch Configuration"
    severity = "MEDIUM"
    description = "Detects kernel launch configurations that limit occupancy"
    
    OPTIMAL_BLOCK_SIZES = {64, 128, 256, 512, 1024}
    
    def detect(self, node: Any, context: Dict[str, Any]) -> List[AnalysisIssue]:
        issues = []
        
        if not LIBCLANG_AVAILABLE:
            return issues
        
        # ✅ FIX: Filter by CursorKind BEFORE expensive operations
        if node.kind != CursorKind.CALL_EXPR:
            return issues
        
        try:
            if hasattr(node, 'get_tokens'):
                tokens = [t.spelling for t in node.get_tokens()]
                
                for i, token in enumerate(tokens):
                    if token.isdigit():
                        try:
                            value = int(token)
                            if value > 0 and value not in self.OPTIMAL_BLOCK_SIZES:
                                if value < 64 or value > 1024:
                                    line = node.location.line if hasattr(node, 'location') and node.location else 0
                                    column = node.location.column if hasattr(node, 'location') and node.location else 0
                                    
                                    issues.append(AnalysisIssue(
                                        rule_id=self.rule_id,
                                        severity=self.severity,
                                        line=line,
                                        column=column,
                                        message=f"Block size {value} may limit occupancy on AMD. "
                                               f"Recommended: {list(self.OPTIMAL_BLOCK_SIZES)}",
                                        recommendation="Adjust block dimensions for optimal occupancy. "
                                                     "Consider 64, 128, 256, 512, or 1024.",
                                        snippet=f"block_size={value}",
                                        confidence=0.5,
                                        metadata={'block_size': value}
                                    ))
                        except ValueError:
                            pass
        
        except Exception as e:
            logger.debug(f"KernelLaunchConfigRule error: {e}")
        
        return issues


# ============================================================================
# RULE ENGINE ORCHESTRATOR
# ============================================================================

class RuleEngine:
    """
    Orchestrates AST traversal and rule execution.
    
    FIXES APPLIED:
    - ✅ Iterative traversal (prevents RecursionError)
    - ✅ Context deep copy (prevents mutation bleeding)
    - ✅ Thread-safe run_rules() method
    """
    
    def __init__(self):
        """Initialize engine with default rules."""
        self._lock: threading.Lock = threading.Lock()
        
        self.rules: List[BaseRule] = [
            HardcodedWarpSizeRule(),
            NvidiaIntrinsicRule(),
            SharedMemoryBankConflictRule(),
            CudaBuiltinRule(),
            KernelLaunchConfigRule(),
        ]
        
        logger.info(f"RuleEngine initialized with {len(self.rules)} rules")
    
    def get_rule_metadata(self) -> List[Dict[str, str]]:
        """Get metadata for all registered rules."""
        return [rule.get_metadata() for rule in self.rules]
    
    # ✅ FIX: Thread-safe and iterative traversal
    def run_rules(self, ast_root: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Traverses the AST and runs all registered rules.
        
        FIXES:
        - Iterative DFS (prevents RecursionError)
        - Context deep copy (prevents mutation bleeding)
        - Thread-safe with lock
        """
        with self._lock:  # ✅ Thread safety
            # ✅ FIX: Create local copy to prevent external dict mutation
            local_context = copy.deepcopy(context) if context else {}
            
            all_issues: List[AnalysisIssue] = []
            local_context['in_kernel'] = False
            local_context['parent_keywords'] = set()
            
            # ✅ FIX: Iterative DFS using a stack to prevent RecursionError
            stack: List[Tuple[Cursor, Dict[str, Any]]] = [(ast_root, local_context)]
            
            while stack:
                current_node, current_context = stack.pop()
                
                # 1. Update Context for current node
                if current_node.kind == CursorKind.FUNCTION_DECL:
                    try:
                        if hasattr(current_node, 'get_tokens'):
                            tokens = [t.spelling for t in current_node.get_tokens()]
                            if '__global__' in tokens or '__device__' in tokens:
                                current_context['in_kernel'] = True
                    except Exception:
                        pass
                        
                if hasattr(current_node, 'spelling') and current_node.spelling:
                    # ✅ FIX: Create a new set to prevent parent keywords bleeding into sibling branches
                    current_context['parent_keywords'] = current_context['parent_keywords'].copy()
                    current_context['parent_keywords'].add(current_node.spelling)
                    
                # 2. Run rules
                for rule in self.rules:
                    try:
                        found = rule.detect(current_node, current_context)
                        all_issues.extend(found)
                    except Exception as e:
                        logger.warning(f"Rule {rule.rule_id} failed on node: {e}")
                        
                # 3. Push children to stack (reversed to maintain left-to-right DFS execution)
                try:
                    children = list(current_node.get_children())
                    if children is not None:
                        for child in reversed(children):
                            # ✅ FIX: Pass a deep copy of the context so children inherit current state safely
                            stack.append((child, copy.deepcopy(current_context)))
                except Exception as e:
                    logger.debug(f"Traversal extraction error: {e}")
            
            # Compute Portability Score (0-100)
            score = self._compute_score(all_issues)
            status = 'PASS' if score >= 70 else ('WARNING' if score >= 40 else 'FAIL')
            
            return {
                'score': score,
                'status': status,
                'issues': [i.to_dict() for i in all_issues],
                'issue_count': len(all_issues),
                'rules_executed': len(self.rules)
            }
    
    def _compute_score(self, issues: List[AnalysisIssue]) -> float:
        """Compute portability score based on issues found."""
        if not issues:
            return 100.0
        
        penalties = {
            'CRITICAL': 25.0,
            'HIGH': 15.0,
            'MEDIUM': 8.0,
            'LOW': 3.0,
            'INFO': 1.0
        }
        
        total_deduction = 0.0
        for issue in issues:
            penalty = penalties.get(issue.severity, 5.0)
            total_deduction += penalty * issue.confidence
        
        score = max(0.0, 100.0 - total_deduction)
        
        return round(score, 2)


import copy