# Definitions of "Anti-Patterns" for AMD Hardware
RULES = {
    "WARP_SIZE": {
        "id": "AMD_001",
        "severity": "HIGH",
        "pattern": "32",
        "message": "Hardcoded Warp Size (32) detected. AMD Wavefronts prefer 64."
    },
    "SHUFFLE_SYNC": {
        "id": "AMD_002",
        "severity": "MEDIUM",
        "pattern": "__shfl_sync",
        "message": "NVIDIA intrinsic '__shfl_sync' detected. Use HIP equivalent."
    },
    "SHARED_MEM_BANK": {
        "id": "AMD_003",
        "severity": "CRITICAL",
        "pattern": "shared_memory",
        "message": "Potential Shared Memory Bank Conflict for 32-bank LDS."
    }
}
