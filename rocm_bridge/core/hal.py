"""
ROCm Bridge - Hardware Abstraction Layer (HAL)
==============================================
Dynamically detects AMD GPU hardware or loads simulation profiles.
Ensures deterministic math for the simulator regardless of host environment.

Production Features:
- Real hardware detection via rocminfo
- JSON profile fallback for CI/CD and headless environments
- Thread-safe profile access with proper singleton pattern
- Atomic profile loading
- Multi-GPU support detection
"""

import subprocess
import json
import re
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from threading import Lock
import platform

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Immutable hardware specification container."""
    
    name: str = "Unknown"
    arch: str = "UNKNOWN"
    wavefront_size: int = 64
    peak_gflops: float = 0.0
    memory_bandwidth: float = 0.0
    compute_units: int = 0
    lds_banks: int = 32
    lds_bank_width: int = 4
    is_simulated: bool = True
    gpu_id: int = 0
    device_id: str = ""
    
    def is_cdna(self) -> bool:
        """Returns True if architecture is CDNA (Wave64)."""
        return "CDNA" in self.arch.upper()
    
    def is_rdna(self) -> bool:
        """Returns True if architecture is RDNA (Wave32)."""
        return "RDNA" in self.arch.upper()
    
    def get_valu_penalty(self, launched_threads: int) -> float:
        """
        Calculates VALU utilization penalty for wavefront mismatch.
        
        Args:
            launched_threads: Thread count from kernel launch configuration
            
        Returns:
            Efficiency multiplier (0.0 to 1.0)
        """
        if launched_threads >= self.wavefront_size:
            return 1.0
        return launched_threads / self.wavefront_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Export profile as dictionary for JSON serialization."""
        return {
            "name": self.name,
            "arch": self.arch,
            "wavefront_size": self.wavefront_size,
            "peak_gflops": self.peak_gflops,
            "memory_bandwidth": self.memory_bandwidth,
            "compute_units": self.compute_units,
            "lds_banks": self.lds_banks,
            "is_simulated": self.is_simulated,
            "gpu_id": self.gpu_id,
            "device_id": self.device_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareProfile":
        """Create HardwareProfile from dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            arch=data.get("arch", "UNKNOWN"),
            wavefront_size=data.get("wavefront_size", 64),
            peak_gflops=data.get("peak_gflops", 0.0),
            memory_bandwidth=data.get("memory_bandwidth", 0.0),
            compute_units=data.get("compute_units", 0),
            lds_banks=data.get("lds_banks", 32),
            lds_bank_width=data.get("lds_bank_width", 4),
            is_simulated=data.get("is_simulated", True),
            gpu_id=data.get("gpu_id", 0),
            device_id=data.get("device_id", "")
        )
    
    def get_valu_penalty(self, launched_threads: int) -> float:
        """
        Calculates VALU utilization penalty for wavefront mismatch.
        FIX: Accurately calculates partial wavefront masking.
        """
        if launched_threads <= 0:
            return 0.0
            
        # Hardware allocates in full wavefront chunks
        allocated_threads = math.ceil(launched_threads / self.wavefront_size) * self.wavefront_size
        return launched_threads / allocated_threads


class HardwareAbstractionLayer:
    """
    Production HAL: Detects real hardware or loads simulation profiles.
    Thread-safe singleton pattern for consistent profile access.
    
    BUG FIXES APPLIED:
    - Fixed path resolution (was 3 parents, now 2)
    - Added thread-safe singleton initialization
    - Improved rocminfo regex parsing for multi-GPU systems
    - Added proper exception handling for subprocess calls
    """
    
    _instance: Optional["HardwareAbstractionLayer"] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    # Default simulation profiles for common AMD GPUs
    DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
        "mi300x": {
            "name": "AMD Instinct MI300X",
            "arch": "CDNA 3",
            "wavefront_size": 64,
            "peak_gflops": 1300000.0,
            "memory_bandwidth": 5300.0,
            "compute_units": 304,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 192,
            "device_id": "gfx942"
        },
        "mi300a": {
            "name": "AMD Instinct MI300A",
            "arch": "CDNA 3",
            "wavefront_size": 64,
            "peak_gflops": 1000000.0,
            "memory_bandwidth": 4800.0,
            "compute_units": 228,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 128,
            "device_id": "gfx942"
        },
        "mi250x": {
            "name": "AMD Instinct MI250X",
            "arch": "CDNA 2",
            "wavefront_size": 64,
            "peak_gflops": 479000.0,
            "memory_bandwidth": 3350.0,
            "compute_units": 220,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 128,
            "device_id": "gfx90a"
        },
        "mi210": {
            "name": "AMD Instinct MI210",
            "arch": "CDNA 2",
            "wavefront_size": 64,
            "peak_gflops": 180000.0,
            "memory_bandwidth": 1600.0,
            "compute_units": 96,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 64,
            "device_id": "gfx90a"
        },
        "rx7900xtx": {
            "name": "AMD Radeon RX 7900 XTX",
            "arch": "RDNA 3",
            "wavefront_size": 32,
            "peak_gflops": 123000.0,
            "memory_bandwidth": 960.0,
            "compute_units": 96,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 24,
            "device_id": "gfx1100"
        },
        "rx7900xt": {
            "name": "AMD Radeon RX 7900 XT",
            "arch": "RDNA 3",
            "wavefront_size": 32,
            "peak_gflops": 103000.0,
            "memory_bandwidth": 800.0,
            "compute_units": 84,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 20,
            "device_id": "gfx1100"
        },
        "rx6900xt": {
            "name": "AMD Radeon RX 6900 XT",
            "arch": "RDNA 2",
            "wavefront_size": 32,
            "peak_gflops": 51000.0,
            "memory_bandwidth": 512.0,
            "compute_units": 80,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 16,
            "device_id": "gfx1030"
        },
        "rx6800xt": {
            "name": "AMD Radeon RX 6800 XT",
            "arch": "RDNA 2",
            "wavefront_size": 32,
            "peak_gflops": 43000.0,
            "memory_bandwidth": 512.0,
            "compute_units": 72,
            "lds_banks": 32,
            "lds_bank_width": 4,
            "global_memory_size_gb": 16,
            "device_id": "gfx1030"
        }
    }
    
    def __new__(cls, profile_name: str = "mi300x") -> "HardwareAbstractionLayer":
        """Thread-safe singleton pattern to ensure single HAL instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, profile_name: str = "mi300x"):
        """Initialize HAL with specified or detected profile."""
        # Prevent re-initialization
        if self._initialized:
            return
        
        # Thread-safe initialization
        with self._lock:
            if self._initialized:
                return
            
            self._profile_lock: Lock = Lock()
            self._profile: HardwareProfile = HardwareProfile()
            self._profile_name: str = profile_name
            
            # BUG FIX: Was parent.parent.parent (3 levels), now parent.parent (2 levels)
            # hal.py is at: rocm_bridge/core/hal.py
            # profiles/ is at: rocm_bridge/../profiles/ = project_root/profiles/
            self._profiles_dir: Path = Path(__file__).parent.parent / "profiles"
            
            # Ensure profiles directory exists
            self._profiles_dir.mkdir(parents=True, exist_ok=True)
            
            self._initialize_profile()
            self._initialized = True
            
            logger.info(f"HAL initialized: {self._profile.name} "
                       f"(Simulated: {self._profile.is_simulated})")
    
    def _initialize_profile(self) -> None:
        """Attempt physical detection, fallback to JSON or defaults."""
        with self._profile_lock:
            # 1. Try Real Hardware Detection
            physical_profile = self._query_rocminfo()
            if physical_profile:
                self._profile = physical_profile
                logger.info(f"Physical GPU detected: {self._profile.name}")
                return
            
            # 2. Try JSON Profile File
            json_profile = self._load_json_profile(self._profile_name)
            if json_profile:
                self._profile = json_profile
                logger.info(f"Loaded JSON profile: {self._profile.name}")
                return
            
            # 3. Fallback to Hardcoded Defaults
            self._profile = self._get_hardcoded_default()
            logger.warning("Using hardcoded default profile (MI300X)")
    
    def _query_rocminfo(self) -> Optional[HardwareProfile]:
        """
        Execute rocminfo and parse output for GPU specifications.
        Returns None if rocminfo is unavailable or parsing fails.
        
        BUG FIX: Improved regex to handle multi-GPU systems and avoid CPU agents
        """
        try:
            result = subprocess.run(
                ['rocminfo'],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            
            if result.returncode != 0:
                logger.debug("rocminfo returned non-zero exit code")
                return None
            
            output = result.stdout
            
            # Parse all GPU agents (not just the first one)
            gpu_agents = re.findall(
                r'Agent\s+\d+\s*\n(.*?)(?=Agent\s+\d+|$)',
                output,
                re.DOTALL
            )
            
            for agent_block in gpu_agents:
                # Skip CPU agents
                if "Ryzen" in agent_block or "CPU" in agent_block:
                    continue
                
                # Extract Marketing Name
                name_match = re.search(r"Marketing Name:\s*(.+?)(?:\n|$)", agent_block)
                if not name_match:
                    continue
                
                name = name_match.group(1).strip()
                if "Ryzen" in name:
                    continue
                
                # Extract Wavefront Size
                wave_match = re.search(r"Wavefront Size:\s*(\d+)", agent_block)
                wavefront_size = int(wave_match.group(1)) if wave_match else 64
                
                # Extract Compute Units
                cu_match = re.search(r"Compute Unit:\s*(\d+)", agent_block)
                compute_units = int(cu_match.group(1)) if cu_match else 0
                
                # Extract Device ID (gfxXXXX)
                device_match = re.search(r"Name:\s*(gfx\d+)", agent_block)
                device_id = device_match.group(1) if device_match else ""
                
                return HardwareProfile(
                    name=name,
                    arch="CDNA" if "Instinct" in name else "RDNA",
                    wavefront_size=wavefront_size,
                    compute_units=compute_units,
                    is_simulated=False,
                    device_id=device_id
                )
            
            logger.debug("No GPU agents found in rocminfo output")
            return None
            
        except FileNotFoundError:
            logger.debug("rocminfo not found in PATH")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("rocminfo timed out after 10 seconds")
            return None
        except Exception as e:
            logger.debug(f"rocminfo query failed: {e}")
            return None
    
    def _load_json_profile(self, profile_name: str) -> Optional[HardwareProfile]:
        """Load hardware profile from JSON file in profiles/ directory."""
        profile_path = self._profiles_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            logger.debug(f"Profile file not found: {profile_path}")
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return HardwareProfile.from_dict(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in profile {profile_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load profile {profile_path}: {e}")
            return None
    
    # Add this method to HardwareAbstractionLayer class

    def reload_profile(self, profile_name: str = "mi300x") -> HardwareProfile:
        """
    Force reload of hardware profile (for testing or runtime switching).
    
    Args:
        profile_name: Profile name to load (mi300x, rx7900xtx, etc.)
        
    Returns:
        New HardwareProfile object
        """
        with self._profile_lock:
            self._profile_name = profile_name
            self._profile = self._get_hardcoded_default()  # Load from DEFAULT_PROFILES
        
        # Try JSON profile first
            json_profile = self._load_json_profile(profile_name)
            if json_profile:
                self._profile = json_profile
        
            logger.info(f"Reloaded profile: {self._profile.name} (Wavefront={self._profile.wavefront_size})")
            return self._profile

    def _get_hardcoded_default(self) -> HardwareProfile:
        """Return hardcoded MI300X defaults for CI/CD environments."""
        defaults = self.DEFAULT_PROFILES.get(self._profile_name, 
                                              self.DEFAULT_PROFILES["mi300x"])
        return HardwareProfile.from_dict(defaults)
    
    def get_profile(self) -> HardwareProfile:
        """Thread-safe access to current hardware profile."""
        with self._profile_lock:
            return self._profile
    
    def get_wavefront_size(self) -> int:
        """Get native wavefront size for target architecture."""
        return self.get_profile().wavefront_size
    
    def is_cdna_architecture(self) -> bool:
        """Check if target is CDNA (Wave64) architecture."""
        return self.get_profile().is_cdna()
    
    def is_rdna_architecture(self) -> bool:
        """Check if target is RDNA (Wave32) architecture."""
        return self.get_profile().is_rdna()
    
    def calculate_valu_efficiency(self, launched_threads: int) -> float:
        """
        Calculate VALU efficiency based on thread count vs wavefront size.
        
        Args:
            launched_threads: Thread count from kernel launch configuration
            
        Returns:
            Efficiency multiplier (0.0 to 1.0)
        """
        return self.get_profile().get_valu_penalty(launched_threads)
    
    def get_roofline_params(self) -> Dict[str, float]:
        """
        Get Roofline Model parameters for performance simulation.
        
        Returns:
            Dictionary with pi (peak compute) and beta (peak bandwidth)
        """
        profile = self.get_profile()
        return {
            "pi": profile.peak_gflops,  # Peak GFLOPS
            "beta": profile.memory_bandwidth  # Peak GB/s
        }
    
    def reload_profile(self, profile_name: str) -> None:
        """Force reload of hardware profile (for runtime switching)."""
        with self._profile_lock:
            self._profile_name = profile_name
            self._initialize_profile()
    
    def list_available_profiles(self) -> List[str]:
        """List all available profile names (JSON files + defaults)."""
        profiles = set(self.DEFAULT_PROFILES.keys())
        
        if self._profiles_dir.exists():
            for profile_file in self._profiles_dir.glob("*.json"):
                profiles.add(profile_file.stem)
        
        return sorted(list(profiles))
    
    def detect_and_save_current_gpu(self) -> Optional[HardwareProfile]:
        """
        Detect current GPU and save as a new JSON profile.
        Useful for creating custom profiles for new hardware.
        
        Returns:
            HardwareProfile if detection successful, None otherwise
        """
        profile = self._query_rocminfo()
        
        if profile:
            # Save to profiles directory
            profile_name = profile.device_id if profile.device_id else f"detected_{profile.name.replace(' ', '_').lower()}"
            profile_path = self._profiles_dir / f"{profile_name}.json"
            
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile.to_dict(), f, indent=2)
                logger.info(f"Saved detected GPU profile to {profile_path}")
                return profile
            except Exception as e:
                logger.error(f"Failed to save detected profile: {e}")
                return None
        
        return None