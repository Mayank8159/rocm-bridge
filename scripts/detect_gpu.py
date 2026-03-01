#!/usr/bin/env python3
"""
ROCm Bridge - GPU Detection Script
===================================
Detects available AMD GPUs and creates custom JSON profiles.
Useful for adding support for new/unsupported hardware.

Usage:
    python scripts/detect_gpu.py [--save-profile]
"""

import sys
import json
import subprocess
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rocm_bridge.core.hal import HardwareAbstractionLayer, HardwareProfile


def detect_gpu_rocminfo() -> list:
    """Detect GPUs using rocminfo."""
    try:
        result = subprocess.run(
            ['rocminfo'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        gpus = []
        output = result.stdout
        
        # Parse all GPU agents
        gpu_agents = re.findall(
            r'Agent\s+\d+\s*\n(.*?)(?=Agent\s+\d+|$)',
            output,
            re.DOTALL
        )
        
        for agent_block in gpu_agents:
            # Skip CPU agents
            if "Ryzen" in agent_block or "CPU" in agent_block:
                continue
            
            name_match = re.search(r"Marketing Name:\s*(.+?)(?:\n|$)", agent_block)
            wave_match = re.search(r"Wavefront Size:\s*(\d+)", agent_block)
            cu_match = re.search(r"Compute Unit:\s*(\d+)", agent_block)
            device_match = re.search(r"Name:\s*(gfx\d+)", agent_block)
            
            if name_match:
                gpu = {
                    "name": name_match.group(1).strip(),
                    "wavefront_size": int(wave_match.group(1)) if wave_match else 64,
                    "compute_units": int(cu_match.group(1)) if cu_match else 0,
                    "device_id": device_match.group(1) if device_match else "unknown"
                }
                gpus.append(gpu)
        
        return gpus
    
    except FileNotFoundError:
        print("❌ rocminfo not found. Is ROCm installed?")
        return []
    except Exception as e:
        print(f"❌ Error detecting GPU: {e}")
        return []

# ============================================================================
# FIX detect_gpu_clinfo FUNCTION
# ============================================================================

def detect_gpu_clinfo() -> list:
    """Fallback GPU detection using clinfo."""
    try:
        result = subprocess.run(
            ['clinfo'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        gpus = []
        output = result.stdout
        
        # FIX #9: Safe index access
        for line in output.split('\n'):
            if 'Device Name' in line:
                parts = line.split(':')
                if len(parts) > 1:  # Check length before accessing
                    name = parts[1].strip()
                    if 'AMD' in name or 'Radeon' in name or 'Instinct' in name:
                        gpus.append({"name": name, "source": "clinfo"})
        
        return gpus
    
    except Exception:
        return []
    
def create_profile_from_detection(gpu_info: dict) -> HardwareProfile:
    """Create a HardwareProfile from detected GPU info."""
    name = gpu_info.get("name", "Unknown")
    
    # Determine architecture from name
    arch = "UNKNOWN"
    if "Instinct" in name:
        arch = "CDNA 3" if "MI300" in name else "CDNA 2"
    elif "Radeon" in name:
        if "7900" in name:
            arch = "RDNA 3"
        elif "6900" in name or "6800" in name:
            arch = "RDNA 2"
        else:
            arch = "RDNA"
    
    # Determine wavefront size from architecture
    wavefront_size = 64 if "CDNA" in arch else 32
    
    return HardwareProfile(
        name=name,
        arch=arch,
        wavefront_size=wavefront_size,
        compute_units=gpu_info.get("compute_units", 0),
        device_id=gpu_info.get("device_id", "unknown"),
        is_simulated=False
    )


def save_profile(profile: HardwareProfile, profiles_dir: Path) -> Path:
    """Save profile to JSON file."""
    profile_name = profile.device_id if profile.device_id != "unknown" else profile.name.replace(' ', '_').lower()
    profile_path = profiles_dir / f"{profile_name}.json"
    
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile.to_dict(), f, indent=2)
    
    return profile_path


def main():
    print("=" * 60)
    print("ROCm Bridge - GPU Detection Script")
    print("=" * 60)
    print()
    
    # Try rocminfo first
    print("🔍 Detecting GPUs via rocminfo...")
    gpus = detect_gpu_rocminfo()
    
    # Fallback to clinfo
    if not gpus:
        print("🔍 Falling back to clinfo...")
        gpus = detect_gpu_clinfo()
    
    if not gpus:
        print("❌ No AMD GPUs detected.")
        print()
        print("Available simulation profiles:")
        hal = HardwareAbstractionLayer()
        for profile_name in hal.list_available_profiles():
            print(f"  - {profile_name}")
        return
    
    print(f"✅ Found {len(gpus)} AMD GPU(s):")
    print()
    
    profiles_dir = Path(__file__).parent.parent / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    for i, gpu in enumerate(gpus, 1):
        print(f"[{i}] {gpu.get('name', 'Unknown')}")
        print(f"    Device ID: {gpu.get('device_id', 'Unknown')}")
        print(f"    Wavefront Size: {gpu.get('wavefront_size', 'Unknown')}")
        print(f"    Compute Units: {gpu.get('compute_units', 'Unknown')}")
        print()
        
        # Create and optionally save profile
        profile = create_profile_from_detection(gpu)
        
        if "--save-profile" in sys.argv:
            profile_path = save_profile(profile, profiles_dir)
            print(f"    💾 Profile saved to: {profile_path}")
            print()
    
    print("=" * 60)
    print("To use a detected GPU as simulation target:")
    print(f"  HardwareAbstractionLayer(profile_name='<device_id>')")
    print("=" * 60)


if __name__ == "__main__":
    main()