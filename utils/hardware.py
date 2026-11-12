"""Hardware detection and environment diagnostics utility for macOS and Windows.

Supports Apple Silicon (MLX, MPS), NVIDIA CUDA (Windows/Linux), DirectML, and CPU fallbacks.
"""

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import psutil


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    architecture: str
    cpu_brand: str
    total_memory_gb: float
    available_memory_gb: float
    is_apple_silicon: bool
    is_windows: bool
    is_unified_memory: bool
    mlx_available: bool
    mlx_gpu_available: bool
    mps_available: bool
    cuda_available: bool
    cuda_device_name: Optional[str]
    cuda_device_count: int
    cuda_vram_gb: Optional[float]
    directml_available: bool
    bitsandbytes_available: bool
    recommended_backend: str
    backend_status_message: str


def get_cpu_brand() -> str:
    """Retrieve detailed CPU brand string across macOS, Windows, and Linux."""
    system = platform.system()
    if system == "Darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if brand:
                return brand
        except Exception:
            pass
    elif system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if brand and str(brand).strip():
                return str(brand).strip()
        except Exception:
            pass
        try:
            brand = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            lines = [l.strip() for l in brand.splitlines() if l.strip() and "name" not in l.lower()]
            if lines:
                return lines[0]
        except Exception:
            pass

    return platform.processor() or platform.machine() or "Generic Processor"


def get_memory_info() -> Dict[str, float]:
    """Retrieve system memory metrics in gigabytes."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
        "used_gb": round(vm.used / (1024**3), 2),
        "percent_used": vm.percent,
    }


def check_mlx_support() -> Dict[str, bool]:
    """Check whether Apple MLX is installed and whether Metal GPU acceleration is active."""
    if platform.system() != "Darwin":
        # MLX is only available on Apple Silicon / macOS
        return {"installed": False, "gpu_available": False}

    import importlib.util

    mlx_installed = (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )
    mlx_gpu = False

    if mlx_installed:
        try:
            import mlx.core as mx

            device = mx.default_device()
            mlx_gpu = "gpu" in str(device).lower()
        except Exception:
            mlx_gpu = False

    return {"installed": mlx_installed, "gpu_available": mlx_gpu}


def check_torch_accelerators() -> Dict[str, Any]:
    """Check PyTorch acceleration devices (MPS for Apple Silicon, CUDA for NVIDIA, DirectML on Windows)."""
    mps_available = False
    cuda_available = False
    cuda_device_name = None
    cuda_count = 0
    cuda_vram_gb = None
    directml_available = False

    try:
        import torch

        # MPS on macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_available = True

        # CUDA on NVIDIA (Windows / Linux)
        if torch.cuda.is_available():
            cuda_available = True
            cuda_count = torch.cuda.device_count()
            if cuda_count > 0:
                cuda_device_name = torch.cuda.get_device_name(0)
                try:
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    cuda_vram_gb = round(total_bytes / (1024**3), 2)
                except Exception:
                    pass

        # DirectML (Optional for Windows AMD/Intel GPUs)
        try:
            import torch_directml  # type: ignore

            if torch_directml.is_available():
                directml_available = True
        except Exception:
            pass

    except Exception:
        pass

    return {
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "cuda_count": cuda_count,
        "cuda_device_name": cuda_device_name,
        "cuda_vram_gb": cuda_vram_gb,
        "directml_available": directml_available,
    }


def check_bitsandbytes() -> bool:
    """Check if bitsandbytes is available (standard on Windows/Linux with CUDA)."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def detect_hardware() -> HardwareProfile:
    """Comprehensive hardware detection and cross-platform backend recommendation."""
    sys_name = platform.system()
    machine = platform.machine()
    cpu_brand = get_cpu_brand()
    mem_info = get_memory_info()

    is_windows = sys_name == "Windows"
    is_apple_silicon = sys_name == "Darwin" and (
        "arm64" in machine.lower() or "apple" in cpu_brand.lower()
    )
    is_unified_memory = is_apple_silicon

    mlx_info = check_mlx_support()
    torch_info = check_torch_accelerators()
    bnb_avail = check_bitsandbytes()

    # Determine recommended backend based on OS and detected hardware
    if is_apple_silicon:
        recommended_backend = "mlx"
        if mlx_info["installed"]:
            status_msg = (
                f"Apple Silicon ({cpu_brand}) detected with {mem_info['total_gb']} GB Unified Memory. "
                f"Native Apple MLX and MLX-LM backend is active for high-performance LoRA & 4-bit QLoRA."
            )
        else:
            status_msg = (
                f"Apple Silicon detected, but MLX is not installed. Please install mlx and mlx-lm."
            )
    elif torch_info["cuda_available"]:
        recommended_backend = "cuda"
        vram_str = f" with {torch_info['cuda_vram_gb']} GB VRAM" if torch_info["cuda_vram_gb"] else ""
        bnb_str = " (bitsandbytes 4-bit NF4 active)" if bnb_avail else ""
        status_msg = (
            f"NVIDIA CUDA detected on {sys_name} ({torch_info['cuda_device_name']}{vram_str}){bnb_str}. "
            f"Hugging Face PEFT + BitsAndBytes QLoRA backend is recommended."
        )
    elif is_windows:
        recommended_backend = "cpu"
        status_msg = (
            f"Windows ({cpu_brand}) detected with {mem_info['total_gb']} GB RAM. "
            f"No NVIDIA CUDA GPU detected. Running in Hugging Face PEFT CPU mode."
        )
    else:
        recommended_backend = "cpu"
        status_msg = (
            f"{sys_name} ({cpu_brand}) detected with {mem_info['total_gb']} GB RAM. "
            f"No GPU accelerator detected. Running in CPU fallback mode."
        )

    return HardwareProfile(
        os_name=sys_name,
        os_version=platform.version() if is_windows else (platform.mac_ver()[0] if sys_name == "Darwin" else platform.version()),
        architecture=machine,
        cpu_brand=cpu_brand,
        total_memory_gb=mem_info["total_gb"],
        available_memory_gb=mem_info["available_gb"],
        is_apple_silicon=is_apple_silicon,
        is_windows=is_windows,
        is_unified_memory=is_unified_memory,
        mlx_available=mlx_info["installed"],
        mlx_gpu_available=mlx_info["gpu_available"],
        mps_available=torch_info["mps_available"],
        cuda_available=torch_info["cuda_available"],
        cuda_device_name=torch_info["cuda_device_name"],
        cuda_device_count=torch_info["cuda_count"],
        cuda_vram_gb=torch_info["cuda_vram_gb"],
        directml_available=torch_info["directml_available"],
        bitsandbytes_available=bnb_avail,
        recommended_backend=recommended_backend,
        backend_status_message=status_msg,
    )


def get_current_telemetry() -> Dict[str, Any]:
    """Fast real-time telemetry for dashboard metrics and progress bars."""
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)

    telemetry = {
        "cpu_percent": cpu_pct,
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_free_gb": round(vm.available / (1024**3), 2),
        "ram_percent": vm.percent,
    }

    # Add MLX peak memory if available on Apple Silicon
    try:
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
            peak_bytes = mx.get_peak_memory()
            active_bytes = mx.get_active_memory()
            telemetry["mlx_peak_mb"] = round(peak_bytes / (1024**2), 2)
            telemetry["mlx_active_mb"] = round(active_bytes / (1024**2), 2)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            peak_bytes = mx.metal.get_peak_memory()
            active_bytes = mx.metal.get_active_memory()
            telemetry["mlx_peak_mb"] = round(peak_bytes / (1024**2), 2)
            telemetry["mlx_active_mb"] = round(active_bytes / (1024**2), 2)
    except Exception:
        pass

    # Add PyTorch CUDA memory if available (Windows/Linux)
    try:
        import torch

        if torch.cuda.is_available():
            telemetry["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
            telemetry["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 2)
    except Exception:
        pass

    return telemetry
