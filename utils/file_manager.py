"""File and directory management utilities."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml with defaults fallback."""
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Ensure all required studio directories exist."""
    root = get_project_root()
    cfg = config or load_config()
    storage = cfg.get("storage", {})

    dirs = {
        "data": root / storage.get("data_dir", "data"),
        "train": root / storage.get("data_dir", "data") / "train",
        "validation": root / storage.get("data_dir", "data") / "validation",
        "test": root / storage.get("data_dir", "data") / "test",
        "models": root / storage.get("models_dir", "models/base"),
        "adapters": root / storage.get("adapters_dir", "models/adapters"),
        "experiments": root / storage.get("experiments_dir", "experiments"),
        "database": (root / storage.get("database_path", "database/experiments.db")).parent,
        "sample_datasets": root / storage.get("sample_data_dir", "sample_datasets"),
    }

    for name, directory in dirs.items():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def format_size(bytes_val: int) -> str:
    """Format byte count to readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_dir_size(directory: str | Path) -> int:
    """Calculate total size of all files in a directory."""
    path = Path(directory)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def list_adapters() -> List[Dict[str, Any]]:
    """Scan and list all trained adapter directories in models/adapters/."""
    root = get_project_root()
    adapters_dir = root / "models" / "adapters"
    if not adapters_dir.exists():
        return []

    adapters = []
    for item in adapters_dir.iterdir():
        if item.is_dir():
            # Check for MLX adapter weights or HF adapter_model.safetensors / adapter_config.json
            files = [f.name for f in item.iterdir() if f.is_file()]
            has_mlx = any(f.endswith(".safetensors") or f.endswith(".npz") for f in files)
            has_hf = "adapter_config.json" in files or "adapter_model.safetensors" in files
            has_config = "adapter_config.json" in files or "config.yaml" in files or "adapter_info.json" in files

            if has_mlx or has_hf or has_config or files:
                size_bytes = get_dir_size(item)
                adapters.append({
                    "name": item.name,
                    "path": str(item.resolve()),
                    "size": format_size(size_bytes),
                    "size_bytes": size_bytes,
                    "type": "MLX" if ("adapters.safetensors" in files or not has_hf) else "HF/PEFT",
                    "files": files,
                    "modified": item.stat().st_mtime,
                })

    adapters.sort(key=lambda x: x["modified"], reverse=True)
    return adapters


def safe_delete_adapter(adapter_name: str) -> bool:
    """Safely delete an adapter directory without touching base models."""
    root = get_project_root()
    adapters_dir = root / "models" / "adapters"
    target = (adapters_dir / adapter_name).resolve()

    # Security check: must reside inside adapters_dir and not be the root
    if not str(target).startswith(str(adapters_dir.resolve())):
        return False
    if target == adapters_dir.resolve():
        return False

    if target.exists() and target.is_dir():
        shutil.rmtree(target)
        return True
    return False

