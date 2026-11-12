

"""Model discovery, inspection, compatibility checking, and downloading."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from huggingface_hub import HfApi, snapshot_download
from utils.file_manager import format_size, get_dir_size, get_project_root, load_config
from utils.hardware import detect_hardware


class ModelLoader:
    def __init__(self):
        self.config = load_config()
        self.root = get_project_root()
        self.models_dir = self.root / self.config.get("storage", {}).get("models_dir", "models/base")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hardware = detect_hardware()

    def get_catalog_models(self) -> List[Dict[str, Any]]:
        """Return predefined supported models from config.yaml."""
        return self.config.get("supported_models", [])

    def list_local_models(self) -> List[Dict[str, Any]]:
        """Scan models/base directory for locally downloaded models."""
        local_models = []
        if not self.models_dir.exists():
            return []

        # HF snapshot download often creates directories with org/repo or flat dirs
        for path in self.models_dir.glob("*"):
            if path.is_dir():
                subdirs = [p for p in path.iterdir() if p.is_dir()]
                # Check if it's an org directory (e.g. models/base/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
                candidate_paths = [path] + subdirs
                for cand in candidate_paths:
                    files = [f.name for f in cand.iterdir() if f.is_file()]
                    has_config = "config.json" in files
                    has_weights = any(
                        f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".npz")
                        for f in files
                    )
                    if has_config or has_weights:
                        size_bytes = get_dir_size(cand)
                        # Determine model id
                        rel_id = str(cand.relative_to(self.models_dir))
                        model_info = self.inspect_model_metadata(str(cand))
                        local_models.append({
                            "id": rel_id,
                            "path": str(cand.resolve()),
                            "name": cand.name,
                            "size": format_size(size_bytes),
                            "size_bytes": size_bytes,
                            "files_count": len(files),
                            "quantization": model_info.get("quantization", "Unknown"),
                            "params": model_info.get("params", "Unknown"),
                            "backend_compatibility": self.check_compatibility(rel_id),
                        })
        return local_models

    def inspect_model_metadata(self, model_path_or_id: str) -> Dict[str, Any]:
        """Inspect config.json and metadata for parameter count and quantization."""
        info = {
            "quantization": "None (fp16/bf16)",
            "params": "Unknown",
            "arch": "Unknown",
        }

        # Check local folder
        local_path = Path(model_path_or_id)
        if not local_path.is_absolute():
            local_path = self.models_dir / model_path_or_id

        config_file = local_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)

                # Architecture
                archs = cfg_data.get("architectures", [])
                if archs:
                    info["arch"] = archs[0]

                # Quantization
                if "quantization" in cfg_data or "quantization_config" in cfg_data:
                    q = cfg_data.get("quantization", {}) or cfg_data.get("quantization_config", {})
                    bits = q.get("bits", 4)
                    info["quantization"] = f"{bits}-bit"
                elif "4bit" in model_path_or_id.lower() or "4-bit" in model_path_or_id.lower():
                    info["quantization"] = "4-bit"
                elif "8bit" in model_path_or_id.lower() or "8-bit" in model_path_or_id.lower():
                    info["quantization"] = "8-bit"

                # Hidden size & layers estimation
                num_layers = cfg_data.get("num_hidden_layers", cfg_data.get("n_layer", 0))
                hidden_dim = cfg_data.get("hidden_size", cfg_data.get("n_embd", 0))
                if num_layers and hidden_dim:
                    approx_params = num_layers * hidden_dim * hidden_dim * 12
                    if approx_params > 1e9:
                        info["params"] = f"{approx_params / 1e9:.1f}B"
                    else:
                        info["params"] = f"{approx_params / 1e6:.0f}M"
            except Exception:
                pass

        if info["params"] == "Unknown":
            if "0.5b" in model_path_or_id.lower():
                info["params"] = "0.5B"
            elif "1.5b" in model_path_or_id.lower():
                info["params"] = "1.5B"
            elif "7b" in model_path_or_id.lower():
                info["params"] = "7B"

        if info["quantization"] == "None (fp16/bf16)":
            if "4bit" in model_path_or_id.lower():
                info["quantization"] = "4-bit"
            elif "8bit" in model_path_or_id.lower():
                info["quantization"] = "8-bit"

        return info

    def check_compatibility(self, model_id: str) -> Dict[str, Any]:
        """Check whether the model is compatible with Apple Silicon MLX or Windows/Linux CUDA/PEFT."""
        is_mlx_model = "mlx-community" in model_id.lower() or "mlx" in model_id.lower()
        is_apple = self.hardware.is_apple_silicon
        is_win = self.hardware.is_windows

        compatible_backends = []
        warnings = []

        if is_apple:
            compatible_backends.append("MLX (Native Apple Silicon)")
            if not is_mlx_model:
                compatible_backends.append("Hugging Face Transformers (MPS/CPU)")
                warnings.append("Hugging Face model will run via Transformers or be loaded into MLX.")
        elif is_win:
            if is_mlx_model:
                warnings.append(
                    "MLX models are native to Apple Silicon only. On Windows, please select standard Hugging Face model repositories (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')."
                )
            if self.hardware.cuda_available:
                compatible_backends.append("NVIDIA CUDA (PEFT + BitsAndBytes QLoRA)")
            compatible_backends.append("Hugging Face PEFT (CPU Mode)")
        else:
            # Linux or generic Unix
            if self.hardware.cuda_available:
                compatible_backends.append("NVIDIA CUDA (PEFT + BitsAndBytes QLoRA)")
            compatible_backends.append("Hugging Face PEFT (CPU Mode)")
            if is_mlx_model:
                warnings.append("MLX models require Apple Silicon macOS.")

        recommended = "MLX" if is_apple else ("CUDA" if self.hardware.cuda_available else "CPU")

        return {
            "supported_backends": compatible_backends,
            "recommended": recommended,
            "warnings": warnings,
            "is_mlx_native": is_mlx_model,
        }

    def download_model(
        self,
        repo_id: str,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Download model weights and configuration from Hugging Face Hub."""
        clean_id = repo_id.strip()
        target_dir = self.models_dir / clean_id
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = snapshot_download(
                repo_id=clean_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            size_bytes = get_dir_size(target_dir)
            meta = self.inspect_model_metadata(str(target_dir))
            return {
                "success": True,
                "path": str(downloaded_path),
                "size": format_size(size_bytes),
                "params": meta.get("params"),
                "quantization": meta.get("quantization"),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "path": None,
                "size": "0 B",
                "error": str(e),
            }
