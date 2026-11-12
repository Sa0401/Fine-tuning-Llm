"""Native Apple Silicon Quantized LoRA (QLoRA) fine-tuning backend using MLX-LM."""

from pathlib import Path
from typing import Any, Dict, List
import yaml


class MLXQLoRATrainer:
    """Manages MLX-LM 4-bit Quantized LoRA fine-tuning on Apple Silicon."""

    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.output_dir / "train_config.yaml"

    def prepare_config(
        self,
        model_path_or_id: str,
        data_dir: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """Create MLX-LM compatible YAML configuration file tailored for quantized models."""
        rank = int(hyperparameters.get("rank", 8))
        alpha = float(hyperparameters.get("alpha", 16.0))
        dropout = float(hyperparameters.get("dropout", 0.05))
        scale = float(alpha / rank) if rank > 0 else 2.0

        # Build MLX-LM config structure optimized for 4-bit quantized base models
        mlx_cfg = {
            "model": model_path_or_id,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": hyperparameters.get("optimizer", "adamw"),
            "data": data_dir,
            "batch_size": int(hyperparameters.get("batch_size", 4)),
            "iters": int(hyperparameters.get("iterations", 100)),
            "val_batches": int(hyperparameters.get("val_batches", 10)),
            # Slightly higher learning rate is typical for quantized LoRA
            "learning_rate": float(hyperparameters.get("learning_rate", 2e-4)),
            "steps_per_report": int(hyperparameters.get("steps_per_report", 5)),
            "steps_per_eval": int(hyperparameters.get("steps_per_eval", 25)),
            "adapter_path": str(self.output_dir.resolve()),
            "save_every": int(hyperparameters.get("save_every", 50)),
            "max_seq_length": int(hyperparameters.get("max_seq_length", 512)),
            "grad_checkpoint": bool(hyperparameters.get("grad_checkpoint", False)),
            "grad_accumulation_steps": int(hyperparameters.get("grad_accumulation_steps", 1)),
            "lora_parameters": {
                "rank": rank,
                "dropout": dropout,
                "scale": scale,
            },
            "mask_prompt": bool(hyperparameters.get("mask_prompt", True)),
        }

        num_layers = hyperparameters.get("num_layers")
        if num_layers is not None:
            mlx_cfg["num_layers"] = int(num_layers)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(mlx_cfg, f, default_flow_style=False)

        return str(self.config_path.resolve())

    def build_command(self, python_bin: str) -> List[str]:
        """Construct the execution command line."""
        return [
            python_bin,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(self.config_path.resolve()),
        ]

