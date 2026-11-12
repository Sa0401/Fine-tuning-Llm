"""Adapter merging, format export, and deployment utilities."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from utils.file_manager import format_size, get_dir_size, get_project_root, load_config
from utils.hardware import detect_hardware


class ModelExporter:
    def __init__(self):
        self.root = get_project_root()
        self.config = load_config()
        self.hardware = detect_hardware()

    def merge_adapter(
        self,
        base_model_id: str,
        adapter_path: str,
        output_dir: str,
        dequantize: bool = False,
    ) -> Tuple[bool, str]:
        """Merge LoRA adapter weights directly into the base model weights.

        For MLX-LM, uses `mlx_lm.fuse`. For Hugging Face, uses `merge_and_unload()`.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            if self.hardware.is_apple_silicon:
                # MLX-LM Fuse: execute using python -m mlx_lm fuse
                python_bin = sys.executable
                venv_py = self.root / "venv" / "bin" / "python3"
                if venv_py.exists():
                    python_bin = str(venv_py)

                cmd = [
                    python_bin,
                    "-m",
                    "mlx_lm",
                    "fuse",
                    "--model",
                    base_model_id,
                    "--adapter-path",
                    adapter_path,
                    "--save-path",
                    str(out_path.resolve()),
                ]
                if dequantize:
                    cmd.append("--dequantize")

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.root.resolve()),
                    check=True,
                )
                size_str = format_size(get_dir_size(out_path))
                return True, f"Successfully merged MLX adapter into '{out_path.name}' ({size_str})."

            else:
                # Hugging Face PEFT merge (Windows / Linux)
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel

                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch_dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                )
                peft_model = PeftModel.from_pretrained(base_model, adapter_path)
                merged_model = peft_model.merge_and_unload()

                merged_model.save_pretrained(str(out_path.resolve()))
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
                tokenizer.save_pretrained(str(out_path.resolve()))

                size_str = format_size(get_dir_size(out_path))
                return True, f"Successfully merged PEFT adapter into '{out_path.name}' ({size_str})."

        except Exception as e:
            return False, f"Failed to merge adapter: {str(e)}"

    def export_experiment_report(self, experiment_data: Dict[str, Any], output_path: str) -> str:
        """Export comprehensive experiment report as formatted Markdown."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        hp = experiment_data.get("hyperparameters", {})
        metrics = experiment_data.get("metrics", {})
        hw = experiment_data.get("hardware_info", {})

        md = f"""# Experiment Report: {experiment_data.get('name', 'N/A')}

- **Experiment ID**: `{experiment_data.get('id')}`
- **Execution Date**: {experiment_data.get('created_at')}
- **Status**: `{experiment_data.get('status')}`
- **Duration**: {experiment_data.get('duration_seconds', 0):.1f}s

## Model & Architecture
- **Base Model**: `{experiment_data.get('model_id')}`
- **Method**: `{experiment_data.get('method', '').upper()}`
- **Backend**: `{experiment_data.get('backend')}`
- **Adapter Location**: `{experiment_data.get('adapter_path')}`

## Hyperparameters
| Parameter | Value |
|---|---|
| LoRA Rank (r) | {hp.get('rank', 'N/A')} |
| LoRA Alpha | {hp.get('alpha', 'N/A')} |
| Dropout | {hp.get('dropout', 'N/A')} |
| Learning Rate | {hp.get('learning_rate', 'N/A')} |
| Batch Size | {hp.get('batch_size', 'N/A')} |
| Iterations | {hp.get('iterations', 'N/A')} |
| Max Sequence Length | {hp.get('max_seq_length', 'N/A')} |
| Optimizer | {hp.get('optimizer', 'adamw')} |

## Training & Evaluation Metrics
- **Final Training Loss**: {experiment_data.get('final_train_loss', 'N/A')}
- **Final Validation Loss**: {experiment_data.get('final_val_loss', 'N/A')}
- **Metrics Summary**: `{json.dumps(metrics, indent=2)}`

## Hardware Environment
- **Device**: {hw.get('cpu', 'N/A')}
- **Total Memory**: {hw.get('ram_total_gb', 'N/A')} GB
- **Platform**: Apple Silicon Unified Memory
"""
        with open(p, "w", encoding="utf-8") as f:
            f.write(md)

        return str(p.resolve())
