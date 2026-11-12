"""Hugging Face PEFT LoRA training backend for NVIDIA CUDA environments."""

from pathlib import Path
from typing import Any, Dict, List
import yaml
from utils.file_manager import get_project_root


class CUDALoRATrainer:
    """Generates and manages Hugging Face PEFT LoRA fine-tuning for CUDA systems."""

    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = self.output_dir / "run_cuda_lora.py"

    def prepare_script(
        self,
        model_id: str,
        data_dir: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """Generate a standalone Python script using PyTorch, Transformers, and PEFT."""
        rank = int(hyperparameters.get("rank", 8))
        alpha = float(hyperparameters.get("alpha", 16.0))
        dropout = float(hyperparameters.get("dropout", 0.05))
        lr = float(hyperparameters.get("learning_rate", 1e-4))
        batch_size = int(hyperparameters.get("batch_size", 4))
        max_seq = int(hyperparameters.get("max_seq_length", 512))
        iters = int(hyperparameters.get("iterations", 100))

        clean_out = Path(self.output_dir).resolve().as_posix()
        clean_data = Path(data_dir).resolve().as_posix()

        script_content = f'''"""Hugging Face PEFT LoRA Training Script (Cross-Platform Windows & Linux)."""
import os
import sys
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_id = "{model_id}"
data_dir = r"{clean_data}"
output_dir = r"{clean_out}"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Hugging Face PEFT LoRA on device: {{device}}")
    if device == "cuda":
        print(f"[INFO] GPU: {{torch.cuda.get_device_name(0)}} ({{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB VRAM)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("[INFO] Loading base model weights...")
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r={rank},
        lora_alpha={alpha},
        lora_dropout={dropout},
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    data_files = {{
        "train": os.path.join(data_dir, "train.jsonl")
    }}
    val_path = os.path.join(data_dir, "valid.jsonl")
    if os.path.exists(val_path):
        data_files["validation"] = val_path

    dataset = load_dataset("json", data_files=data_files)

    def format_prompt(sample):
        if "messages" in sample:
            text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        else:
            text = sample.get("text", "")
        tokenized = tokenizer(text, max_length={max_seq}, truncation=True, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size={batch_size},
        learning_rate={lr},
        max_steps={iters},
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Saving adapter to:", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Training finished successfully.")

if __name__ == "__main__":
    main()
'''
        with open(self.script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        return str(self.script_path.resolve())

    def build_command(self, python_bin: str) -> List[str]:
        return [python_bin, str(self.script_path.resolve())]

