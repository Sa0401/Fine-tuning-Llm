"""Hugging Face PEFT + BitsAndBytes QLoRA training backend for NVIDIA CUDA environments."""

from pathlib import Path
from typing import Any, Dict, List
import yaml
from utils.file_manager import get_project_root


class CUDAQLoRATrainer:
    """Generates and manages Hugging Face PEFT + bitsandbytes QLoRA fine-tuning for CUDA systems."""

    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = self.output_dir / "run_cuda_qlora.py"

    def prepare_script(
        self,
        model_id: str,
        data_dir: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """Generate a standalone Python script utilizing bitsandbytes 4-bit NF4 quantization."""
        rank = int(hyperparameters.get("rank", 8))
        alpha = float(hyperparameters.get("alpha", 16.0))
        dropout = float(hyperparameters.get("dropout", 0.05))
        lr = float(hyperparameters.get("learning_rate", 2e-4))
        batch_size = int(hyperparameters.get("batch_size", 4))
        max_seq = int(hyperparameters.get("max_seq_length", 512))
        iters = int(hyperparameters.get("iterations", 100))
        double_quant = bool(hyperparameters.get("double_quant", True))
        quant_type = str(hyperparameters.get("quant_type", "nf4"))

        clean_out = Path(self.output_dir).resolve().as_posix()
        clean_data = Path(data_dir).resolve().as_posix()

        script_content = f'''"""Hugging Face PEFT + BitsAndBytes QLoRA Training Script (Windows & Linux CUDA)."""
import os
import sys
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

model_id = "{model_id}"
data_dir = r"{clean_data}"
output_dir = r"{clean_out}"

def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. BitsAndBytes 4-bit QLoRA strictly requires an NVIDIA GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] Initializing Windows/Linux CUDA QLoRA with BitsAndBytes on device: {{gpu_name}} ({{vram:.2f}} GB VRAM)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Configuration
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="{quant_type}",
        bnb_4bit_use_double_quant={double_quant},
        bnb_4bit_compute_dtype=compute_dtype
    )

    print("[INFO] Loading base model with 4-bit NormalFloat4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("[INFO] Starting QLoRA training on GPU...")
    trainer.train()
    print("[INFO] Saving adapter to:", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] QLoRA training finished successfully.")

if __name__ == "__main__":
    main()
'''
        with open(self.script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        return str(self.script_path.resolve())

    def build_command(self, python_bin: str) -> List[str]:
        return [python_bin, str(self.script_path.resolve())]

