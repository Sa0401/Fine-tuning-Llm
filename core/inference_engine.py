"""Inference engine supporting base models and fine-tuned LoRA/QLoRA adapters."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from utils.file_manager import get_project_root, load_config
from utils.hardware import detect_hardware


class InferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.root = get_project_root()
        self.hardware = detect_hardware()
        self.current_model_id: Optional[str] = None
        self.current_adapter_path: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.backend_type: Optional[str] = None
        self._initialized = True

    def load_model(
        self,
        model_id_or_path: str,
        adapter_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load base model and optionally apply LoRA adapter weights with caching."""
        # Check if already loaded identical model and adapter
        clean_adapter = str(Path(adapter_path).resolve()) if adapter_path else None
        if (
            self.model is not None
            and self.current_model_id == model_id_or_path
            and self.current_adapter_path == clean_adapter
        ):
            return True, "Model and adapter already loaded in memory."

        self.unload_model()

        # Decide backend based on hardware and model
        try:
            if self.hardware.is_apple_silicon:
                # Use MLX-LM
                import mlx_lm

                adapter_arg = clean_adapter if clean_adapter and os.path.isdir(clean_adapter) else None
                model, tokenizer = mlx_lm.load(
                    model_id_or_path,
                    adapter_path=adapter_arg,
                )
                self.model = model
                self.tokenizer = tokenizer
                self.current_model_id = model_id_or_path
                self.current_adapter_path = clean_adapter
                self.backend_type = "mlx"
                adapter_str = f" with adapter '{Path(clean_adapter).name}'" if clean_adapter else " (base model)"
                return True, f"Successfully loaded MLX model '{model_id_or_path}'{adapter_str} into unified memory."
            else:
                # Use Hugging Face / Transformers
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel

                device_map = "auto" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )

                if clean_adapter and os.path.isdir(clean_adapter):
                    model = PeftModel.from_pretrained(base_model, clean_adapter)
                else:
                    model = base_model

                self.model = model
                self.tokenizer = tokenizer
                self.current_model_id = model_id_or_path
                self.current_adapter_path = clean_adapter
                self.backend_type = "transformers"
                adapter_str = f" with adapter '{Path(clean_adapter).name}'" if clean_adapter else ""
                return True, f"Successfully loaded Transformers model '{model_id_or_path}'{adapter_str}."

        except Exception as e:
            self.unload_model()
            return False, f"Failed to load model: {str(e)}"

    def unload_model(self) -> None:
        """Free loaded model weights and clear cache."""
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        self.current_adapter_path = None
        self.backend_type = None

        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Generate response given a conversation history list of messages."""
        if self.model is None or self.tokenizer is None:
            return {
                "success": False,
                "response": "No model loaded. Please load a base model or adapter first.",
                "latency_sec": 0.0,
                "tokens_count": 0,
            }

        start_time = time.time()

        try:
            # Format prompt using chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback manual conversation formatting
                parts = []
                for m in messages:
                    parts.append(f"{m['role'].capitalize()}: {m['content']}")
                parts.append("Assistant:")
                prompt = "\n".join(parts)

            if self.backend_type == "mlx":
                import mlx_lm
                from mlx_lm.sample_utils import make_sampler

                # Create sampler with temperature and top_p
                sampler = make_sampler(temp=temperature, top_p=top_p)

                # MLX generate
                raw_response = mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                output_text = raw_response.strip()

            else:
                import torch

                inputs = self.tokenizer(prompt, return_tensors="pt")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 0.01),
                        top_p=top_p,
                        do_sample=temperature > 0.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            duration = round(time.time() - start_time, 3)
            token_count = max(1, len(output_text.split()))

            return {
                "success": True,
                "response": output_text,
                "latency_sec": duration,
                "tokens_count": token_count,
                "tokens_per_sec": round(token_count / duration, 1) if duration > 0 else 0.0,
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"Generation failed: {str(e)}",
                "latency_sec": round(time.time() - start_time, 3),
                "tokens_count": 0,
            }
