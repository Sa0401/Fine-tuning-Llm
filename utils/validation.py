"""Validation utilities for datasets, hyperparameters, and model configurations."""

from typing import Any, Dict, List, Optional, Tuple


def validate_hyperparameters(hp: Dict[str, Any], method: str = "lora") -> Tuple[bool, List[str]]:
    """Validate hyperparameter values and constraints."""
    errors = []

    # Rank
    rank = hp.get("rank")
    if rank is None or not isinstance(rank, int) or rank <= 0 or rank > 256:
        errors.append("LoRA rank 'r' must be a positive integer between 1 and 256 (typical: 4, 8, 16, 32).")

    # Alpha
    alpha = hp.get("alpha")
    if alpha is None or not (isinstance(alpha, (int, float)) and alpha > 0):
        errors.append("LoRA alpha must be a positive number (typical: 16 or 2 * rank).")

    # Dropout
    dropout = hp.get("dropout", 0.0)
    if not (isinstance(dropout, (int, float)) and 0.0 <= dropout < 1.0):
        errors.append("LoRA dropout must be between 0.0 and 0.9.")

    # Learning rate
    lr = hp.get("learning_rate")
    if lr is None or not (isinstance(lr, (int, float)) and 1e-7 <= lr <= 1e-1):
        errors.append("Learning rate must be between 1e-7 and 0.1 (recommended: 1e-4 to 3e-4).")

    # Batch size
    batch_size = hp.get("batch_size")
    if batch_size is None or not isinstance(batch_size, int) or batch_size < 1 or batch_size > 128:
        errors.append("Batch size must be a positive integer between 1 and 128 (on 16GB Mac, 2 to 4 is ideal).")

    # Iterations / Epochs
    iterations = hp.get("iterations")
    if iterations is None or not isinstance(iterations, int) or iterations < 1:
        errors.append("Training iterations must be an integer >= 1.")

    # Max sequence length
    max_seq = hp.get("max_seq_length", 512)
    if not isinstance(max_seq, int) or max_seq < 64 or max_seq > 8192:
        errors.append("Max sequence length must be between 64 and 8192 tokens (512 is safe for 16GB memory).")

    # QLoRA specific checks
    if method == "qlora":
        q_bits = hp.get("quantization_bits", 4)
        if q_bits not in [4, 8]:
            errors.append("Quantization bits must be 4 or 8.")

    return len(errors) == 0, errors


def validate_conversational_item(item: Any) -> Tuple[bool, str]:
    """Validate if an item matches the standard conversational messages schema."""
    if not isinstance(item, dict):
        return False, "Record is not a JSON object/dictionary."

    # Format 1: messages list
    if "messages" in item:
        messages = item["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False, "'messages' must be a list containing at least user and assistant turns."
        has_user = False
        has_assistant = False
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return False, "Each message must have 'role' and 'content' keys."
            if not str(msg.get("content", "")).strip():
                return False, "Message content cannot be empty."
            if msg["role"] == "user":
                has_user = True
            elif msg["role"] == "assistant":
                has_assistant = True

        if not (has_user and has_assistant):
            return False, "Conversation must contain at least one 'user' and one 'assistant' message."
        return True, "Valid messages format"

    # Format 2: prompt / completion or instruction / output
    has_prompt = any(k in item for k in ["prompt", "instruction", "input", "question"])
    has_completion = any(k in item for k in ["completion", "output", "response", "answer"])

    if has_prompt and has_completion:
        p_val = str(item.get("prompt") or item.get("instruction") or item.get("input") or item.get("question") or "").strip()
        c_val = str(item.get("completion") or item.get("output") or item.get("response") or item.get("answer") or "").strip()
        if not p_val or not c_val:
            return False, "Prompt or completion text is empty."
        return True, "Valid prompt/completion format"

    # Format 3: raw text
    if "text" in item and str(item["text"]).strip():
        return True, "Valid raw text format"

    return False, "Unrecognized format. Must contain 'messages', 'prompt'/'completion', or 'text'."

