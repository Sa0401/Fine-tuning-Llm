"""Dataset loading, validation, cleaning, token statistics, and train/val/test splitting."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from utils.file_manager import get_project_root, load_config
from utils.validation import validate_conversational_item


class DatasetProcessor:
    def __init__(self):
        self.config = load_config()
        self.root = get_project_root()
        self.data_dir = self.root / self.config.get("storage", {}).get("data_dir", "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_from_file(self, file_content: bytes | str, filename: str) -> List[Dict[str, Any]]:
        """Parse raw file bytes or string into a list of dictionaries based on file extension."""
        ext = Path(filename).suffix.lower()
        records: List[Dict[str, Any]] = []

        if isinstance(file_content, bytes):
            text = file_content.decode("utf-8", errors="replace")
        else:
            text = file_content

        if ext in [".jsonl", ".txt"]:
            for line_idx, line in enumerate(text.splitlines()):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if isinstance(data, dict):
                        records.append(data)
                except Exception:
                    # If it's plain text line in .txt
                    if ext == ".txt":
                        records.append({"text": line_str})
        elif ext == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    records = [d for d in data if isinstance(d, dict)]
                elif isinstance(data, dict):
                    # Check if there is an inner list like {"data": [...]} or {"items": [...]}
                    for k in ["data", "items", "rows", "records", "samples"]:
                        if k in data and isinstance(data[k], list):
                            records = [d for d in data[k] if isinstance(d, dict)]
                            break
                    if not records:
                        records = [data]
            except Exception as e:
                raise ValueError(f"Failed to parse JSON file: {e}")
        elif ext == ".csv":
            try:
                df = pd.read_csv(io.StringIO(text))
                records = df.to_dict(orient="records")
            except Exception as e:
                raise ValueError(f"Failed to parse CSV file: {e}")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .jsonl, .json, .csv, .txt")

        return records

    def normalize_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert various schemas into standardized conversational messages format."""
        # 1. Already conversational
        if "messages" in item and isinstance(item["messages"], list):
            clean_messages = []
            for m in item["messages"]:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    clean_messages.append({
                        "role": str(m["role"]).strip().lower(),
                        "content": str(m["content"]).strip(),
                    })
            return {"messages": clean_messages}

        # 2. Instruction / Output or Prompt / Completion
        prompt_val = (
            item.get("instruction")
            or item.get("prompt")
            or item.get("input")
            or item.get("question")
            or item.get("human")
        )
        completion_val = (
            item.get("output")
            or item.get("completion")
            or item.get("response")
            or item.get("answer")
            or item.get("assistant")
        )

        if prompt_val is not None and completion_val is not None:
            # If there's an optional system message or input context
            system_val = item.get("system")
            messages = []
            if system_val and str(system_val).strip():
                messages.append({"role": "system", "content": str(system_val).strip()})
            messages.append({"role": "user", "content": str(prompt_val).strip()})
            messages.append({"role": "assistant", "content": str(completion_val).strip()})
            return {"messages": messages}

        # 3. Plain text format
        if "text" in item and str(item["text"]).strip():
            return {"text": str(item["text"]).strip()}

        return item

    def validate_and_clean(
        self,
        records: List[Dict[str, Any]],
        remove_duplicates: bool = True,
        filter_empty: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate dataset, detect duplicates, remove empty records, and calculate stats."""
        total_raw = len(records)
        cleaned_records = []
        seen_hashes = set()
        duplicate_count = 0
        empty_count = 0
        invalid_format_count = 0

        for r in records:
            norm = self.normalize_record(r)
            is_valid, reason = validate_conversational_item(norm)

            if not is_valid:
                if filter_empty and ("empty" in reason.lower() or "missing" in reason.lower()):
                    empty_count += 1
                    continue
                invalid_format_count += 1
                continue

            # Deduplication key
            if "messages" in norm:
                repr_str = "||".join(f"{m.get('role')}:{m.get('content')}" for m in norm["messages"])
            else:
                repr_str = str(norm.get("text", ""))

            if remove_duplicates:
                h = hash(repr_str)
                if h in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(h)

            cleaned_records.append(norm)

        stats = {
            "total_raw": total_raw,
            "cleaned_count": len(cleaned_records),
            "duplicates_removed": duplicate_count,
            "empty_removed": empty_count,
            "invalid_format_count": invalid_format_count,
        }
        return cleaned_records, stats

    def calculate_token_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute token length distribution across records (approx 1 token ~= 4 chars or whitespace split)."""
        if not records:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "lengths": []}

        lengths = []
        for r in records:
            if "messages" in r:
                full_text = " ".join(m.get("content", "") for m in r["messages"])
            else:
                full_text = str(r.get("text", ""))
            # Accurate token estimate: words * 1.3 or char_len / 3.8
            est_tokens = max(1, int(len(full_text) / 3.8))
            lengths.append(est_tokens)

        s = pd.Series(lengths)
        return {
            "min": int(s.min()),
            "max": int(s.max()),
            "mean": round(float(s.mean()), 1),
            "median": int(s.median()),
            "p95": int(s.quantile(0.95)),
            "lengths": lengths,
        }

    def split_dataset(
        self,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Split cleaned records into train, validation, and test partitions."""
        import random

        rec_copy = list(records)
        if shuffle:
            random.seed(seed)
            random.shuffle(rec_copy)

        n = len(rec_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data = rec_copy[:train_end]
        val_data = rec_copy[train_end:val_end]
        test_data = rec_copy[val_end:]

        # Edge case: if dataset is small, ensure at least 1 record in val if possible
        if len(train_data) > 2 and len(val_data) == 0:
            val_data = [train_data.pop()]

        return {
            "train": train_data,
            "valid": val_data,
            "test": test_data,
        }

    def save_split_dataset(
        self,
        splits: Dict[str, List[Dict[str, Any]]],
        dataset_name: str,
    ) -> Dict[str, str]:
        """Save train.jsonl, valid.jsonl, and test.jsonl in MLX-LM compatible directory."""
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
        target_dir = self.data_dir / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for split_key in ["train", "valid", "test"]:
            file_path = target_dir / f"{split_key}.jsonl"
            records = splits.get(split_key, [])
            if not records:
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                continue
            with open(file_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            paths[split_key] = str(file_path.resolve())

        # Also write summary metadata
        meta_path = target_dir / "dataset_info.json"
        meta = {
            "name": dataset_name,
            "train_samples": len(splits.get("train", [])),
            "valid_samples": len(splits.get("valid", [])),
            "test_samples": len(splits.get("test", [])),
            "total_samples": sum(len(s) for s in splits.values()),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        paths["dir"] = str(target_dir.resolve())
        paths["info"] = str(meta_path.resolve())
        return paths

    def list_saved_datasets(self) -> List[Dict[str, Any]]:
        """List all prepared datasets in data/."""
        datasets = []
        if not self.data_dir.exists():
            return []

        for p in self.data_dir.iterdir():
            if p.is_dir():
                info_file = p / "dataset_info.json"
                train_file = p / "train.jsonl"
                valid_file = p / "valid.jsonl"
                if train_file.exists():
                    info = {}
                    if info_file.exists():
                        try:
                            with open(info_file, "r", encoding="utf-8") as f:
                                info = json.load(f)
                        except Exception:
                            pass
                    datasets.append({
                        "name": info.get("name", p.name),
                        "path": str(p.resolve()),
                        "train_samples": info.get("train_samples", sum(1 for _ in open(train_file, "r", encoding="utf-8"))),
                        "valid_samples": info.get("valid_samples", sum(1 for _ in open(valid_file, "r", encoding="utf-8")) if valid_file.exists() else 0),
                        "total_samples": info.get("total_samples", 0),
                    })
        return datasets
