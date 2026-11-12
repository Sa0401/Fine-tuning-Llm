"""Model evaluation and comparative metrics engine.

Computes ROUGE, validation loss / perplexity estimates, latency, and memory metrics.
Compares base model against LoRA and QLoRA fine-tuned adapters on standardized prompts.
"""

import math
import time
from typing import Any, Dict, List, Optional
import psutil
from rouge_score import rouge_scorer
from utils.hardware import get_current_telemetry


class EvaluationEngine:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        r1_list = []
        r2_list = []
        rl_list = []

        for pred, ref in zip(predictions, references):
            p_clean = str(pred).strip()
            r_clean = str(ref).strip()
            if not p_clean or not r_clean:
                continue
            scores = self.scorer.score(r_clean, p_clean)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        if not r1_list:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        return {
            "rouge1": round(sum(r1_list) / len(r1_list) * 100, 2),
            "rouge2": round(sum(r2_list) / len(r2_list) * 100, 2),
            "rougeL": round(sum(rl_list) / len(rl_list) * 100, 2),
        }

    def compute_perplexity_from_loss(self, val_loss: Optional[float]) -> Optional[float]:
        """Calculate perplexity as exp(validation_loss)."""
        if val_loss is None or val_loss <= 0.0:
            return None
        try:
            ppl = math.exp(val_loss)
            return round(ppl, 2) if ppl < 100000 else 99999.0
        except OverflowError:
            return 99999.0

    def evaluate_generation_performance(
        self,
        inference_fn,
        test_prompts: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Run benchmark on test prompts, recording outputs, latencies, and memory telemetry."""
        results = []
        latencies = []
        tokens_per_sec_list = []
        mem_before = psutil.virtual_memory().used / (1024**2)

        for item in test_prompts:
            user_msg = item.get("prompt") or item.get("user") or item.get("instruction") or ""
            expected_ref = item.get("reference") or item.get("expected") or item.get("output") or ""

            messages = [{"role": "user", "content": user_msg}]
            t0 = time.time()
            gen_result = inference_fn(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - t0

            response_text = gen_result.get("response", "")
            latencies.append(elapsed)
            tokens_count = gen_result.get("tokens_count", max(1, len(response_text.split())))
            tps = gen_result.get("tokens_per_sec", round(tokens_count / elapsed, 1) if elapsed > 0 else 0.0)
            tokens_per_sec_list.append(tps)

            results.append({
                "prompt": user_msg,
                "reference": expected_ref,
                "response": response_text,
                "latency_sec": round(elapsed, 3),
                "tokens_count": tokens_count,
                "tokens_per_sec": tps,
            })

        mem_after = psutil.virtual_memory().used / (1024**2)
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        avg_tps = round(sum(tokens_per_sec_list) / len(tokens_per_sec_list), 1) if tokens_per_sec_list else 0.0

        # Compute ROUGE if references are provided
        has_refs = any(bool(r.get("reference")) for r in results)
        rouge_scores = {}
        if has_refs:
            preds = [r["response"] for r in results]
            refs = [r["reference"] for r in results]
            rouge_scores = self.compute_rouge_scores(preds, refs)

        return {
            "num_samples": len(results),
            "avg_latency_sec": avg_latency,
            "avg_tokens_per_sec": avg_tps,
            "memory_delta_mb": round(max(0, mem_after - mem_before), 1),
            "rouge_scores": rouge_scores,
            "samples": results,
        }

