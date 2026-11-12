"""Model evaluation, ROUGE benchmarking, and quantitative metrics page."""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from core.evaluation_engine import EvaluationEngine
from core.experiment_tracker import ExperimentTracker
from core.inference_engine import InferenceEngine
from core.model_loader import ModelLoader
from utils.file_manager import list_adapters
from utils.hardware import detect_hardware, get_current_telemetry


def render():
    st.markdown("## 📊 Model Evaluation & Benchmarking")
    st.caption("Evaluate loss metrics, ROUGE scores, perplexity, inference latency, and memory footprints.")

    # Core cautionary warning required by specifications
    st.warning(
        "⚠️ **Evaluation Best Practice:** Never claim an adapter is better based on training loss alone! "
        "A low training loss often indicates memorization or overfitting. Always evaluate against held-out validation data, ROUGE metrics, and qualitative human relevance."
    )

    eval_engine = EvaluationEngine()
    infer_engine = InferenceEngine()
    loader = ModelLoader()
    tracker = ExperimentTracker()
    adapters = list_adapters()
    catalog_models = loader.get_catalog_models()

    # 1. Model Configuration
    st.markdown("### 1. Select Model & Adapter to Benchmark")
    col1, col2 = st.columns(2)

    with col1:
        base_options = [m["id"] for m in catalog_models]
        curr_m = st.session_state.get("selected_model", base_options[0])
        idx = base_options.index(curr_m) if curr_m in base_options else 0
        eval_base_model = st.selectbox("Base Pretrained Model", base_options, index=idx)

    with col2:
        adapter_options = ["None (Base Model Only)"] + [a["name"] for a in adapters]
        eval_adapter = st.selectbox("Attach Trained Adapter", adapter_options, index=0)

    chosen_adapter_path = None
    if eval_adapter != "None (Base Model Only)":
        match_a = next((a for a in adapters if a["name"] == eval_adapter), None)
        if match_a:
            chosen_adapter_path = match_a["path"]

    # 2. Test Suite Prompts
    st.markdown("### 2. Evaluation Test Suite")
    st.write("Specify test questions and expected reference outputs for ROUGE score calculation:")

    default_test_suite = [
        {
            "prompt": "What is Parameter-Efficient Fine-Tuning (PEFT)?",
            "reference": "PEFT fine-tunes large language models by updating a small subset of parameters (like LoRA adapters), preserving base weights and saving memory.",
        },
        {
            "prompt": "Explain LoRA in simple terms.",
            "reference": "LoRA freezes the pretrained model and adds low-rank decomposition matrices into transformer layers to drastically reduce trainable parameters.",
        },
        {
            "prompt": "What is the key benefit of 4-bit QLoRA on Apple Silicon?",
            "reference": "QLoRA loads weights in 4-bit precision to fit within unified memory while backpropagating gradients into high-precision adapter matrices.",
        },
    ]

    test_suite_text = st.text_area(
        "Test Prompts JSON",
        value=json.dumps(default_test_suite, indent=2),
        height=180,
        help="JSON array of test items with 'prompt' and 'reference' keys.",
    )

    try:
        test_prompts = json.loads(test_suite_text)
    except Exception:
        test_prompts = default_test_suite
        st.error("Invalid JSON format in test prompts. Reverting to default test suite.")

    # 3. Execution Parameters
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        temp = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1)
    with col_p2:
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
    with col_p3:
        max_tokens = st.slider("Max New Tokens", 32, 512, 128, 32)

    # 4. Run Benchmark
    if st.button("🚀 Run Full Evaluation Benchmark", type="primary", use_container_width=True):
        with st.status("Running evaluation benchmark...", expanded=True) as status:
            st.write(f"Loading '{eval_base_model}' with adapter: `{eval_adapter}`...")
            load_success, load_msg = infer_engine.load_model(eval_base_model, adapter_path=chosen_adapter_path)

            if not load_success:
                status.update(label="Model Loading Failed", state="error")
                st.error(load_msg)
                return

            st.write(f"Generating responses across {len(test_prompts)} test prompts...")
            benchmark_results = eval_engine.evaluate_generation_performance(
                inference_fn=infer_engine.generate_response,
                test_prompts=test_prompts,
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            status.update(label="Benchmark Completed Successfully!", state="complete")

        st.session_state["last_eval_results"] = benchmark_results
        st.session_state["last_eval_target"] = f"{eval_base_model} + {eval_adapter}"

    # Display Results if available
    eval_res = st.session_state.get("last_eval_results")
    if eval_res:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown(f"### 📈 Evaluation Summary: `{st.session_state.get('last_eval_target')}`")

        # Top metric cards
        r_scores = eval_res.get("rouge_scores", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Avg Latency", f"{eval_res['avg_latency_sec']}s")
        m2.metric("Tokens / Sec", f"{eval_res['avg_tokens_per_sec']}")
        m3.metric("ROUGE-1 F1", f"{r_scores.get('rouge1', 0.0)}%")
        m4.metric("ROUGE-2 F1", f"{r_scores.get('rouge2', 0.0)}%")
        m5.metric("ROUGE-L F1", f"{r_scores.get('rougeL', 0.0)}%")

        # Detailed Sample Outputs Table
        st.markdown("#### 🔬 Generated Responses vs References")
        for idx, sample in enumerate(eval_res.get("samples", [])):
            with st.expander(f"Prompt #{idx + 1}: {sample['prompt'][:80]}...", expanded=True):
                st.markdown(f"**Prompt:** {sample['prompt']}")
                if sample.get("reference"):
                    st.markdown(f"**Target Reference:** *{sample['reference']}*")
                st.markdown(f"**Model Output:**")
                st.info(sample["response"])
                st.caption(f"Latency: {sample['latency_sec']}s • Output Tokens: {sample['tokens_count']} • Generation Speed: {sample['tokens_per_sec']} tokens/s")

        # Download Report Button
        st.download_button(
            "📥 Download Evaluation Results (JSON)",
            data=json.dumps(eval_res, indent=2),
            file_name="evaluation_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    render()

