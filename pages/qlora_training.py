"""Quantized LoRA (QLoRA) fine-tuning page."""

import streamlit as st
from core.dataset_processor import DatasetProcessor
from core.model_loader import ModelLoader
from core.training_manager import TrainingManager
from utils.hardware import detect_hardware, get_current_telemetry


def render():
    st.markdown("## 🧩 QLoRA (Quantized LoRA) Studio")
    st.caption("Fine-tune 4-bit quantized base models with high-precision LoRA adapters for maximum memory efficiency.")

    hw = detect_hardware()
    loader = ModelLoader()
    dset_proc = DatasetProcessor()
    tm = TrainingManager()
    telemetry = get_current_telemetry()

    # Architectural Breakdown Comparison Box
    st.markdown(
        """
        <div class="studio-box">
            <div class="studio-box-header">Architecture: Apple Silicon MLX 4-bit vs NVIDIA CUDA QLoRA</div>
            <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
                <p><strong>Apple Silicon MLX Quantized LoRA:</strong> In MLX, models are stored with native 4-bit grouped affine quantization (<code>QuantizedLinear</code>). During backpropagation, MLX computes activations and routes gradients directly through frozen 4-bit weights into FP16/FP32 LoRA matrices without requiring the Linux-only <code>bitsandbytes</code> CUDA library.</p>
                <p style="margin-bottom: 0;"><strong>NVIDIA CUDA QLoRA:</strong> Standard CUDA QLoRA uses NormalFloat4 (NF4) quantization with double quantization (quantizing the quantization constants) via <code>bitsandbytes</code>, dequantizing to 16-bit brain float (BF16) on-the-fly during the forward pass.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model & Dataset Selection
    st.markdown("### 1. Model & Dataset Selection")
    col_m, col_d = st.columns(2)

    with col_m:
        catalog = loader.get_catalog_models()
        # Highlight 4-bit quantized models
        q_models = [m for m in catalog if "4bit" in m["id"].lower() or m.get("type") == "qlora"]
        all_options = [m["id"] for m in q_models] + [m["id"] for m in catalog if m not in q_models]

        current_sel = st.session_state.get("selected_model", all_options[0])
        idx = all_options.index(current_sel) if current_sel in all_options else 0

        selected_model_id = st.selectbox(
            "4-bit Quantized Base Model",
            options=all_options,
            index=idx,
            help="Select a 4-bit quantized model for native Apple Silicon training (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit).",
        )
        st.session_state["selected_model"] = selected_model_id

    with col_d:
        saved_dsets = dset_proc.list_saved_datasets()
        if not saved_dsets:
            st.warning("No prepared datasets found. Please prepare a dataset in the Dataset Manager first.")
            dset_options = ["None"]
            sel_dset_idx = 0
        else:
            dset_options = [d["name"] for d in saved_dsets]
            curr_dset = st.session_state.get("selected_dataset_name")
            sel_dset_idx = dset_options.index(curr_dset) if curr_dset in dset_options else 0

        selected_dset_name = st.selectbox(
            "Training Dataset",
            options=dset_options,
            index=sel_dset_idx,
            help="Prepared dataset with train.jsonl and valid.jsonl partitions.",
        )

        selected_dset_info = next((d for d in saved_dsets if d["name"] == selected_dset_name), None)
        if selected_dset_info:
            st.session_state["selected_dataset_path"] = selected_dset_info["path"]
            st.session_state["selected_dataset_name"] = selected_dset_name
            st.session_state["selected_dataset_samples"] = selected_dset_info["train_samples"]

    # 2. Experiment Details
    st.markdown("### 2. Experiment Details")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        exp_name = st.text_input(
            "Experiment Name",
            value=f"qlora_{selected_dset_name}_run",
            help="Unique experiment identifier.",
        )
    with exp_col2:
        if hw.is_apple_silicon:
            backend_options = ["mlx"]
            backend_help = "Apple Silicon native 4-bit MLX Quantized LoRA."
        elif hw.cuda_available:
            backend_options = ["cuda"]
            backend_help = "NVIDIA CUDA BitsAndBytes 4-bit (NF4) QLoRA."
        else:
            backend_options = ["cpu"]
            backend_help = "4-bit quantization on Windows requires an NVIDIA GPU with CUDA. CPU fallback will run standard PEFT."

        backend_choice = st.selectbox(
            f"Compute Backend ({hw.os_name})",
            options=backend_options,
            index=0,
            help=backend_help,
        )

    # 3. QLoRA Hyperparameters
    st.markdown("### 3. QLoRA Hyperparameters & Quantization Settings")

    q_col1, q_col2, q_col3 = st.columns(3)

    with q_col1:
        rank = st.select_slider(
            "LoRA Rank (r)",
            options=[4, 8, 16, 32, 64],
            value=8,
            help="Inner rank dimension. For QLoRA, rank=8 or 16 captures excellent adaptation fidelity with almost negligible memory.",
        )
        alpha = st.number_input(
            "LoRA Alpha",
            min_value=1.0,
            max_value=128.0,
            value=16.0,
            step=1.0,
            help="Adapter scaling factor (scaling = alpha / r).",
        )
        dropout = st.slider(
            "LoRA Dropout",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Dropout rate to regularize adapter weights.",
        )

    with q_col2:
        lr = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=2e-4,
            format="%.2e",
            step=1e-5,
            help="Slightly higher learning rates (~2e-4) are optimal for 4-bit QLoRA.",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
            help="Batch size per step. With 4-bit quantization, batch_size=4 is fast and safe.",
        )
        grad_accum = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="Accumulate gradients across steps before weight update.",
        )

    with q_col3:
        iterations = st.number_input(
            "Training Iterations",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            help="Number of training iterations (steps).",
        )
        max_seq_len = st.number_input(
            "Max Sequence Length",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Maximum token sequence length.",
        )
        quant_bits = st.selectbox(
            "Quantization Precision",
            options=[4, 8],
            index=0,
            help="4-bit affine quantization yields the lowest memory footprint (~1.2 GB VRAM for 0.5B model).",
        )

    # Real-time memory telemetry monitor
    st.markdown(
        f"""
        <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 10px 14px; border-radius: 4px; font-size: 0.85rem; margin-top: 15px;">
            📊 <strong>Current System Memory:</strong> {telemetry['ram_used_gb']} GB used of {telemetry['ram_total_gb']} GB ({telemetry['ram_free_gb']} GB available).
            QLoRA on 4-bit weights will only allocate ~300-600MB of additional memory for optimizer states and gradients!
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Launch Button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Start QLoRA Training Job", type="primary", use_container_width=True):
        if not selected_dset_info:
            st.error("Please select a valid dataset before starting training.")
            return

        hyperparameters = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
            "grad_accumulation_steps": int(grad_accum),
            "iterations": int(iterations),
            "max_seq_length": int(max_seq_len),
            "quantization_bits": int(quant_bits),
            "optimizer": "adamw",
            "steps_per_report": 5,
            "steps_per_eval": 25,
            "save_every": 50,
        }

        with st.spinner("Starting QLoRA background training process..."):
            success, msg, exp_id = tm.start_training(
                name=exp_name,
                model_id=selected_model_id,
                backend=backend_choice,
                method="qlora",
                dataset_path=selected_dset_info["path"],
                dataset_name=selected_dset_name,
                dataset_samples=selected_dset_info["train_samples"],
                hyperparameters=hyperparameters,
            )

        if success:
            st.success(f"🎉 {msg}")
            st.session_state["active_training_exp_id"] = exp_id
            st.info("QLoRA job started! Switch to **Training Monitor** in the sidebar to observe the real-time loss curve.")
        else:
            st.error(f"Failed to start QLoRA training: {msg}")


if __name__ == "__main__":
    render()

