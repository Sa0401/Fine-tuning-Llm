"""Standard LoRA fine-tuning page."""

import streamlit as st
from core.dataset_processor import DatasetProcessor
from core.model_loader import ModelLoader
from core.training_manager import TrainingManager
from utils.hardware import detect_hardware


def render():
    st.markdown("## ⚡ LoRA Fine-Tuning Studio")
    st.caption("Train parameter-efficient Low-Rank Adaptation (LoRA) matrices on full-precision base model weights.")

    hw = detect_hardware()
    loader = ModelLoader()
    dset_proc = DatasetProcessor()
    tm = TrainingManager()

    # Hardware Info Callout
    st.info(
        f"🖥️ **Training Backend:** {hw.backend_status_message}"
    )

    # 1. Selection: Model & Dataset
    st.markdown("### 1. Model & Dataset Configuration")
    col_m, col_d = st.columns(2)

    with col_m:
        catalog = loader.get_catalog_models()
        # Filter for models suitable for full-precision LoRA or let user pick any
        model_options = [m["id"] for m in catalog]
        current_sel_model = st.session_state.get("selected_model", model_options[2] if len(model_options) > 2 else model_options[0])
        idx = model_options.index(current_sel_model) if current_sel_model in model_options else 0

        selected_model_id = st.selectbox(
            "Base Pretrained Model",
            options=model_options,
            index=idx,
            help="Select base model. For standard LoRA, unquantized or full-precision weights are adapted.",
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
            value=f"lora_{selected_dset_name}_run",
            help="Unique human-readable experiment identifier.",
        )
    with exp_col2:
        if hw.is_apple_silicon:
            backend_options = ["mlx"]
            backend_help = "Native Apple Silicon MLX & MLX-LM backend."
        elif hw.cuda_available:
            backend_options = ["cuda", "cpu"]
            backend_help = "NVIDIA CUDA (Hugging Face PEFT) or CPU fallback."
        else:
            backend_options = ["cpu"]
            backend_help = "Hugging Face PEFT running in CPU mode."

        backend_choice = st.selectbox(
            f"Compute Backend ({hw.os_name})",
            options=backend_options,
            index=0,
            help=backend_help,
        )

    # 3. LoRA Hyperparameter Tuning with Tooltips
    st.markdown("### 3. Hyperparameter Configuration")
    st.caption("Adjust adapter architecture and optimization parameters. Hover over ❓ icons for technical explanations.")

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        rank = st.select_slider(
            "LoRA Rank (r)",
            options=[2, 4, 8, 16, 32, 64, 128],
            value=8,
            help="Low-rank dimension inner size. Lower rank (r=4 or 8) trains fewer parameters with minimal memory; higher rank (r=32 or 64) captures more complex behaviors.",
        )
        alpha = st.number_input(
            "LoRA Alpha (Scaling)",
            min_value=1.0,
            max_value=256.0,
            value=16.0,
            step=1.0,
            help="Scaling factor for adapter weights (scaling factor = alpha / r). Standard rule of thumb is setting alpha = 2 * r.",
        )
        dropout = st.slider(
            "LoRA Dropout",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Dropout applied to LoRA input features to prevent overfitting on small datasets.",
        )

    with h_col2:
        lr = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-4,
            format="%.2e",
            step=1e-5,
            help="Optimization learning rate. For LoRA on LLMs, 1e-4 to 2e-4 is typically recommended.",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=4,
            step=1,
            help="Number of sequences processed in each training step. On 16GB unified memory, 2 to 4 prevents memory spikes.",
        )
        grad_accum = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="Accumulates gradients over multiple mini-batches before updating weights, simulating larger batch sizes without extra memory.",
        )

    with h_col3:
        iterations = st.number_input(
            "Training Iterations (Steps)",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            help="Total number of optimizer steps. For quick adaptation on small datasets, 50 to 100 steps completes in minutes.",
        )
        max_seq_len = st.number_input(
            "Max Sequence Length",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Maximum token length for input + completion. Shorter lengths (e.g. 512) conserve unified memory.",
        )
        optimizer = st.selectbox(
            "Optimizer",
            options=["adamw", "adam", "sgd"],
            index=0,
            help="Optimization algorithm. AdamW with decoupled weight decay is standard for Transformers.",
        )

    # Memory Safety Recommendation
    st.markdown(
        """
        <div style="background: rgba(59, 130, 246, 0.08); border-left: 3px solid #3b82f6; padding: 10px 14px; border-radius: 4px; font-size: 0.85rem; margin-top: 15px;">
            💡 <strong>16GB Unified Memory Tip:</strong> With rank=8, batch_size=4, and max_seq_length=512, memory consumption will typically stay below 3-4 GB, leaving plenty of headroom for system tasks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Launch Button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Start LoRA Training Job", type="primary", use_container_width=True):
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
            "optimizer": optimizer,
            "steps_per_report": 5,
            "steps_per_eval": 25,
            "save_every": 50,
        }

        with st.spinner("Initializing training environment and starting background process..."):
            success, msg, exp_id = tm.start_training(
                name=exp_name,
                model_id=selected_model_id,
                backend=backend_choice,
                method="lora",
                dataset_path=selected_dset_info["path"],
                dataset_name=selected_dset_name,
                dataset_samples=selected_dset_info["train_samples"],
                hyperparameters=hyperparameters,
            )

        if success:
            st.success(f"🎉 {msg}")
            st.session_state["active_training_exp_id"] = exp_id
            st.info("Training is running asynchronously in the background. Switch to **Training Monitor** in the sidebar to view live logs and loss curves.")
        else:
            st.error(f"Failed to start training: {msg}")


if __name__ == "__main__":
    render()

