"""Settings, hardware diagnostics, adapter management, and export/merge utilities."""

from pathlib import Path
import streamlit as st
from core.model_exporter import ModelExporter
from core.model_loader import ModelLoader
from utils.file_manager import format_size, get_dir_size, get_project_root, list_adapters, safe_delete_adapter
from utils.hardware import detect_hardware, get_current_telemetry


def render():
    st.markdown("## ⚙️ Studio Settings & Model Deployment")
    st.caption("Inspect system hardware, manage storage, fuse/merge LoRA adapters, and configure the API server.")

    hw = detect_hardware()
    telemetry = get_current_telemetry()
    exporter = ModelExporter()
    loader = ModelLoader()
    root = get_project_root()
    adapters = list_adapters()

    tab_hw, tab_merge, tab_adapters, tab_api = st.tabs([
        "💻 Hardware Diagnostics",
        "🔀 Fuse & Merge Adapters",
        "📦 Adapter Storage",
        "🚀 FastAPI Deployment",
    ])

    with tab_hw:
        st.markdown("### System & Hardware Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="studio-box">
                    <div class="studio-box-header">Processor & Operating System</div>
                    <div><strong>CPU Brand:</strong> {hw.cpu_brand}</div>
                    <div><strong>Architecture:</strong> {hw.architecture}</div>
                    <div><strong>Operating System:</strong> {hw.os_name} {hw.os_version}</div>
                    <div><strong>Platform Type:</strong> {'Apple Silicon macOS' if hw.is_apple_silicon else ('Microsoft Windows' if hw.is_windows else 'Linux / Unix')}</div>
                    <div><strong>Memory Architecture:</strong> {'Unified Memory (Zero-Copy UMA)' if hw.is_unified_memory else 'Dedicated System RAM'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            cuda_status = f"Active ({hw.cuda_device_name} - {hw.cuda_vram_gb} GB VRAM)" if hw.cuda_available else "Not active / No CUDA GPU"
            st.markdown(
                f"""
                <div class="studio-box">
                    <div class="studio-box-header">Compute Accelerators</div>
                    <div><strong>Apple MLX:</strong> {'Active (macOS ARM64)' if hw.mlx_available else 'Not applicable on Windows / Non-Apple'}</div>
                    <div><strong>PyTorch MPS:</strong> {'Active (macOS Metal)' if hw.mps_available else 'Disabled / Unavailable'}</div>
                    <div><strong>NVIDIA CUDA:</strong> {cuda_status}</div>
                    <div><strong>DirectML:</strong> {'Available' if hw.directml_available else 'Not detected'}</div>
                    <div><strong>bitsandbytes:</strong> {'Installed (4-bit NF4 enabled)' if hw.bitsandbytes_available else 'Not installed (MLX used on Mac)'}</div>
                    <div><strong>Active Backend:</strong> <code>{hw.recommended_backend.upper()}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Real-Time Memory Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total RAM", f"{telemetry['ram_total_gb']} GB")
        m2.metric("Used RAM", f"{telemetry['ram_used_gb']} GB")
        m3.metric("Free RAM", f"{telemetry['ram_free_gb']} GB")
        m4.metric("Memory Pressure", f"{telemetry['ram_percent']}%")

    with tab_merge:
        st.markdown("### Fuse LoRA Adapter into Base Model")
        st.write(
            "Merging mathematically folds the low-rank delta matrices \\(W = W_0 + \\frac{\\alpha}{r} (B \\times A)\\) "
            "directly into the base model weights, eliminating adapter loading overhead for zero-latency standalone deployment."
        )

        if not adapters:
            st.info("No trained adapters available to merge. Complete a training run first.")
        else:
            merge_col1, merge_col2 = st.columns(2)
            with merge_col1:
                cat_models = loader.get_catalog_models()
                base_to_fuse = st.selectbox("Base Model to Fuse Into", [m["id"] for m in cat_models], index=0)
            with merge_col2:
                adapter_to_fuse = st.selectbox("Adapter to Merge", [a["name"] for a in adapters], index=0)

            out_dir_name = st.text_input("Merged Model Output Folder", value=f"merged_{adapter_to_fuse}")
            dequantize_flag = st.checkbox("Dequantize base weights to FP16 (if base model is 4-bit)", value=False)

            if st.button("🔀 Execute Model Fuse & Merge", type="primary"):
                selected_ad = next((a for a in adapters if a["name"] == adapter_to_fuse), None)
                if not selected_ad:
                    st.error("Selected adapter not found.")
                else:
                    target_out = str((root / "models" / "base" / out_dir_name).resolve())
                    with st.spinner("Fusing adapter into base model weights..."):
                        success, merge_msg = exporter.merge_adapter(
                            base_model_id=base_to_fuse,
                            adapter_path=selected_ad["path"],
                            output_dir=target_out,
                            dequantize=dequantize_flag,
                        )
                    if success:
                        st.success(f"🎉 {merge_msg}")
                    else:
                        st.error(f"Merge failed: {merge_msg}")

    with tab_adapters:
        st.markdown("### Saved Adapter Storage")
        if not adapters:
            st.info("No trained adapters found in models/adapters/.")
        else:
            for ad in adapters:
                col_i, col_d = st.columns([3, 1])
                with col_i:
                    st.markdown(f"**{ad['name']}**")
                    st.caption(f"Size: {ad['size']} • Type: {ad['type']} • Path: `{ad['path']}`")
                with col_d:
                    if st.button("🗑️ Delete Adapter", key=f"del_ad_{ad['name']}", type="secondary"):
                        ok = safe_delete_adapter(ad["name"])
                        if ok:
                            st.success(f"Safely deleted adapter '{ad['name']}'.")
                            st.rerun()
                        else:
                            st.error("Could not delete adapter directory.")

    with tab_api:
        st.markdown("### REST API Serving (FastAPI & Uvicorn)")
        st.write("Serve fine-tuned models with an OpenAI-compatible REST API:")

        st.code(
            """# Start the API server in your terminal:
./venv/bin/python serve_api.py --port 8000

# Send a chat completion request:
curl http://127.0.0.1:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "adapter": "models/adapters/qlora_run1",
    "messages": [{"role": "user", "content": "What is LoRA?"}]
  }'
""",
            language="bash",
        )


if __name__ == "__main__":
    render()

