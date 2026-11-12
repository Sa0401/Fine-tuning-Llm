"""Model Selection & Inspection Page."""

import streamlit as st
from core.model_loader import ModelLoader
from utils.hardware import detect_hardware


def render():
    st.markdown("## 🧠 Model Selection & Management")
    st.caption("Inspect, download, and configure pretrained models for LoRA and QLoRA fine-tuning.")

    loader = ModelLoader()
    hw = detect_hardware()
    catalog_models = loader.get_catalog_models()
    local_models = loader.list_local_models()

    # Hardware Compatibility Alert
    st.info(f"💡 **Active Device:** {hw.cpu_brand} ({hw.total_memory_gb} GB RAM). Recommended backend: **{hw.recommended_backend.upper()}**.")

    tab_catalog, tab_custom, tab_local = st.tabs([
        "📚 Supported Model Catalog",
        "🌐 Custom Hugging Face Model",
        "💾 Locally Downloaded Models",
    ])

    with tab_catalog:
        st.markdown("### Pre-Configured Compatible Models")
        st.write("These models have been verified for memory footprint, architecture compatibility, and training stability.")

        for m in catalog_models:
            is_mlx = m.get("backend") == "mlx"
            badge_color = "badge-blue" if is_mlx else "badge-emerald"
            type_badge = "badge-purple" if m.get("type") == "qlora" else "badge-amber"

            with st.container():
                st.markdown(
                    f"""
                    <div class="studio-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <h3 style="margin: 0; font-size: 1.2rem;">{m['name']}</h3>
                                <div style="margin-top: 6px;">
                                    <span class="badge {badge_color}">{m['backend'].upper()}</span>
                                    <span class="badge {type_badge}">{m.get('type', 'lora').upper()}</span>
                                    <span class="badge badge-blue">{m.get('quantization', 'FP16')}</span>
                                    <span class="badge badge-emerald">Params: {m.get('params', 'N/A')}</span>
                                </div>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">{m['description']}</p>
                                <code style="font-size: 0.8rem; color: #60a5fa;">{m['id']}</code>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_col1, btn_col2 = st.columns([1, 4])
                with btn_col1:
                    is_selected = st.session_state.get("selected_model") == m["id"]
                    if st.button(
                        "✅ Selected" if is_selected else "Select Model",
                        key=f"sel_cat_{m['id']}",
                        type="primary" if is_selected else "secondary",
                    ):
                        st.session_state["selected_model"] = m["id"]
                        st.session_state["selected_model_meta"] = m
                        st.success(f"Selected '{m['name']}' as active model for training and inference.")
                        st.rerun()

                with btn_col2:
                    if st.button("⬇️ Download / Cache Weights", key=f"dl_cat_{m['id']}"):
                        with st.spinner(f"Downloading snapshot for '{m['id']}' from Hugging Face Hub..."):
                            res = loader.download_model(m["id"])
                            if res["success"]:
                                st.success(f"Successfully downloaded {m['id']} ({res['size']}) to local storage!")
                            else:
                                st.error(f"Download failed: {res['error']}")

    with tab_custom:
        st.markdown("### Load Custom Hugging Face Model")
        st.caption("Enter any Hugging Face model repository ID (e.g. mlx-community/Qwen2.5-1.5B-Instruct-4bit or HuggingFaceH4/zephyr-7b-beta).")

        custom_id = st.text_input(
            "Hugging Face Model ID",
            placeholder="e.g., mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            help="Hugging Face repository ID. For Apple Silicon MLX, models from mlx-community are recommended.",
        )

        if custom_id:
            compat = loader.check_compatibility(custom_id)
            meta = loader.inspect_model_metadata(custom_id)

            st.markdown(
                f"""
                <div class="studio-box">
                    <div class="studio-box-header">Model Compatibility Report</div>
                    <div><strong>Detected Quantization:</strong> {meta['quantization']}</div>
                    <div><strong>Estimated Parameters:</strong> {meta['params']}</div>
                    <div><strong>Recommended Backend:</strong> {compat['recommended']}</div>
                    <div><strong>Supported Backends:</strong> {', '.join(compat['supported_backends'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if compat["warnings"]:
                for w in compat["warnings"]:
                    st.warning(f"⚠️ {w}")

            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("Select This Model", type="primary", key="btn_sel_custom"):
                    st.session_state["selected_model"] = custom_id.strip()
                    st.session_state["selected_model_meta"] = {
                        "id": custom_id.strip(),
                        "name": custom_id.strip(),
                        "backend": "mlx" if compat["is_mlx_native"] else "cuda",
                        "quantization": meta["quantization"],
                        "params": meta["params"],
                    }
                    st.success(f"Selected '{custom_id}' as active model.")
                    st.rerun()

            with c_btn2:
                if st.button("⬇️ Download Custom Model", key="btn_dl_custom"):
                    with st.spinner(f"Downloading {custom_id}..."):
                        res = loader.download_model(custom_id)
                        if res["success"]:
                            st.success(f"Downloaded successfully ({res['size']})!")
                        else:
                            st.error(f"Download failed: {res['error']}")

    with tab_local:
        st.markdown("### Local Models Storage")
        if not local_models:
            st.info("No local models found in models/base/. Models downloaded above or loaded directly from Hugging Face will appear here.")
        else:
            st.write(f"Found {len(local_models)} downloaded model directories:")
            for lm in local_models:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{lm['name']}** (`{lm['id']}`)")
                    st.caption(f"Size: {lm['size']} • Files: {lm['files_count']} • Quantization: {lm['quantization']}")
                with col2:
                    is_active = st.session_state.get("selected_model") == lm["path"]
                    if st.button("Select", key=f"sel_local_{lm['id']}", type="primary" if is_active else "secondary"):
                        st.session_state["selected_model"] = lm["path"]
                        st.session_state["selected_model_meta"] = lm
                        st.success(f"Selected local model '{lm['name']}'.")
                        st.rerun()
                with col3:
                    st.code(lm["path"], language="text")

    # Current Selection Banner
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
    curr_sel = st.session_state.get("selected_model")
    if curr_sel:
        meta = st.session_state.get("selected_model_meta", {})
        st.markdown(
            f"""
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 14px 20px;">
                <span style="color: #60a5fa; font-weight: 600;">Currently Active Model:</span>
                <span style="color: #f8fafc; font-weight: 600; margin-left: 8px;">{curr_sel}</span>
                <span style="color: #94a3b8; margin-left: 12px;">(Backend: {meta.get('backend', 'auto')}, Quantization: {meta.get('quantization', 'unknown')})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Default selection if none selected
        default_model = catalog_models[0]["id"]
        st.session_state["selected_model"] = default_model
        st.session_state["selected_model_meta"] = catalog_models[0]
        st.info(f"Default model selected: **{default_model}**")


if __name__ == "__main__":
    render()

