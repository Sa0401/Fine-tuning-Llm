"""Side-by-side comparison between Base Pretrained Model and Fine-Tuned Adapters."""

import time
import plotly.graph_objects as go
import streamlit as st
from core.inference_engine import InferenceEngine
from core.model_loader import ModelLoader
from utils.file_manager import list_adapters


def render():
    st.markdown("## ⚖️ Side-by-Side Model Comparison")
    st.caption("Directly compare responses, response quality, and latency between base models and fine-tuned LoRA / QLoRA adapters.")

    loader = ModelLoader()
    engine = InferenceEngine()
    catalog = loader.get_catalog_models()
    adapters = list_adapters()

    # Model Configuration
    st.markdown("### 1. Configure Comparison Candidates")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Candidate A (e.g., Base Model)")
        model_a_id = st.selectbox("Model A Base ID", [m["id"] for m in catalog], index=0, key="comp_base_a")
        adapter_a_opts = ["None (Pure Base)"] + [a["name"] for a in adapters]
        adapter_a_name = st.selectbox("Model A Adapter", adapter_a_opts, index=0, key="comp_ad_a")
        adapter_a_path = None
        if adapter_a_name != "None (Pure Base)":
            ad = next((a for a in adapters if a["name"] == adapter_a_name), None)
            if ad:
                adapter_a_path = ad["path"]

    with col_b:
        st.markdown("#### Candidate B (e.g., Fine-Tuned Adapter)")
        model_b_id = st.selectbox("Model B Base ID", [m["id"] for m in catalog], index=0, key="comp_base_b")
        adapter_b_opts = ["None (Pure Base)"] + [a["name"] for a in adapters]
        default_b_idx = 1 if len(adapter_b_opts) > 1 else 0
        adapter_b_name = st.selectbox("Model B Adapter", adapter_b_opts, index=default_b_idx, key="comp_ad_b")
        adapter_b_path = None
        if adapter_b_name != "None (Pure Base)":
            ad = next((a for a in adapters if a["name"] == adapter_b_name), None)
            if ad:
                adapter_b_path = ad["path"]

    # Prompt & Generation Parameters
    st.markdown("### 2. Prompt & Generation Settings")
    test_prompt = st.text_area(
        "Evaluation Prompt",
        value="Explain the difference between LoRA and full fine-tuning in machine learning.",
        height=90,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1, key="comp_temp")
    with c2:
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05, key="comp_topp")
    with c3:
        max_tokens = st.slider("Max Output Tokens", 32, 512, 160, 32, key="comp_max")

    if st.button("⚡ Run Side-by-Side Generation", type="primary", use_container_width=True):
        if not test_prompt.strip():
            st.error("Please enter a non-empty prompt.")
            return

        with st.spinner("Generating response with Candidate A..."):
            engine.load_model(model_a_id, adapter_path=adapter_a_path)
            res_a = engine.generate_response(
                messages=[{"role": "user", "content": test_prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        with st.spinner("Generating response with Candidate B..."):
            engine.load_model(model_b_id, adapter_path=adapter_b_path)
            res_b = engine.generate_response(
                messages=[{"role": "user", "content": test_prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        st.session_state["comparison_results"] = {
            "prompt": test_prompt,
            "res_a": res_a,
            "res_b": res_b,
            "label_a": f"{model_a_id.split('/')[-1]} ({adapter_a_name})",
            "label_b": f"{model_b_id.split('/')[-1]} ({adapter_b_name})",
        }

    # Display Side-by-Side outputs
    comp = st.session_state.get("comparison_results")
    if comp:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Side-by-Side Output Comparison")

        out_col1, out_col2 = st.columns(2)

        with out_col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(180deg, rgba(22, 33, 56, 0.7) 0%, rgba(13, 20, 36, 0.85) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">🅰️ Candidate A</span>
                        <span class="badge badge-blue">{comp['label_a']}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; min-height: 180px; white-space: pre-wrap;">{comp['res_a'].get('response', '')}</div>
                    <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); display: flex; gap: 14px; font-size: 0.8rem; color: #94a3b8;">
                        <span>⏱️ <strong>{comp['res_a'].get('latency_sec', 0)}s</strong></span>
                        <span>🚀 <strong>{comp['res_a'].get('tokens_per_sec', 0)} tok/s</strong></span>
                        <span>📝 <strong>{comp['res_a'].get('tokens_count', 0)} tokens</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with out_col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(180deg, rgba(26, 38, 68, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 12px; padding: 20px; box-shadow: 0 8px 30px rgba(37, 99, 235, 0.15);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
                        <span style="font-weight: 700; color: #60a5fa; font-size: 1.05rem;">🅱️ Candidate B</span>
                        <span class="badge badge-purple">{comp['label_b']}</span>
                    </div>
                    <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.6; min-height: 180px; white-space: pre-wrap;">{comp['res_b'].get('response', '')}</div>
                    <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); display: flex; gap: 14px; font-size: 0.8rem; color: #94a3b8;">
                        <span>⏱️ <strong style="color: #60a5fa;">{comp['res_b'].get('latency_sec', 0)}s</strong></span>
                        <span>🚀 <strong style="color: #34d399;">{comp['res_b'].get('tokens_per_sec', 0)} tok/s</strong></span>
                        <span>📝 <strong>{comp['res_b'].get('tokens_count', 0)} tokens</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Latency & Speed Chart
        st.markdown("#### 📊 Quantitative Latency Comparison")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Latency (seconds)",
                x=[comp["label_a"], comp["label_b"]],
                y=[comp["res_a"].get("latency_sec", 0), comp["res_b"].get("latency_sec", 0)],
                marker_color="#38bdf8",
                text=[f"{comp['res_a'].get('latency_sec', 0)}s", f"{comp['res_b'].get('latency_sec', 0)}s"],
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Tokens / Second",
                x=[comp["label_a"], comp["label_b"]],
                y=[comp["res_a"].get("tokens_per_sec", 0), comp["res_b"].get("tokens_per_sec", 0)],
                marker_color="#10b981",
                text=[f"{comp['res_a'].get('tokens_per_sec', 0)}/s", f"{comp['res_b'].get('tokens_per_sec', 0)}/s"],
                textposition="auto",
            )
        )
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render()

