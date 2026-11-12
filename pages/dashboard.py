"""Dashboard page: High-level overview, metrics, telemetry, and training history."""

from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.experiment_tracker import ExperimentTracker
from core.model_loader import ModelLoader
from utils.file_manager import list_adapters
from utils.hardware import detect_hardware, get_current_telemetry


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">LLM Fine-Tuning Studio</h1>
            <p class="hero-subtitle">High-Performance Parameter-Efficient Fine-Tuning (LoRA & QLoRA) on Apple Silicon & NVIDIA CUDA</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tracker = ExperimentTracker()
    model_loader = ModelLoader()
    hw = detect_hardware()
    stats = tracker.get_summary_stats()
    local_models = model_loader.list_local_models()
    adapters = list_adapters()
    telemetry = get_current_telemetry()

    # Hardware Banner
    hw_accent = "#38bdf8" if hw.is_apple_silicon else "#10b981"
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid {hw_accent}; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
            <span style="font-weight: 600; color: #f8fafc;">Hardware Environment:</span>
            <span style="color: #94a3b8; margin-left: 8px;">{hw.backend_status_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Key Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label"><span>📊</span> Total Experiments</div>
                <div class="metric-value">{stats['total_experiments']}</div>
                <div class="metric-sub">
                    <span class="badge badge-emerald" style="margin:0; font-size: 0.65rem;">{stats['completed_experiments']} Done</span>
                    <span class="badge badge-blue" style="margin:0; font-size: 0.65rem;">{stats['running_experiments']} Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label"><span>🧠</span> Base Models</div>
                <div class="metric-value">{len(local_models) + len(model_loader.get_catalog_models())}</div>
                <div class="metric-sub">
                    <span class="badge badge-purple" style="margin:0; font-size: 0.65rem;">{len(local_models)} Local</span>
                    <span style="margin-left: 6px;">Available to fine-tune</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label"><span>📦</span> Saved Adapters</div>
                <div class="metric-value">{len(adapters)}</div>
                <div class="metric-sub">
                    <span class="badge badge-amber" style="margin:0; font-size: 0.65rem;">Ready</span>
                    <span style="margin-left: 6px;">Inference / Fuse</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        ram_pct = telemetry["ram_percent"]
        ram_color = "#10b981" if ram_pct < 75 else ("#f59e0b" if ram_pct < 90 else "#ef4444")
        mem_label = "Unified Memory" if hw.is_unified_memory else "System Memory"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label"><span>⚡</span> {mem_label}</div>
                <div class="metric-value" style="color: {ram_color};">{telemetry['ram_used_gb']} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">/ {telemetry['ram_total_gb']} GB</span></div>
                <div class="metric-sub">
                    <span class="badge badge-blue" style="margin:0; font-size: 0.65rem;">{ram_pct}%</span>
                    <span style="margin-left: 6px;">{telemetry['ram_free_gb']} GB free</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Recent Experiments & Training History Charts
    exp_col, chart_col = st.columns([1, 1])

    with exp_col:
        st.markdown("### 📋 Recent Experiments")
        experiments = tracker.list_experiments(limit=5)
        if not experiments:
            st.info("No training experiments recorded yet. Start your first fine-tuning run in LoRA or QLoRA Training!")
        else:
            for exp in experiments:
                status = exp.get("status", "unknown")
                badge_class = {
                    "completed": "badge-emerald",
                    "running": "badge-blue",
                    "failed": "badge-rose",
                    "stopped": "badge-amber",
                }.get(status, "badge-blue")

                loss_str = f"Loss: {exp['final_train_loss']:.4f}" if exp.get("final_train_loss") is not None else "In Progress"
                dur_str = f"{exp.get('duration_seconds', 0):.1f}s"

                st.markdown(
                    f"""
                    <div style="background: #162032; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">{exp.get('name')}</div>
                            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 2px;">
                                {exp.get('model_id')} • <span class="badge {badge_class}">{status.upper()}</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #38bdf8; font-weight: 600; font-size: 0.9rem;">{loss_str}</div>
                            <div style="color: #64748b; font-size: 0.75rem;">{dur_str}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with chart_col:
        st.markdown("### 📈 Training Loss Overview")
        all_exps = tracker.list_experiments(limit=20)
        completed_exps = [e for e in all_exps if e.get("final_train_loss") is not None]

        if not completed_exps:
            st.info("Interactive loss trends will appear here once experiments finish.")
        else:
            names = [e["name"][:16] for e in reversed(completed_exps)]
            losses = [e["final_train_loss"] for e in reversed(completed_exps)]
            methods = [e["method"].upper() for e in reversed(completed_exps)]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=losses,
                    text=[f"{l:.3f}" for l in losses],
                    textposition="auto",
                    marker_color=["#3b82f6" if m == "LORA" else "#06b6d4" for m in methods],
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                margin=dict(l=20, r=20, t=20, b=30),
                height=260,
                yaxis_title="Final Train Loss",
                xaxis_title="Experiment",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Quick Action Banner
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Studio Workflow")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("**1. Select Model**")
        st.caption("Choose base weights (Qwen2.5-0.5B / 1.5B 4-bit) tailored for Apple Silicon.")
    with w2:
        st.markdown("**2. Prepare Dataset**")
        st.caption("Upload and validate conversational JSONL, CSV, or Alpaca instructions.")
    with w3:
        st.markdown("**3. Train Adapter**")
        st.caption("Run background LoRA or QLoRA fine-tuning with live loss tracking.")
    with w4:
        st.markdown("**4. Test & Chat**")
        st.caption("Compare against base model in the interactive chat playground.")


if __name__ == "__main__":
    render()

