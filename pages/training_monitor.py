"""Real-time training monitor with live logs, loss curves, and job controls."""

import time
import plotly.graph_objects as go
import streamlit as st
from core.experiment_tracker import ExperimentTracker
from core.training_manager import TrainingManager
from utils.hardware import get_current_telemetry


def render():
    st.markdown("## 📡 Training Monitor")
    st.caption("Live streaming training logs, loss curves, and background execution status.")

    tm = TrainingManager()
    tracker = ExperimentTracker()
    telemetry = get_current_telemetry()

    # List recent experiments to monitor
    all_exps = tracker.list_experiments(limit=30)
    if not all_exps:
        st.info("No training jobs found. Start a run in LoRA or QLoRA Training to begin monitoring.")
        return

    exp_options = {f"{e['name']} ({e['id']}) [{e['status'].upper()}]": e["id"] for e in all_exps}

    # Default to active job if one exists, otherwise most recent
    default_id = st.session_state.get("active_training_exp_id", all_exps[0]["id"])
    default_idx = 0
    for idx, (label, eid) in enumerate(exp_options.items()):
        if eid == default_id:
            default_idx = idx
            break

    m_col1, m_col2 = st.columns([3, 1])
    with m_col1:
        selected_label = st.selectbox("Select Training Experiment to Monitor:", list(exp_options.keys()), index=default_idx)
        selected_exp_id = exp_options[selected_label]
        st.session_state["active_training_exp_id"] = selected_exp_id
    with m_col2:
        auto_refresh = st.checkbox("Auto-refresh live metrics (2s)", value=True)

    job_data = tm.get_job_status(selected_exp_id)
    status = job_data.get("status", "unknown")

    # Header Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    status_color = {
        "running": "#38bdf8",
        "completed": "#10b981",
        "failed": "#ef4444",
        "stopped": "#f59e0b",
    }.get(status, "#94a3b8")

    c1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label"><span>⚙️</span> Execution Status</div>
            <div class="metric-value" style="color: {status_color}; font-size: 1.55rem;">
                <span class="pulsing-dot" style="background-color: {status_color}; box-shadow: 0 0 10px {status_color};"></span>
                {status.upper()}
            </div>
            <div class="metric-sub">{job_data.get('method', '').upper()} • {job_data.get('model_id', '').split('/')[-1][:22]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prog_pct = int(job_data.get("progress", 0.0) * 100)
    c2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label"><span>🎯</span> Progress</div>
            <div class="metric-value">{job_data.get('current_iter', 0)} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">/ {job_data.get('total_iters', 0)} iters</span></div>
            <div class="metric-sub">
                <span class="badge badge-blue" style="margin:0; font-size: 0.65rem;">{prog_pct}%</span>
                <span style="margin-left: 6px;">Iterations completed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_loss = job_data.get("train_loss")
    t_loss_str = f"{t_loss:.4f}" if t_loss is not None else "Awaiting report..."
    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label"><span>📉</span> Current Train Loss</div>
            <div class="metric-value" style="color: #60a5fa;">{t_loss_str}</div>
            <div class="metric-sub">Evaluated at step {job_data.get('current_iter', 0)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    elapsed_sec = job_data.get("elapsed_sec", 0.0)
    c4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label"><span>⏱️</span> Elapsed Duration</div>
            <div class="metric-value">{elapsed_sec:.1f}s</div>
            <div class="metric-sub">RAM: {telemetry['ram_used_gb']} / {telemetry['ram_total_gb']} GB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress Bar
    st.progress(job_data.get("progress", 0.0))

    # Job Actions: Cancel if running
    if status == "running":
        cancel_col1, cancel_col2 = st.columns([1, 4])
        with cancel_col1:
            if st.button("🛑 Safe Cancel Job", type="secondary"):
                success, cancel_msg = tm.cancel_training(selected_exp_id)
                if success:
                    st.warning(cancel_msg)
                    st.rerun()
                else:
                    st.error(cancel_msg)
        with cancel_col2:
            st.caption("Canceling terminates the background process cleanly without corrupting checkpoints or base models.")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Live Loss Graph
    st.markdown("### 📉 Live Loss Convergence Curve")
    loss_history = tracker.get_loss_history(selected_exp_id)

    if not loss_history:
        st.info("Loss history points are recorded every report interval (steps_per_report). Waiting for first report...")
    else:
        steps = [h["step"] for h in loss_history if h.get("train_loss") is not None]
        train_losses = [h["train_loss"] for h in loss_history if h.get("train_loss") is not None]

        val_steps = [h["step"] for h in loss_history if h.get("val_loss") is not None]
        val_losses = [h["val_loss"] for h in loss_history if h.get("val_loss") is not None]

        fig = go.Figure()
        if steps:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=train_losses,
                    mode="lines+markers",
                    name="Training Loss",
                    line=dict(color="#38bdf8", width=2.5),
                    marker=dict(size=6),
                )
            )
        if val_steps:
            fig.add_trace(
                go.Scatter(
                    x=val_steps,
                    y=val_losses,
                    mode="lines+markers",
                    name="Validation Loss",
                    line=dict(color="#10b981", width=2.5, dash="dash"),
                    marker=dict(size=7),
                )
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font_color="#cbd5e1",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Training Steps (Iterations)",
            yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Live Terminal Logs with Mac-style window styling
    st.markdown("### 🖥️ Real-Time Standard Output & Training Logs")
    logs_content = job_data.get("logs", "No logs available.")
    st.markdown(
        f"""
        <div class="terminal-window">
            <div class="terminal-header">
                <div class="terminal-dots">
                    <span class="terminal-dot terminal-dot-red"></span>
                    <span class="terminal-dot terminal-dot-yellow"></span>
                    <span class="terminal-dot terminal-dot-green"></span>
                </div>
                <span class="terminal-title">studio_train_runner — {selected_exp_id} — {status.upper()}</span>
            </div>
            <div class="log-terminal">{logs_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Auto refresh trigger if running
    if auto_refresh and status == "running":
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    render()

