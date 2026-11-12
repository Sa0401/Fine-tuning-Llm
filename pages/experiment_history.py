"""Experiment History and Comparison Tracker page."""

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.experiment_tracker import ExperimentTracker
from core.model_exporter import ModelExporter
from utils.file_manager import format_size, get_dir_size


def render():
    st.markdown("## 📜 Experiment History & Tracking")
    st.caption("Browse, inspect, compare, and manage your fine-tuning experiments recorded in SQLite.")

    tracker = ExperimentTracker()
    exporter = ModelExporter()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.selectbox("Filter Status", ["All", "completed", "running", "failed", "stopped"])
    with col_f2:
        method_filter = st.selectbox("Filter Method", ["All", "lora", "qlora"])
    with col_f3:
        search_query = st.text_input("Search Name or Model", placeholder="e.g., qwen or run1")

    experiments = tracker.list_experiments(
        status=status_filter if status_filter != "All" else None,
        method=method_filter if method_filter != "All" else None,
    )

    if search_query:
        sq = search_query.lower()
        experiments = [
            e for e in experiments
            if sq in e.get("name", "").lower() or sq in e.get("model_id", "").lower()
        ]

    if not experiments:
        st.info("No matching experiments found.")
        return

    # Table Summary
    table_rows = []
    for e in experiments:
        t_loss = f"{e['final_train_loss']:.4f}" if e.get("final_train_loss") is not None else "N/A"
        v_loss = f"{e['final_val_loss']:.4f}" if e.get("final_val_loss") is not None else "N/A"
        table_rows.append({
            "ID": e["id"],
            "Name": e["name"],
            "Method": e["method"].upper(),
            "Base Model": e["model_id"].split("/")[-1],
            "Dataset": e["dataset_name"],
            "Status": e["status"].upper(),
            "Duration": f"{e.get('duration_seconds', 0):.1f}s",
            "Final Loss": t_loss,
            "Created": e["created_at"][:16].replace("T", " "),
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Detailed Inspector
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🔎 Experiment Details & Logs")

    selected_exp_id = st.selectbox(
        "Select Experiment to Inspect:",
        options=[e["id"] for e in experiments],
        format_func=lambda x: next((f"{e['name']} ({e['id']}) - {e['status'].upper()}" for e in experiments if e["id"] == x), x),
    )

    exp_data = tracker.get_experiment(selected_exp_id)
    if exp_data:
        # Top metric row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Method", exp_data["method"].upper())
        m2.metric("Status", exp_data["status"].upper())
        m3.metric("Final Train Loss", exp_data.get("final_train_loss") or "N/A")
        m4.metric("Duration", f"{exp_data.get('duration_seconds', 0):.1f}s")

        tab_hp, tab_loss, tab_logs, tab_actions = st.tabs([
            "⚙️ Hyperparameters",
            "📉 Loss Curve",
            "📄 Execution Logs",
            "🛠️ Actions & Export",
        ])

        with tab_hp:
            hp = exp_data.get("hyperparameters", {})
            if hp:
                st.json(hp)
            else:
                st.write("No hyperparameters recorded.")

        with tab_loss:
            loss_history = tracker.get_loss_history(selected_exp_id)
            if not loss_history:
                st.info("No intermediate step losses recorded for this run.")
            else:
                steps = [h["step"] for h in loss_history if h.get("train_loss") is not None]
                losses = [h["train_loss"] for h in loss_history if h.get("train_loss") is not None]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines+markers", line=dict(color="#38bdf8", width=2)))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    height=240,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Step",
                    yaxis_title="Train Loss",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab_logs:
            log_path = Path(exp_data.get("log_file_path", ""))
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text = f.read()
                    st.markdown(
                        f"""
                        <div class="terminal-window">
                            <div class="terminal-header">
                                <div class="terminal-dots">
                                    <span class="terminal-dot terminal-dot-red"></span>
                                    <span class="terminal-dot terminal-dot-yellow"></span>
                                    <span class="terminal-dot terminal-dot-green"></span>
                                </div>
                                <span class="terminal-title">{selected_exp_id} — execution.log</span>
                            </div>
                            <div class="log-terminal">{log_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error reading log file: {e}")
            else:
                st.info("No log file found on disk.")

        with tab_actions:
            st.markdown("#### Export and Safety Management")
            col_a1, col_a2 = st.columns(2)

            with col_a1:
                # Markdown Report Export
                report_path = f"experiments/{selected_exp_id}/report.md"
                md_content = f"# Experiment: {exp_data['name']}\nModel: {exp_data['model_id']}\nLoss: {exp_data.get('final_train_loss')}\n"
                try:
                    rep_file = exporter.export_experiment_report(exp_data, report_path)
                    with open(rep_file, "r", encoding="utf-8") as f:
                        md_content = f.read()
                except Exception:
                    pass

                st.download_button(
                    "📥 Download Markdown Report",
                    data=md_content,
                    file_name=f"{selected_exp_id}_report.md",
                    mime="text/markdown",
                )

            with col_a2:
                # Delete metadata safely
                if st.button("🗑️ Delete Experiment Metadata", type="secondary"):
                    tracker.delete_experiment(selected_exp_id)
                    st.success(f"Deleted experiment '{selected_exp_id}' metadata (base models preserved).")
                    st.rerun()


if __name__ == "__main__":
    render()

