"""LLM Fine-Tuning Studio – LoRA & QLoRA

Main application entry point and navigation dispatcher.
"""

from pathlib import Path
import streamlit as st
from utils.file_manager import ensure_directories, get_project_root, load_config
from utils.hardware import detect_hardware, get_current_telemetry

# 1. Page Configuration
st.set_page_config(
    page_title="LLM Fine-Tuning Studio – LoRA & QLoRA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Initialize Directories & Config
root = get_project_root()
ensure_directories()
config = load_config()

# 3. Inject Custom Dark Navy CSS
css_path = root / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 4. Import Page Modules
import pages.comparison as p_comparison
import pages.dashboard as p_dashboard
import pages.dataset_manager as p_dataset_manager
import pages.evaluation as p_evaluation
import pages.experiment_history as p_experiment_history
import pages.lora_training as p_lora_training
import pages.model_selection as p_model_selection
import pages.playground as p_playground
import pages.qlora_training as p_qlora_training
import pages.settings as p_settings
import pages.training_monitor as p_training_monitor

# 5. Define Pages Navigation with explicit unique url_path for each page
pages_list = [
    st.Page(p_dashboard.render, title="Dashboard", icon="📊", url_path="dashboard", default=True),
    st.Page(p_model_selection.render, title="Model Selection", icon="🧠", url_path="model-selection"),
    st.Page(p_dataset_manager.render, title="Dataset Manager", icon="📁", url_path="dataset-manager"),
    st.Page(p_lora_training.render, title="LoRA Training", icon="⚡", url_path="lora-training"),
    st.Page(p_qlora_training.render, title="QLoRA Training", icon="🧩", url_path="qlora-training"),
    st.Page(p_training_monitor.render, title="Training Monitor", icon="📡", url_path="training-monitor"),
    st.Page(p_evaluation.render, title="Evaluation", icon="📈", url_path="evaluation"),
    st.Page(p_comparison.render, title="Model Comparison", icon="⚖️", url_path="model-comparison"),
    st.Page(p_playground.render, title="LLM Playground", icon="💬", url_path="playground"),
    st.Page(p_experiment_history.render, title="Experiment History", icon="📜", url_path="experiment-history"),
    st.Page(p_settings.render, title="Settings", icon="⚙️", url_path="settings"),
]

# 6. Sidebar Diagnostics & Telemetry
hw = detect_hardware()
telemetry = get_current_telemetry()

with st.sidebar:
    # Premium Branding Header
    os_badge = "Apple Silicon MLX" if hw.is_apple_silicon else (f"{hw.os_name} CUDA" if hw.cuda_available else f"{hw.os_name} CPU")
    st.markdown(
        f"""
        <div style="padding: 14px 6px 18px 6px; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 18px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 1.25rem; font-weight: 800; background: linear-gradient(90deg, #ffffff, #60a5fa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">
                    ⚡ Studio Pro
                </div>
                <div style="display: inline-flex; align-items: center; font-size: 0.7rem; font-weight: 600; color: #34d399; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: 9999px; padding: 2px 8px;">
                    <span class="pulsing-dot"></span> Online
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; font-weight: 400;">
                PEFT • LoRA & QLoRA • {os_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hardware Card
    if hw.is_apple_silicon:
        hw_type = "macOS (Apple Silicon UMA)"
        hw_accent_color = "#38bdf8"
    elif hw.is_windows:
        hw_type = f"Windows ({'NVIDIA CUDA' if hw.cuda_available else 'CPU Mode'})"
        hw_accent_color = "#10b981" if hw.cuda_available else "#94a3b8"
    elif hw.cuda_available:
        hw_type = "Linux (NVIDIA CUDA)"
        hw_accent_color = "#10b981"
    else:
        hw_type = f"{hw.os_name} (CPU Mode)"
        hw_accent_color = "#94a3b8"

    st.markdown(
        f"""
        <div style="background: linear-gradient(180deg, rgba(22, 33, 56, 0.7) 0%, rgba(13, 20, 36, 0.85) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; color: #94a3b8;">Active Compute</div>
                <span class="badge badge-blue" style="margin: 0; font-size: 0.68rem;">{hw.recommended_backend.upper()}</span>
            </div>
            <div style="font-size: 0.92rem; font-weight: 700; color: #f8fafc; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {hw.cpu_brand}
            </div>
            <div style="font-size: 0.74rem; color: {hw_accent_color}; margin-top: 2px;">
                {hw_type} • {hw.total_memory_gb} GB RAM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Memory Telemetry Progress
    ram_used = telemetry["ram_used_gb"]
    ram_total = telemetry["ram_total_gb"]
    ram_ratio = min(1.0, max(0.0, ram_used / ram_total)) if ram_total > 0 else 0.0
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">
            <span>Memory Pressure</span>
            <span style="font-weight: 600; color: {'#10b981' if telemetry['ram_percent'] < 80 else '#f59e0b'};">{telemetry['ram_percent']}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(ram_ratio)
    st.caption(f"{ram_used} GB used of {ram_total} GB ({telemetry['ram_free_gb']} GB available)")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# 7. Render Selected Page
nav = st.navigation(pages_list)

# 8. Sidebar Status Footers (Active Model & Dataset)
with st.sidebar:
    st.markdown("<div style='height: 20px; border-top: 1px solid rgba(255,255,255,0.07); margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 8px;">
            Active Session State
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_m = st.session_state.get("selected_model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.78rem;">
            <div style="color: #94a3b8;">Model</div>
            <div style="color: #60a5fa; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{active_m.split('/')[-1]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_d = st.session_state.get("selected_dataset_name", "None")
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.78rem;">
            <div style="color: #94a3b8;">Dataset</div>
            <div style="color: #34d399; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{active_d}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_exp = st.session_state.get("active_training_exp_id")
    if active_exp:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 10px; font-size: 0.78rem;">
                <div style="color: #94a3b8;">Active Run</div>
                <div style="color: #f59e0b; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{active_exp[:18]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Run navigation
nav.run()

