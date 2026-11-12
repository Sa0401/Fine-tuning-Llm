"""Dataset management, validation, token statistics, and train/val/test splitting."""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from core.dataset_processor import DatasetProcessor
from utils.file_manager import get_project_root


def render():
    st.markdown("## 📊 Dataset Manager")
    st.caption("Upload, validate, inspect token lengths, and partition conversational instruction datasets.")

    processor = DatasetProcessor()
    root = get_project_root()
    sample_dir = root / "sample_datasets"

    tab_upload, tab_samples, tab_saved = st.tabs([
        "📤 Upload Custom Dataset",
        "📂 Pre-loaded Sample Datasets",
        "🗂️ Prepared Datasets on Disk",
    ])

    loaded_records = None
    loaded_dataset_name = ""

    with tab_upload:
        st.markdown("### Upload Dataset File")
        st.write("Supported formats: **JSONL**, **JSON**, **CSV**, **TXT**. Formats supported include multi-turn `messages`, Alpaca `instruction`/`output`, and `prompt`/`completion`.")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["jsonl", "json", "csv", "txt"],
            help="Upload raw instruction or chat data.",
        )

        if uploaded_file is not None:
            dataset_name_input = st.text_input(
                "Dataset Identifier Name",
                value=Path(uploaded_file.name).stem,
            )
            try:
                raw_bytes = uploaded_file.read()
                raw_records = processor.load_from_file(raw_bytes, uploaded_file.name)
                loaded_records = raw_records
                loaded_dataset_name = dataset_name_input
                st.success(f"Parsed {len(raw_records)} records from '{uploaded_file.name}'.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with tab_samples:
        st.markdown("### Quick Start: Use Pre-Built Sample Datasets")
        sample_files = list(sample_dir.glob("*")) if sample_dir.exists() else []

        if not sample_files:
            st.info("No sample datasets found.")
        else:
            sample_choices = {f.name: f for f in sample_files if f.is_file() and not f.name.startswith(".")}
            selected_sample = st.selectbox("Select a sample dataset:", list(sample_choices.keys()))

            if st.button("Load Selected Sample Dataset", type="primary"):
                target_path = sample_choices[selected_sample]
                with open(target_path, "rb") as f:
                    raw_records = processor.load_from_file(f.read(), target_path.name)
                st.session_state["active_dataset_records"] = raw_records
                st.session_state["active_dataset_name"] = target_path.stem
                st.success(f"Loaded {len(raw_records)} sample records from '{target_path.name}'.")

    with tab_saved:
        st.markdown("### Ready-to-Train Datasets")
        saved_dsets = processor.list_saved_datasets()
        if not saved_dsets:
            st.info("No prepared datasets found in data/ yet. Process and save one below.")
        else:
            for d in saved_dsets:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{d['name']}**")
                    st.caption(f"Train samples: {d['train_samples']} • Valid samples: {d['valid_samples']} • Path: `{d['path']}`")
                with col2:
                    if st.button("Set for Training", key=f"set_dset_{d['name']}"):
                        st.session_state["selected_dataset_path"] = d["path"]
                        st.session_state["selected_dataset_name"] = d["name"]
                        st.session_state["selected_dataset_samples"] = d["train_samples"]
                        st.success(f"Set '{d['name']}' as active training dataset!")

    # Check session state for loaded records
    if loaded_records is not None:
        st.session_state["active_dataset_records"] = loaded_records
        st.session_state["active_dataset_name"] = loaded_dataset_name

    records = st.session_state.get("active_dataset_records")
    current_name = st.session_state.get("active_dataset_name", "custom_dataset")

    if records:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown(f"### 🔍 Validation & Inspection: `{current_name}`")

        # 1. Validation & Cleaning
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            remove_dups = st.checkbox("Deduplicate identical records", value=True)
            filter_empty = st.checkbox("Filter empty prompts and completions", value=True)

        cleaned_records, clean_stats = processor.validate_and_clean(
            records,
            remove_duplicates=remove_dups,
            filter_empty=filter_empty,
        )

        # Validation Metric Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Input Records", clean_stats["total_raw"])
        c2.metric("Valid Cleaned Records", clean_stats["cleaned_count"])
        c3.metric("Duplicates Removed", clean_stats["duplicates_removed"])
        c4.metric("Empty / Invalid Filtered", clean_stats["empty_removed"] + clean_stats["invalid_format_count"])

        # 2. Token Length Distribution
        st.markdown("#### 📏 Token Length Statistics")
        token_stats = processor.calculate_token_statistics(cleaned_records)

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Min Tokens", token_stats["min"])
        t2.metric("Median Tokens", token_stats["median"])
        t3.metric("Mean Tokens", token_stats["mean"])
        t4.metric("95th Percentile", token_stats["p95"])
        t5.metric("Max Tokens", token_stats["max"])

        if token_stats["lengths"]:
            fig = px.histogram(
                x=token_stats["lengths"],
                nbins=20,
                title="Token Length Distribution",
                labels={"x": "Estimated Token Count", "y": "Frequency"},
                color_discrete_sequence=["#38bdf8"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=240,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3. Preview Records
        st.markdown("#### 👁️ Preview Processed Conversational Records")
        preview_samples = []
        for i, r in enumerate(cleaned_records[:3]):
            if "messages" in r:
                user_msg = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
                preview_samples.append({
                    "Sample #": i + 1,
                    "User Input": user_msg[:120] + "..." if len(user_msg) > 120 else user_msg,
                    "Assistant Target": asst_msg[:120] + "..." if len(asst_msg) > 120 else asst_msg,
                })
            else:
                preview_samples.append({
                    "Sample #": i + 1,
                    "Text": str(r.get("text", ""))[:200],
                })
        st.dataframe(pd.DataFrame(preview_samples), use_container_width=True)

        # 4. Train / Val / Test Partitioning
        st.markdown("#### ✂️ Train / Validation / Test Splitting")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            train_pct = st.slider("Training Split %", 50, 95, 80, step=5)
            val_pct = st.slider("Validation Split %", 5, 40, 10, step=5)
            test_pct = max(0, 100 - train_pct - val_pct)
            st.caption(f"Resulting Split: **{train_pct}% Train** ({int(len(cleaned_records)*train_pct/100)} rows) | **{val_pct}% Valid** ({int(len(cleaned_records)*val_pct/100)} rows) | **{test_pct}% Test** ({int(len(cleaned_records)*test_pct/100)} rows)")

        with s_col2:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Split & Save Dataset for Training", type="primary", use_container_width=True):
                splits = processor.split_dataset(
                    cleaned_records,
                    train_ratio=train_pct / 100.0,
                    val_ratio=val_pct / 100.0,
                    test_ratio=test_pct / 100.0,
                )
                saved_paths = processor.save_split_dataset(splits, current_name)

                st.session_state["selected_dataset_path"] = saved_paths["dir"]
                st.session_state["selected_dataset_name"] = current_name
                st.session_state["selected_dataset_samples"] = len(splits["train"])

                st.success(
                    f"✅ Dataset '{current_name}' successfully processed and saved to `{saved_paths['dir']}`! "
                    f"Ready for LoRA & QLoRA training."
                )

        # Download Processed Dataset
        st.markdown("#### 📥 Export Processed Dataset")
        jsonl_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned_records)
        st.download_button(
            label="Download Cleaned JSONL",
            data=jsonl_str,
            file_name=f"{current_name}_cleaned.jsonl",
            mime="application/jsonlines",
        )

    # Active dataset status indicator
    active_dset = st.session_state.get("selected_dataset_path")
    if active_dset:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px 18px; margin-top: 25px;">
                <span style="color: #34d399; font-weight: 600;">Active Training Dataset:</span>
                <span style="color: #f8fafc; margin-left: 8px;">{st.session_state.get('selected_dataset_name')} ({st.session_state.get('selected_dataset_samples', 0)} training samples)</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Path: <code>{active_dset}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render()

