"""Conversational chatbot playground with loaded base models and adapters."""

import streamlit as st
from core.inference_engine import InferenceEngine
from core.model_loader import ModelLoader
from utils.file_manager import list_adapters


def render():
    st.markdown("## 💬 LLM Chatbot Playground")
    st.caption("Interact with your fine-tuned LoRA & QLoRA models in a modern conversational interface.")

    engine = InferenceEngine()
    loader = ModelLoader()
    catalog = loader.get_catalog_models()
    adapters = list_adapters()

    # Chat Settings & Model Selection Sidebar / Expander
    with st.expander("⚙️ Model & Generation Settings", expanded=False):
        col_m, col_a = st.columns(2)
        with col_m:
            model_options = [m["id"] for m in catalog]
            curr_sel = st.session_state.get("selected_model", model_options[0])
            idx = model_options.index(curr_sel) if curr_sel in model_options else 0
            chat_model = st.selectbox("Base Pretrained Model", model_options, index=idx)

        with col_a:
            adapter_opts = ["None (Base Model Only)"] + [a["name"] for a in adapters]
            selected_adapter_name = st.selectbox("Active LoRA Adapter", adapter_opts, index=0)

        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Higher values produce more creative output; lower values are deterministic.")
        with c2:
            top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05, help="Cumulative probability cutoff for nucleus sampling.")
        with c3:
            max_tokens = st.slider("Max Response Tokens", 32, 1024, 256, 32, help="Maximum number of tokens to generate per turn.")

    # Model Active Status Bar
    active_adapter_path = None
    if selected_adapter_name != "None (Base Model Only)":
        ad = next((a for a in adapters if a["name"] == selected_adapter_name), None)
        if ad:
            active_adapter_path = ad["path"]

    # Header Bar
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        adapter_label = f"• Adapter: <span style='color: #a78bfa; font-weight:600;'>{selected_adapter_name}</span>" if active_adapter_path else "<span style='color: #64748b;'>• Pure Base Weights</span>"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(22, 33, 56, 0.7) 0%, rgba(13, 20, 36, 0.85) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="pulsing-dot"></span>
                    <span style="color: #60a5fa; font-weight: 700; font-size: 0.95rem;">{chat_model.split('/')[-1]}</span>
                    <span style="color: #94a3b8; font-size: 0.82rem;">{adapter_label}</span>
                </div>
                <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                    Temp: {temperature} | Top-P: {top_p}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display chat history using st.chat_message
    for msg in st.session_state["chat_messages"]:
        role = msg["role"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if "telemetry" in msg:
                t = msg["telemetry"]
                st.caption(f"⏱️ {t.get('latency_sec', 0)}s • {t.get('tokens_per_sec', 0)} tokens/s • {t.get('tokens_count', 0)} tokens")

    # Chat input
    user_prompt = st.chat_input("Ask your fine-tuned model a question...")

    if user_prompt:
        # Append user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_prompt)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating response..."):
                # Ensure model is loaded
                if (
                    engine.model is None
                    or engine.current_model_id != chat_model
                    or engine.current_adapter_path != active_adapter_path
                ):
                    load_ok, load_err = engine.load_model(chat_model, adapter_path=active_adapter_path)
                    if not load_ok:
                        st.error(f"Error loading model: {load_err}")
                        return

                # Send full conversation messages
                conv_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chat_messages"]
                ]

                gen_result = engine.generate_response(
                    messages=conv_history,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

                if gen_result["success"]:
                    response_text = gen_result["response"]
                    st.markdown(response_text)
                    st.caption(
                        f"⏱️ {gen_result['latency_sec']}s • {gen_result.get('tokens_per_sec', 0)} tokens/s • {gen_result.get('tokens_count', 0)} tokens"
                    )

                    # Save to chat history
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "telemetry": {
                            "latency_sec": gen_result["latency_sec"],
                            "tokens_per_sec": gen_result.get("tokens_per_sec", 0),
                            "tokens_count": gen_result.get("tokens_count", 0),
                        },
                    })
                else:
                    st.error(f"Inference error: {gen_result['response']}")


if __name__ == "__main__":
    render()

