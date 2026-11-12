# 🧠 LLM Fine-Tuning Studio – LoRA & QLoRA (macOS & Windows)

A complete, modular, production-grade LLM Fine-Tuning Studio built with **Python**, **Streamlit**, and **Apple MLX-LM** (for macOS Apple Silicon) and **Hugging Face PEFT + BitsAndBytes** (for Windows / Linux with NVIDIA CUDA or CPU).

Engineered to run seamlessly across both **macOS (Apple Silicon M-Series)** and **Microsoft Windows (10/11 with NVIDIA CUDA or CPU)**.

---

## 🚀 Key Features

- **🍎 macOS Native (Apple Silicon MLX & MLX-LM):**
  - Zero-copy unified memory utilization with native Metal GPU acceleration.
  - Native 4-bit grouped affine quantization (`QuantizedLinear`) without requiring Linux-only `bitsandbytes`.
  - Keeps 16GB memory consumption strictly under control.
- **🪟 Windows Native (NVIDIA CUDA & CPU Mode):**
  - Full Hugging Face **PEFT** + **Transformers** + **BitsAndBytes** 4-bit NormalFloat4 (NF4) QLoRA.
  - Automatic fallback to CPU PEFT mode if no NVIDIA GPU is detected.
  - Multi-process safe process group management (`CREATE_NEW_PROCESS_GROUP`).
- **📁 Dataset Management & Cleaning:** Upload JSONL, JSON, CSV, or TXT datasets. Automatic deduplication, empty record filtering, token statistics visualization, and train/val/test splitting in MLX-LM / Hugging Face format.
- **📡 Background Training Manager:** Non-blocking asynchronous background execution (`subprocess.Popen`), capturing real-time stdout/stderr, live loss extraction, and safe job cancellation on both OSs.
- **📈 Interactive Training Monitor:** Live Plotly loss convergence curves, progress bars, hardware telemetry, and real-time terminal output streaming.
- **⚖️ Side-by-Side Model Comparison:** Compare raw base models against fine-tuned adapters on identical test prompts with latency, speed, and qualitative response analysis.
- **📊 Quantitative Evaluation Engine:** Measures ROUGE-1, ROUGE-2, and ROUGE-L scores, perplexity estimates, token throughput, and memory consumption.
- **💬 Conversational Chatbot Playground:** Chat with fine-tuned models using `st.chat_message` and `st.chat_input`, complete with runtime speed telemetry and cached weight inference.
- **🔀 Adapter Fusing & Deployment:** Mathematically fold trained LoRA adapters back into base model weights for zero-latency deployment, or serve via the included OpenAI-compatible **FastAPI** server.

---

## 🏗️ Architecture & Project Structure

```text
LLM-Fine-Tuning-Studio/
├── app.py                      # Main Streamlit application entry point & navigation
├── config.yaml                 # Studio defaults, paths, supported models catalog
├── requirements.txt            # Cross-platform dependencies (environment markers for macOS & Windows)
├── README.md                   # Comprehensive documentation and developer guide
├── serve_api.py                # Standalone FastAPI server for OpenAI-compatible REST API
│
├── run_studio.sh               # macOS / Linux one-click launcher
├── run_studio.bat              # Windows one-click launcher
├── serve_api.sh                # macOS / Linux API server launcher
├── serve_api.bat               # Windows API server launcher
│
├── pages/                      # Modular Streamlit Pages (st.navigation)
│   ├── dashboard.py            # Overview cards, telemetry, and training history
│   ├── model_selection.py      # Catalog & local model discovery, HF downloader
│   ├── dataset_manager.py      # Uploads, validation, token length stats, splitting
│   ├── lora_training.py        # Standard LoRA configuration and job launcher
│   ├── qlora_training.py       # 4-bit QLoRA configuration and launcher
│   ├── training_monitor.py     # Live loss curves, terminal logs, and safe cancellation
│   ├── evaluation.py           # ROUGE scoring, latency, and test suite benchmarking
│   ├── comparison.py           # Side-by-side prompt testing against base models
│   ├── playground.py           # Conversational chat interface (st.chat_input)
│   ├── experiment_history.py   # SQLite experiment explorer and Markdown report export
│   └── settings.py             # Hardware diagnostics, adapter management & fusing
│
├── core/                       # Core Studio Business Logic & Engines
│   ├── model_loader.py         # Model discovery, inspection, and HF hub download
│   ├── dataset_processor.py    # Schema normalization, cleaning, token stats, splitting
│   ├── training_manager.py     # Background process execution, log parsing, persistence
│   ├── evaluation_engine.py    # ROUGE-1/2/L, latency, and memory evaluation
│   ├── inference_engine.py     # MLX-LM & Transformers inference with memory caching
│   ├── experiment_tracker.py   # SQLite database CRUD operations and stats aggregation
│   └── model_exporter.py       # Adapter fusing (mlx_lm.fuse & PEFT merge) and reports
│
├── training/                   # Hardware-Specific Training Backends
│   ├── mlx_lora.py             # MLX-LM standard LoRA backend (macOS)
│   ├── mlx_qlora.py            # MLX-LM 4-bit quantized LoRA backend (macOS)
│   ├── cuda_lora.py            # Hugging Face PEFT LoRA backend (Windows / Linux)
│   └── cuda_qlora.py           # Hugging Face PEFT + BitsAndBytes QLoRA (Windows / Linux)
│
├── utils/                      # Helper Utilities
│   ├── hardware.py             # macOS & Windows hardware, Metal/CUDA, RAM telemetry
│   ├── validation.py           # Dataset schema and hyperparameter bounds validation
│   ├── logging_config.py       # Centralized structured logging
│   └── file_manager.py         # Directory management, file sizes, adapter listing
│
├── sample_datasets/            # Pre-loaded sample instruction & chat datasets
├── data/                       # Prepared train/valid/test datasets
├── models/base/                # Downloaded base model weights
├── models/adapters/            # Trained adapter checkpoints
└── database/                   # SQLite database (experiments.db)
```

---

## 💻 Hardware Detection & OS Differences

### Platform Comparison Matrix
| Feature | macOS (Apple Silicon) | Windows (NVIDIA CUDA) | Windows (CPU Fallback) |
|---|---|---|---|
| **Primary Backend** | Apple MLX & MLX-LM | Hugging Face PEFT + TRL | Hugging Face PEFT |
| **Quantization Format** | Native 4-bit Grouped Affine | NormalFloat4 (NF4) | 8-bit / Float32 |
| **Quantization Engine** | Apple MLX Core (`QuantizedLinear`) | `bitsandbytes` | Torch Float32 / BFloat16 |
| **Acceleration** | Metal Shaders (UMA) | CUDA Tensor Cores | Multi-threaded CPU AVX2/512 |
| **Adapter Fusing** | `mlx_lm.fuse` | `peft.merge_and_unload()` | `peft.merge_and_unload()` |
| **Virtualenv Path** | `venv/bin/python3` | `venv\Scripts\python.exe` | `venv\Scripts\python.exe` |
| **Process Management** | `os.setsid` & `psutil` | `CREATE_NEW_PROCESS_GROUP` | `CREATE_NEW_PROCESS_GROUP` |

---

## 🛠️ Installation & Setup

### A. On Windows (10 / 11)

#### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Install from python.org, check "Add Python to PATH")
- (Recommended) NVIDIA GPU with latest NVIDIA Game Ready or Studio Drivers + CUDA

#### 2. One-Click Launch (Windows)
Double-click `run_studio.bat` or run in Command Prompt / PowerShell:
```cmd
run_studio.bat
```
This script will automatically create a virtual environment, install the Windows-compatible dependencies (including `bitsandbytes`, `transformers`, `torch`), and start the Studio at `http://localhost:8501`.

#### 3. Manual Installation (Windows)
```cmd
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

### B. On macOS (Apple Silicon M1/M2/M3/M4/M5)

#### 1. Prerequisites
- macOS Sonoma or later
- Python 3.11
- Apple Silicon (M1 or later with 16 GB unified memory recommended)

#### 2. One-Click Launch (macOS)
```bash
chmod +x run_studio.sh
./run_studio.sh
```

#### 3. Manual Installation (macOS)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

## 📖 Step-by-Step Fine-Tuning Workflow

1. **Model Selection:**
   - On macOS: Choose `mlx-community/Qwen2.5-0.5B-Instruct-4bit` or `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
   - On Windows: Choose `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` (or any Hugging Face model ID).
   - Click **Download / Cache Weights**.

2. **Dataset Preparation:**
   - Go to **Dataset Manager**.
   - Upload your custom JSONL/JSON/CSV dataset or pick a sample from the **Pre-loaded Sample Datasets** tab.
   - Review the automatic deduplication and token length histogram.
   - Click **Split & Save Dataset for Training** (generates `train.jsonl` and `valid.jsonl`).

3. **LoRA or QLoRA Training:**
   - Navigate to **QLoRA Training** (for 4-bit models) or **LoRA Training** (for full precision).
   - The studio auto-detects your OS:
     - On macOS: Uses native MLX backend.
     - On Windows: Uses CUDA (or CPU fallback) backend.
   - Configure hyperparameters (Rank $r=8$, Alpha $=16$, Learning Rate $=2\times 10^{-4}$, Batch Size $=2\text{ to }4$).
   - Click **Start Training Job**. The process runs asynchronously in the background.

4. **Training Monitor:**
   - Watch the live loss convergence curve update in real-time.
   - Monitor elapsed time, throughput (tokens/sec), and streaming terminal logs.
   - Safely cancel any active job if needed with one click.

5. **Evaluation & Comparison:**
   - Benchmark your trained adapter in **Evaluation** to measure ROUGE-1, ROUGE-2, and ROUGE-L scores against test prompts.
   - Compare Candidate A (Base Model) vs Candidate B (Adapter) side-by-side in **Model Comparison**.

6. **Interactive Chat Playground:**
   - Go to **LLM Playground**.
   - Select your base model and attach your newly trained adapter.
   - Chat in real-time with full generation latency and token speed telemetry.

---

## 🚀 REST API Deployment (FastAPI)

Serve your fine-tuned models and adapters through an OpenAI-compatible REST API on either Windows or Mac:

### On Windows:
```cmd
serve_api.bat
```

### On macOS / Linux:
```bash
./serve_api.sh
```

### Example Curl Request:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/base/mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "adapter": "models/adapters/qlora_qwen_test2_20261112_102340",
    "messages": [
      {"role": "user", "content": "What is Parameter-Efficient Fine-Tuning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

---

## 🔀 Adapter Fusing (Merge into Base Model)

To deploy without adapter runtime dependencies:
1. Navigate to **Settings** -> **Fuse & Merge Adapters**.
2. Select the base model and the trained adapter.
3. Click **Execute Model Fuse & Merge**.
4. On macOS, uses `mlx_lm.fuse`. On Windows, uses `peft.merge_and_unload()`.
5. Outputs a standalone model directory with complete weights and tokenizer files.

---

## 📄 License
MIT License. Built for Apple Silicon (macOS) and Windows / Linux AI developers.
