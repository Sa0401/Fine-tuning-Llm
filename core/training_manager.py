"""Background training manager and job execution engine.

Executes training jobs in non-blocking background processes, captures real logs,
parses live loss metrics, tracks progress, and allows safe cancellation.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import psutil
from core.experiment_tracker import ExperimentTracker
from training.cuda_lora import CUDALoRATrainer
from training.cuda_qlora import CUDAQLoRATrainer
from training.mlx_lora import MLXLoRATrainer
from training.mlx_qlora import MLXQLoRATrainer
from utils.file_manager import get_project_root, load_config
from utils.hardware import detect_hardware
from utils.logging_config import setup_logger
from utils.validation import validate_hyperparameters

logger = setup_logger("training_manager")


class TrainingManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TrainingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.root = get_project_root()
        self.config = load_config()
        self.experiments_dir = self.root / self.config.get("storage", {}).get("experiments_dir", "experiments")
        self.adapters_dir = self.root / self.config.get("adapters_dir", "models/adapters")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker()
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self._monitor_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._python_bin = sys.executable
        self._initialized = True

    def get_python_binary(self) -> str:
        """Find the virtualenv python binary across macOS, Linux, and Windows."""
        # Windows virtual environment path
        windows_py = self.root / "venv" / "Scripts" / "python.exe"
        if windows_py.exists():
            return str(windows_py)

        # macOS / Linux virtual environment path
        posix_py = self.root / "venv" / "bin" / "python3"
        if posix_py.exists():
            return str(posix_py)

        posix_py_alt = self.root / "venv" / "bin" / "python"
        if posix_py_alt.exists():
            return str(posix_py_alt)

        return sys.executable

    def start_training(
        self,
        name: str,
        model_id: str,
        backend: str,  # 'mlx' or 'cuda'
        method: str,   # 'lora' or 'qlora'
        dataset_path: str,
        dataset_name: str,
        dataset_samples: int,
        hyperparameters: Dict[str, Any],
    ) -> Tuple[bool, str, Optional[str]]:
        """Validate, configure, and launch a training job in the background."""
        # 1. Validation
        is_valid, errors = validate_hyperparameters(hyperparameters, method=method)
        if not is_valid:
            return False, f"Validation failed: {'; '.join(errors)}", None

        # Check dataset existence
        data_dir_path = Path(dataset_path)
        if not (data_dir_path / "train.jsonl").exists():
            return False, f"Missing train.jsonl in dataset directory: {dataset_path}", None

        # 2. Check hardware compatibility
        hw = detect_hardware()
        if backend == "cuda" and not hw.cuda_available:
            return False, "Cannot run CUDA backend: No NVIDIA CUDA GPU detected on this system. Choose CPU or MLX.", None
        if backend == "mlx" and not hw.mlx_available:
            return False, "Cannot run MLX backend: MLX is only supported on Apple Silicon macOS. On Windows or Linux, please choose CUDA or CPU.", None

        # 3. Create unique experiment ID and directories
        slug_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip().lower())
        exp_id = f"{slug_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        exp_dir = self.experiments_dir / exp_id
        adapter_output = self.adapters_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        adapter_output.mkdir(parents=True, exist_ok=True)

        log_file = exp_dir / "training.log"
        state_file = exp_dir / "state.json"

        # 4. Initialize Trainer based on backend and method
        python_bin = self.get_python_binary()

        try:
            if backend == "mlx":
                if method == "qlora":
                    trainer = MLXQLoRATrainer(exp_id, str(adapter_output.resolve()))
                else:
                    trainer = MLXLoRATrainer(exp_id, str(adapter_output.resolve()))
                cfg_path = trainer.prepare_config(model_id, str(data_dir_path.resolve()), hyperparameters)
                cmd = trainer.build_command(python_bin)
            else:
                if method == "qlora":
                    trainer = CUDAQLoRATrainer(exp_id, str(adapter_output.resolve()))
                else:
                    trainer = CUDALoRATrainer(exp_id, str(adapter_output.resolve()))
                script_path = trainer.prepare_script(model_id, str(data_dir_path.resolve()), hyperparameters)
                cmd = trainer.build_command(python_bin)
        except Exception as e:
            return False, f"Failed to prepare training configuration: {str(e)}", None

        # 5. Insert record into database
        exp_record = {
            "id": exp_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "model_id": model_id,
            "backend": backend,
            "method": method,
            "dataset_name": dataset_name,
            "dataset_samples": dataset_samples,
            "hyperparameters": hyperparameters,
            "status": "running",
            "duration_seconds": 0.0,
            "final_train_loss": None,
            "final_val_loss": None,
            "metrics": {},
            "adapter_path": str(adapter_output.resolve()),
            "log_file_path": str(log_file.resolve()),
            "hardware_info": {
                "cpu": hw.cpu_brand,
                "ram_total_gb": hw.total_memory_gb,
                "backend": backend,
                "is_apple_silicon": hw.is_apple_silicon,
            },
        }
        self.tracker.create_experiment(exp_record)

        # 6. Launch background process
        try:
            log_handle = open(log_file, "w", encoding="utf-8")
            log_handle.write(f"=== Training Job Started: {datetime.now().isoformat()} ===\n")
            log_handle.write(f"Command: {' '.join(cmd)}\n")
            log_handle.write("=" * 60 + "\n\n")
            log_handle.flush()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            preexec = None
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            elif hasattr(os, "setsid"):
                preexec = os.setsid

            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(self.root.resolve()),
                env=env,
                preexec_fn=preexec,
                creationflags=creationflags,
            )

            self.active_processes[exp_id] = proc

            # Save state file
            state_data = {
                "exp_id": exp_id,
                "pid": proc.pid,
                "status": "running",
                "start_time": time.time(),
                "log_file": str(log_file.resolve()),
                "adapter_path": str(adapter_output.resolve()),
                "total_iters": int(hyperparameters.get("iterations", 100)),
                "current_iter": 0,
                "train_loss": None,
                "val_loss": None,
            }
            with open(state_file, "w", encoding="utf-8") as sf:
                json.dump(state_data, sf, indent=2)

            # 7. Start monitor thread
            stop_event = threading.Event()
            self._stop_events[exp_id] = stop_event
            monitor_thread = threading.Thread(
                target=self._monitor_job,
                args=(exp_id, proc, log_handle, log_file, state_file, stop_event),
                daemon=True,
            )
            self._monitor_threads[exp_id] = monitor_thread
            monitor_thread.start()

            return True, f"Training job '{name}' started successfully (PID: {proc.pid})", exp_id

        except Exception as e:
            self.tracker.update_experiment(exp_id, {"status": "failed"})
            return False, f"Failed to launch training process: {str(e)}", None

    def _monitor_job(
        self,
        exp_id: str,
        proc: subprocess.Popen,
        log_handle: Any,
        log_file: Path,
        state_file: Path,
        stop_event: threading.Event,
    ) -> None:
        """Monitor logs, parse metrics, and update database until process finishes."""
        start_time = time.time()
        last_read_pos = 0

        # Regex for MLX-LM outputs:
        # Format A (MLX-LM 0.32+ Rich table):
        # "     1    val 3.238    1.45s"
        # "     5    2.974 ▼       85      0.5k"
        # "    10    1.069 ▼    1,113      1.1k"
        mlx_val_pattern = re.compile(r"^\s*(\d+)\s+val\s+([\d\.]+)", re.MULTILINE)
        mlx_train_pattern = re.compile(r"^\s*(\d+)\s+([\d\.]+)\s*(?:▼|▲)?\s+[\d,]+", re.MULTILINE)

        # Format B (Classic MLX-LM):
        classic_train_pattern = re.compile(
            r"Iter\s+(\d+):\s+Train loss\s+([\d\.]+)(?:,\s+Learning Rate\s+([\d\.eE\-+]+))?",
            re.IGNORECASE,
        )
        classic_val_pattern = re.compile(
            r"Iter\s+(\d+):\s+Val loss\s+([\d\.]+)",
            re.IGNORECASE,
        )

        # Format C (Hugging Face Trainer / Windows CUDA):
        hf_train_pattern = re.compile(
            r"\{'loss':\s*([\d\.]+)(?:,\s*'grad_norm'[^}]*)?,\s*'learning_rate':\s*([\d\.eE\-+]+)",
            re.IGNORECASE,
        )
        hf_eval_pattern = re.compile(
            r"\{'eval_loss':\s*([\d\.]+)",
            re.IGNORECASE,
        )
        hf_step_pattern = re.compile(
            r"'epoch':\s*[\d\.]+(?:,\s*'step':\s*(\d+))?",
            re.IGNORECASE,
        )

        last_train_loss = None
        last_val_loss = None
        current_iter = 0

        while not stop_event.is_set():
            retcode = proc.poll()

            # Read new log lines
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_read_pos)
                        new_content = f.read()
                        last_read_pos = f.tell()

                        if new_content:
                            for raw_line in new_content.splitlines():
                                line = raw_line.strip()
                                if not line:
                                    continue

                                # 1. Check MLX rich val pattern
                                m_mlx_v = mlx_val_pattern.match(line)
                                if m_mlx_v:
                                    step = int(m_mlx_v.group(1))
                                    v_loss = float(m_mlx_v.group(2))
                                    last_val_loss = v_loss
                                    current_iter = max(current_iter, step)
                                    self.tracker.record_loss_step(exp_id, step=step, val_loss=v_loss)
                                    continue

                                # 2. Check MLX rich train pattern
                                m_mlx_t = mlx_train_pattern.match(line)
                                if m_mlx_t:
                                    step = int(m_mlx_t.group(1))
                                    t_loss = float(m_mlx_t.group(2))
                                    last_train_loss = t_loss
                                    current_iter = max(current_iter, step)
                                    self.tracker.record_loss_step(exp_id, step=step, train_loss=t_loss)
                                    continue

                                # 3. Classic train pattern
                                m_train = classic_train_pattern.search(line)
                                if m_train:
                                    step = int(m_train.group(1))
                                    t_loss = float(m_train.group(2))
                                    lr = float(m_train.group(3)) if m_train.group(3) else None
                                    current_iter = max(current_iter, step)
                                    last_train_loss = t_loss
                                    self.tracker.record_loss_step(
                                        exp_id, step=step, train_loss=t_loss, learning_rate=lr
                                    )
                                    continue

                                # 4. Classic val pattern
                                m_val = classic_val_pattern.search(line)
                                if m_val:
                                    step = int(m_val.group(1))
                                    v_loss = float(m_val.group(2))
                                    last_val_loss = v_loss
                                    current_iter = max(current_iter, step)
                                    self.tracker.record_loss_step(
                                        exp_id, step=step, val_loss=v_loss
                                    )
                                    continue

                                # 5. Hugging Face Trainer pattern (Windows CUDA/CPU)
                                m_hf_t = hf_train_pattern.search(line)
                                if m_hf_t:
                                    t_loss = float(m_hf_t.group(1))
                                    lr = float(m_hf_t.group(2)) if m_hf_t.group(2) else None
                                    current_iter += 5
                                    last_train_loss = t_loss
                                    self.tracker.record_loss_step(
                                        exp_id, step=current_iter, train_loss=t_loss, learning_rate=lr
                                    )
                                    continue

                                m_hf_v = hf_eval_pattern.search(line)
                                if m_hf_v:
                                    v_loss = float(m_hf_v.group(1))
                                    last_val_loss = v_loss
                                    self.tracker.record_loss_step(
                                        exp_id, step=current_iter, val_loss=v_loss
                                    )
                                    continue
                except Exception:
                    pass

            # Update state file
            try:
                if state_file.exists():
                    with open(state_file, "r", encoding="utf-8") as sf:
                        st_data = json.load(sf)
                    st_data["current_iter"] = current_iter
                    st_data["train_loss"] = last_train_loss
                    st_data["val_loss"] = last_val_loss
                    st_data["elapsed_sec"] = round(time.time() - start_time, 1)
                    with open(state_file, "w", encoding="utf-8") as sf:
                        json.dump(st_data, sf, indent=2)
            except Exception:
                pass

            if retcode is not None:
                # Process has completed
                duration = round(time.time() - start_time, 2)
                final_status = "completed" if retcode == 0 else "failed"

                updates = {
                    "status": final_status,
                    "duration_seconds": duration,
                    "final_train_loss": last_train_loss,
                    "final_val_loss": last_val_loss,
                }
                self.tracker.update_experiment(exp_id, updates)

                # Write footer in state file
                try:
                    if state_file.exists():
                        with open(state_file, "r", encoding="utf-8") as sf:
                            st_data = json.load(sf)
                        st_data["status"] = final_status
                        st_data["exit_code"] = retcode
                        st_data["duration_sec"] = duration
                        with open(state_file, "w", encoding="utf-8") as sf:
                            json.dump(st_data, sf, indent=2)
                except Exception:
                    pass

                log_handle.close()
                if exp_id in self.active_processes:
                    del self.active_processes[exp_id]
                break

            time.sleep(1.0)

    def cancel_training(self, exp_id: str) -> Tuple[bool, str]:
        """Safely terminate a running training job."""
        proc = self.active_processes.get(exp_id)
        pid = None

        if proc is not None:
            pid = proc.pid
        else:
            # Check state file
            state_file = self.experiments_dir / exp_id / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r", encoding="utf-8") as sf:
                        data = json.load(sf)
                    pid = data.get("pid")
                except Exception:
                    pass

        if not pid:
            self.tracker.update_experiment(exp_id, {"status": "stopped"})
            return False, "Could not find active process PID for job."

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
            parent.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        except Exception:
            try:
                if hasattr(signal, "SIGKILL"):
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        if exp_id in self._stop_events:
            self._stop_events[exp_id].set()
        if exp_id in self.active_processes:
            del self.active_processes[exp_id]

        self.tracker.update_experiment(exp_id, {"status": "stopped"})

        # Update state file
        state_file = self.experiments_dir / exp_id / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as sf:
                    st_data = json.load(sf)
                st_data["status"] = "stopped"
                with open(state_file, "w", encoding="utf-8") as sf:
                    json.dump(st_data, sf, indent=2)
            except Exception:
                pass

        return True, f"Training job '{exp_id}' successfully stopped."

    def get_job_status(self, exp_id: str) -> Dict[str, Any]:
        """Fetch real-time status, telemetry, and recent logs for a specific experiment."""
        exp = self.tracker.get_experiment(exp_id)
        if not exp:
            return {"status": "not_found", "logs": "", "progress": 0.0}

        log_path = Path(exp.get("log_file_path", ""))
        recent_logs = ""
        if log_path.exists():
            try:
                # Read last 80 lines
                with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                    lines = lf.readlines()
                    recent_logs = "".join(lines[-80:])
            except Exception:
                recent_logs = "Error reading log file."

        # Read state file
        state_file = self.experiments_dir / exp_id / "state.json"
        state_info = {}
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as sf:
                    state_info = json.load(sf)
            except Exception:
                pass

        # Progress calculation
        hp = exp.get("hyperparameters", {})
        total_iters = int(hp.get("iterations", 100))
        current_iter = state_info.get("current_iter", 0)
        progress = min(1.0, current_iter / total_iters) if total_iters > 0 else 0.0

        return {
            "id": exp_id,
            "name": exp.get("name"),
            "status": exp.get("status"),
            "model_id": exp.get("model_id"),
            "method": exp.get("method"),
            "current_iter": current_iter,
            "total_iters": total_iters,
            "progress": progress,
            "train_loss": state_info.get("train_loss") or exp.get("final_train_loss"),
            "val_loss": state_info.get("val_loss") or exp.get("final_val_loss"),
            "elapsed_sec": state_info.get("elapsed_sec") or exp.get("duration_seconds", 0.0),
            "logs": recent_logs,
            "adapter_path": exp.get("adapter_path"),
            "is_active": exp.get("status") == "running",
        }
