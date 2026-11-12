"""SQLite experiment tracking database for LLM Fine-Tuning Studio."""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.file_manager import get_project_root, load_config


class ExperimentTracker:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cfg = load_config()
            rel_path = cfg.get("storage", {}).get("database_path", "database/experiments.db")
            self.db_path = str((get_project_root() / rel_path).resolve())
        else:
            self.db_path = db_path

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    method TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_samples INTEGER DEFAULT 0,
                    hyperparameters TEXT DEFAULT '{}',
                    status TEXT NOT NULL,
                    duration_seconds REAL DEFAULT 0.0,
                    final_train_loss REAL,
                    final_val_loss REAL,
                    metrics TEXT DEFAULT '{}',
                    adapter_path TEXT,
                    log_file_path TEXT,
                    hardware_info TEXT DEFAULT '{}'
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS loss_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    learning_rate REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def create_experiment(self, exp_data: Dict[str, Any]) -> str:
        """Insert a new experiment record."""
        exp_id = exp_data.get("id") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        created_at = exp_data.get("created_at") or datetime.now().isoformat()

        hp_str = json.dumps(exp_data.get("hyperparameters", {}))
        metrics_str = json.dumps(exp_data.get("metrics", {}))
        hw_str = json.dumps(exp_data.get("hardware_info", {}))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (
                    id, name, created_at, model_id, backend, method,
                    dataset_name, dataset_samples, hyperparameters,
                    status, duration_seconds, final_train_loss, final_val_loss,
                    metrics, adapter_path, log_file_path, hardware_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    exp_data.get("name", exp_id),
                    created_at,
                    exp_data.get("model_id", "unknown"),
                    exp_data.get("backend", "mlx"),
                    exp_data.get("method", "lora"),
                    exp_data.get("dataset_name", "custom"),
                    int(exp_data.get("dataset_samples", 0)),
                    hp_str,
                    exp_data.get("status", "queued"),
                    float(exp_data.get("duration_seconds", 0.0)),
                    exp_data.get("final_train_loss"),
                    exp_data.get("final_val_loss"),
                    metrics_str,
                    exp_data.get("adapter_path"),
                    exp_data.get("log_file_path"),
                    hw_str,
                ),
            )
            conn.commit()
        return exp_id

    def update_experiment(self, exp_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of an existing experiment."""
        fields = []
        values = []

        for key, val in updates.items():
            if key in ["hyperparameters", "metrics", "hardware_info"] and isinstance(val, (dict, list)):
                fields.append(f"{key} = ?")
                values.append(json.dumps(val))
            else:
                fields.append(f"{key} = ?")
                values.append(val)

        if not fields:
            return False

        values.append(exp_id)
        query = f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(values))
            conn.commit()
            return cursor.rowcount > 0

    def record_loss_step(
        self,
        exp_id: str,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Record an intermediate training loss step."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO loss_history (experiment_id, step, train_loss, val_loss, learning_rate, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (exp_id, step, train_loss, val_loss, learning_rate, datetime.now().isoformat()),
            )
            conn.commit()

    def get_loss_history(self, exp_id: str) -> List[Dict[str, Any]]:
        """Retrieve recorded loss history for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step, train_loss, val_loss, learning_rate, timestamp FROM loss_history WHERE experiment_id = ? ORDER BY step ASC",
                (exp_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single experiment by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            for json_field in ["hyperparameters", "metrics", "hardware_info"]:
                if res.get(json_field):
                    try:
                        res[json_field] = json.loads(res[json_field])
                    except Exception:
                        pass
            return res

    def list_experiments(
        self,
        status: Optional[str] = None,
        method: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments"
        conditions = []
        params = []

        if status and status != "All":
            conditions.append("status = ?")
            params.append(status.lower())
        if method and method != "All":
            conditions.append("method = ?")
            params.append(method.lower())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                for f in ["hyperparameters", "metrics", "hardware_info"]:
                    if item.get(f):
                        try:
                            item[f] = json.loads(item[f])
                        except Exception:
                            pass
                results.append(item)
            return results

    def delete_experiment(self, exp_id: str) -> bool:
        """Safely delete experiment metadata without touching base models."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM loss_history WHERE experiment_id = ?", (exp_id,))
            cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_summary_stats(self) -> Dict[str, Any]:
        """Aggregate stats for dashboard overview."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'completed'")
            completed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'running'")
            running = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'failed'")
            failed = cursor.fetchone()[0]

            cursor.execute(
                "SELECT MIN(final_train_loss) FROM experiments WHERE status = 'completed' AND final_train_loss IS NOT NULL"
            )
            min_loss_row = cursor.fetchone()
            best_loss = min_loss_row[0] if min_loss_row and min_loss_row[0] is not None else None

            cursor.execute(
                "SELECT AVG(duration_seconds) FROM experiments WHERE status = 'completed'"
            )
            avg_dur_row = cursor.fetchone()
            avg_duration = avg_dur_row[0] if avg_dur_row and avg_dur_row[0] is not None else 0.0

            cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1")
            last_run = cursor.fetchone()
            last_activity = dict(last_run) if last_run else None

            return {
                "total_experiments": total,
                "completed_experiments": completed,
                "running_experiments": running,
                "failed_experiments": failed,
                "best_train_loss": round(best_loss, 4) if best_loss is not None else None,
                "avg_duration_sec": round(avg_duration, 1),
                "last_activity": last_activity,
            }

