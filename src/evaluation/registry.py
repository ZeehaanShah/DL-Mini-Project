"""
Experiment tracking registry utility for DL MiniProject.

Manages experiments/experiment_registry.csv with a consistent, validated schema:
- experiment_id
- timestamp
- dataset
- task
- model_name
- partition
- features
- target
- sequence_length
- forecast_horizon
- config
- random_seed
- n_samples
- mae
- rmse
- r2
- metric_unit
- status
- notes
"""
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
import pandas as pd

REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "dataset",
    "task",
    "model_name",
    "partition",
    "features",
    "target",
    "sequence_length",
    "forecast_horizon",
    "config",
    "random_seed",
    "n_samples",
    "mae",
    "rmse",
    "r2",
    "metric_unit",
    "status",
    "notes",
]


class ExperimentRegistry:
    """Manages experiment records in CSV format."""

    def __init__(self, registry_path: Union[str, Path] = "experiments/experiment_registry.csv"):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create empty CSV with header if missing."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            df = pd.DataFrame(columns=REGISTRY_COLUMNS)
            df.to_csv(self.path, index=False)
        else:
            # Check existing columns; re-initialize if empty or incompatible
            existing = pd.read_csv(self.path)
            if len(existing) == 0 and set(existing.columns) != set(REGISTRY_COLUMNS):
                df = pd.DataFrame(columns=REGISTRY_COLUMNS)
                df.to_csv(self.path, index=False)

    def log_experiment(
        self,
        experiment_id: str,
        dataset: str,
        task: str,
        model_name: str,
        partition: str,
        features: str,
        target: str,
        n_samples: int,
        mae: float,
        rmse: float,
        r2: float,
        metric_unit: str,
        sequence_length: int = 0,
        forecast_horizon: int = 0,
        config: str = "",
        random_seed: int = 42,
        status: str = "COMPLETED",
        notes: str = "",
        timestamp: Optional[str] = None,
    ) -> Dict[str, Union[str, int, float]]:
        """Append an experiment record to the registry CSV."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "dataset": dataset,
            "task": task,
            "model_name": model_name,
            "partition": partition,
            "features": features,
            "target": target,
            "sequence_length": int(sequence_length),
            "forecast_horizon": int(forecast_horizon),
            "config": config,
            "random_seed": int(random_seed),
            "n_samples": int(n_samples),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4) if pd.notna(r2) else "nan",
            "metric_unit": metric_unit,
            "status": status,
            "notes": notes,
        }

        # Read, append, and save
        if self.path.exists() and self.path.stat().st_size > 0:
            df = pd.read_csv(self.path)
            # Remove any existing row with same experiment_id to allow idempotent re-runs
            if not df.empty:
                df = df[df["experiment_id"] != experiment_id]
            if df.empty:
                new_df = pd.DataFrame([record], columns=REGISTRY_COLUMNS)
            else:
                new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            new_df = pd.DataFrame([record], columns=REGISTRY_COLUMNS)

        new_df.to_csv(self.path, index=False)
        return record

    def get_records(self) -> pd.DataFrame:
        """Read all registered records as a DataFrame."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=REGISTRY_COLUMNS)
        return pd.read_csv(self.path)
