"""
Standardized training and evaluation loop for regression deep learning models.

Provides:
- RegressionTrainer: Manages epoch loops, metric tracking, early stopping, and checkpointing.
- Lookahead-free and unit-safe evaluation (supports target unscaling for physical units).
- JSON history export and PyTorch checkpoint management.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_regression_metrics


class RegressionTrainer:
    """
    Manages model training, validation, early stopping, and metric logging.

    Attributes:
        model: PyTorch neural network module.
        optimizer: PyTorch optimizer.
        criterion: Loss function (default: nn.MSELoss()).
        device: Target torch device ('cpu' or 'cuda').
        experiment_id: Unique identifier string for tracking.
        checkpoint_dir: Directory where model weights and histories are saved.
        early_stopping_patience: Number of epochs without improvement before stopping.
        target_unscaler: Optional callable to transform predictions back into physical units.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        experiment_id: str = "EXP_01",
        checkpoint_dir: Union[str, Path] = "checkpoints",
        early_stopping_patience: int = 5,
        min_delta: float = 1e-4,
        target_unscaler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        grad_clip_norm: Optional[float] = 5.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cpu")
        self.experiment_id = experiment_id
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        self.target_unscaler = target_unscaler
        self.grad_clip_norm = grad_clip_norm

        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch across dataloader and return mean batch loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x, y = batch[0], batch[1]
            x = x.to(self.device)
            y = y.to(self.device).float()

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            self.optimizer.zero_grad()
            preds = self.model(x)

            if preds.shape != y.shape:
                preds = preds.view_as(y)

            loss = self.criterion(preds, y)
            loss.backward()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation dataloader.

        Returns:
            Dictionary with 'loss', 'mae', 'rmse', 'r2', and 'n_samples'.
            Metrics are evaluated in physical units if target_unscaler is configured.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0], batch[1]
                x = x.to(self.device)
                y = y.to(self.device).float()

                if y.ndim == 1:
                    y = y.unsqueeze(-1)

                preds = self.model(x)
                if preds.shape != y.shape:
                    preds = preds.view_as(y)

                loss = self.criterion(preds, y)
                total_loss += float(loss.item())
                n_batches += 1

                y_true_list.append(y.detach().cpu().numpy().ravel())
                y_pred_list.append(preds.detach().cpu().numpy().ravel())

        mean_loss = total_loss / max(n_batches, 1)

        y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
        y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])

        # Unscale to physical units if configured (e.g. normalized weather features -> Wh/m²)
        if self.target_unscaler is not None and len(y_true) > 0:
            y_true_eval = self.target_unscaler(y_true)
            y_pred_eval = self.target_unscaler(y_pred)
        else:
            y_true_eval = y_true
            y_pred_eval = y_pred

        metrics = calculate_regression_metrics(y_true_eval, y_pred_eval)

        return {
            "loss": round(mean_loss, 6),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "n_samples": metrics["n_samples"],
        }

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for all samples in dataloader.

        Returns:
            Tuple of (y_true_physical, y_pred_physical).
        """
        self.model.eval()
        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0], batch[1]
                x = x.to(self.device)
                preds = self.model(x)

                y_true_list.append(y.detach().cpu().numpy().ravel())
                y_pred_list.append(preds.detach().cpu().numpy().ravel())

        y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
        y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])

        if self.target_unscaler is not None and len(y_true) > 0:
            return self.target_unscaler(y_true), self.target_unscaler(y_pred)

        return y_true, y_pred

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Maximum epoch budget.
            verbose: If True, prints per-epoch progress to console.

        Returns:
            Dictionary containing best metrics, training history, and duration.
        """
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = -1
        best_metrics: Dict[str, float] = {}
        patience_counter = 0
        early_stopped = False

        start_time = time.time()
        best_model_path = self.checkpoint_dir / "best_model.pt"

        if verbose:
            print(f"\n--- Starting Experiment: {self.experiment_id} ---")
            print(f"Device: {self.device} | Max Epochs: {max_epochs} | Early Stopping Patience: {self.early_stopping_patience}")

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # Train epoch
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_eval = self.evaluate(val_loader)
            val_loss = val_eval["loss"]
            val_mae = val_eval["mae"]
            val_rmse = val_eval["rmse"]
            val_r2 = val_eval["r2"]

            # Current learning rate
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_duration = time.time() - epoch_start

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "lr": current_lr,
                "epoch_duration_sec": round(epoch_duration, 2),
            }
            history.append(record)

            if verbose:
                r2_str = f"{val_r2:.4f}" if np.isfinite(val_r2) else "nan"
                print(
                    f"Epoch {epoch:02d}/{max_epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R^2: {r2_str} | "
                    f"Time: {epoch_duration:.1f}s"
                )

            # Checkpoint saving on improvement
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = val_eval
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_mae": val_mae,
                        "val_rmse": val_rmse,
                        "val_r2": val_r2,
                        "experiment_id": self.experiment_id,
                    },
                    best_model_path,
                )
                if verbose:
                    print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"  [Early Stopping Triggered] No improvement for {self.early_stopping_patience} epochs.")
                    early_stopped = True
                    break

        total_duration = time.time() - start_time

        # Load best checkpoint weights back into model
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if verbose:
                print(f"Restored best weights from epoch {best_epoch} ({best_model_path})")

        # Save training history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": self.experiment_id,
                    "total_duration_sec": round(total_duration, 2),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_metrics": best_metrics,
                    "early_stopped": early_stopped,
                    "history": history,
                },
                f,
                indent=2,
            )

        return {
            "experiment_id": self.experiment_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "total_duration_sec": round(total_duration, 2),
            "early_stopped": early_stopped,
            "history": history,
            "checkpoint_path": str(best_model_path),
            "history_path": str(history_path),
        }
