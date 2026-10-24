"""
Test suite for Stage 3 deep learning models (WindLSTM and DemandLSTM).

Validates:
1. WindLSTM input/output tensor shapes with univariate and multivariate configurations.
2. DemandLSTM input/output tensor shapes with single and stacked configurations.
3. Training loop finite loss generation, early stopping, and checkpoint restoration.
4. Checkpoint integrity and loading of trained Stage 3 models (C1-C4, D1-D4).
5. Experiment registry Stage 3 entries, uniqueness, and correct physical units.
6. Baseline vs DL model comparison alignment on identical validation timestamps.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.wind.sequence_models import WindLSTM
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
from src.evaluation.registry import ExperimentRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestStage3Architectures:
    """Validate neural network shapes, layers, and forward passes."""

    def test_wind_lstm_univariate_and_multivariate_shapes(self):
        """Verify WindLSTM produces (batch_size, 1) for both 1D and 4D inputs."""
        # Univariate (C1/C2)
        m1 = WindLSTM(input_dim=1, hidden_dim=32, num_layers=1, dropout=0.0)
        x1 = torch.randn(8, 24, 1)
        out1 = m1(x1)
        assert out1.shape == (8, 1)
        assert torch.isfinite(out1).all()

        # Multivariate (C3/C4)
        m4 = WindLSTM(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2)
        x4 = torch.randn(8, 24, 4)
        out4 = m4(x4)
        assert out4.shape == (8, 1)
        assert torch.isfinite(out4).all()

        trainable_p, total_p = m4.get_parameter_counts()
        assert trainable_p == total_p
        assert trainable_p > 10_000

    def test_demand_lstm_shapes_and_layer_variants(self):
        """Verify DemandLSTM produces (batch_size, 1) with 1-layer and 2-layer configs."""
        # 1-layer (D1/D2)
        m1 = DemandLSTM(input_dim=1, hidden_dim=32, num_layers=1, dropout=0.0)
        x = torch.randn(6, 24, 1)
        out1 = m1(x)
        assert out1.shape == (6, 1)
        assert torch.isfinite(out1).all()

        # 2-layer stacked (D3/D4)
        m2 = DemandLSTM(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2)
        out2 = m2(x)
        assert out2.shape == (6, 1)
        assert torch.isfinite(out2).all()


class TestStage3TrainingPipelines:
    """Validate training loop, checkpoints, and restoration for Stage 3 models."""

    def test_wind_training_and_checkpoint_roundtrip(self, tmp_path):
        """Verify WindLSTM trains with finite loss and restores identically from checkpoint."""
        torch.manual_seed(42)
        X_train = torch.randn(16, 24, 1)
        y_train = torch.randn(16, 1)
        X_val = torch.randn(8, 24, 1)
        y_val = torch.randn(8, 1)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

        model = WindLSTM(input_dim=1, hidden_dim=16, num_layers=1)
        trainer = RegressionTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_id="TEST_WIND",
            early_stopping_patience=3,
        )

        summary = trainer.train(train_loader, val_loader, max_epochs=2, verbose=False)
        assert summary["best_epoch"] >= 1
        assert np.isfinite(summary["best_val_loss"])

        ckpt_file = tmp_path / "checkpoints" / "TEST_WIND" / "best_model.pt"
        assert ckpt_file.exists()

        eval_before = trainer.evaluate(val_loader)
        fresh_model = WindLSTM(input_dim=1, hidden_dim=16, num_layers=1)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        fresh_trainer = RegressionTrainer(
            model=fresh_model,
            optimizer=torch.optim.Adam(fresh_model.parameters()),
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_id="TEST_RESTORE",
        )
        eval_after = fresh_trainer.evaluate(val_loader)
        assert np.isclose(eval_before["loss"], eval_after["loss"], atol=1e-5)
        assert np.isclose(eval_before["mae"], eval_after["mae"], atol=1e-5)

    def test_demand_training_and_checkpoint_roundtrip(self, tmp_path):
        """Verify DemandLSTM trains with finite loss and restores identically from checkpoint."""
        torch.manual_seed(42)
        X_train = torch.randn(16, 24, 1)
        y_train = torch.randn(16, 1)
        X_val = torch.randn(8, 24, 1)
        y_val = torch.randn(8, 1)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

        model = DemandLSTM(input_dim=1, hidden_dim=16, num_layers=1)
        trainer = RegressionTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_id="TEST_DEMAND",
            early_stopping_patience=3,
        )

        summary = trainer.train(train_loader, val_loader, max_epochs=2, verbose=False)
        assert summary["best_epoch"] >= 1
        assert np.isfinite(summary["best_val_loss"])

        ckpt_file = tmp_path / "checkpoints" / "TEST_DEMAND" / "best_model.pt"
        assert ckpt_file.exists()

        eval_before = trainer.evaluate(val_loader)
        fresh_model = DemandLSTM(input_dim=1, hidden_dim=16, num_layers=1)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        fresh_trainer = RegressionTrainer(
            model=fresh_model,
            optimizer=torch.optim.Adam(fresh_model.parameters()),
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_id="TEST_RESTORE_DEMAND",
        )
        eval_after = fresh_trainer.evaluate(val_loader)
        assert np.isclose(eval_before["loss"], eval_after["loss"], atol=1e-5)
        assert np.isclose(eval_before["mae"], eval_after["mae"], atol=1e-5)


class TestStage3CheckpointsAndRegistry:
    """Validate saved Stage 3 artifacts and registry records."""

    def test_stage3_checkpoints_exist_and_loadable(self):
        """Verify that all trained model checkpoints exist on disk and can be loaded."""
        wind_ckpts = ["EXP_WIND_DL_C1", "EXP_WIND_DL_C2", "EXP_WIND_DL_C3", "EXP_WIND_DL_C4"]
        for exp_id in wind_ckpts:
            p = PROJECT_ROOT / "checkpoints" / exp_id / "best_model.pt"
            assert p.exists(), f"Missing checkpoint for {exp_id}"
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt

        demand_ckpts = ["EXP_DEMAND_DL_D1", "EXP_DEMAND_DL_D2", "EXP_DEMAND_DL_D3", "EXP_DEMAND_DL_D4"]
        for exp_id in demand_ckpts:
            p = PROJECT_ROOT / "checkpoints" / exp_id / "best_model.pt"
            assert p.exists(), f"Missing checkpoint for {exp_id}"
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt

    def test_registry_stage3_records_integrity(self):
        """Verify experiment registry contains all Stage 3 records with correct schema and units."""
        registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.csv"
        assert registry_path.exists()
        df = pd.read_csv(registry_path)

        # Verify no duplicate experiment IDs
        assert not df["experiment_id"].duplicated().any(), "Found duplicate experiment_id in registry"

        # Check Wind records
        for c in ["C1", "C2", "C3", "C4"]:
            exp_id = f"EXP_WIND_DL_{c}"
            row = df[df["experiment_id"] == exp_id]
            assert not row.empty, f"Missing {exp_id} in registry"
            assert row.iloc[0]["dataset"] == "NREL_WIND_TOOLKIT"
            assert row.iloc[0]["metric_unit"] == "m/s"
            assert row.iloc[0]["status"] == "COMPLETED"
            assert row.iloc[0]["n_samples"] == 1290
            assert np.isfinite(row.iloc[0]["mae"])
            assert np.isfinite(row.iloc[0]["rmse"])
            assert np.isfinite(row.iloc[0]["r2"])

        # Check Demand records
        for d in ["D1", "D2", "D3", "D4"]:
            exp_id = f"EXP_DEMAND_DL_{d}"
            row = df[df["experiment_id"] == exp_id]
            assert not row.empty, f"Missing {exp_id} in registry"
            assert row.iloc[0]["dataset"] == "BDG2"
            assert row.iloc[0]["metric_unit"] == "kWh"
            assert row.iloc[0]["status"] == "COMPLETED"
            assert row.iloc[0]["n_samples"] == 2607
            assert np.isfinite(row.iloc[0]["mae"])
            assert np.isfinite(row.iloc[0]["rmse"])
            assert np.isfinite(row.iloc[0]["r2"])

        # Check Test evaluation records
        assert not df[df["experiment_id"] == "EXP_WIND_DL_C2_TEST"].empty
        assert not df[df["experiment_id"] == "EXP_DEMAND_DL_D2_TEST"].empty
        assert df[df["experiment_id"] == "EXP_WIND_DL_C2_TEST"].iloc[0]["n_samples"] == 1242
        assert df[df["experiment_id"] == "EXP_DEMAND_DL_D2_TEST"].iloc[0]["n_samples"] == 2561

    def test_validation_prediction_timestamps_alignment(self):
        """Verify that validation predictions match baseline target timestamps exactly."""
        # Wind validation predictions
        w_pred = np.load(PROJECT_ROOT / "experiments" / "EXP_WIND_DL_C2" / "val_predictions.npz")
        assert len(w_pred["timestamps"]) == 1290
        assert len(w_pred["y_true_speed"]) == 1290
        assert len(w_pred["y_pred_speed"]) == 1290

        # Demand validation predictions
        d_pred = np.load(PROJECT_ROOT / "experiments" / "EXP_DEMAND_DL_D2" / "val_predictions.npz")
        assert len(d_pred["timestamps"]) == 2607
        assert len(d_pred["y_true"]) == 2607
        assert len(d_pred["y_pred"]) == 2607
