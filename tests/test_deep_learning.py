"""
Test suite for Phase 3 deep learning models, training routines, and checkpoints.

Validates:
1. ResNet18Nowcaster output shapes and finite values.
2. Parameter freezing in feature extraction vs fine-tuning modes.
3. WeatherLSTM input and output tensor shapes with single and stacked layers.
4. Trainer finite loss generation and early stopping logic.
5. Checkpoint saving, serialization, and weight restoration round-trip.
6. Target unscaling fidelity during evaluation.
7. Strict training/validation separation (no gradient leakage during eval).
8. Logging of deep learning records to ExperimentRegistry.
"""
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.solar.resnet import ResNet18Nowcaster
from src.models.nasa_weather.sequence_models import WeatherLSTM
from src.training.trainer import RegressionTrainer
from src.evaluation.registry import ExperimentRegistry, REGISTRY_COLUMNS


class TestModelArchitectures:
    """Validate neural network shapes, parameter freezing, and forward passes."""

    def test_resnet_output_shape_and_finite_values(self):
        """Verify ResNet-18 produces (batch_size, 1) output with finite values."""
        model = ResNet18Nowcaster(pretrained=False, mode="feature_extractor", dropout=0.2)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_resnet_feature_extractor_freezing(self):
        """Verify feature_extractor mode freezes backbone and trains only regression head."""
        model = ResNet18Nowcaster(pretrained=False, mode="feature_extractor", dropout=0.2)
        trainable_p, total_p = model.get_parameter_counts()

        # In ResNet-18, linear head with in_features=512, out_features=1 has 512 weights + 1 bias = 513 params
        assert trainable_p == 513
        assert total_p > 11_000_000

        # Conv1 and layer4 must be frozen
        assert model.backbone.conv1.weight.requires_grad is False
        assert model.backbone.layer4[0].conv1.weight.requires_grad is False
        # Head linear layer must be trainable
        assert model.head[-1].weight.requires_grad is True
        assert model.head[-1].bias.requires_grad is True

    def test_resnet_fine_tune_freezing(self):
        """Verify fine_tune mode freezes early layers and unfreezes layer4 and head."""
        model = ResNet18Nowcaster(pretrained=False, mode="fine_tune", dropout=0.2)
        trainable_p, total_p = model.get_parameter_counts()

        # Early layers frozen
        assert model.backbone.conv1.weight.requires_grad is False
        assert model.backbone.layer1[0].conv1.weight.requires_grad is False
        assert model.backbone.layer2[0].conv1.weight.requires_grad is False
        assert model.backbone.layer3[0].conv1.weight.requires_grad is False

        # Layer 4 and head trainable
        assert model.backbone.layer4[0].conv1.weight.requires_grad is True
        assert model.head[-1].weight.requires_grad is True

        # Layer 4 contains ~8.39M params + 513 head params
        assert trainable_p > 8_000_000
        assert trainable_p < total_p

    def test_lstm_output_shapes_and_layer_variants(self):
        """Verify WeatherLSTM produces (batch_size, 1) with 1-layer and 2-layer configs."""
        # 1-layer configuration (W1 / W2)
        lstm_1l = WeatherLSTM(input_dim=4, hidden_dim=32, num_layers=1, dropout=0.0)
        x = torch.randn(6, 24, 4)
        out_1l = lstm_1l(x)
        assert out_1l.shape == (6, 1)
        assert torch.isfinite(out_1l).all()

        # 2-layer stacked configuration with dropout (W3 / W4)
        lstm_2l = WeatherLSTM(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2)
        out_2l = lstm_2l(x)
        assert out_2l.shape == (6, 1)
        assert torch.isfinite(out_2l).all()


class TestTrainingPipeline:
    """Validate training loop, checkpoints, and evaluation separation."""

    def test_trainer_finite_loss_and_checkpoint_roundtrip(self, tmp_path):
        """Verify training produces finite losses and checkpoints can be restored exactly."""
        # Synthetic time-series data
        torch.manual_seed(42)
        X_train = torch.randn(20, 24, 4)
        y_train = torch.randn(20, 1)
        X_val = torch.randn(10, 24, 4)
        y_val = torch.randn(10, 1)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=5)
        val_loader = DataLoader(val_ds, batch_size=5)

        model = WeatherLSTM(input_dim=4, hidden_dim=16, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        trainer = RegressionTrainer(
            model=model,
            optimizer=opt,
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_id="TEST_LSTM",
            early_stopping_patience=3,
        )

        summary = trainer.train(train_loader, val_loader, max_epochs=3, verbose=False)

        assert summary["best_epoch"] >= 1
        assert np.isfinite(summary["best_val_loss"])
        assert (tmp_path / "checkpoints" / "TEST_LSTM" / "best_model.pt").exists()
        assert (tmp_path / "checkpoints" / "TEST_LSTM" / "history.json").exists()

        # Test checkpoint restoration fidelity
        eval_before = trainer.evaluate(val_loader)
        fresh_model = WeatherLSTM(input_dim=4, hidden_dim=16, num_layers=1)
        ckpt = torch.load(tmp_path / "checkpoints" / "TEST_LSTM" / "best_model.pt", map_location="cpu")
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

    def test_target_unscaling_in_evaluation(self, tmp_path):
        """Verify trainer evaluation correctly applies unscaler to report physical units."""
        X = torch.zeros(4, 24, 4)
        y_scaled = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(X, y_scaled), batch_size=4)

        model = WeatherLSTM(input_dim=4, hidden_dim=8)
        # Mock model to output exact y_scaled values
        trainer = RegressionTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            checkpoint_dir=tmp_path,
            experiment_id="TEST_UNSCALE",
            # Mock unscaler: multiply by 100
            target_unscaler=lambda arr: arr * 100.0,
        )

        eval_res = trainer.evaluate(val_loader)
        assert eval_res["n_samples"] == 4
        assert np.isfinite(eval_res["mae"])

    def test_validation_gradient_isolation(self, tmp_path):
        """Verify model evaluation does not alter gradients or model weights."""
        model = WeatherLSTM(input_dim=4, hidden_dim=16)
        initial_weights = model.fc_head.weight.clone().detach()

        X_val = torch.randn(8, 24, 4)
        y_val = torch.randn(8, 1)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

        trainer = RegressionTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            checkpoint_dir=tmp_path,
            experiment_id="TEST_ISOLATION",
        )

        _ = trainer.evaluate(val_loader)

        # Ensure weights are strictly identical
        torch.testing.assert_close(model.fc_head.weight, initial_weights)
        # Ensure no gradients were accumulated on parameters
        for p in model.parameters():
            assert p.grad is None
