"""
ResNet-18 transfer learning regression model for SKIPP'D solar PV nowcasting.

Predicts instantaneous PV power (kW) from 64x64 RGB sky imagery.
Supports:
- Feature extraction mode: pretrained backbone frozen, regression head trained.
- Fine-tuning mode: early layers frozen, layer4 and regression head trained.
- Training from scratch: all layers randomly initialized and trainable.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Nowcaster(nn.Module):
    """
    ResNet-18 model tailored for contemporaneous PV power estimation.

    Attributes:
        mode: One of 'feature_extractor', 'fine_tune', or 'scratch'.
        dropout_rate: Dropout probability applied before final linear output.
        backbone: ResNet-18 feature extraction backbone.
        head: Linear regression head producing scalar PV prediction in kW.
    """

    def __init__(
        self,
        pretrained: bool = True,
        mode: str = "feature_extractor",
        dropout: float = 0.2,
        head_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        valid_modes = ("feature_extractor", "fine_tune", "scratch")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")

        self.mode = mode
        self.dropout_rate = float(dropout)
        self.pretrained = bool(pretrained)

        # 1. Load ResNet-18 backbone
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features  # 512

        # 2. Build regression head
        if head_hidden_dim is not None and head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, 1),
            )

        # Replace classification fc with Identity
        self.backbone.fc = nn.Identity()

        # 3. Configure layer freezing
        self._configure_freezing()

    def _configure_freezing(self) -> None:
        """Freeze or unfreeze backbone layers according to chosen mode."""
        if self.mode == "feature_extractor":
            # Freeze all parameters in backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Regression head is trainable
            for param in self.head.parameters():
                param.requires_grad = True

        elif self.mode == "fine_tune":
            # Freeze earlier blocks: conv1, bn1, layer1, layer2, layer3
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 (final residual block)
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            # Regression head is trainable
            for param in self.head.parameters():
                param.requires_grad = True

        elif self.mode == "scratch":
            # All parameters trainable
            for param in self.parameters():
                param.requires_grad = True

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return tuple of (trainable_parameters, total_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of RGB images, shape (batch_size, 3, 64, 64).

        Returns:
            Scalar predictions in kW, shape (batch_size, 1).
        """
        features = self.backbone(x)  # Shape: (batch_size, 512)
        out = self.head(features)    # Shape: (batch_size, 1)
        return out
