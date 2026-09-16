"""DeepFakeBuster-style Adaptive Fusion Ensemble for GRACE-DF baselines.

Implements:
  1. StaticEnsemble: Uniform average of the five detector backbones.
  2. AdaptiveFusionEnsemble: Input-dependent gating network w_i(I) predicting
     weights across the five CNN baselines based on degradation indicators.
     Following DeepFakeBuster (Wang et al., 2024 / IEEE TIFS), weights sum to 1
     via softmax: y_ensemble = sum_i w_i(I) * f_i(I).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines import BaselineClassifier


class GatingNetwork(nn.Module):
    """Predicts ensemble weights w_i(I) from input image or pooled features."""

    def __init__(self, in_channels: int = 3, num_experts: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Lightweight convolutional feature extractor for image quality estimation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax gating weights [B, Num_experts]."""
        feat = self.encoder(x)
        logits = self.fc(feat)
        return F.softmax(logits, dim=-1)


class AdaptiveFusionEnsemble(nn.Module):
    """Adaptive gated ensemble over multiple deepfake detectors.

    Parameters
    ----------
    models : Sequence of nn.Module
        Pretrained/instantiated detector models.
    adaptive : bool
        If True, uses input-adaptive gating w_i(I). If False, uses uniform weighting.
    """

    def __init__(self, models: List[nn.Module], adaptive: bool = True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_experts = len(models)
        self.adaptive = adaptive

        if adaptive:
            self.gating = GatingNetwork(in_channels=3, num_experts=self.num_experts)
        else:
            self.gating = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        Returns
        -------
        logits : torch.Tensor
            Ensemble logits [B, Num_classes].
        """
        # Collect individual model predictions
        expert_logits = [m(x) for m in self.models]  # list of [B, Num_classes]
        stacked = torch.stack(expert_logits, dim=1)  # [B, Num_experts, Num_classes]

        if self.adaptive and self.gating is not None:
            weights = self.gating(x)  # [B, Num_experts]
            weights = weights.unsqueeze(-1)  # [B, Num_experts, 1]
            blended = (stacked * weights).sum(dim=1)  # [B, Num_classes]
        else:
            blended = stacked.mean(dim=1)

        return blended

    def get_last_conv_layer(self) -> nn.Module:
        """Target layer for explanation extraction (defaults to first expert)."""
        if hasattr(self.models[0], "get_last_conv_layer"):
            return self.models[0].get_last_conv_layer()
        return self.models[0]
