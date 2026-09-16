"""Baseline architectures for deepfake detection.

Implements the five architectures from Shakya et al. (IEEE ISDFS 2026):
  1. ResNet-18
  2. MobileNetV3-Small
  3. EfficientNet-B0
  4. ConvNeXt-Tiny
  5. ViT-B/16

Plus ARC-Net from PLOS ONE 2026 (Residual Attention on EfficientNet-B0).
All models expose `get_last_conv_layer()` for explanation extraction (Grad-CAM, Score-CAM).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResidualAttentionBlock(nn.Module):
    """Residual Attention mechanism following ARC-Net (PLOS ONE 2026).

    Applies a spatial-channel convolutional gate to feature maps and combines
    the modulated features residually: F_out = F + F * M(F).
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(16, in_channels // reduction)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_conv(x)
        return x + x * mask


class ARCNet(nn.Module):
    """ARC-Net: Attention and Residual Convolutional Network (PLOS ONE 2026).

    Built upon EfficientNet-B0 with a residual attention module applied to the
    penultimate feature representation before global pooling and classification.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = tv_models.efficientnet_b0(weights=weights)

        # Feature extractor up to final conv block
        self.features = base.features
        in_channels = 1280  # EfficientNet-B0 output channels

        # Residual attention block
        self.attention_block = ResidualAttentionBlock(in_channels)

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Target layer for Grad-CAM."""
        return self.attention_block


class BaselineClassifier(nn.Module):
    """Wrapper around torchvision architectures with uniform API and Grad-CAM hooks."""

    def __init__(
        self,
        arch: str,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.num_classes = num_classes

        if self.arch in ["resnet18", "resnet-18"]:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = tv_models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            self._cam_layer = self.model.layer4[-1]

        elif self.arch in ["mobilenet_v3_small", "mobilenetv3_s", "mobilenetv3"]:
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = tv_models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            self._cam_layer = self.model.features[-1]

        elif self.arch in ["efficientnet_b0", "effnet_b0"]:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = tv_models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=drop_rate, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._cam_layer = self.model.features[-1]

        elif self.arch in ["convnext_tiny", "convnext_t", "convnext"]:
            weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = tv_models.convnext_tiny(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            self._cam_layer = self.model.features[-1]

        elif self.arch in ["vit_b_16", "vit", "vit_b"]:
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.model = tv_models.vit_b_16(weights=weights)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_classes)
            # For ViT, CAM targets the last encoder block
            self._cam_layer = self.model.encoder.layers[-1]

        elif self.arch in ["arcnet", "arc_net"]:
            self.model = ARCNet(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)
            self._cam_layer = self.model.get_last_conv_layer()

        else:
            raise ValueError(
                f"Unsupported architecture '{arch}'. Choose from: "
                "resnet18, mobilenet_v3_small, efficientnet_b0, convnext_tiny, vit_b_16, arcnet"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities across classes: (B, num_classes)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the penultimate convolutional layer or block for Grad-CAM."""
        return self._cam_layer


def build_model(
    config_or_arch: Union[str, Dict[str, Any], Any],
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> nn.Module:
    """Model factory for GRACE-DF baselines.

    Args:
        config_or_arch: Architecture name (str) or Config dict containing `model.arch`.
        num_classes: Number of output classes (default: 2 for real/fake).
        pretrained: Whether to load ImageNet pre-trained weights.
        drop_rate: Dropout rate for classifier heads.

    Returns:
        Initialized nn.Module.
    """
    if isinstance(config_or_arch, str):
        arch = config_or_arch
    elif hasattr(config_or_arch, "model") and hasattr(config_or_arch.model, "arch"):
        arch = config_or_arch.model.arch
        num_classes = getattr(config_or_arch.model, "num_classes", num_classes)
        pretrained = getattr(config_or_arch.model, "pretrained", pretrained)
    elif isinstance(config_or_arch, dict) and "model" in config_or_arch:
        m_cfg = config_or_arch["model"]
        arch = m_cfg.get("arch", "efficientnet_b0")
        num_classes = m_cfg.get("num_classes", num_classes)
        pretrained = m_cfg.get("pretrained", pretrained)
    else:
        arch = str(config_or_arch)

    return BaselineClassifier(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
