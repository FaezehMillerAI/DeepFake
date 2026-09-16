"""Tests for DeepFakeBuster-style Adaptive Fusion Ensemble."""
from __future__ import annotations

import pytest
import torch

from src.eval.explain import explain
from src.models import build_model
from src.models.fusion import AdaptiveFusionEnsemble, GatingNetwork


def test_gating_network():
    gate = GatingNetwork(in_channels=3, num_experts=5)
    x = torch.rand(2, 3, 64, 64)
    weights = gate(x)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (weights >= 0.0).all()


def test_adaptive_fusion_ensemble():
    m1 = build_model("resnet18", pretrained=False)
    m2 = build_model("mobilenetv3_s", pretrained=False)

    ensemble = AdaptiveFusionEnsemble([m1, m2], adaptive=True)
    ensemble.eval()

    x = torch.rand(2, 3, 64, 64)
    out = ensemble(x)
    assert out.shape == (2, 2)

    # Static uniform mode
    static_ensemble = AdaptiveFusionEnsemble([m1, m2], adaptive=False)
    out_static = static_ensemble(x)
    assert out_static.shape == (2, 2)

    # Test Grad-CAM on ensemble
    cam = explain(ensemble, x, target_class=1, method="gradcam")
    assert cam.shape == (2, 1, 64, 64)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0 + 1e-5
