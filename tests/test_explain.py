"""Tests for explanation extraction module (Grad-CAM, Score-CAM, IG, Attention Rollout)."""
from __future__ import annotations

import pytest
import torch

from src.eval.explain import (
    AttentionRollout,
    GradCAM,
    IntegratedGradients,
    ScoreCAM,
    explain,
)
from src.models.baselines import BaselineClassifier


@pytest.fixture(scope="module")
def resnet_model():
    model = BaselineClassifier("resnet18", pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def vit_model():
    model = BaselineClassifier("vit_b_16", pretrained=False)
    model.eval()
    return model


def test_gradcam_cnn(resnet_model):
    x = torch.rand(2, 3, 224, 224)
    cam = GradCAM(resnet_model)
    heatmap = cam.generate(x, target_class=1)

    assert heatmap.shape == (2, 1, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    # Check normalization: max per sample should be ~1.0
    assert (heatmap[0].max() - 1.0).abs() < 1e-3
    assert (heatmap[1].max() - 1.0).abs() < 1e-3


def test_gradcam_vit(vit_model):
    x = torch.rand(2, 3, 224, 224)
    cam = GradCAM(vit_model)
    heatmap = cam.generate(x, target_class=0)

    assert heatmap.shape == (2, 1, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_scorecam_cnn(resnet_model):
    x = torch.rand(2, 3, 224, 224)
    scorecam = ScoreCAM(resnet_model, max_channels=4)
    heatmap = scorecam.generate(x, target_class=1)

    assert heatmap.shape == (2, 1, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_integrated_gradients(resnet_model):
    x = torch.rand(2, 3, 224, 224)
    ig = IntegratedGradients(resnet_model, steps=4)
    heatmap = ig.generate(x, target_class=1)

    assert heatmap.shape == (2, 1, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_attention_rollout_vit(vit_model):
    x = torch.rand(2, 3, 224, 224)
    rollout = AttentionRollout(vit_model)
    heatmap = rollout.generate(x)

    assert heatmap.shape == (2, 1, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_unified_explain_interface(resnet_model, vit_model):
    x = torch.rand(2, 3, 224, 224)

    # GradCAM
    h_grad = explain(resnet_model, x, target_class=1, method="gradcam")
    assert h_grad.shape == (2, 1, 224, 224)

    # ScoreCAM
    h_score = explain(resnet_model, x, target_class=1, method="scorecam", max_channels=2)
    assert h_score.shape == (2, 1, 224, 224)

    # IG
    h_ig = explain(resnet_model, x, target_class=1, method="ig", steps=2)
    assert h_ig.shape == (2, 1, 224, 224)

    # Rollout
    h_roll = explain(vit_model, x, method="rollout")
    assert h_roll.shape == (2, 1, 224, 224)

    # Invalid method error
    with pytest.raises(ValueError):
        explain(resnet_model, x, method="invalid_method_xyz")
