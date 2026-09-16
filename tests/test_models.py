"""Unit tests for baseline deepfake detection models."""
import pytest
import torch

from src.models import build_model, ARCNet, BaselineClassifier


@pytest.mark.parametrize("arch", [
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "convnext_tiny",
    "arcnet",
])
def test_baseline_forward_and_cam_layer(arch):
    model = build_model(arch, num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    # 1. Forward pass produces (B, 2) logits
    logits = model(x)
    assert logits.shape == (2, 2)

    # 2. Probability method produces valid simplex
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    # 3. Last conv layer is accessible
    cam_layer = model.get_last_conv_layer()
    assert isinstance(cam_layer, torch.nn.Module)


def test_arcnet_attention_module():
    model = ARCNet(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
    assert hasattr(model, "attention_block")


def test_build_model_from_config_dict():
    cfg = {
        "model": {
            "arch": "resnet18",
            "num_classes": 2,
            "pretrained": False,
        }
    }
    model = build_model(cfg)
    assert isinstance(model, BaselineClassifier)
    assert model.arch == "resnet18"
