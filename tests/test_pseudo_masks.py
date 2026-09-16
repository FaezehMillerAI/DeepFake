"""Unit tests for the pseudo-mask generation pipeline (Section 5.2 / Gate G0)."""
import numpy as np
import pytest
from PIL import Image

from src.data.pseudo_masks import generate_pseudo_mask, validate_pseudo_masks


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_base_image(rng):
    """Create a textured background image (128x128x3)."""
    base = rng.integers(50, 200, size=(128, 128, 3), dtype=np.uint8)
    return base


def test_identical_images_rejected_as_noop(synthetic_base_image):
    res = generate_pseudo_mask(synthetic_base_image, synthetic_base_image)
    assert not res.valid
    assert res.reason == "noop_or_subpixel"
    assert res.mask.sum() == 0


def test_localized_object_edit_accepted_and_localized(synthetic_base_image):
    real = synthetic_base_image.copy()
    edit = synthetic_base_image.copy()

    # Simulate an object edit: replace a 30x30 region (40:70, 40:70) with distinct color
    edit[40:70, 40:70] = np.array([240, 20, 30], dtype=np.uint8)
    gt_mask = np.zeros((128, 128), dtype=np.uint8)
    gt_mask[40:70, 40:70] = 1

    res = generate_pseudo_mask(real, edit)
    assert res.valid
    assert res.reason == "accepted"

    # Intersection over Union against ground truth should be very high
    intersection = np.logical_and(res.mask, gt_mask).sum()
    union = np.logical_or(res.mask, gt_mask).sum()
    score = intersection / union
    assert score > 0.85


def test_global_style_change_rejected(synthetic_base_image):
    real = synthetic_base_image.copy()
    # Uniform brightness increase across the entire image (> 60% area affected)
    edit = np.clip(real.astype(np.int16) + 60, 0, 255).astype(np.uint8)

    res = generate_pseudo_mask(real, edit, max_area_fraction=0.60)
    assert not res.valid
    assert res.reason == "global_or_style_change"
    assert res.area_fraction > 0.60


def test_validation_function_gate_g0(synthetic_base_image):
    pairs = []
    gts = []

    # 4 clean local edits
    for i in range(4):
        real = synthetic_base_image.copy()
        edit = synthetic_base_image.copy()
        y, x = 20 + i * 15, 20 + i * 15
        edit[y:y+25, x:x+25] = np.array([255, 0, 100], dtype=np.uint8)
        gt = np.zeros((128, 128), dtype=np.uint8)
        gt[y:y+25, x:x+25] = 1
        pairs.append((real, edit))
        gts.append(gt)

    report = validate_pseudo_masks(pairs, gts, gate_g0_threshold=0.70)
    assert report["n_samples"] == 4
    assert report["mean_iou"] > 0.70
    assert report["gate_g0_passed"] is True
    assert report["accepted_count"] == 4
