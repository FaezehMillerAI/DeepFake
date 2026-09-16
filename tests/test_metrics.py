"""Week 1 acceptance tests for the metric definitions.

These are the tests named in the plan: identical maps score 1.0, random maps
score about 0, and the rank variant survives an intensity shift.
"""
import numpy as np
import pytest

from src.metrics import (brier, ece, esi, esi_rho, grounded_accuracy, iou,
                         normalise_map, drift_row,
                         counterfactual_swap, sample_control_mask, counterfactual_faithfulness)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---- ESI -----------------------------------------------------------------

def test_esi_identical_maps_is_one(rng):
    m = rng.random((64, 64))
    assert esi(m, m) == pytest.approx(1.0, abs=1e-9)


def test_esi_random_maps_is_near_zero(rng):
    a, b = rng.random((64, 64)), rng.random((64, 64))
    assert esi(a, b) < 0.2


def test_esi_rho_survives_monotone_intensity_shift(rng):
    m = rng.random((64, 64))
    shifted = 0.3 * m + 0.5          # monotone: ordering preserved
    assert esi_rho(m, shifted) == pytest.approx(1.0, abs=1e-6)


def test_esi_rejects_mismatched_shapes(rng):
    with pytest.raises(ValueError):
        esi(rng.random((64, 64)), rng.random((32, 32)))


# ---- IoU / GAcc ----------------------------------------------------------

def test_iou_perfect_overlap():
    m = np.zeros((16, 16)); m[4:12, 4:12] = 1.0
    assert iou(m, m > 0.5) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.zeros((16, 16)); a[0:4, 0:4] = 1.0
    g = np.zeros((16, 16), bool); g[12:16, 12:16] = True
    assert iou(a, g) == pytest.approx(0.0)


def test_iou_two_empty_masks_agree():
    assert iou(np.zeros((8, 8)), np.zeros((8, 8), bool)) == 1.0


def test_grounded_accuracy_needs_both_label_and_region():
    good = np.zeros((16, 16)); good[4:12, 4:12] = 1.0
    gt = good > 0.5
    bad = np.zeros((16, 16)); bad[0:3, 0:3] = 1.0

    # right label, right region
    assert grounded_accuracy([1], [1], [good], [gt]) == 1.0
    # right label, wrong region -> not grounded
    assert grounded_accuracy([1], [1], [bad], [gt]) == 0.0
    # wrong label, right region -> not grounded
    assert grounded_accuracy([0], [1], [good], [gt]) == 0.0


# ---- calibration ---------------------------------------------------------

def test_ece_is_zero_for_perfectly_calibrated_confident_model():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_high_for_confidently_wrong_model():
    probs = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([0, 0, 1, 1])
    assert ece(probs, labels) > 0.9


def test_brier_bounds():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)
    assert brier(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(1.0)


# ---- drift ---------------------------------------------------------------

def test_drift_is_positive_when_evidence_falls_faster_than_accuracy():
    row = drift_row(acc_clean=0.97, acc_corrupt=0.96, esi_value=0.40)
    assert row["drift"] > 0          # this is what C1 predicts


def test_normalise_map_handles_constant_input():
    assert np.all(normalise_map(np.full((8, 8), 3.0)) == 0.0)


# ---- counterfactual faithfulness (CF) ------------------------------------

def test_counterfactual_swap_replaces_only_masked_pixels():
    fake = np.zeros((10, 10, 3), dtype=np.uint8)
    real = np.full((10, 10, 3), 255, dtype=np.uint8)
    mask = np.zeros((10, 10))
    mask[2:5, 2:5] = 1.0

    swapped = counterfactual_swap(fake, real, mask)
    # Masked region should match real
    assert np.all(swapped[2:5, 2:5] == 255)
    # Unmasked region should match fake
    assert np.all(swapped[0:2, :] == 0)
    assert np.all(swapped[5:, :] == 0)


def test_sample_control_mask_preserves_area(rng):
    mask = np.zeros((32, 32))
    mask[5:15, 10:20] = 1.0  # 100 pixels
    control = sample_control_mask(mask, seed=42)

    assert int(control.sum()) == int(mask.sum())
    assert control.shape == mask.shape


def test_counterfactual_faithfulness_positive_gap():
    # If restoring predicted region drops fake probability to 0.1 (flipping verdict to real)
    # while restoring control region stays at 0.9 (fake verdict preserved):
    probs_pred = [0.1, 0.2, 0.15, 0.05]
    probs_ctrl = [0.8, 0.9, 0.85, 0.75]
    res = counterfactual_faithfulness(probs_pred, probs_ctrl, decision_threshold=0.5)

    assert res["flip_pred"] == 1.0
    assert res["flip_control"] == 0.0
    assert res["cf_gap"] == pytest.approx(1.0)
    assert res["delta_conf"] > 0.6


def test_counterfactual_faithfulness_zero_gap_on_uninformative():
    probs_pred = [0.9, 0.8]
    probs_ctrl = [0.9, 0.8]
    res = counterfactual_faithfulness(probs_pred, probs_ctrl)

    assert res["cf_gap"] == pytest.approx(0.0)

