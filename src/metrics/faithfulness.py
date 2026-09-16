"""Counterfactual Faithfulness (CF) metric.

Evaluates whether a model's visual explanation is causally sound:
  - If we restore the predicted manipulated region using authentic source pixels,
    the detector's verdict should flip from 'fake' to 'real'.
  - If we restore an equal-area random control region, the verdict should NOT flip.

The Counterfactual Faithfulness score is the causal flip-rate gap:
    CF = FlipRate(restored_pred) - FlipRate(restored_control)

CF in [0, 1]: higher is more faithful. CF <= 0 indicates the explanation is
unfaithful or no better than random attribution.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np


def counterfactual_swap(img_fake: np.ndarray, img_real: np.ndarray, mask: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
    """Replace pixels in `img_fake` where `mask >= threshold` with `img_real`.

    Args:
        img_fake: (H, W) or (H, W, C) manipulated image array.
        img_real: Matching shape authentic source image array.
        mask: (H, W) or (H, W, 1) evidence / manipulation mask.
        threshold: Binarization threshold for `mask`.

    Returns:
        Swapped image array of the same shape and dtype as `img_fake`.
    """
    fake = np.asarray(img_fake)
    real = np.asarray(img_real)
    m = np.asarray(mask)

    if fake.shape != real.shape:
        raise ValueError(f"Shape mismatch: fake {fake.shape} vs real {real.shape}")

    # Standardize mask shape to match spatial dimensions
    m_bin = (m >= threshold)
    if fake.ndim == 3 and m_bin.ndim == 2:
        m_bin = np.expand_dims(m_bin, axis=-1)

    out = np.where(m_bin, real, fake)
    return out.astype(fake.dtype)


def sample_control_mask(mask: np.ndarray, seed: int = 0,
                        threshold: float = 0.5) -> np.ndarray:
    """Generate an equal-area, spatially coherent random control mask.

    Preserves the exact pixel count and spatial structure (bounding aspect ratio)
    of the original mask to ensure fair causal comparison.

    Args:
        mask: (H, W) array.
        seed: Random seed for deterministic placement.
        threshold: Binarization threshold for `mask`.

    Returns:
        (H, W) float64 binary mask (0.0 or 1.0) with matching active pixel count.
    """
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    h, w = m.shape
    m_bin = m >= threshold
    target_area = int(m_bin.sum())

    if target_area == 0:
        return np.zeros((h, w), dtype=np.float64)
    if target_area >= h * w:
        return np.ones((h, w), dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Find bounding box of original mask to approximate shape
    y_indices, x_indices = np.where(m_bin)
    h_box = max(1, int(y_indices.max() - y_indices.min() + 1))
    w_box = max(1, int(x_indices.max() - x_indices.min() + 1))
    aspect = w_box / h_box

    # Size of the control rectangle: area = h_c * w_c ≈ target_area
    h_c = max(1, min(h, int(np.round(np.sqrt(target_area / aspect)))))
    w_c = max(1, min(w, int(np.round(target_area / h_c))))

    # Place at random valid coordinate
    max_y = max(0, h - h_c)
    max_x = max(0, w - w_c)
    top_y = rng.integers(0, max_y + 1) if max_y > 0 else 0
    left_x = rng.integers(0, max_x + 1) if max_x > 0 else 0

    control = np.zeros((h, w), dtype=np.float64)
    control[top_y:top_y + h_c, left_x:left_x + w_c] = 1.0

    # Fine-adjust pixels if rounding caused slight mismatch
    current_area = int(control.sum())
    diff = target_area - current_area

    if diff > 0:
        zero_y, zero_x = np.where(control == 0.0)
        if len(zero_y) > 0:
            add_idx = rng.choice(len(zero_y), size=min(diff, len(zero_y)), replace=False)
            control[zero_y[add_idx], zero_x[add_idx]] = 1.0
    elif diff < 0:
        one_y, one_x = np.where(control == 1.0)
        remove_idx = rng.choice(len(one_y), size=abs(diff), replace=False)
        control[one_y[remove_idx], one_x[remove_idx]] = 0.0

    return control


def counterfactual_faithfulness(
    probs_pred: Sequence[float] | np.ndarray,
    probs_control: Sequence[float] | np.ndarray,
    probs_original: Optional[Sequence[float] | np.ndarray] = None,
    decision_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute Counterfactual Faithfulness metrics from model predictions.

    Args:
        probs_pred: Model's P(fake) on counterfactual images where predicted
                    manipulation region was restored with authentic pixels.
        probs_control: Model's P(fake) on counterfactual images where equal-area
                       random control region was restored.
        probs_original: Model's P(fake) on original manipulated images (optional).
        decision_threshold: Threshold below which a prediction flips to 'real'.

    Returns:
        Dict with keys:
            - 'flip_pred': fraction of samples flipped by restoring predicted region
            - 'flip_control': fraction of samples flipped by restoring control region
            - 'cf_gap': flip_pred - flip_control (the main CF metric)
            - 'delta_conf': mean drop in P(fake) from control to predicted restoration
    """
    p_pred = np.asarray(probs_pred, dtype=np.float64)
    p_ctrl = np.asarray(probs_control, dtype=np.float64)

    if len(p_pred) != len(p_ctrl):
        raise ValueError(f"Lengths differ: {len(p_pred)} vs {len(p_ctrl)}")
    if len(p_pred) == 0:
        return {"flip_pred": 0.0, "flip_control": 0.0, "cf_gap": 0.0, "delta_conf": 0.0}

    flips_pred = (p_pred < decision_threshold).astype(np.float64)
    flips_ctrl = (p_ctrl < decision_threshold).astype(np.float64)

    flip_pred_rate = float(np.mean(flips_pred))
    flip_ctrl_rate = float(np.mean(flips_ctrl))
    cf_gap = float(flip_pred_rate - flip_ctrl_rate)
    delta_conf = float(np.mean(p_ctrl - p_pred))

    out = {
        "flip_pred": flip_pred_rate,
        "flip_control": flip_ctrl_rate,
        "cf_gap": cf_gap,
        "delta_conf": delta_conf,
    }

    if probs_original is not None:
        p_orig = np.asarray(probs_original, dtype=np.float64)
        if len(p_orig) == len(p_pred):
            out["flip_pred_from_orig"] = float(np.mean((p_orig >= decision_threshold) & (p_pred < decision_threshold)))
            out["flip_ctrl_from_orig"] = float(np.mean((p_orig >= decision_threshold) & (p_ctrl < decision_threshold)))

    return out
