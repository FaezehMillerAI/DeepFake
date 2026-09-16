"""Evidence and localisation metrics.

Definitions follow Section 4 of the research plan. Everything takes plain numpy
arrays so it works from any framework.

    M : evidence map      float (H, W), any range — normalised internally
    G : ground-truth mask bool or 0/1 (H, W)

FROZEN IN WEEK 1. Changing a definition after experiments have run invalidates
every number already in runs/.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def normalise_map(m: np.ndarray) -> np.ndarray:
    """Min-max a map to [0, 1]. A constant map becomes all zeros."""
    m = np.asarray(m, dtype=np.float64)
    lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between a (soft) evidence map and a binary mask.

    The map is min-max normalised, then thresholded. Two empty masks score 1.0.
    """
    p = normalise_map(pred) >= threshold
    g = np.asarray(gt).astype(bool)
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def grounded_accuracy(y_pred: Sequence[int], y_true: Sequence[int],
                      maps: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                      tau: float = 0.5, threshold: float = 0.5) -> float:
    """GAcc — right for the right reason.

    Fraction of samples that are BOTH classified correctly AND localised with
    IoU >= tau. Metric 1, Section 4.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if not (len(y_pred) == len(y_true) == len(maps) == len(gts)):
        raise ValueError("y_pred, y_true, maps and gts must be the same length")
    if len(y_pred) == 0:
        return 0.0

    hits = 0
    for pred, true, m, g in zip(y_pred, y_true, maps, gts):
        if pred == true and iou(m, g, threshold) >= tau:
            hits += 1
    return hits / len(y_pred)


def esi(map_clean: np.ndarray, map_corrupted: np.ndarray) -> float:
    """Evidence Stability Index — SSIM between the clean and perturbed maps.

    1.0 = the explanation did not move. 0.0 = it is unrecognisable.
    Metric 2, Section 4. This is the core novel measurement of the project.
    """
    a = normalise_map(map_clean)
    b = normalise_map(map_corrupted)
    if a.shape != b.shape:
        raise ValueError(f"map shapes differ: {a.shape} vs {b.shape}")

    win = min(7, min(a.shape) if a.ndim == 2 else min(a.shape[:2]))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise ValueError("maps are too small for SSIM (need at least 3x3)")

    return float(structural_similarity(a, b, data_range=1.0, win_size=win))


def esi_rho(map_clean: np.ndarray, map_corrupted: np.ndarray) -> float:
    """Rank-correlation variant of ESI — robust to monotone intensity shifts.

    Report alongside `esi`: a high rho with a low SSIM means the ordering of
    evidence survived but its scale or contrast changed.
    """
    a = normalise_map(map_clean).ravel()
    b = normalise_map(map_corrupted).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def drift_row(acc_clean: float, acc_corrupt: float,
              esi_value: float, esi_rho_value: float | None = None) -> Dict[str, float]:
    """One row of the evidence-drift table.

    Evidence drift is the case where the label survives but the evidence does
    not: small `delta_acc`, large `delta_esi`. Plot delta_acc against delta_esi;
    points in the high-drift quadrant are the finding (Metric 3, Section 4).
    """
    delta_acc = float(acc_clean - acc_corrupt)
    delta_esi = float(1.0 - esi_value)
    row = {
        "acc_clean": float(acc_clean),
        "acc_corrupt": float(acc_corrupt),
        "delta_acc": delta_acc,
        "esi": float(esi_value),
        "delta_esi": delta_esi,
        "drift": delta_esi - delta_acc,   # > 0 means evidence degraded faster than the label
    }
    if esi_rho_value is not None:
        row["esi_rho"] = float(esi_rho_value)
    return row
