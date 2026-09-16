"""Calibration metrics.

ARC-Net reports Brier overall; we report both ECE and Brier *per corruption
level*, which no baseline does.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _as_confidence(probs: Sequence[float] | np.ndarray, labels: np.ndarray):
    """Accept either (N,) probability-of-positive or (N, C) full distribution."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(int)

    if p.ndim == 1:
        conf = np.where(p >= 0.5, p, 1.0 - p)
        pred = (p >= 0.5).astype(int)
    elif p.ndim == 2:
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
    else:
        raise ValueError("probs must be 1-D or 2-D")

    return conf, pred, y


def reliability_bins(probs, labels, bins: int = 15) -> List[Dict[str, float]]:
    """Per-bin confidence, accuracy and count — the reliability diagram data."""
    conf, pred, y = _as_confidence(probs, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: List[Dict[str, float]] = []

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "confidence": 0.0, "accuracy": 0.0})
            continue
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "confidence": float(conf[mask].mean()),
            "accuracy": float((pred[mask] == y[mask]).mean()),
        })
    return out


def ece(probs, labels, bins: int = 15) -> float:
    """Expected Calibration Error — 15 bins by default, as in the plan."""
    rows = reliability_bins(probs, labels, bins)
    total = sum(r["n"] for r in rows)
    if total == 0:
        return 0.0
    return float(sum(r["n"] / total * abs(r["accuracy"] - r["confidence"]) for r in rows))


def brier(probs, labels) -> float:
    """Brier score. Binary: mean squared error of P(positive) against the label."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(int)

    if p.ndim == 1:
        return float(np.mean((p - y) ** 2))

    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


# Aliases
expected_calibration_error = ece
brier_score = brier
