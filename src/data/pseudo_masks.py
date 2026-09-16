"""Pseudo-mask generation and validation pipeline.

Section 5.2 of the GRACE-DF Research Plan:
Derives pixel manipulation masks from (I_real, I_edit) pairs without manual
annotation:
  1. Per-pixel difference in CIE LAB space (perceptually uniform)
  2. Gaussian blur (sigma=2.0)
  3. Binarization via Otsu's threshold
  4. Morphological opening (noise reduction) and closing (hole filling)
  5. Retain largest-k connected components
  6. Rejection filtering:
       - area > 0.60 * HW  (style-change / global illumination)
       - area < 0.001 * HW (sub-pixel noise / no-op edits)

Gate G0: mean IoU against ground-truth masks (e.g. SID-Set) >= 0.70.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk

from src.metrics.maps import iou


@dataclass
class PseudoMaskResult:
    """Result of pseudo-mask generation."""
    mask: np.ndarray          # (H, W) uint8 (0 or 1)
    valid: bool               # True if passed area filtering
    area_fraction: float      # active_pixels / (H * W)
    reason: str               # 'accepted', 'global_or_style_change', 'noop_or_subpixel'
    threshold: float          # Otsu threshold value applied


def _ensure_rgb_array(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Ensure image is an RGB uint8 or float ndarray (H, W, 3)."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8 and arr.max() <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def generate_pseudo_mask(
    img_real: Union[np.ndarray, Image.Image],
    img_edit: Union[np.ndarray, Image.Image],
    sigma: float = 2.0,
    max_components: int = 3,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.60,
    disk_radius_open: int = 2,
    disk_radius_close: int = 3,
) -> PseudoMaskResult:
    """Generate a manipulation mask from a real and edited image pair.

    Args:
        img_real: Authentic source image.
        img_edit: Partially edited manipulated image.
        sigma: Gaussian smoothing radius for difference map.
        max_components: Maximum number of connected components to retain.
        min_area_fraction: Lower bound on mask area (below is no-op / noise).
        max_area_fraction: Upper bound on mask area (above is global style change).
        disk_radius_open: Structuring element radius for morphological opening.
        disk_radius_close: Structuring element radius for morphological closing.

    Returns:
        PseudoMaskResult containing the binary mask and quality metadata.
    """
    arr_real = _ensure_rgb_array(img_real)
    arr_edit = _ensure_rgb_array(img_edit)

    if arr_real.shape[:2] != arr_edit.shape[:2]:
        raise ValueError(f"Spatial shapes differ: {arr_real.shape[:2]} vs {arr_edit.shape[:2]}")

    h, w = arr_real.shape[:2]
    total_pixels = h * w

    # 1. CIE LAB color space conversion
    lab_real = rgb2lab(arr_real)
    lab_edit = rgb2lab(arr_edit)

    # 2. Euclidean color difference map
    diff = np.sqrt(np.sum((lab_edit - lab_real) ** 2, axis=-1))

    # Guard: if images are identical
    diff_range = float(diff.max() - diff.min())
    if diff_range < 1e-6:
        empty = np.zeros((h, w), dtype=np.uint8)
        return PseudoMaskResult(
            mask=empty,
            valid=False,
            area_fraction=0.0,
            reason="noop_or_subpixel",
            threshold=0.0,
        )

    # 3. Gaussian smoothing
    diff_blur = gaussian_filter(diff, sigma=float(sigma))

    # 4. Otsu adaptive binarization
    try:
        thresh = float(threshold_otsu(diff_blur))
        binary = diff_blur >= thresh
    except Exception:
        # Fallback if distribution is degenerate
        thresh = float(np.median(diff_blur))
        binary = diff_blur > thresh

    # 5. Morphological cleaning: opening then closing
    if disk_radius_open > 0:
        binary = binary_opening(binary, disk(disk_radius_open))
    if disk_radius_close > 0:
        binary = binary_closing(binary, disk(disk_radius_close))

    # 6. Connected components filtering: retain top-k largest
    lbl, num_features = label(binary, return_num=True)
    if num_features > 0:
        component_sizes = np.bincount(lbl.ravel())
        component_sizes[0] = 0  # Ignore background
        # Keep top max_components
        if num_features > max_components:
            keep_labels = np.argsort(component_sizes)[-max_components:]
            binary = np.isin(lbl, keep_labels)

    mask_uint8 = binary.astype(np.uint8)
    active_pixels = int(mask_uint8.sum())
    area_frac = float(active_pixels / total_pixels)

    # 7. Area rejection filters
    if area_frac > max_area_fraction:
        return PseudoMaskResult(
            mask=mask_uint8,
            valid=False,
            area_fraction=area_frac,
            reason="global_or_style_change",
            threshold=thresh,
        )
    if area_frac < min_area_fraction:
        return PseudoMaskResult(
            mask=mask_uint8,
            valid=False,
            area_fraction=area_frac,
            reason="noop_or_subpixel",
            threshold=thresh,
        )

    return PseudoMaskResult(
        mask=mask_uint8,
        valid=True,
        area_fraction=area_frac,
        reason="accepted",
        threshold=thresh,
    )


def validate_pseudo_masks(
    pairs_real_edit: Sequence[Tuple[Union[np.ndarray, Image.Image], Union[np.ndarray, Image.Image]]],
    gt_masks: Sequence[np.ndarray],
    tau: float = 0.5,
    gate_g0_threshold: float = 0.70,
) -> Dict[str, Union[float, int, bool, List[float]]]:
    """Validate pseudo-mask pipeline against ground-truth annotations (Gate G0).

    Args:
        pairs_real_edit: Sequence of (real, edited) image pairs.
        gt_masks: Matching sequence of ground-truth binary masks.
        tau: IoU threshold for counting a detection hit.
        gate_g0_threshold: Mean IoU required to pass Gate G0 (default 0.70).

    Returns:
        Validation report dict with mean IoU, pass rate, and Gate G0 status.
    """
    if len(pairs_real_edit) != len(gt_masks):
        raise ValueError(f"Counts differ: {len(pairs_real_edit)} pairs vs {len(gt_masks)} masks")

    ious: List[float] = []
    accepted_count = 0
    rejected_count = 0
    reasons: Dict[str, int] = {}

    for (real, edit), gt in zip(pairs_real_edit, gt_masks):
        res = generate_pseudo_mask(real, edit)
        reasons[res.reason] = reasons.get(res.reason, 0) + 1

        if res.valid:
            accepted_count += 1
            score = iou(res.mask, gt)
            ious.append(score)
        else:
            rejected_count += 1
            # For rejected cases, IoU is 0.0 against non-empty ground truth
            score = 1.0 if np.asarray(gt).sum() == 0 else 0.0
            ious.append(score)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    hit_rate = float(np.mean([s >= tau for s in ious])) if ious else 0.0

    return {
        "n_samples": len(pairs_real_edit),
        "mean_iou": mean_iou,
        "hit_rate_tau": hit_rate,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "acceptance_rate": float(accepted_count / len(pairs_real_edit)) if pairs_real_edit else 0.0,
        "reasons": reasons,
        "gate_g0_passed": bool(mean_iou >= gate_g0_threshold),
        "ious": [round(x, 4) for x in ious],
    }
