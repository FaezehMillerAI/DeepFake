"""Gate G0 validation script: Evaluates pseudo-masks against ground truth masks.

Usage:
  python scripts/validate_gate_g0.py \
      --data-dir data/sid_set \
      --output-json runs/pseudo_mask_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.pseudo_masks import (
    evaluate_pseudo_mask_viability,
    generate_pseudo_mask,
)
from src.metrics.maps import iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_gate_g0")


def evaluate_pairs_against_gt(
    pairs: List[Tuple[Path, Path, Path]],  # (edit_path, source_path, gt_mask_path)
    tau: float = 0.5,
) -> Dict:
    """Evaluates pseudo-mask pipeline against ground-truth masks.

    Returns Gate G0 decision dictionary.
    """
    ious: List[float] = []
    accepted = 0
    rejected = 0

    for edit_p, src_p, gt_p in tqdm(pairs, desc="Gate G0 Eval"):
        mask, area_ratio, is_valid = generate_pseudo_mask(edit_p, src_p)
        if not is_valid:
            rejected += 1
            continue
        accepted += 1

        # Load ground truth mask
        gt_img = Image.open(gt_p).convert("L")
        gt_arr = np.array(gt_img) > 127

        # Ensure spatial match
        if mask.shape != gt_arr.shape:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                gt_img.size, Image.NEAREST
            )
            mask = (np.array(mask_pil) > 127).astype(np.float32)

        sample_iou = iou(mask, gt_arr, threshold=tau)
        ious.append(sample_iou)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    std_iou = float(np.std(ious)) if ious else 0.0

    # Decision logic as specified in Gate G0 of research plan:
    if mean_iou >= 0.70:
        decision = "Full plan: DFBench becomes a localisation benchmark."
        branch = "A"
    elif mean_iou >= 0.50:
        decision = "Use pseudo-masks for training only; evaluate localisation on annotated sets only."
        branch = "B"
    else:
        decision = "Drop DFBench localisation. Train and evaluate masks on annotated sets only."
        branch = "C"

    return {
        "gate": "G0",
        "name": "Pseudo-mask viability",
        "total_pairs_evaluated": len(pairs),
        "accepted_masks": accepted,
        "rejected_masks": rejected,
        "yield_ratio": round(accepted / max(len(pairs), 1), 4),
        "mean_iou": round(mean_iou, 4),
        "median_iou": round(median_iou, 4),
        "std_iou": round(std_iou, 4),
        "gate_branch": branch,
        "decision": decision,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Gate G0 Pseudo-mask Validation.")
    parser.add_argument("--edit-dir", type=str, default=None, help="Path to edited images")
    parser.add_argument("--source-dir", type=str, default=None, help="Path to source images")
    parser.add_argument("--mask-dir", type=str, default=None, help="Path to ground truth masks")
    parser.add_argument(
        "--output-json",
        type=str,
        default="runs/pseudo_mask_validation.json",
        help="Path to output json report",
    )
    args = parser.parse_args()

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.edit_dir and args.source_dir and args.mask_dir:
        edit_dir = Path(args.edit_dir)
        source_dir = Path(args.source_dir)
        mask_dir = Path(args.mask_dir)

        # Match files by stem
        pairs = []
        for edit_p in edit_dir.glob("*.*"):
            stem = edit_p.stem.split("_")[0]
            src_candidates = list(source_dir.glob(f"{stem}.*"))
            mask_candidates = list(mask_dir.glob(f"{edit_p.stem}.*")) or list(
                mask_dir.glob(f"{stem}.*")
            )
            if src_candidates and mask_candidates:
                pairs.append((edit_p, src_candidates[0], mask_candidates[0]))

        logger.info("Found %d matched triplets for Gate G0 evaluation.", len(pairs))
        report = evaluate_pairs_against_gt(pairs)
    else:
        logger.info("No ground-truth directory passed. Generating baseline Gate G0 schema...")
        report = {
            "gate": "G0",
            "name": "Pseudo-mask viability",
            "status": "Awaiting annotated benchmark download on Kaggle",
            "benchmark_candidates": ["SID-Set", "MagicBrush", "AutoSplice", "INP-X"],
            "thresholds": {
                ">= 0.70": "Branch A: Full plan, DFBench becomes localisation benchmark",
                "0.50 - 0.70": "Branch B: Training only, evaluate on annotated sets",
                "< 0.50": "Branch C: Drop DFBench localisation, fallback to INP-X / SID-Set",
            },
        }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Gate G0 report written to %s", out_file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
