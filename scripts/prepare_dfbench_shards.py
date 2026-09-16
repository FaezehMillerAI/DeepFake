"""CLI script to build WebDataset shards and pristine subset from DFBench pairs.

Usage:
  python scripts/prepare_dfbench_shards.py \
      --source-dir data/dfbench/partial_source \
      --edit-dir data/dfbench/edit \
      --output-dir data/shards \
      --pristine-count 5000 \
      --target-size 384
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.acquire import find_dfbench_pairs
from src.data.pseudo_masks import generate_pseudo_mask
from src.data.shards import ShardSample, ShardWriter, verify_read_throughput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_shards")


def main():
    parser = argparse.ArgumentParser(description="Prepare GRACE-DF WebDataset shards from DFBench.")
    parser.add_argument("--source-dir", type=str, required=True, help="Path to partial_source/ directory")
    parser.add_argument("--edit-dir", type=str, required=True, help="Path to edit/ directory")
    parser.add_argument("--output-dir", type=str, default="data/shards", help="Output shard directory")
    parser.add_argument("--pristine-count", type=int, default=5000, help="Number of pristine PNG samples")
    parser.add_argument("--target-size", type=int, default=384, help="Target resolution for training shards")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")

    args = parser.parse_args()
    np.random.seed(args.seed)

    source_dir = Path(args.source_dir)
    edit_dir = Path(args.edit_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Matching DFBench source-edit pairs...")
    pairs = find_dfbench_pairs(edit_dir=edit_dir, source_dir=source_dir)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    logger.info("Found %d total pairs.", len(pairs))

    # Permute indices
    shuffled_idx = np.random.permutation(len(pairs))
    n_pristine = min(args.pristine_count, len(pairs))
    pristine_idx = shuffled_idx[:n_pristine]
    train_idx = shuffled_idx[n_pristine:]

    # 1. Pristine 5K PNG shards
    pristine_dir = out_dir / "pristine_5k"
    pristine_writer = ShardWriter(
        output_dir=pristine_dir, prefix="pristine_5k", max_samples_per_shard=1000
    )
    logger.info("Writing %d pristine PNG samples to %s...", len(pristine_idx), pristine_dir)
    for idx in tqdm(pristine_idx, desc="Pristine PNGs"):
        edit_p, src_p, sample_id, cat = pairs[idx]
        mask, area_ratio, is_valid = generate_pseudo_mask(edit_p, src_p)
        if not is_valid:
            continue
        img = Image.open(edit_p).convert("RGB")
        pristine_writer.add_sample(
            ShardSample(
                sample_id=sample_id,
                image=img,
                mask=mask,
                label=1,
                meta={"category": cat, "area_ratio": float(area_ratio), "is_pristine": True},
                image_format="png",
            )
        )
    pristine_shards = pristine_writer.close()
    logger.info("Wrote %d pristine shard archives.", len(pristine_shards))

    # 2. Resized 384px training shards
    train_dir = out_dir / f"train_{args.target_size}px"
    train_writer = ShardWriter(
        output_dir=train_dir, prefix=f"train_{args.target_size}px", max_samples_per_shard=1000
    )
    sz = (args.target_size, args.target_size)
    accepted, rejected = 0, 0

    logger.info("Writing training shards to %s...", train_dir)
    for idx in tqdm(train_idx, desc="Train Shards"):
        edit_p, src_p, sample_id, cat = pairs[idx]
        mask, area_ratio, is_valid = generate_pseudo_mask(edit_p, src_p)
        if not is_valid:
            rejected += 1
            continue
        accepted += 1
        img = Image.open(edit_p).convert("RGB").resize(sz, Image.BILINEAR)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(sz, Image.NEAREST)
        train_writer.add_sample(
            ShardSample(
                sample_id=sample_id,
                image=img,
                mask=mask_pil,
                label=1,
                meta={"category": cat, "area_ratio": float(area_ratio), "is_pristine": False},
                image_format="jpg",
            )
        )
    train_shards = train_writer.close()
    logger.info("Training shards complete: %d accepted, %d rejected.", accepted, rejected)

    # 3. Throughput check
    if train_shards:
        bench = verify_read_throughput(train_shards, max_samples=200, target_size=sz)
        logger.info(
            "Benchmark: %d samples in %.2fs -> %.1f img/s (Kaggle threshold: %s)",
            bench["samples_read"],
            bench["elapsed_sec"],
            bench["img_per_sec"],
            bench["meets_kaggle_threshold"],
        )
    else:
        bench = {"img_per_sec": 0, "meets_kaggle_threshold": False}

    # 4. Gate G0 report
    report = {
        "gate": "G0",
        "total_pairs": len(pairs),
        "pristine_subset_size": len(pristine_idx),
        "training_accepted": accepted,
        "training_rejected": rejected,
        "yield": round(accepted / max(accepted + rejected, 1), 4),
        "target_size": args.target_size,
        "throughput_img_per_sec": bench.get("img_per_sec", 0),
        "meets_kaggle_threshold": bench.get("meets_kaggle_threshold", False),
    }
    report_file = out_dir / "pseudo_mask_validation.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved validation report to %s", report_file)


if __name__ == "__main__":
    main()
