"""Benchmark acquisition helper for SID-Set, MagicBrush, AutoSplice, and INP-X.

Usage:
  python scripts/acquire_benchmarks.py --benchmark inpx
  python scripts/acquire_benchmarks.py --benchmark magicbrush
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_benchmarks")

BENCHMARK_REGISTRY = {
    "inpx": {
        "name": "INP-X (Inpainting Exchange)",
        "source": "https://www.kaggle.com/datasets/emirhanbilgic/inpainting-exchange/data",
        "description": "90K triplets (real, inpainted, exchanged) with ground truth masks",
        "kaggle_dataset": "emirhanbilgic/inpainting-exchange",
        "license": "MIT",
    },
    "magicbrush": {
        "name": "MagicBrush",
        "source": "https://huggingface.co/datasets/osunlp/MagicBrush",
        "description": "~10K real-to-edit triplets with turn index, instruction prompts, and ground truth masks",
        "hf_dataset": "osunlp/MagicBrush",
        "license": "CC-BY 4.0",
    },
    "autosplice": {
        "name": "AutoSplice",
        "source": "https://github.com/Shan-S/AutoSplice",
        "description": "Spliced images with ground truth manipulation masks",
        "license": "Academic",
    },
    "sid_set": {
        "name": "SID-Set (SIDA)",
        "source": "Academic Request / Inpainting benchmark",
        "description": "~20K tampered images with pixel-level ground truth masks",
        "license": "Research Only",
    },
}


def acquire_benchmark(benchmark_key: str, out_dir: Path) -> Dict:
    meta = BENCHMARK_REGISTRY.get(benchmark_key.lower())
    if not meta:
        raise ValueError(f"Unknown benchmark: {benchmark_key}. Available: {list(BENCHMARK_REGISTRY.keys())}")

    target_dir = out_dir / benchmark_key
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Acquiring benchmark metadata for %s into %s...", meta['name'], target_dir)

    info_file = target_dir / "info.json"
    with open(info_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Benchmark %s ready for cloud/local mounting.", meta['name'])
    return meta


def main():
    parser = argparse.ArgumentParser(description="Acquire and inspect validation benchmarks.")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "inpx", "magicbrush", "autosplice", "sid_set"])
    parser.add_argument("--out-dir", type=str, default="data/benchmarks")
    args = parser.parse_args()

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "all":
        for k in BENCHMARK_REGISTRY:
            acquire_benchmark(k, out_p)
    else:
        acquire_benchmark(args.benchmark, out_p)


if __name__ == "__main__":
    main()
