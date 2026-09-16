"""Data acquisition, metadata inspection, and provenance logging.

Handles metadata downloading from Hugging Face (IntMeGroup/DFBench), verifies
real-to-edited image pairing for pseudo-mask generation, and maintains
data/MANIFEST.md for the Week 17 license audit.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DFBENCH_REPO = "IntMeGroup/DFBench"
HF_BASE_URL = f"https://huggingface.co/datasets/{DFBENCH_REPO}/raw/main"

MANIFEST_TEMPLATE = """# Data Manifest & Provenance

This manifest documents all datasets, versions, licenses, and checksums used in
the GRACE-DF project. Required for the Week 17 license audit and Week 22 code release.

| Dataset | Source / URL | Split / Subset | Format | License | Status |
|---|---|---|---|---|---|
| **DFBench** | `https://huggingface.co/datasets/IntMeGroup/DFBench` | Real (45K), AI-edit (15K), AI-gen sampled | WebDataset / PNG | Research only | Metadata inspected |
| **SID-Set (SIDA)** | Public / Academic request | ~20K tampered images with masks | JPEG + PNG masks | Academic | Pending acquisition |
| **MagicBrush** | `https://github.com/OSU-NLP-Group/MagicBrush` | ~10K edit triplets + masks | PNG | CC-BY 4.0 | Pending acquisition |
| **AutoSplice** | `https://github.com/Shan-S/AutoSplice` | ~5K spliced + masks | JPEG + masks | Academic | Request form |
| **INP-X** | `https://github.com/emirhanbilgic/INP-X` | 90K inpainting benchmark | PNG | Academic | Diagnostic |
| **FaceForensics++** | Technical Univ. Munich | 10K face frames | MP4/PNG | Non-commercial | Secondary |
| **Celeb-DFv2** | Deepfake detection benchmark | 5K face frames | MP4/PNG | Research only | Secondary |

---
*Last updated: {updated_at}*
"""


def download_hf_metadata(filename: str, out_dir: str | Path = "data/metadata") -> Path:
    """Download a small metadata JSON/JSONL file from Hugging Face Hub."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename

    url = f"{HF_BASE_URL}/{filename}"
    print(f"Fetching {url} -> {target}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "grace-df-research/1.0"}
    )
    with urllib.request.urlopen(req) as resp, open(target, "wb") as f:
        f.write(resp.read())

    print(f"Downloaded {target} ({target.stat().st_size / 1024:.1f} KB)")
    return target


def inspect_dfbench_metadata(path: str | Path) -> Dict[str, Any]:
    """Parse and inspect a DFBench metadata JSON/JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            records = json.loads(content)
        else:
            for line in content.splitlines():
                if line.strip():
                    records.append(json.loads(line))

    categories: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    has_pair_info = 0

    for r in records:
        lbl = str(r.get("label", r.get("type", "unknown")))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        cat = r.get("category", r.get("sub_category", "none"))
        categories[cat] = categories.get(cat, 0) + 1

        if "real_path" in r or "source_image" in r or "src" in r or "original" in r:
            has_pair_info += 1

    summary = {
        "file": str(path),
        "total_records": len(records),
        "label_counts": label_counts,
        "categories": categories,
        "explicit_pairs_found": has_pair_info,
        "sample_record": records[0] if records else {},
    }
    return summary


def update_manifest(manifest_path: str | Path = "data/MANIFEST.md") -> Path:
    """Initialize or update the data/MANIFEST.md document."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = MANIFEST_TEMPLATE.format(updated_at=now_str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def find_dfbench_pairs(edit_dir: str | Path, source_dir: str | Path) -> List[Tuple[Path, Path]]:
    """Match DFBench edited images with their authentic source images.

    In DFBench, edited images follow the naming format:
        edit/{source_id}_{prompt/instruction}.png
    And the authentic source image is:
        partial_source/{source_id}.jpg (or .png/.jpeg)

    Returns:
        List of (source_path, edit_path) tuples.
    """
    edit_p = Path(edit_dir)
    src_p = Path(source_dir)

    if not edit_p.exists():
        raise FileNotFoundError(f"Edit directory not found: {edit_p}")
    if not src_p.exists():
        raise FileNotFoundError(f"Source directory not found: {src_p}")

    # Build source index by stem
    source_index: Dict[str, Path] = {}
    for f in src_p.iterdir():
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            source_index[f.stem] = f

    matched_pairs: List[Tuple[Path, Path]] = []
    for edit_file in sorted(edit_p.iterdir()):
        if not edit_file.is_file() or edit_file.name.startswith("."):
            continue

        # Extract source_id before first underscore
        stem = edit_file.stem
        source_id = stem.split("_")[0]

        if source_id in source_index:
            matched_pairs.append((source_index[source_id], edit_file))

    return matched_pairs


def main():
    parser = argparse.ArgumentParser(description="GRACE-DF data acquisition & metadata helper")
    parser.add_argument("--fetch-meta", action="store_true", help="Download DFBench metadata from HuggingFace")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect a downloaded metadata JSON/JSONL file")
    parser.add_argument("--init-manifest", action="store_true", help="Generate or update data/MANIFEST.md")

    args = parser.parse_args()

    if args.init_manifest:
        p = update_manifest()
        print(f"Manifest written to {p}")

    if args.fetch_meta:
        for f in ["img_test.json", "img_val.json"]:
            try:
                download_hf_metadata(f)
            except Exception as e:
                print(f"Could not fetch {f}: {e}")

    if args.inspect:
        summary = inspect_dfbench_metadata(args.inspect)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
