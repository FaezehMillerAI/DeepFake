"""JSON run logging.

Every experiment writes exactly one JSON file. Tables and figures in the paper
are generated from these files — no number is ever typed by hand.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or 'nogit' outside a repository."""
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    if short:
        args.append("HEAD")
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"


def _dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except Exception:
        return False


class RunLogger:
    """Accumulates metrics for one run and writes a single JSON on close.

        with RunLogger("baseline_effnet_b0", cfg, seed=0) as log:
            log.metric("val/acc", 0.97, epoch=3)
            log.result("test/acc", 0.965)
    """

    def __init__(self, name: str, config: Optional[Dict] = None,
                 seed: Optional[int] = None, out_dir: str | Path = "runs"):
        self.name = name
        self.dir = Path(out_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.payload: Dict[str, Any] = {
            "name": name,
            "seed": seed,
            "git_sha": git_sha(),
            "git_dirty": _dirty(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "config": dict(config) if config else {},
            "history": [],
            "results": {},
            "notes": [],
        }

    # ---- during the run -------------------------------------------------
    def metric(self, key: str, value: float, **context: Any) -> None:
        """A value that changes over training (per epoch, per step)."""
        entry = {"key": key, "value": float(value), "t": round(time.time() - self.started, 2)}
        entry.update(context)
        self.payload["history"].append(entry)

    def result(self, key: str, value: Any) -> None:
        """A final number that may appear in the paper."""
        self.payload["results"][key] = value

    def note(self, text: str) -> None:
        self.payload["notes"].append(text)

    # ---- lifecycle ------------------------------------------------------
    def close(self, status: str = "ok") -> Path:
        self.payload["status"] = status
        self.payload["duration_s"] = round(time.time() - self.started, 1)
        self.payload["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self.dir / "run.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2)
        return path

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close("ok" if exc_type is None else f"failed: {exc_type.__name__}")
        return False


def load_runs(root: str | Path = "runs") -> list:
    """Every run.json under `root`, for table generation."""
    out = []
    for path in sorted(Path(root).rglob("run.json")):
        with open(path, "r", encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out
