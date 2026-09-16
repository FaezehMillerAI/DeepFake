"""Kaggle-proof checkpoint and resume harness.

Enforces Rule 1 of GRACE-DF:
    "Every training script resumes from runs/<name>/ckpt_last.pt on start.
     Assume every Kaggle session dies at 11h58m."

Features:
  1. Atomic writes via temporary files to prevent corrupt checkpoints on hard aborts.
  2. Full state preservation: model, optimizer, lr_scheduler, GradScaler (fp16), epoch, step.
  3. RNG state preservation: python.random, numpy, torch, torch.cuda (exact resumption).
  4. Automatic best-checkpoint mirroring (ckpt_best.pt).
"""
from __future__ import annotations

import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def find_resume_checkpoint(dir_or_path: Union[str, Path]) -> Optional[Path]:
    """Find the checkpoint to resume from.

    If given a directory, looks for `ckpt_last.pt`. If given a file path, checks
    if that file exists.
    """
    p = Path(dir_or_path)
    if p.is_dir():
        candidate = p / "ckpt_last.pt"
        return candidate if candidate.is_file() else None
    return p if p.is_file() else None


def save_checkpoint(
    target: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    step: int = 0,
    best_score: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> Path:
    """Atomically save a full training checkpoint.

    Args:
        target: Target directory (will save as `ckpt_last.pt`) or full file path.
        model: PyTorch model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        scaler: AMP GradScaler instance.
        epoch: Current epoch index (0-indexed).
        step: Global iteration step.
        best_score: Highest validation metric achieved so far.
        metrics: Current epoch/step evaluation metrics.
        extra: Any supplementary metadata (e.g. config dict).
        is_best: If True, also writes `ckpt_best.pt` in the same directory.

    Returns:
        Path to the saved checkpoint.
    """
    target_path = Path(target)
    if target_path.is_dir() or target_path.suffix != ".pt":
        target_path.mkdir(parents=True, exist_ok=True)
        final_file = target_path / "ckpt_last.pt"
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        final_file = target_path

    tmp_file = final_file.with_name(f"{final_file.stem}.tmp_{os.getpid()}.pt")

    # Capture complete RNG state for exact reproducibility across session boundaries
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

    # Unwrap DataParallel or DDP if wrapped
    raw_model = model.module if hasattr(model, "module") else model

    payload: Dict[str, Any] = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "best_score": float(best_score) if best_score is not None else None,
        "metrics": dict(metrics) if metrics is not None else {},
        "extra": dict(extra) if extra is not None else {},
        "rng_states": rng_states,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # 1. Write to temporary file first
    torch.save(payload, tmp_file)

    # 2. Atomic rename (replaces target atomically on POSIX)
    os.replace(tmp_file, final_file)

    # 3. Mirror to ckpt_best.pt if marked as best
    if is_best:
        best_file = final_file.parent / "ckpt_best.pt"
        best_tmp = final_file.parent / f"ckpt_best.tmp_{os.getpid()}.pt"
        torch.save(payload, best_tmp)
        os.replace(best_tmp, best_file)

    return final_file


def load_checkpoint(
    target: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Resume model and training state from checkpoint if available.

    Args:
        target: Checkpoint directory or file path.
        model: Model instance to load weights into.
        optimizer: Optimizer instance to restore state into.
        scheduler: Scheduler instance to restore state into.
        scaler: GradScaler instance to restore state into.
        device: Device to map weights onto ('cpu' or 'cuda').
        restore_rng: Whether to restore random number generator states.

    Returns:
        Dictionary indicating resume status and restored state:
        {
            'resumed': bool,
            'epoch': int,
            'step': int,
            'best_score': Optional[float],
            'metrics': dict,
            'extra': dict,
            'path': Optional[str]
        }
    """
    ckpt_path = find_resume_checkpoint(target)
    if ckpt_path is None:
        return {
            "resumed": False,
            "epoch": 0,
            "step": 0,
            "best_score": None,
            "metrics": {},
            "extra": {},
            "path": None,
        }

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions where weights_only argument doesn't exist
        checkpoint = torch.load(ckpt_path, map_location=device)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    if restore_rng and "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu() if isinstance(rng["torch"], torch.Tensor) else rng["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return {
        "resumed": True,
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_score": checkpoint.get("best_score"),
        "metrics": checkpoint.get("metrics", {}),
        "extra": checkpoint.get("extra", {}),
        "path": str(ckpt_path),
    }
