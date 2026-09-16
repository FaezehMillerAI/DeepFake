"""Baseline training engine with Kaggle-proof checkpointing and JSON logging.

Fulfills:
  - Rule 1: Always resume from runs/<name>/ckpt_last.pt on start.
  - Rule 2: Log all metrics to runs/<name>/run.json.
  - Mixed precision fp16 + GradScaler.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.metrics.calibration import brier, ece
from src.models import build_model
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.utils.runlog import RunLogger
from src.utils.seed import set_seed


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Run validation evaluation: accuracy, loss, ECE, and Brier score."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    if total_samples == 0:
        return {"val/loss": 0.0, "val/acc": 0.0, "val/ece": 0.0, "val/brier": 0.0}

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    preds = (probs_arr >= 0.5).astype(int)

    acc = float(np.mean(preds == targets_arr))
    avg_loss = float(total_loss / total_samples)
    val_ece = float(ece(probs_arr, targets_arr, bins=15))
    val_brier = float(brier(probs_arr, targets_arr))

    return {
        "val/loss": avg_loss,
        "val/acc": acc,
        "val/ece": val_ece,
        "val/brier": val_brier,
    }


def train_baseline(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a baseline detector for the configured number of epochs.

    Automatically resumes from runs/<name>/ckpt_last.pt if present.
    """
    set_seed(cfg.seed)

    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    run_dir = Path(cfg.log.out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = build_model(cfg)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    # Scheduler: Cosine Annealing
    total_epochs = int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=1e-6,
    )

    # Mixed precision: fp16 only on CUDA (P100 / T4)
    use_cuda_amp = (cfg.train.amp == "fp16") and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp) if use_cuda_amp else None

    # Checkpoint resume check (Rule 1)
    resume_info = load_checkpoint(run_dir, model, optimizer, scheduler, scaler, device=str(device))
    start_epoch = resume_info["epoch"]
    global_step = resume_info["step"]
    best_score = resume_info["best_score"] or 0.0

    criterion = nn.CrossEntropyLoss()

    with RunLogger(cfg.name, config=dict(cfg), seed=cfg.seed, out_dir=cfg.log.out_dir) as log:
        if resume_info["resumed"]:
            log.note(f"Resumed from epoch {start_epoch}, step {global_step}, best={best_score:.4f}")
            print(f"Resumed {cfg.name} from epoch {start_epoch} (step {global_step})")

        grad_accum = max(1, int(getattr(cfg.train, "grad_accum", 1)))

        for epoch in range(start_epoch, total_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_samples = 0
            t0 = time.time()

            optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                batch_size = images.size(0)

                # Autocast block
                if use_cuda_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(images)
                        loss = criterion(logits, labels) / grad_accum
                    scaler.scale(loss).backward()
                else:
                    logits = model(images)
                    loss = criterion(logits, labels) / grad_accum
                    loss.backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * grad_accum * batch_size
                epoch_samples += batch_size

            scheduler.step()
            train_loss = epoch_loss / max(1, epoch_samples)
            log.metric("train/loss", train_loss, epoch=epoch, step=global_step)

            # Validation pass
            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, criterion)
                for k, v in val_metrics.items():
                    log.metric(k, v, epoch=epoch)

                val_acc = val_metrics.get("val/acc", 0.0)
                is_best = val_acc > best_score
                if is_best:
                    best_score = val_acc

                print(
                    f"Epoch {epoch+1:02d}/{total_epochs:02d} [{time.time()-t0:.1f}s] "
                    f"train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f} "
                    f"{'★ BEST' if is_best else ''}"
                )
            else:
                is_best = False
                print(f"Epoch {epoch+1:02d}/{total_epochs:02d} [{time.time()-t0:.1f}s] train_loss: {train_loss:.4f}")

            # Save checkpoint atomically at end of each epoch
            save_checkpoint(
                run_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                step=global_step,
                best_score=best_score,
                metrics=val_metrics,
                is_best=is_best,
            )

        log.result("best_val_acc", best_score)
        log.result("final_step", global_step)

    return {
        "name": cfg.name,
        "best_val_acc": best_score,
        "final_step": global_step,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a GRACE-DF baseline classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'mps', 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    print(f"Loaded config: {args.config} (model: {cfg.model.arch}, epochs: {cfg.train.epochs})")


if __name__ == "__main__":
    main()
