"""Unit tests for Kaggle checkpoint and resume harness (Rule 1)."""
import random
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.train.checkpoint import find_resume_checkpoint, load_checkpoint, save_checkpoint


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tmp_run_dir(tmp_path):
    d = tmp_path / "test_run"
    d.mkdir()
    return d


def test_save_and_load_restores_exact_weights_and_step(tmp_run_dir):
    model = SimpleNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 optimization step to mutate model and optimizer state
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Save checkpoint
    saved_path = save_checkpoint(
        tmp_run_dir,
        model=model,
        optimizer=optimizer,
        epoch=3,
        step=120,
        best_score=0.88,
        metrics={"val/acc": 0.88},
    )

    assert saved_path.exists()
    assert saved_path.name == "ckpt_last.pt"

    # Create fresh model with different weights
    new_model = SimpleNet()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    # Confirm weights differ before resuming
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert not torch.allclose(p1, p2)

    # Resume
    res = load_checkpoint(tmp_run_dir, new_model, new_optimizer)

    assert res["resumed"] is True
    assert res["epoch"] == 3
    assert res["step"] == 120
    assert res["best_score"] == pytest.approx(0.88)
    assert res["metrics"]["val/acc"] == pytest.approx(0.88)

    # Confirm weights now match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_best_checkpoint_mirroring(tmp_run_dir):
    model = SimpleNet()
    save_checkpoint(tmp_run_dir, model, epoch=1, step=50, is_best=True)

    assert (tmp_run_dir / "ckpt_last.pt").exists()
    assert (tmp_run_dir / "ckpt_best.pt").exists()


def test_missing_checkpoint_returns_clean_defaults(tmp_run_dir):
    empty_dir = tmp_run_dir / "empty"
    empty_dir.mkdir()
    model = SimpleNet()

    res = load_checkpoint(empty_dir, model)
    assert res["resumed"] is False
    assert res["epoch"] == 0
    assert res["step"] == 0
    assert res["path"] is None


def test_rng_restoration_reproduces_identical_sequence(tmp_run_dir):
    model = SimpleNet()

    # Set seeds and save
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    save_checkpoint(tmp_run_dir, model, epoch=0, step=0)

    # Generate sequence A
    seq_py_a = [random.random() for _ in range(5)]
    seq_np_a = np.random.randn(5).tolist()
    seq_th_a = torch.randn(5).tolist()

    # Advance generators further
    _ = [random.random() for _ in range(10)]

    # Restore from checkpoint
    load_checkpoint(tmp_run_dir, model)

    # Generate sequence B after restore
    seq_py_b = [random.random() for _ in range(5)]
    seq_np_b = np.random.randn(5).tolist()
    seq_th_b = torch.randn(5).tolist()

    # Sequences must be identical
    assert seq_py_a == seq_py_b
    assert seq_np_a == seq_np_b
    assert seq_th_a == seq_th_b
