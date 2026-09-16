"""End-to-end smoke test for the evidence drift evaluation pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.eval.evaluate_drift import evaluate_model_drift
from src.models import build_model


class TinyEvalDataset(Dataset):
    """Synthetic in-memory dataset for pipeline smoke testing."""

    def __init__(self, n: int = 8, size: int = 64):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Semi-deterministic image and mask
        torch.manual_seed(idx)
        img = torch.rand(3, self.size, self.size)
        mask = torch.zeros(1, self.size, self.size)
        mask[:, self.size // 4 : 3 * self.size // 4, self.size // 4 : 3 * self.size // 4] = 1.0
        label = 1 if idx % 2 == 1 else 0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


def test_evaluate_model_drift_end_to_end():
    model = build_model("resnet18", pretrained=False)
    model.eval()

    dataset = TinyEvalDataset(n=6, size=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Run smoke test on CPU with 1 corruption and 1 severity for speed
    results = evaluate_model_drift(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        explanation_method="gradcam",
        conditions=["blur"],
        severities=[1.0],
        include_adversarial=True,
        max_eval_samples=6,
    )

    assert results["explanation_method"] == "gradcam"
    assert results["total_samples"] == 6
    assert results["conditions_evaluated"] >= 3  # clean, blur, pgd_eps2, pgd_eps4, pgd_eps8

    condition_names = [r["condition"] for r in results["results"]]
    assert "clean" in condition_names
    assert "blur" in condition_names

    # Clean condition check
    clean_res = [r for r in results["results"] if r["condition"] == "clean"][0]
    assert clean_res["mean_esi"] == 1.0
    assert clean_res["mean_esi_rho"] == 1.0
    assert 0.0 <= clean_res["accuracy"] <= 1.0
    assert 0.0 <= clean_res["ece_15bin"] <= 1.0

    # Perturbed condition check
    corrupt_res = [r for r in results["results"] if r["condition"] == "blur"][0]
    assert 0.0 <= corrupt_res["mean_esi"] <= 1.0
    assert -1.0 <= corrupt_res["mean_esi_rho"] <= 1.0
    assert 0.0 <= corrupt_res["mean_gacc"] <= 1.0
    assert 0.0 <= corrupt_res["mean_iou"] <= 1.0
