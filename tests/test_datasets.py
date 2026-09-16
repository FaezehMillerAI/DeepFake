"""Unit tests for PyTorch dataset classes and splitting helpers."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from src.data.datasets import (
    PairedEditDataset,
    MaskSupervisedDataset,
    split_dataset_indices,
    default_transform,
    mask_transform,
)
from src.data.acquire import find_dfbench_pairs



@pytest.fixture
def dummy_image_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()

    # Create synthetic real and edited images
    real_path = d / "real_01.png"
    edit_path = d / "edit_01.png"
    mask_path = d / "mask_01.png"

    img_real = Image.fromarray(np.full((100, 100, 3), 100, dtype=np.uint8))
    img_edit = Image.fromarray(np.full((100, 100, 3), 100, dtype=np.uint8))
    # Modify a patch
    arr_edit = np.asarray(img_edit).copy()
    arr_edit[20:50, 20:50] = 220
    img_edit = Image.fromarray(arr_edit)

    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[20:50, 20:50] = 255
    img_mask = Image.fromarray(mask_arr)

    img_real.save(real_path)
    img_edit.save(edit_path)
    img_mask.save(mask_path)

    return {
        "dir": d,
        "real": real_path,
        "edit": edit_path,
        "mask": mask_path,
    }


def test_split_dataset_indices_properties():
    total = 200
    splits = split_dataset_indices(total, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(splits["train"]) == 140
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 30

    # Ensure no overlap and complete coverage
    all_indices = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_indices) == total
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0


def test_paired_edit_dataset_shapes_and_labels(dummy_image_dir):
    pairs = [(dummy_image_dir["real"], dummy_image_dir["edit"])]
    ds = PairedEditDataset(pairs, include_reals=True, image_size=(128, 128))

    # 1 pair with include_reals=True -> 2 samples
    assert len(ds) == 2

    # First is edit
    sample_edit = ds[0]
    assert sample_edit["label"] == 1
    assert sample_edit["image"].shape == (3, 128, 128)
    assert sample_edit["mask"].shape == (1, 128, 128)
    assert sample_edit["mask"].sum() > 0

    # Second is real
    sample_real = ds[1]
    assert sample_real["label"] == 0
    assert sample_real["image"].shape == (3, 128, 128)
    assert sample_real["mask"].shape == (1, 128, 128)
    assert sample_real["mask"].sum() == 0.0


def test_mask_supervised_dataset(dummy_image_dir):
    samples = [
        (dummy_image_dir["edit"], dummy_image_dir["mask"], 1),
        (dummy_image_dir["real"], None, 0),
    ]
    ds = MaskSupervisedDataset(samples, image_size=(64, 64))

    assert len(ds) == 2

    s1 = ds[0]
    assert s1["label"] == 1
    assert s1["image"].shape == (3, 64, 64)
    assert s1["mask"].shape == (1, 64, 64)
    assert s1["mask"].sum() > 0

    s2 = ds[1]
    assert s2["label"] == 0
    assert s2["mask"].sum() == 0.0


def test_find_dfbench_pairs(tmp_path):
    src_dir = tmp_path / "partial_source"
    edit_dir = tmp_path / "edit"
    src_dir.mkdir()
    edit_dir.mkdir()

    # Create dummy source files
    (src_dir / "10013.jpg").write_bytes(b"dummy")
    (src_dir / "10014.jpg").write_bytes(b"dummy")
    (src_dir / "99999.jpg").write_bytes(b"dummy")

    # Create dummy edit files following DFBench naming convention: {id}_{prompt}.png
    (edit_dir / "10013_Add_a_butterfly.png").write_bytes(b"dummy")
    (edit_dir / "10013_Change_texture_to_marble.png").write_bytes(b"dummy")
    (edit_dir / "10014_Make_skin_rough.png").write_bytes(b"dummy")
    (edit_dir / "12345_Unmatched_edit.png").write_bytes(b"dummy")

    pairs = find_dfbench_pairs(edit_dir, src_dir)

    # Should match 3 edits (2 for 10013, 1 for 10014)
    assert len(pairs) == 3
    for source_path, edit_path in pairs:
        source_id = edit_path.stem.split("_")[0]
        assert source_path.stem == source_id
        assert source_path.exists()
        assert edit_path.exists()

