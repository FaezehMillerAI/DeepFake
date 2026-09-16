"""Tests for WebDataset shards and frequency extraction."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.freq import (
    SRMFilterLayer,
    compute_2d_dct,
    get_srm_filters,
    load_frequency_map_fp16,
    save_frequency_map_fp16,
)
from src.data.shards import ShardDataset, ShardSample, ShardWriter, verify_read_throughput


def test_srm_filters_and_layer():
    filters = get_srm_filters()
    assert filters.shape == (3, 1, 5, 5)

    layer = SRMFilterLayer()
    x = torch.rand(2, 3, 64, 64)
    residuals = layer(x)
    assert residuals.shape == (2, 3, 64, 64)
    # Grayscale image input
    x_gray = torch.rand(2, 1, 64, 64)
    res_gray = layer(x_gray)
    assert res_gray.shape == (2, 3, 64, 64)


def test_compute_2d_dct_and_fp16_serialization():
    img = np.random.rand(64, 64).astype(np.float32)
    dct_map = compute_2d_dct(img, block_size=8)
    assert dct_map.shape == (64, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp16_path = Path(tmpdir) / "dct.npy"
        save_frequency_map_fp16(fp16_path, dct_map)
        assert fp16_path.exists()

        loaded = load_frequency_map_fp16(fp16_path)
        assert isinstance(loaded, torch.Tensor)
        assert loaded.shape == (64, 64)
        # Check precision preserved within fp16 tolerance
        assert torch.allclose(loaded, torch.from_numpy(dct_map), atol=1e-2, rtol=1e-2)


def test_shard_writer_and_dataset_streaming():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ShardWriter(output_dir=tmpdir, prefix="test_shard", max_samples_per_shard=3)

        for i in range(5):
            img_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            mask_arr = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255
            sample = ShardSample(
                sample_id=f"sample_{i}",
                image=img_arr,
                mask=mask_arr,
                label=i % 2,
                meta={"category": "object_swap", "index": i},
                image_format="jpg" if i % 2 == 0 else "png",
            )
            writer.add_sample(sample)

        shard_files = writer.close()
        # With 5 samples and max_samples_per_shard=3, should produce 2 shard tar files
        assert len(shard_files) == 2
        for sf in shard_files:
            assert sf.exists()

        # Stream back using ShardDataset
        dataset = ShardDataset(shard_files, target_size=(64, 64))
        samples_read = list(dataset)
        assert len(samples_read) == 5

        for i, s in enumerate(samples_read):
            assert "image" in s
            assert "mask" in s
            assert "label" in s
            assert "meta" in s
            assert s["image"].shape == (3, 64, 64)
            assert s["mask"].shape == (1, 64, 64)
            assert s["label"].item() in [0, 1]
            assert s["meta"]["sample_id"] == f"sample_{i}"

        # Throughput test
        bench = verify_read_throughput(shard_files, max_samples=5, target_size=(64, 64))
        assert bench["samples_read"] == 5
        assert bench["img_per_sec"] > 0
