"""PyTorch Dataset implementations for GRACE-DF.

Supports:
  1. PairedEditDataset: pairs of (real, edit) images with on-the-fly or cached pseudo-masks.
  2. MaskSupervisedDataset: datasets with ground-truth pixel masks (SID-Set, MagicBrush).
  3. StandardImageDataset: single images with classification labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Graceful fallback for non-PyTorch inspection environments
    class Dataset:
        pass

from src.data.pseudo_masks import generate_pseudo_mask


def default_transform(img: Image.Image, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Default image transform: resize to `size` and scale to [0, 1] float32 (C, H, W)."""
    resized = img.convert("RGB").resize(size, Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def mask_transform(mask: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize binary mask to `size` and format as (1, H, W) float32."""
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    resized = pil.resize(size, Image.NEAREST)
    arr = (np.asarray(resized, dtype=np.float32) > 127).astype(np.float32)
    return np.expand_dims(arr, axis=0)


class PairedEditDataset(Dataset):
    """Dataset for paired (real, edited) images, deriving or loading pseudo-masks.

    Each sample corresponds to one edited image (label=1) with its derived mask,
    or optionally interleaves the authentic source image (label=0, empty mask).
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[Union[str, Path], Union[str, Path]]],
        include_reals: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        cached_masks: Optional[Sequence[Union[str, Path, np.ndarray]]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            pairs: List of (real_image_path, edit_image_path) tuples.
            include_reals: If True, authentic real images are included as negative samples.
            image_size: Target (width, height) for resizing.
            cached_masks: Optional precomputed mask paths or arrays.
            transform: Optional custom image transformation callable.
        """
        self.pairs = list(pairs)
        self.include_reals = include_reals
        self.image_size = image_size
        self.cached_masks = list(cached_masks) if cached_masks else None
        self.transform = transform or (lambda img: default_transform(img, self.image_size))

        # Build index: if include_reals is True, index maps to either (pair_idx, 'edit') or (pair_idx, 'real')
        self.index_map: List[Tuple[int, str]] = []
        for i in range(len(self.pairs)):
            self.index_map.append((i, "edit"))
            if self.include_reals:
                self.index_map.append((i, "real"))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_idx, sample_type = self.index_map[idx]
        real_path, edit_path = self.pairs[pair_idx]

        real_img = Image.open(real_path).convert("RGB")

        if sample_type == "real":
            img_tensor = self.transform(real_img)
            # Empty mask for authentic images
            empty_mask = np.zeros((1, self.image_size[1], self.image_size[0]), dtype=np.float32)
            return {
                "image": torch.from_numpy(img_tensor) if "torch" in globals() else img_tensor,
                "label": 0,
                "mask": torch.from_numpy(empty_mask) if "torch" in globals() else empty_mask,
                "is_valid_mask": True,
                "path": str(real_path),
            }

        edit_img = Image.open(edit_path).convert("RGB")
        img_tensor = self.transform(edit_img)

        # Load or generate mask
        if self.cached_masks is not None:
            raw_mask = self.cached_masks[pair_idx]
            if isinstance(raw_mask, (str, Path)):
                mask_arr = np.asarray(Image.open(raw_mask))
            else:
                mask_arr = np.asarray(raw_mask)
            mask_t = mask_transform(mask_arr, self.image_size)
            is_valid = True
        else:
            pseudo_res = generate_pseudo_mask(real_img, edit_img)
            mask_t = mask_transform(pseudo_res.mask, self.image_size)
            is_valid = pseudo_res.valid

        return {
            "image": torch.from_numpy(img_tensor) if "torch" in globals() else img_tensor,
            "label": 1,
            "mask": torch.from_numpy(mask_t) if "torch" in globals() else mask_t,
            "is_valid_mask": is_valid,
            "path": str(edit_path),
        }


class MaskSupervisedDataset(Dataset):
    """Dataset with ground-truth manipulation masks (SID-Set, MagicBrush, AutoSplice)."""

    def __init__(
        self,
        samples: Sequence[Tuple[Union[str, Path], Optional[Union[str, Path]], int]],
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of (image_path, mask_path_or_None, label) tuples.
                     For real images (label=0), mask_path can be None.
            image_size: Target (width, height).
        """
        self.samples = list(samples)
        self.image_size = image_size
        self.transform = transform or (lambda img: default_transform(img, self.image_size))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        if mask_path is not None and Path(mask_path).exists():
            mask_arr = np.asarray(Image.open(mask_path))
            mask_t = mask_transform(mask_arr, self.image_size)
        else:
            mask_t = np.zeros((1, self.image_size[1], self.image_size[0]), dtype=np.float32)

        return {
            "image": torch.from_numpy(img_t) if "torch" in globals() else img_t,
            "label": int(label),
            "mask": torch.from_numpy(mask_t) if "torch" in globals() else mask_t,
            "path": str(img_path),
        }


def split_dataset_indices(
    total_samples: int,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 0
) -> Dict[str, List[int]]:
    """Deterministically partition indices into train, val, and test splits."""
    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    n_train = int(np.round(total_samples * train_ratio))
    n_val = int(np.round(total_samples * val_ratio))

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
