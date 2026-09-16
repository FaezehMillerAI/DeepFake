"""Frequency-domain and high-pass residual extraction for deepfake analysis.

Implements:
  1. Spatial Rich Model (SRM) high-pass residual filters (1st, 2nd, 3rd order).
  2. 2D Discrete Cosine Transform (DCT) block and spectrum extraction.
  3. Batch extraction and fp16 .npy serialization for fast Kaggle caching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.fft
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_srm_filters() -> torch.Tensor:
    """Returns the 3 canonical SRM (Spatial Rich Model) high-pass kernels.

    Shape: [3, 1, 5, 5] (padded to 5x5 for unified convolution).
      Kernel 0: 1st-order edge filter
      Kernel 1: 2nd-order Laplacian filter
      Kernel 2: 3rd-order / square 5x5 steganographic residual filter
    """
    k1 = np.zeros((5, 5), dtype=np.float32)
    k1[2, 1:4] = [-1.0, 2.0, -1.0]

    k2 = np.zeros((5, 5), dtype=np.float32)
    k2[1:4, 1:4] = np.array([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    k3 = np.array([
        [-1.0,  2.0,  -2.0,  2.0, -1.0],
        [ 2.0, -6.0,   8.0, -6.0,  2.0],
        [-2.0,  8.0, -12.0,  8.0, -2.0],
        [ 2.0, -6.0,   8.0, -6.0,  2.0],
        [-1.0,  2.0,  -2.0,  2.0, -1.0],
    ], dtype=np.float32)

    # Normalize by central coefficient scale
    k1 /= 2.0
    k2 /= 4.0
    k3 /= 12.0

    kernels = np.stack([k1, k2, k3], axis=0)[:, np.newaxis, :, :]  # [3, 1, 5, 5]
    return torch.from_numpy(kernels)


class SRMFilterLayer(nn.Module):
    """SRM High-pass residual extractor layer (fixed non-trainable weights)."""

    def __init__(self):
        super().__init__()
        filters = get_srm_filters()  # [3, 1, 5, 5]
        # Register as fixed buffer
        self.register_buffer("weights", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SRM filters across color channels or grayscale.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape [B, 3, H, W] or [B, 1, H, W], values in [0, 1].

        Returns
        -------
        residuals : torch.Tensor
            Shape [B, 3, H, W] containing the 3 SRM residuals (computed on luminance).
        """
        if x.shape[1] == 3:
            # Standard Rec.601 RGB to grayscale luminance
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Convolution with reflection padding to preserve spatial size
        padded = F.pad(gray, (2, 2, 2, 2), mode="reflect")
        out = F.conv2d(padded, self.weights)
        return out


def compute_2d_dct(image: Union[np.ndarray, torch.Tensor], block_size: int = 8) -> np.ndarray:
    """Computes 2D DCT log-magnitude spectrum or block-DCT coefficients.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Image in [0, 1] or [0, 255], shape [H, W] or [C, H, W].
    block_size : int
        Block size for block-DCT. If None or 0, computes full-image 2D DCT.

    Returns
    -------
    dct_map : np.ndarray (float32)
        Log-magnitude DCT spectrum, shape [H, W].
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        # [C, H, W] -> [H, W] luminance
        if img_np.shape[0] == 3:
            img_np = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        else:
            img_np = img_np[0]
    elif img_np.ndim == 3 and img_np.shape[-1] in [1, 3]:
        if img_np.shape[-1] == 3:
            img_np = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
        else:
            img_np = img_np[:, :, 0]

    h, w = img_np.shape
    if block_size and block_size > 1:
        # Block-wise 2D DCT (JPEG style)
        h_pad = (block_size - (h % block_size)) % block_size
        w_pad = (block_size - (w % block_size)) % block_size
        if h_pad > 0 or w_pad > 0:
            img_np = np.pad(img_np, ((0, h_pad), (0, w_pad)), mode="edge")

        h_padded, w_padded = img_np.shape
        blocks = img_np.reshape(
            h_padded // block_size, block_size, w_padded // block_size, block_size
        ).transpose(0, 2, 1, 3)
        dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")
        # Reassemble
        dct_map = dct_blocks.transpose(0, 2, 1, 3).reshape(h_padded, w_padded)[:h, :w]
    else:
        # Full-frame 2D DCT
        dct_map = scipy.fft.dctn(img_np, axes=(0, 1), norm="ortho")

    log_dct = np.log(np.abs(dct_map) + 1e-6)
    return log_dct.astype(np.float32)


def save_frequency_map_fp16(
    out_path: Union[str, Path],
    freq_data: Union[np.ndarray, torch.Tensor],
) -> None:
    """Saves frequency representation as compressed fp16 .npy file to save Kaggle storage."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(freq_data, torch.Tensor):
        freq_np = freq_data.detach().cpu().numpy()
    else:
        freq_np = np.asarray(freq_data)

    freq_fp16 = freq_np.astype(np.float16)
    np.save(str(p), freq_fp16)


def load_frequency_map_fp16(in_path: Union[str, Path]) -> torch.Tensor:
    """Loads fp16 .npy frequency file and returns float32 torch.Tensor."""
    freq_fp16 = np.load(str(in_path))
    return torch.from_numpy(freq_fp16.astype(np.float32))
