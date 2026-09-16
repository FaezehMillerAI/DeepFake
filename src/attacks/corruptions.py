"""Deterministic corruption bank.

Every corruption is a pure function of (image, severity, seed) so a sweep is
exactly reproducible. Severities match Section 5 / E2 of the plan.

    from src.attacks import corruption_grid, apply_corruption
    for name, sev in corruption_grid():
        out = apply_corruption(img, name, sev, seed=0)

IMPORTANT: run compression experiments on the pristine-PNG subset. Studying
JPEG artefacts on already-JPEG data measures the wrong thing.
"""
from __future__ import annotations

import io
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def jpeg(img, quality: int) -> Image.Image:
    """Single JPEG round-trip at the given quality."""
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def webp(img, quality: int) -> Image.Image:
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img, sigma: float) -> Image.Image:
    return _to_pil(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def gaussian_noise(img, sigma: float, seed: int = 0) -> Image.Image:
    """Additive Gaussian noise in 0-255 units. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(_to_pil(img), dtype=np.float64)
    noisy = arr + rng.normal(0.0, float(sigma), arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def downscale(img, size: int) -> Image.Image:
    """Downscale to `size` on the short side, then back up — the resolution test."""
    pil = _to_pil(img)
    w, h = pil.size
    small = pil.resize((max(1, int(size)), max(1, int(size))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def resave_chain(img, n: int, quality: int = 70) -> Image.Image:
    """`n` successive JPEG saves — the social-media transmission model.

    DeepFakeBuster models the resulting decay as A(q,n) = A0 * exp(-lambda*n*(100-q)/100).
    """
    out = _to_pil(img)
    for _ in range(int(n)):
        out = jpeg(out, quality)
    return out


CORRUPTION_BANK: Dict[str, Tuple[Callable, List]] = {
    "jpeg":      (jpeg,          [90, 70, 50, 30, 10]),
    "blur":      (blur,          [0.5, 1.0, 2.0, 3.0]),
    "noise":     (gaussian_noise,[5, 10, 20]),
    "downscale": (downscale,     [256, 128, 64, 32]),
    "webp":      (webp,          [90, 70, 50, 30]),
    "resave":    (resave_chain,  [1, 3, 5, 10]),
}


def corruption_grid(names: List[str] | None = None) -> List[Tuple[str, float]]:
    """Every (name, severity) pair to sweep. 25 points by default."""
    keys = names or list(CORRUPTION_BANK.keys())
    return [(k, s) for k in keys for s in CORRUPTION_BANK[k][1]]


def apply_corruption(img, name: str, severity, seed: int = 0) -> Image.Image:
    """Dispatch to the named corruption. Deterministic given (name, severity, seed)."""
    if name not in CORRUPTION_BANK:
        raise KeyError(f"unknown corruption {name!r}; have {sorted(CORRUPTION_BANK)}")
    fn = CORRUPTION_BANK[name][0]
    if fn is gaussian_noise:
        return fn(img, severity, seed=seed)
    return fn(img, severity)
