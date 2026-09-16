"""The corruption bank must be deterministic — a sweep has to be reproducible."""
import numpy as np
import pytest

from src.attacks import apply_corruption, corruption_grid, CORRUPTION_BANK


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_grid_covers_every_corruption():
    names = {n for n, _ in corruption_grid()}
    assert names == set(CORRUPTION_BANK)


def test_grid_has_expected_size():
    assert len(corruption_grid()) == sum(len(v[1]) for v in CORRUPTION_BANK.values())


@pytest.mark.parametrize("name", list(CORRUPTION_BANK))
def test_corruption_is_deterministic(img, name):
    sev = CORRUPTION_BANK[name][1][0]
    a = np.asarray(apply_corruption(img, name, sev, seed=0))
    b = np.asarray(apply_corruption(img, name, sev, seed=0))
    assert np.array_equal(a, b)


def test_noise_seed_changes_output(img):
    a = np.asarray(apply_corruption(img, "noise", 20, seed=0))
    b = np.asarray(apply_corruption(img, "noise", 20, seed=1))
    assert not np.array_equal(a, b)


def test_corruption_preserves_shape(img):
    for name, sev in corruption_grid():
        out = np.asarray(apply_corruption(img, name, sev, seed=0))
        assert out.shape == img.shape, f"{name}@{sev} changed shape"


def test_unknown_corruption_raises(img):
    with pytest.raises(KeyError):
        apply_corruption(img, "sharpen", 1)
