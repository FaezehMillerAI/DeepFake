"""Configuration: dataclass-backed, YAML-driven, single source of truth.

Configs support one level of inheritance through a `_base_` key so that
experiment files stay short and diffable.

    cfg = load_config("configs/baselines/effnet_b0.yaml")
    cfg.train.lr
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a single `_base_` reference if present."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is not None:
        base = load_config((path.parent / base_ref).resolve())
        raw = _deep_merge(dict(base), raw)

    return Config(raw)


def save_config(cfg: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, sort_keys=False)
