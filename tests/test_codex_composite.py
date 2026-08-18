"""Deterministic unit tests for CODEX composite rendering."""

from __future__ import annotations

import numpy as np
import pytest

from hexif.pipeline.codex_composite import build_codex_composite
from hexif.scaling import QuantileScaler


def _scaler(channels: int = 53) -> QuantileScaler:
    scaler = QuantileScaler(C=channels)
    scaler.qlo = np.zeros(channels, dtype=np.float32)
    scaler.qhi = np.ones(channels, dtype=np.float32)
    return scaler


def test_composite_shape_channels_and_dtype(tmp_path):
    rng = np.random.default_rng(7)
    array = rng.integers(0, 1000, size=(53, 32, 32), dtype=np.uint16)
    path = tmp_path / "codex.npy"
    np.save(path, array)
    rgb, info = build_codex_composite(path, _scaler())
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
    assert [(row["idx"], row["name"], row["rgb"]) for row in info["channels"]] == [
        (7, "CD45", "red"),
        (52, "panCK", "green"),
        (0, "DAPI", "blue"),
    ]


def test_uniform_nonzero_channels_are_visible(tmp_path):
    array = np.zeros((20, 20, 53), dtype=np.float32)
    array[..., 0] = np.float32(np.log1p(0.5))
    array[..., 7] = np.float32(np.log1p(0.25))
    array[..., 52] = np.float32(np.log1p(0.75))
    path = tmp_path / "codex.npy"
    np.save(path, array)
    rgb, _ = build_codex_composite(path, _scaler())
    assert np.all(rgb == 255)


def test_rejects_invalid_shape_and_percentiles(tmp_path):
    path = tmp_path / "codex.npy"
    np.save(path, np.zeros((32, 32), dtype=np.uint16))
    with pytest.raises(ValueError, match="must be 3-D"):
        build_codex_composite(path, _scaler())
    with pytest.raises(ValueError, match="percentile_window"):
        build_codex_composite(path, _scaler(), percentile_window=(50.0, 50.0))
