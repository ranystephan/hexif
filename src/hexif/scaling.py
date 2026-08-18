"""Serializable channel-wise quantile scaling parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class QuantileScaler:
    """Container for fitted per-channel lower and upper quantiles."""

    def __init__(self, q_low: float = 1.0, q_high: float = 99.5, C: int = 20) -> None:
        if C < 1:
            raise ValueError("C must be positive")
        if not 0 <= q_low < q_high <= 100:
            raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 100")
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.C = int(C)
        self.qlo = np.zeros(self.C, dtype=np.float32)
        self.qhi = np.ones(self.C, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "q_low": self.q_low,
            "q_high": self.q_high,
            "qlo": self.qlo.tolist(),
            "qhi": self.qhi.tolist(),
            "C": self.C,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> QuantileScaler:
        scaler = cls(payload["q_low"], payload["q_high"], payload["C"])
        scaler.qlo = np.asarray(payload["qlo"], dtype=np.float32)
        scaler.qhi = np.asarray(payload["qhi"], dtype=np.float32)
        if scaler.qlo.shape != (scaler.C,) or scaler.qhi.shape != (scaler.C,):
            raise ValueError("qlo and qhi must each contain C values")
        if np.any(scaler.qhi <= scaler.qlo):
            raise ValueError("every fitted upper quantile must exceed its lower quantile")
        return scaler

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> QuantileScaler:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"scaler artifact does not exist: {source}")
        return cls.from_dict(json.loads(source.read_text()))
