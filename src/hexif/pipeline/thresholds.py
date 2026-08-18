"""Validated operating thresholds for cell and phenotype calls.

Thresholds are scientific outputs. They must be fitted on the designated
training partition, recorded in JSON, and supplied explicitly. This module
never substitutes a default threshold or a historical run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES

VALID_CONFIDENCE = frozenset({"strong", "moderate", "weak"})


@dataclass(frozen=True)
class CalibratedThresholds:
    """Complete thresholds and optional evidence-derived confidence labels."""

    marker_thresholds: dict[int, float]
    marker_thresholds_by_name: dict[str, float]
    phenotype_thresholds: dict[str, float]
    marker_confidence: dict[str, str]
    phenotype_confidence: dict[str, str]
    provenance: dict[str, Any]
    source: Path

    def marker_threshold(self, channel_or_name: int | str) -> float:
        try:
            if isinstance(channel_or_name, int):
                return self.marker_thresholds[channel_or_name]
            return self.marker_thresholds_by_name[channel_or_name]
        except KeyError as exc:
            raise KeyError(f"no calibrated marker threshold for {channel_or_name!r}") from exc

    def phenotype_threshold(self, name: str) -> float:
        try:
            return self.phenotype_thresholds[name]
        except KeyError as exc:
            raise KeyError(f"no calibrated phenotype threshold for {name!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "marker_thresholds": {str(k): v for k, v in self.marker_thresholds.items()},
            "phenotype_thresholds": dict(self.phenotype_thresholds),
            "marker_confidence": dict(self.marker_confidence),
            "phenotype_confidence": dict(self.phenotype_confidence),
            "provenance": dict(self.provenance),
        }


def _probability_map(raw: object, required: set[str], label: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    values = {str(key): float(value) for key, value in raw.items()}
    invalid = {key: value for key, value in values.items() if not 0.0 <= value <= 1.0}
    if invalid:
        raise ValueError(f"{label} values must be in [0, 1]: {invalid}")
    return values


def _confidence_map(raw: object, required: set[str], label: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    values = {str(key): str(value) for key, value in raw.items()}
    invalid = {key: value for key, value in values.items() if value not in VALID_CONFIDENCE}
    if invalid:
        raise ValueError(f"{label} contains invalid strata: {invalid}")
    return values


def load_thresholds(path: str | Path) -> CalibratedThresholds:
    """Load a complete threshold artifact; missing or malformed data is fatal."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"threshold artifact does not exist: {source}")
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("threshold artifact must have schema_version 1")

    channel_keys = {str(int(channel)) for channel in FOCUSED_MARKERS}
    marker_values = _probability_map(
        payload.get("marker_thresholds"), channel_keys, "marker_thresholds"
    )
    phenotype_values = _probability_map(
        payload.get("phenotype_thresholds"), set(PHENOTYPE_NAMES), "phenotype_thresholds"
    )
    marker_by_channel = {int(key): value for key, value in marker_values.items()}
    marker_by_name = {
        name: marker_by_channel[int(channel)]
        for channel, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True)
    }
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("threshold artifact provenance must be a non-empty JSON object")

    return CalibratedThresholds(
        marker_thresholds=marker_by_channel,
        marker_thresholds_by_name=marker_by_name,
        phenotype_thresholds=phenotype_values,
        marker_confidence=_confidence_map(
            payload.get("marker_confidence"), set(MARKER_NAMES), "marker_confidence"
        ),
        phenotype_confidence=_confidence_map(
            payload.get("phenotype_confidence"),
            set(PHENOTYPE_NAMES),
            "phenotype_confidence",
        ),
        provenance=provenance,
        source=source.resolve(),
    )


def save_thresholds_json(thresholds: CalibratedThresholds, path: str | Path) -> None:
    Path(path).write_text(json.dumps(thresholds.to_dict(), indent=2) + "\n")
