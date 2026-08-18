"""Unit tests for validated operating-threshold artifacts."""

from __future__ import annotations

import json

import pytest

from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES
from hexif.pipeline.thresholds import load_thresholds


def _payload() -> dict:
    return {
        "schema_version": 1,
        "marker_thresholds": {
            str(channel): 0.1 + index / 100 for index, channel in enumerate(FOCUSED_MARKERS)
        },
        "phenotype_thresholds": {
            name: 0.2 + index / 100 for index, name in enumerate(PHENOTYPE_NAMES)
        },
        "marker_confidence": {name: "moderate" for name in MARKER_NAMES},
        "phenotype_confidence": {name: "weak" for name in PHENOTYPE_NAMES},
        "provenance": {"fixture": "deterministic unit test"},
    }


def _write(tmp_path, payload: dict | None = None):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(_payload() if payload is None else payload))
    return path


def test_loads_complete_artifact(tmp_path):
    thresholds = load_thresholds(_write(tmp_path))
    assert set(thresholds.marker_thresholds) == set(FOCUSED_MARKERS)
    assert set(thresholds.marker_thresholds_by_name) == set(MARKER_NAMES)
    assert set(thresholds.phenotype_thresholds) == set(PHENOTYPE_NAMES)
    assert thresholds.marker_confidence == {name: "moderate" for name in MARKER_NAMES}


def test_missing_artifact_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_thresholds(tmp_path / "missing.json")


def test_missing_threshold_is_fatal(tmp_path):
    payload = _payload()
    payload["marker_thresholds"].pop(str(FOCUSED_MARKERS[0]))
    with pytest.raises(ValueError, match="keys mismatch"):
        load_thresholds(_write(tmp_path, payload))


def test_out_of_range_threshold_is_fatal(tmp_path):
    payload = _payload()
    payload["phenotype_thresholds"][PHENOTYPE_NAMES[0]] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_thresholds(_write(tmp_path, payload))


def test_missing_provenance_is_fatal(tmp_path):
    payload = _payload()
    payload["provenance"] = {}
    with pytest.raises(ValueError, match="provenance"):
        load_thresholds(_write(tmp_path, payload))
