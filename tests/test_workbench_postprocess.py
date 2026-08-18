"""Unit tests for hexif.pipeline.postprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hexif.cell_phenotype import FOCUSED_MARKERS, PHENOTYPE_NAMES
from hexif.pipeline.postprocess import (
    assign_marker_calls,
    assign_phenotype_calls,
    attach_tma_tissue,
    per_core_composition,
)
from hexif.pipeline.thresholds import CalibratedThresholds


def _thresholds(tmp_path) -> CalibratedThresholds:
    marker = {int(channel): 0.25 for channel in FOCUSED_MARKERS}
    return CalibratedThresholds(
        marker_thresholds=marker,
        marker_thresholds_by_name={},
        phenotype_thresholds={name: 0.35 for name in PHENOTYPE_NAMES},
        marker_confidence={},
        phenotype_confidence={},
        provenance={"fixture": "deterministic unit test"},
        source=tmp_path / "thresholds.json",
    )


def _toy_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "basename": ["ccRCC_TMA1__A-1"] * n + ["ccOC_TMA1__A-2"] * n,
            "cell_id": list(range(n)) * 2,
            "centroid_x": rng.integers(0, 1000, 2 * n),
            "centroid_y": rng.integers(0, 1000, 2 * n),
        }
    )
    for ch in FOCUSED_MARKERS:
        df[f"ch{ch:02d}_pred_v1_1"] = rng.uniform(0, 1, 2 * n)
        df[f"ch{ch:02d}_pos"] = rng.integers(0, 2, 2 * n)
    for name in PHENOTYPE_NAMES:
        # Use the bare convention (matches the v1.1 eval CSV's output)
        df[f"phenotype_{name}_score"] = rng.uniform(0, 1, 2 * n)
        df[f"phenotype_{name}_label"] = rng.integers(0, 2, 2 * n)
    return df


def test_assign_marker_calls_adds_call_column_per_marker(tmp_path):
    df = _toy_predictions()
    t = _thresholds(tmp_path)
    out = assign_marker_calls(df, t, model_id="v1_1")
    for ch in FOCUSED_MARKERS:
        col = f"ch{ch:02d}_call_v1_1"
        assert col in out.columns, f"missing {col}"
        assert out[col].dtype.kind in "iu"
        # Match: prediction >= threshold
        thr = t.marker_threshold(int(ch))
        expected = (out[f"ch{ch:02d}_pred_v1_1"] >= thr).astype("int8")
        assert (out[col] == expected).all()


def test_assign_phenotype_calls_handles_bare_score_convention(tmp_path):
    df = _toy_predictions()
    t = _thresholds(tmp_path)
    out = assign_phenotype_calls(df, t, model_id="v1_1")
    for name in PHENOTYPE_NAMES:
        col = f"phenotype_{name}_call_v1_1"
        assert col in out.columns
        thr = t.phenotype_threshold(name)
        expected = (out[f"phenotype_{name}_score"] >= thr).astype("int8")
        assert (out[col] == expected).all()


def test_per_core_composition_one_row_per_core(tmp_path):
    df = _toy_predictions()
    t = _thresholds(tmp_path)
    df = assign_marker_calls(df, t, model_id="v1_1")
    df = assign_phenotype_calls(df, t, model_id="v1_1")
    comp = per_core_composition(df, model_ids=("v1_1",))
    assert len(comp) == 2
    assert set(comp["basename"]) == {"ccRCC_TMA1__A-1", "ccOC_TMA1__A-2"}
    # Fractions for each marker/phenotype match the in-core mean of the
    # corresponding call column.
    for _, r in comp.iterrows():
        sub = df[df.basename == r.basename]
        for ch in FOCUSED_MARKERS:
            col = f"frac_ch{ch:02d}_pos_v1_1"
            if col in r.index:
                assert abs(float(r[col]) - float(sub[f"ch{ch:02d}_call_v1_1"].mean())) < 1e-9


def test_attach_tma_tissue_parses_basename():
    df = pd.DataFrame({"basename": ["ccRCC_TMA1__A-1", "ccOC_TMA2__G-5"]})
    out = attach_tma_tissue(df)
    assert out["tma"].tolist() == ["ccRCC_TMA1", "ccOC_TMA2"]
    assert out["tissue"].tolist() == ["ccRCC", "ccOC"]
