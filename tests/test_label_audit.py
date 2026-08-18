"""Unit tests for ``hexif.audit.label_audit`` (label audit label audit).

Spec: ``experiments/configs/label_audit_v1.md`` ("Unit (audit computation)" subsection).

These tests use synthetic numpy data only — no bundle dependency. The
real-data integration test against ``ccRCC_TMA1__A-10`` is owned by real-data validation
and lives in :file:`tests/test_label_audit_integration.py`.

The Leiden-based tests (``TestPositivityLeidenZscore`` and the
``audit_core`` end-to-end class) need scanpy + igraph, which only live in
the ``spacec_cpu`` conda env. They skip cleanly on the default ``hexif[dev]``
install where those packages aren't required.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexif.audit import (
    MARKERS_DEFAULT,
    MarkerAuditResult,
    audit_core,
    load_cohort_thresholds,
    positivity_leiden_zscore,
    positivity_per_core_otsu,
)

_HAS_SCANPY = (
    importlib.util.find_spec("scanpy") is not None
    and importlib.util.find_spec("igraph") is not None
)
needs_scanpy = pytest.mark.skipif(
    not _HAS_SCANPY,
    reason="scanpy/igraph not installed; Leiden tests require the spacec_cpu env",
)

# ---------------------------------------------------------------------------
# Lens 2 — per-core Otsu
# ---------------------------------------------------------------------------


class TestPositivityPerCoreOtsu:
    def test_bimodal_mixture_separates_modes(self) -> None:
        rng = np.random.default_rng(42)
        low = rng.normal(loc=0.5, scale=0.1, size=500)
        high = rng.normal(loc=3.0, scale=0.1, size=500)
        log1p_intensities = np.concatenate([low, high])

        pos, threshold = positivity_per_core_otsu(log1p_intensities)

        # Otsu should land in the gap between the two modes (~0.5 and ~3.0).
        assert 0.7 < threshold < 2.7
        # ~half the cells should be positive (the "high" mode).
        assert pos.sum() == pytest.approx(500, abs=20)
        assert pos.dtype == bool
        assert pos.shape == log1p_intensities.shape

    def test_all_zeros_input_is_degenerate(self) -> None:
        pos, threshold = positivity_per_core_otsu(np.zeros(100, dtype=np.float64))

        assert pos.shape == (100,)
        assert pos.dtype == bool
        assert not pos.any()
        # Spec: degenerate inputs return a finite-or-inf threshold (never NaN).
        assert np.isinf(threshold) or np.isfinite(threshold)
        assert not np.isnan(threshold)

    def test_single_unique_value_returns_all_false_and_inf(self) -> None:
        log1p_intensities = np.full(50, 2.5, dtype=np.float64)

        pos, threshold = positivity_per_core_otsu(log1p_intensities)

        assert pos.shape == (50,)
        assert pos.dtype == bool
        assert not pos.any()
        assert threshold == float("inf")

    def test_empty_array_returns_empty_bool_array_and_inf(self) -> None:
        pos, threshold = positivity_per_core_otsu(np.array([], dtype=np.float64))

        assert pos.shape == (0,)
        assert pos.dtype == bool
        assert threshold == float("inf")


# ---------------------------------------------------------------------------
# Lens 3 — Leiden cluster z-score
# ---------------------------------------------------------------------------


def _two_cluster_synthetic_matrix(
    n_per_cluster: int = 200,
    n_markers: int = 12,
    cd68_index: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Build a synthetic two-cluster intensity matrix.

    One cluster is bright in CD68, one is dim. After the function's
    ``log1p`` + per-marker z-score the CD68 column has one mode at roughly
    ``+1`` z-score and one at ``-1`` z-score (clipped at ``±scale_max``),
    while all other markers are near-constant background (so they contribute
    no clustering signal). This guarantees Leiden separates exactly two
    populations at the default resolution.

    Matches the spec's "Synthetic two-cluster data" assertion (one cluster
    centered at +3 in CD68, one at -3, ``np.random.default_rng(0)``).
    """
    rng = np.random.default_rng(seed)
    marker_names = [name for name, _ in MARKERS_DEFAULT][:n_markers]

    # Background markers: tiny shared noise (positive intensities) — same
    # distribution in both clusters so they contribute no separating signal.
    # Each marker needs >0 variance to survive ``sc.pp.scale`` without NaN.
    raw_matrix = (
        np.abs(rng.normal(loc=1.0, scale=0.02, size=(2 * n_per_cluster, n_markers)))
    ).astype(np.float32)

    # CD68: bright cluster centered at raw intensity exp(3)-1 ≈ 19.1; dim
    # cluster centered near 0. The notebook's prescription is
    # "+3 vs -3 in log1p space" — log1p clamps negatives, so we use raw
    # values near 0 for the dim mode (equivalent to log1p ~= 0 << +3 — a
    # clear separation after scale).
    cd68_bright = np.exp(rng.normal(loc=3.0, scale=0.1, size=n_per_cluster)) - 1.0
    cd68_dim = np.clip(rng.normal(loc=0.0, scale=0.05, size=n_per_cluster), 0.0, None)
    raw_matrix[:n_per_cluster, cd68_index] = cd68_bright.astype(np.float32)
    raw_matrix[n_per_cluster:, cd68_index] = cd68_dim.astype(np.float32)

    return raw_matrix, marker_names


@needs_scanpy
class TestPositivityLeidenZscore:
    def test_two_cluster_synthetic_finds_exactly_two_clusters(self) -> None:
        raw_matrix, marker_names = _two_cluster_synthetic_matrix(seed=0)
        cd68_name = "CD68"
        assert cd68_name in marker_names

        cluster_id_per_cell, cluster_zscore, positive = positivity_leiden_zscore(
            raw_matrix,
            marker_names=marker_names,
            zscore_tau=0.5,
            random_state=0,
        )

        # Spec: exactly 2 Leiden clusters.
        assert cluster_zscore.shape[0] == 2
        assert set(np.unique(cluster_id_per_cell)) <= {0, 1}
        # Spec: exactly one cluster is CD68-positive at tau=0.5.
        assert len(positive[cd68_name]) == 1
        # Sanity: that cluster must be the one with the higher mean CD68 z-score.
        positive_cluster_id = positive[cd68_name][0]
        cd68_z = cluster_zscore[cd68_name]
        assert cd68_z[positive_cluster_id] == cd68_z.max()

    def test_determinism_with_same_random_state(self) -> None:
        raw_matrix, marker_names = _two_cluster_synthetic_matrix(seed=0)

        cid1, _, pos1 = positivity_leiden_zscore(
            raw_matrix,
            marker_names=marker_names,
            random_state=0,
        )
        cid2, _, pos2 = positivity_leiden_zscore(
            raw_matrix,
            marker_names=marker_names,
            random_state=0,
        )

        np.testing.assert_array_equal(cid1, cid2)
        assert pos1 == pos2

    def test_marker_name_length_mismatch_raises(self) -> None:
        raw_matrix = np.zeros((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="marker_names length"):
            positivity_leiden_zscore(raw_matrix, marker_names=["a", "b"])


# ---------------------------------------------------------------------------
# MarkerAuditResult derived properties
# ---------------------------------------------------------------------------


class TestMarkerAuditResultProperties:
    def test_disagreement_score_is_l1_sum_of_gaps(self) -> None:
        # n_cells=100, consensus=10 → frac=0.10.
        # otsu=30 → gap = +0.20; leiden=-10 not possible, so set leiden=0 → gap = -0.10? Use 30 vs 0.
        # We need opposite-signed gaps summing to 0.4: otsu-consensus = +0.20,
        # leiden-consensus = -0.20.  consensus=20 (0.20), otsu=40 (0.40), leiden=0 (0.0).
        result = MarkerAuditResult(
            name="CD68",
            channel=27,
            n_cells=100,
            n_pos_cohort_consensus=20,
            n_pos_per_core_otsu=40,
            n_pos_leiden_zscore=0,
            per_core_otsu_threshold_log1p=1.0,
            cohort_thresholds_log1p={"gmm": 0.3, "otsu": 0.32, "quantile": 0.3},
            leiden_positive_cluster_ids=(),
            max_leiden_cluster_zscore=-0.5,
        )

        assert result.frac_cohort_consensus == pytest.approx(0.20, rel=1e-5)
        assert result.frac_per_core_otsu == pytest.approx(0.40, rel=1e-5)
        assert result.frac_leiden_zscore == pytest.approx(0.0, abs=1e-5)

        # +0.20 + |-0.20| = 0.40 — L1, not signed sum.
        assert result.disagreement_score == pytest.approx(0.40, rel=1e-5)

    def test_suspicious_zero_consensus_pattern(self) -> None:
        # The CD68 / A-10 pattern: consensus=0 but Otsu / Leiden >5%.
        result = MarkerAuditResult(
            name="CD68",
            channel=27,
            n_cells=1000,
            n_pos_cohort_consensus=0,
            n_pos_per_core_otsu=100,  # 10%
            n_pos_leiden_zscore=80,  # 8%
            per_core_otsu_threshold_log1p=0.2,
            cohort_thresholds_log1p={"gmm": 0.3, "otsu": 0.32, "quantile": 0.3},
            leiden_positive_cluster_ids=("3",),
            max_leiden_cluster_zscore=1.5,
        )
        assert result.suspicious_zero_consensus is True

        # Strong-marker case (consensus agrees, nothing suspicious).
        ok = MarkerAuditResult(
            name="DAPI",
            channel=0,
            n_cells=1000,
            n_pos_cohort_consensus=900,
            n_pos_per_core_otsu=910,
            n_pos_leiden_zscore=900,
            per_core_otsu_threshold_log1p=0.5,
            cohort_thresholds_log1p={"gmm": 0.4, "otsu": 0.5, "quantile": 0.45},
            leiden_positive_cluster_ids=("0", "1"),
            max_leiden_cluster_zscore=2.0,
        )
        assert ok.suspicious_zero_consensus is False


# ---------------------------------------------------------------------------
# audit_core end-to-end on synthetic frame
# ---------------------------------------------------------------------------


@needs_scanpy
class TestAuditCore:
    def test_end_to_end_two_marker_synthetic(self) -> None:
        rng = np.random.default_rng(0)
        n_cells = 60

        # Two markers from MARKERS_DEFAULT — DAPI (ch 0) and CD68 (ch 27).
        # Build a frame with exactly the required columns.
        markers = (("DAPI", 0), ("CD68", 27))

        # Two synthetic populations: half bright in CD68, half dim.
        cd68_bright = rng.normal(loc=5.0, scale=0.5, size=n_cells // 2)
        cd68_dim = rng.normal(loc=0.05, scale=0.05, size=n_cells // 2)
        cd68_raw = np.clip(np.concatenate([cd68_bright, cd68_dim]), 0.0, None)

        dapi_raw = rng.normal(loc=4.0, scale=0.5, size=n_cells)
        dapi_raw = np.clip(dapi_raw, 0.0, None)

        # consensus column: half of bright are called positive in CD68; DAPI
        # mostly positive.
        cd68_consensus = np.zeros(n_cells, dtype=bool)
        cd68_consensus[: n_cells // 4] = True
        dapi_consensus = np.ones(n_cells, dtype=bool)

        df = pd.DataFrame(
            {
                "basename": ["synth_core"] * n_cells,
                "ch00_true": dapi_raw.astype(np.float32),
                "ch27_true": cd68_raw.astype(np.float32),
                "ch00_pos_consensus": dapi_consensus,
                "ch27_pos_consensus": cd68_consensus,
            }
        )

        cohort_thresholds: dict[int, dict[str, float]] = {
            0: {"gmm": 0.5, "otsu": 0.6, "quantile": 0.55, "target_pos": 0.9},
            27: {"gmm": 0.3, "otsu": 0.32, "quantile": 0.3, "target_pos": 0.5},
        }

        result = audit_core(
            df,
            cohort_thresholds=cohort_thresholds,
            markers=markers,
            random_state=0,
        )

        # Types are correct.
        assert result.basename == "synth_core"
        assert result.n_cells == n_cells
        assert set(result.per_marker.keys()) == {"DAPI", "CD68"}
        assert isinstance(result.per_marker["CD68"], MarkerAuditResult)
        assert isinstance(result.leiden_cluster_id_per_cell, np.ndarray)
        assert result.leiden_cluster_id_per_cell.shape == (n_cells,)
        assert isinstance(result.leiden_cluster_zscore_matrix, pd.DataFrame)
        assert list(result.leiden_cluster_zscore_matrix.columns) == ["DAPI", "CD68"]
        assert result.leiden_n_clusters == result.leiden_cluster_zscore_matrix.shape[0]
        assert result.leiden_n_clusters >= 1

        # Consensus counts are passed through verbatim.
        assert result.per_marker["CD68"].n_pos_cohort_consensus == n_cells // 4
        assert result.per_marker["DAPI"].n_pos_cohort_consensus == n_cells

        # Per-core Otsu finds the bimodal split in CD68.
        assert result.per_marker["CD68"].n_pos_per_core_otsu == pytest.approx(n_cells // 2, abs=4)

        # Cohort thresholds are preserved (with float casts).
        assert result.per_marker["CD68"].cohort_thresholds_log1p["otsu"] == pytest.approx(
            0.32, rel=1e-5
        )

    def test_missing_consensus_column_is_warned_not_raised(self, caplog) -> None:
        rng = np.random.default_rng(1)
        n_cells = 30
        markers = (("DAPI", 0),)

        df = pd.DataFrame(
            {
                "basename": ["synth_core"] * n_cells,
                "ch00_true": np.clip(rng.normal(2.0, 0.3, size=n_cells), 0, None).astype(
                    np.float32
                ),
                # No ch00_pos_consensus column — should warn, not crash.
            }
        )
        cohort_thresholds: dict[int, dict[str, float]] = {
            0: {"gmm": 0.5, "otsu": 0.6, "quantile": 0.55},
        }

        with caplog.at_level("WARNING", logger="hexif.audit.label_audit"):
            result = audit_core(
                df,
                cohort_thresholds=cohort_thresholds,
                markers=markers,
                random_state=0,
            )

        assert result.per_marker["DAPI"].n_pos_cohort_consensus == 0
        # The warning mentions the missing column.
        assert any("ch00_pos_consensus" in rec.message for rec in caplog.records)

    def test_missing_true_column_raises(self) -> None:
        # Frame missing the ``chXX_true`` intensity column → error at validation.
        df = pd.DataFrame({"basename": ["synth_core"] * 5, "ch00_true": [1.0] * 5})
        with pytest.raises(ValueError, match="missing required intensity columns"):
            audit_core(
                df,
                cohort_thresholds={27: {"gmm": 0.3, "otsu": 0.3, "quantile": 0.3}},
                markers=(("CD68", 27),),
                random_state=0,
            )


# ---------------------------------------------------------------------------
# load_cohort_thresholds round-trip + key normalization
# ---------------------------------------------------------------------------


class TestLoadCohortThresholds:
    def test_round_trip_with_string_keys(self, tmp_path: Path) -> None:
        payload = {
            "0": {"gmm": 0.3, "otsu": 0.32, "quantile": 0.30, "target_pos": 0.5},
            "27": {"gmm": 0.25, "otsu": 0.28, "quantile": 0.26, "target_pos": 0.4},
        }
        path = tmp_path / "thresholds.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_cohort_thresholds(path)

        # Keys normalized to int.
        assert set(loaded.keys()) == {0, 27}
        assert all(isinstance(k, int) for k in loaded.keys())

        # Values preserved (as floats).
        assert loaded[0]["otsu"] == pytest.approx(0.32, rel=1e-5)
        assert loaded[27]["gmm"] == pytest.approx(0.25, rel=1e-5)
        assert loaded[27]["target_pos"] == pytest.approx(0.4, rel=1e-5)

    def test_non_integer_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"banana": {"gmm": 0.3}}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-integer channel key"):
            load_cohort_thresholds(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cohort_thresholds(tmp_path / "does_not_exist.json")
