"""Unit tests for ``hexif.pipeline.spacec_gating.core`` (SPACEc gating label gen).

Spec: ``experiments/configs/spacec_gating_v1.md`` ("Test plan / Unit
(gating computation)" subsection).

These tests use synthetic numpy data only — no bundle dependency. The
real-data integration test against ``ccRCC_TMA1__A-10`` is owned by
cohort builder and lives in :file:`tests/test_spacec_gating_integration.py`.

Tests that run scanpy's Leiden pipeline need scanpy + igraph, which
only live in the ``spacec_cpu`` conda env. They skip cleanly on the
default ``hexif[dev]`` install where those packages aren't required,
via the same ``@needs_scanpy`` decorator pattern label audit established
in :file:`tests/test_label_audit.py`.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from hexif.pipeline.spacec_gating import (
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    CoreLabelResult,
    assign_marker_positivity_from_clusters,
    cluster_leiden_in_marker_space,
    label_one_core,
    normalize_per_core_log1p_zscore,
)

_HAS_SCANPY = (
    importlib.util.find_spec("scanpy") is not None
    and importlib.util.find_spec("igraph") is not None
)
needs_scanpy = pytest.mark.skipif(
    not _HAS_SCANPY,
    reason="scanpy/igraph not installed; Leiden tests require the spacec_cpu env",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _two_cluster_scaled_input(
    n_per_cluster: int = 200,
    n_markers: int = 12,
    bright_index: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Two-cluster raw intensity matrix matching label audit's fixture.

    One cluster is bright in marker ``bright_index`` (raw intensity
    near ``exp(3) - 1 ≈ 19.1``); the other is dim (raw intensity near
    zero). All other markers carry tiny near-constant positive noise
    so they survive ``log1p`` + per-marker z-score with non-zero
    variance but contribute no separating signal — Leiden must split
    on the bright marker alone at the default resolution (0.6).
    """
    rng = np.random.default_rng(seed)
    marker_names = [f"M{i}" for i in range(n_markers)]

    raw_matrix = np.abs(
        rng.normal(loc=1.0, scale=0.02, size=(2 * n_per_cluster, n_markers))
    ).astype(np.float32)

    bright = np.exp(rng.normal(loc=3.0, scale=0.1, size=n_per_cluster)) - 1.0
    dim = np.clip(rng.normal(loc=0.0, scale=0.05, size=n_per_cluster), 0.0, None)
    raw_matrix[:n_per_cluster, bright_index] = bright.astype(np.float32)
    raw_matrix[n_per_cluster:, bright_index] = dim.astype(np.float32)

    return raw_matrix, marker_names


# --------------------------------------------------------------------------- #
# Step 1 — normalize_per_core_log1p_zscore
# --------------------------------------------------------------------------- #


class TestNormalizePerCoreLog1pZscore:
    """Bimodal mean/std behaviour, clipping, and constant-column safety.

    Implements spec § "Test plan / Unit (gating computation)" item 1
    (``normalize_per_core_log1p_zscore``).
    """

    def test_bimodal_input_yields_zero_mean_unit_std(self) -> None:
        # Build a 3-marker matrix where each marker has a bimodal
        # raw distribution; after log1p + z-score the per-marker
        # mean should be ~0 and std ~1.
        rng = np.random.default_rng(0)
        n_per_mode = 500
        raw = np.empty((2 * n_per_mode, 3), dtype=np.float32)
        for j in range(3):
            low = np.clip(rng.normal(loc=0.5, scale=0.1, size=n_per_mode), 0.0, None)
            high = np.exp(rng.normal(loc=2.5, scale=0.1, size=n_per_mode)) - 1.0
            raw[:n_per_mode, j] = low.astype(np.float32)
            raw[n_per_mode:, j] = high.astype(np.float32)

        # Pick a large clip so no values are clipped — we want raw
        # z-score behaviour for the mean/std check.
        scaled = normalize_per_core_log1p_zscore(
            raw,
            marker_names=["a", "b", "c"],
            scale_max_value=100.0,
        )

        assert scaled.dtype == np.float32
        assert scaled.shape == raw.shape

        per_marker_mean = scaled.mean(axis=0)
        per_marker_std = scaled.std(axis=0)
        assert np.all(np.abs(per_marker_mean) < 0.05)
        assert np.all(np.abs(per_marker_std - 1.0) < 0.1)

    def test_values_exceeding_scale_max_are_clipped(self) -> None:
        # One marker has a single huge outlier; after z-scoring it
        # would land far past ``scale_max_value`` and must be clipped
        # exactly to ``+scale_max_value``.
        rng = np.random.default_rng(1)
        n_cells = 200
        raw = np.abs(rng.normal(loc=1.0, scale=0.1, size=(n_cells, 2))).astype(np.float32)
        # Inject extreme positives and negatives in marker 0.
        raw[0, 0] = 1e6
        raw[1, 0] = 0.0  # log1p(0) = 0 — well below the mean

        scaled_max = 5.0
        scaled = normalize_per_core_log1p_zscore(
            raw, marker_names=["a", "b"], scale_max_value=scaled_max
        )

        assert scaled[0, 0] == pytest.approx(scaled_max, rel=0, abs=1e-6)
        assert scaled.min() >= -scaled_max - 1e-6
        assert scaled.max() <= scaled_max + 1e-6
        # And every column respects the bound.
        assert np.all(scaled >= -scaled_max)
        assert np.all(scaled <= scaled_max)

    def test_constant_marker_returns_zero_column_and_warns(self, caplog) -> None:
        # Marker 0 has identical values across all cells → std == 0.
        # Spec: return a zero column (not NaN) and log a warning
        # naming the marker.
        n_cells = 50
        raw = np.full((n_cells, 2), 1.0, dtype=np.float32)
        # Marker 1 has variance so it normalizes cleanly.
        rng = np.random.default_rng(2)
        raw[:, 1] = np.abs(rng.normal(loc=1.0, scale=0.2, size=n_cells)).astype(np.float32)

        with caplog.at_level("WARNING", logger="hexif.pipeline.spacec_gating.core"):
            scaled = normalize_per_core_log1p_zscore(
                raw,
                marker_names=["FlatMarker", "VariedMarker"],
                scale_max_value=DEFAULT_SCALE_MAX_VALUE,
            )

        # Zero column, not NaN.
        assert np.all(scaled[:, 0] == 0.0)
        assert not np.any(np.isnan(scaled))
        # Warning was emitted with the affected marker name.
        assert any("FlatMarker" in rec.message for rec in caplog.records)

    def test_negative_input_raises(self) -> None:
        raw = np.array([[1.0, -0.5], [0.5, 0.2]], dtype=np.float32)
        with pytest.raises(ValueError, match="negative"):
            normalize_per_core_log1p_zscore(raw, marker_names=["a", "b"])

    def test_nan_input_raises(self) -> None:
        raw = np.array([[1.0, np.nan], [0.5, 0.2]], dtype=np.float32)
        with pytest.raises(ValueError, match=r"NaN|finite"):
            normalize_per_core_log1p_zscore(raw, marker_names=["a", "b"])

    def test_marker_name_length_mismatch_raises(self) -> None:
        raw = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="marker_names length"):
            normalize_per_core_log1p_zscore(raw, marker_names=["a", "b"])


# --------------------------------------------------------------------------- #
# Step 2 — cluster_leiden_in_marker_space
# --------------------------------------------------------------------------- #


@needs_scanpy
class TestClusterLeidenInMarkerSpace:
    """PCA + kNN + Leiden cluster discovery on synthetic data.

    Implements spec § "Test plan / Unit (gating computation)" item 2
    (``cluster_leiden_in_marker_space``).
    """

    def test_two_cluster_synthetic_finds_exactly_two_clusters(self) -> None:
        raw, marker_names = _two_cluster_scaled_input(seed=0)
        scaled = normalize_per_core_log1p_zscore(raw, marker_names=marker_names)

        cluster_ids, cluster_z = cluster_leiden_in_marker_space(
            scaled,
            marker_names=marker_names,
            random_state=0,
        )

        # Spec: exactly two Leiden clusters at the default resolution.
        assert cluster_z.shape[0] == 2
        assert set(np.unique(cluster_ids).tolist()) <= {0, 1}
        # Bright marker (index 0) should have one positive and one
        # negative mean cluster z-score.
        bright_z = cluster_z[marker_names[0]].to_numpy()
        assert bright_z.max() > 0.0
        assert bright_z.min() < 0.0

    def test_determinism_with_same_random_state(self) -> None:
        raw, marker_names = _two_cluster_scaled_input(seed=0)
        scaled = normalize_per_core_log1p_zscore(raw, marker_names=marker_names)

        cid1, zmat1 = cluster_leiden_in_marker_space(
            scaled, marker_names=marker_names, random_state=0
        )
        cid2, zmat2 = cluster_leiden_in_marker_space(
            scaled, marker_names=marker_names, random_state=0
        )

        np.testing.assert_array_equal(cid1, cid2)
        pd.testing.assert_frame_equal(zmat1, zmat2)

    def test_small_core_clamps_neighbors_and_returns_single_cluster(self) -> None:
        # 3 cells, 4 markers — far below the default n_neighbors=20.
        # Must clamp gracefully and return one cluster (Leiden on a
        # ~complete tiny graph collapses to a single community).
        rng = np.random.default_rng(3)
        raw = np.abs(rng.normal(loc=1.0, scale=0.05, size=(3, 4))).astype(np.float32)
        marker_names = ["a", "b", "c", "d"]
        scaled = normalize_per_core_log1p_zscore(raw, marker_names=marker_names)

        cluster_ids, cluster_z = cluster_leiden_in_marker_space(
            scaled,
            marker_names=marker_names,
            leiden_n_neighbors=DEFAULT_LEIDEN_N_NEIGHBORS,
            random_state=0,
        )
        assert cluster_ids.shape == (3,)
        assert cluster_ids.dtype == np.int32
        assert cluster_z.shape == (1, 4)
        assert set(np.unique(cluster_ids).tolist()) == {0}


# --------------------------------------------------------------------------- #
# Step 3 — assign_marker_positivity_from_clusters
# --------------------------------------------------------------------------- #


class TestAssignMarkerPositivityFromClusters:
    """Cluster mean-z → per-cell positivity broadcast.

    Implements spec § "Test plan / Unit (gating computation)" item 3
    (``assign_marker_positivity_from_clusters``).
    """

    def _hand_built_matrix(self) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
        # 3 clusters × 2 markers.
        # Cluster 0: marker0 z=+1.0 (positive at tau=0.5), marker1 z=-1.0.
        # Cluster 1: marker0 z=+0.2 (below tau), marker1 z=0.0.
        # Cluster 2: marker0 z=-0.5, marker1 z=+0.3 (below tau at 0.5).
        cluster_z = pd.DataFrame(
            {
                "marker0": [1.0, 0.2, -0.5],
                "marker1": [-1.0, 0.0, 0.3],
            },
            index=pd.Index(["0", "1", "2"], name="leiden"),
        )
        # 6 cells: 2 per cluster.
        cluster_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        return cluster_ids, cluster_z, ["marker0", "marker1"]

    def test_hand_built_matrix_at_default_tau(self) -> None:
        cluster_ids, cluster_z, names = self._hand_built_matrix()

        marker_pos, pos_clusters = assign_marker_positivity_from_clusters(
            cluster_ids, cluster_z, marker_names=names, zscore_tau=0.5
        )

        # marker0: only cluster 0 has mean z > 0.5.
        assert pos_clusters["marker0"] == ("0",)
        # marker1: no cluster has mean z > 0.5.
        assert pos_clusters["marker1"] == ()

        # Per-cell broadcast: cells in cluster 0 (rows 0,1) are marker0+,
        # everyone else is marker0-. Nobody is marker1+.
        expected = np.array(
            [
                [True, False],
                [True, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(marker_pos, expected)
        assert marker_pos.dtype == bool
        assert marker_pos.shape == (6, 2)

    def test_tau_zero_uses_strict_greater_than(self) -> None:
        # Construct a cluster with mean z exactly 0 — must NOT be
        # positive at tau=0 (strict inequality, per the spec).
        cluster_z = pd.DataFrame(
            {"marker0": [0.0, 0.5]},
            index=pd.Index(["0", "1"], name="leiden"),
        )
        cluster_ids = np.array([0, 1], dtype=np.int32)

        marker_pos, pos_clusters = assign_marker_positivity_from_clusters(
            cluster_ids, cluster_z, marker_names=["marker0"], zscore_tau=0.0
        )

        # Cluster 0 (z == 0) is NOT positive.
        assert pos_clusters["marker0"] == ("1",)
        np.testing.assert_array_equal(marker_pos[:, 0], np.array([False, True]))

    def test_tau_ten_yields_no_positives(self) -> None:
        cluster_ids, cluster_z, names = self._hand_built_matrix()

        marker_pos, pos_clusters = assign_marker_positivity_from_clusters(
            cluster_ids, cluster_z, marker_names=names, zscore_tau=10.0
        )

        assert pos_clusters["marker0"] == ()
        assert pos_clusters["marker1"] == ()
        assert not marker_pos.any()
        assert marker_pos.shape == (6, 2)

    def test_unknown_marker_name_raises(self) -> None:
        cluster_ids, cluster_z, _ = self._hand_built_matrix()
        with pytest.raises(ValueError, match="missing from cluster_zscore_matrix"):
            assign_marker_positivity_from_clusters(
                cluster_ids,
                cluster_z,
                marker_names=["marker0", "not_a_marker"],
                zscore_tau=0.5,
            )


# --------------------------------------------------------------------------- #
# Step 4 — label_one_core
# --------------------------------------------------------------------------- #


def _synthetic_two_marker_core_frame(
    n_cells: int = 60, seed: int = 0
) -> tuple[pd.DataFrame, tuple[tuple[str, int], ...]]:
    """Build a 2-marker, ``n_cells``-row frame with one bright/dim CD68 split.

    Marker ordering matches ``MARKERS_DEFAULT`` semantics:
    ``("DAPI", 0)`` then ``("CD68", 27)``. DAPI is uniformly bright;
    CD68 is bimodal (half bright, half dim) so Leiden should split
    on it.
    """
    rng = np.random.default_rng(seed)
    half = n_cells // 2

    cd68_bright = np.exp(rng.normal(loc=3.0, scale=0.1, size=half)) - 1.0
    cd68_dim = np.clip(rng.normal(loc=0.0, scale=0.05, size=n_cells - half), 0.0, None)
    cd68 = np.concatenate([cd68_bright, cd68_dim]).astype(np.float32)

    dapi = np.clip(rng.normal(loc=4.0, scale=0.3, size=n_cells), 0.0, None).astype(np.float32)

    df = pd.DataFrame(
        {
            "basename": ["synth_core"] * n_cells,
            "ch00_true": dapi,
            "ch27_true": cd68,
        }
    )
    markers = (("DAPI", 0), ("CD68", 27))
    return df, markers


@needs_scanpy
class TestLabelOneCore:
    """End-to-end composition over a synthetic single-core frame.

    Implements spec § "Test plan / Unit (gating computation)" item 4
    (``label_one_core``).
    """

    def test_end_to_end_returns_corelabelresult_with_right_shapes(self) -> None:
        df, markers = _synthetic_two_marker_core_frame(n_cells=60, seed=0)

        result = label_one_core(df, markers=markers, random_state=0)

        # Type and identity.
        assert isinstance(result, CoreLabelResult)
        assert result.basename == "synth_core"
        assert result.n_cells == 60

        # Per-cell arrays.
        assert isinstance(result.cluster_id, np.ndarray)
        assert result.cluster_id.shape == (60,)
        assert result.cluster_id.dtype == np.int32

        assert isinstance(result.scaled_intensities, np.ndarray)
        assert result.scaled_intensities.shape == (60, 2)
        assert result.scaled_intensities.dtype == np.float32

        assert isinstance(result.marker_pos, np.ndarray)
        assert result.marker_pos.shape == (60, 2)
        assert result.marker_pos.dtype == bool

        # Cluster-level matrix.
        assert isinstance(result.cluster_zscore_matrix, pd.DataFrame)
        assert list(result.cluster_zscore_matrix.columns) == ["DAPI", "CD68"]
        assert result.n_clusters == result.cluster_zscore_matrix.shape[0]
        assert result.n_clusters >= 1

        # positive_cluster_ids_per_marker covers every marker.
        assert set(result.positive_cluster_ids_per_marker.keys()) == {"DAPI", "CD68"}

        # Consistency: marker_pos[i, j] iff cluster_id[i] ∈
        # positive_cluster_ids_per_marker[marker_j].
        for j, marker in enumerate(["DAPI", "CD68"]):
            pos_ids = result.positive_cluster_ids_per_marker[marker]
            pos_ids_int = {int(c) for c in pos_ids}
            expected_col = np.array([int(c) in pos_ids_int for c in result.cluster_id], dtype=bool)
            np.testing.assert_array_equal(result.marker_pos[:, j], expected_col)

    def test_missing_basename_column_raises(self) -> None:
        df = pd.DataFrame(
            {
                "ch00_true": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "ch27_true": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            }
        )
        with pytest.raises(ValueError, match="must contain 'basename'"):
            label_one_core(df, markers=(("DAPI", 0), ("CD68", 27)))

    def test_multiple_basenames_in_one_frame_raises(self) -> None:
        df = pd.DataFrame(
            {
                "basename": ["core_a", "core_b", "core_a"],
                "ch00_true": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "ch27_true": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            }
        )
        with pytest.raises(ValueError, match="exactly one unique 'basename'"):
            label_one_core(df, markers=(("DAPI", 0), ("CD68", 27)))

    def test_missing_intensity_column_raises_naming_the_column(self) -> None:
        df = pd.DataFrame(
            {
                "basename": ["synth"] * 3,
                "ch00_true": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                # No ch27_true.
            }
        )
        with pytest.raises(ValueError, match="ch27_true"):
            label_one_core(df, markers=(("DAPI", 0), ("CD68", 27)))

    def test_empty_frame_raises(self) -> None:
        df = pd.DataFrame(
            {
                "basename": pd.Series([], dtype=object),
                "ch00_true": pd.Series([], dtype=np.float32),
                "ch27_true": pd.Series([], dtype=np.float32),
            }
        )
        with pytest.raises(ValueError, match="no cells in core"):
            label_one_core(df, markers=(("DAPI", 0), ("CD68", 27)))

    def test_determinism_with_same_random_state(self) -> None:
        df, markers = _synthetic_two_marker_core_frame(n_cells=60, seed=0)

        r1 = label_one_core(df, markers=markers, random_state=0)
        r2 = label_one_core(df, markers=markers, random_state=0)

        # Frozen-dataclass equality is unreliable for numpy/pandas
        # fields → assert field by field.
        assert r1.basename == r2.basename
        assert r1.n_cells == r2.n_cells
        assert r1.n_clusters == r2.n_clusters
        np.testing.assert_array_equal(r1.cluster_id, r2.cluster_id)
        np.testing.assert_array_equal(r1.scaled_intensities, r2.scaled_intensities)
        np.testing.assert_array_equal(r1.marker_pos, r2.marker_pos)
        pd.testing.assert_frame_equal(r1.cluster_zscore_matrix, r2.cluster_zscore_matrix)
        assert r1.positive_cluster_ids_per_marker == r2.positive_cluster_ids_per_marker


# --------------------------------------------------------------------------- #
# Default-value smoke checks (no scanpy)
# --------------------------------------------------------------------------- #


class TestDefaults:
    """Sanity-check the defaults the spec pins (visible via the public API)."""

    def test_default_zscore_tau(self) -> None:
        assert DEFAULT_ZSCORE_TAU == pytest.approx(0.5)

    def test_default_scale_max_value(self) -> None:
        assert DEFAULT_SCALE_MAX_VALUE == pytest.approx(5.0)
