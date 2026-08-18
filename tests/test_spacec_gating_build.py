"""Unit tests for ``hexif.pipeline.spacec_gating.build`` (SPACEc gating cohort builder).

Spec: ``experiments/configs/spacec_gating_v1.md`` (§ "Test plan / Unit").

These tests exercise the cohort orchestrator + DataFrame emitter +
provenance JSON writer on synthetic data. Real-data assertions against
the validated bundle live in :file:`tests/test_spacec_gating_integration.py`.

The end-to-end ``build_spacec_labels`` test and the ``emit_new_columns``
round-trip need scanpy + igraph (gating computation's ``core.label_one_core`` runs
Leiden clustering). They skip cleanly via the ``@needs_scanpy`` marker
borrowed from :file:`tests/test_label_audit.py` on the default
``hexif[dev]`` install. The provenance + overwrite-refusal tests are
pure-Python and run everywhere.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexif.cell_phenotype import PHENOTYPE_NAMES
from hexif.pipeline.spacec_gating import (
    COL_CLUSTER_ID_SPACEC,
    CohortLabelResult,
    CoreLabelResult,
    col_phenotype_spacec,
    col_pos_spacec,
    col_zscore_spacec,
    emit_new_columns,
    write_spacec_provenance,
)

_HAS_SCANPY = (
    importlib.util.find_spec("scanpy") is not None
    and importlib.util.find_spec("igraph") is not None
)
needs_scanpy = pytest.mark.skipif(
    not _HAS_SCANPY,
    reason="scanpy/igraph not installed; SPACEc gating clustering needs the spacec_cpu env",
)


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #


def _two_marker_two_cluster_frame(
    n_cores: int = 3,
    n_per_cluster: int = 25,
    seed: int = 0,
) -> tuple[pd.DataFrame, tuple[tuple[str, int], ...]]:
    """Build a synthetic cell table with two markers and bimodal CD68 intensities.

    Each core has ``2 * n_per_cluster`` cells: half bright in CD68 (channel
    27), half dim. DAPI (channel 0) is roughly constant background. The
    Leiden lens at the audit defaults reliably finds two clusters and
    flags the bright one as CD68-positive.

    Returns:
        ``(cells, markers)`` — the synthetic DataFrame and the marker
        tuple used (DAPI, CD68). Each cell carries the columns
        ``basename``, ``ch00_true``, ``ch27_true``, ``ch00_pos_consensus``,
        ``ch27_pos_consensus`` — enough for both ``build_spacec_labels``
        (which only reads the ``chXX_true`` columns) and the re-audit
        path (which reads the consensus columns).
    """
    rng = np.random.default_rng(seed)
    markers: tuple[tuple[str, int], ...] = (("DAPI", 0), ("CD68", 27))
    n_per_core = 2 * n_per_cluster

    rows: list[dict] = []
    for core_idx in range(n_cores):
        basename = f"synth_core_{core_idx:02d}"

        cd68_bright = np.exp(rng.normal(loc=3.0, scale=0.1, size=n_per_cluster)) - 1.0
        cd68_dim = np.clip(rng.normal(loc=0.0, scale=0.05, size=n_per_cluster), 0.0, None)
        cd68_raw = np.concatenate([cd68_bright, cd68_dim]).astype(np.float32)

        dapi_raw = np.abs(rng.normal(loc=2.0, scale=0.4, size=n_per_core).astype(np.float32))

        # Pre-existing cohort-consensus labels — half the bright cells
        # for CD68, most cells for DAPI. The build pipeline ignores
        # these; they exist so the re-audit path sees a populated
        # cohort_thresholds column.
        cd68_consensus = np.zeros(n_per_core, dtype=bool)
        cd68_consensus[: n_per_cluster // 2] = True
        dapi_consensus = np.ones(n_per_core, dtype=bool)

        for i in range(n_per_core):
            rows.append(
                {
                    "basename": basename,
                    "ch00_true": float(dapi_raw[i]),
                    "ch27_true": float(cd68_raw[i]),
                    "ch00_pos_consensus": bool(dapi_consensus[i]),
                    "ch27_pos_consensus": bool(cd68_consensus[i]),
                }
            )
    return pd.DataFrame(rows), markers


def _build_synthetic_cohort_result(
    markers: tuple[tuple[str, int], ...],
    n_cores: int = 2,
    n_cells_per_core: int = 8,
    seed: int = 0,
) -> CohortLabelResult:
    """Hand-construct a tiny ``CohortLabelResult`` without invoking Leiden.

    Used by the tests that don't need to verify the clustering itself —
    just that the column emitter / provenance writer correctly read the
    dataclass contract.
    """
    rng = np.random.default_rng(seed)
    n_markers = len(markers)
    cores: list[CoreLabelResult] = []

    for core_idx in range(n_cores):
        basename = f"synth_core_{core_idx:02d}"
        # Alternate cluster IDs so we exercise both clusters in each core.
        cluster_id = np.tile(np.arange(2, dtype=np.int32), n_cells_per_core // 2 + 1)[
            :n_cells_per_core
        ]
        scaled = rng.normal(size=(n_cells_per_core, n_markers)).astype(np.float32)
        # Mark cluster 0 as positive for marker 0, cluster 1 positive for
        # marker 1 (or the last marker if we only have two).
        marker_pos = np.zeros((n_cells_per_core, n_markers), dtype=bool)
        marker_pos[cluster_id == 0, 0] = True
        if n_markers > 1:
            marker_pos[cluster_id == 1, -1] = True

        cluster_zscore = pd.DataFrame(
            np.array(
                [
                    [1.0, -0.5] + [0.0] * max(0, n_markers - 2),
                    [-0.5, 1.0] + [0.0] * max(0, n_markers - 2),
                ]
            )[:, :n_markers],
            index=pd.Index(["0", "1"], name="leiden"),
            columns=[name for name, _ch in markers],
        )
        pos_clusters: dict[str, tuple[str, ...]] = {name: () for name, _ch in markers}
        pos_clusters[markers[0][0]] = ("0",)
        if n_markers > 1:
            pos_clusters[markers[-1][0]] = ("1",)

        cores.append(
            CoreLabelResult(
                basename=basename,
                n_cells=n_cells_per_core,
                cluster_id=cluster_id,
                scaled_intensities=scaled,
                marker_pos=marker_pos,
                cluster_zscore_matrix=cluster_zscore,
                positive_cluster_ids_per_marker=pos_clusters,
                n_clusters=2,
            )
        )

    return CohortLabelResult(
        cores=tuple(cores),
        markers=markers,
        zscore_tau=0.5,
        leiden_resolution=0.6,
        leiden_n_neighbors=20,
        leiden_pca_n_comps=10,
        scale_max_value=5.0,
        random_state=0,
        timestamp="2026-05-14T12:34:56",
        hexif_git_rev="deadbee",
    )


# --------------------------------------------------------------------------- #
# 1. build_spacec_labels end-to-end (needs scanpy)
# --------------------------------------------------------------------------- #


@needs_scanpy
class TestBuildSpacecLabels:
    def test_end_to_end_two_marker_three_core(self) -> None:
        """3 synthetic cores × 2 markers → CohortLabelResult with provenance."""
        from hexif.pipeline.spacec_gating import build_spacec_labels

        cells, markers = _two_marker_two_cluster_frame(n_cores=3, n_per_cluster=25)

        result = build_spacec_labels(
            cells,
            markers=markers,
            random_state=0,
            show_progress=False,
        )

        # Three core results, in sorted basename order.
        assert isinstance(result, CohortLabelResult)
        assert result.n_cores == 3
        assert [c.basename for c in result.cores] == [
            "synth_core_00",
            "synth_core_01",
            "synth_core_02",
        ]
        # Marker tuple round-trips verbatim.
        assert result.markers == markers
        # Each per-core slot has the right shapes.
        for core in result.cores:
            assert core.n_cells == 50
            assert core.cluster_id.shape == (50,)
            assert core.scaled_intensities.shape == (50, 2)
            assert core.marker_pos.shape == (50, 2)
            assert core.marker_pos.dtype == bool
        # Provenance populated.
        assert result.timestamp != ""
        assert len(result.timestamp) == len("YYYY-MM-DDTHH:MM:SS")
        assert isinstance(result.hexif_git_rev, str)
        assert result.hexif_git_rev != ""

    def test_cores_filter_restricts_iteration_order(self) -> None:
        """An explicit ``cores`` iterable controls which cores are rebuilt."""
        from hexif.pipeline.spacec_gating import build_spacec_labels

        cells, markers = _two_marker_two_cluster_frame(n_cores=3, n_per_cluster=20)

        result = build_spacec_labels(
            cells,
            markers=markers,
            cores=["synth_core_02", "synth_core_00"],
            random_state=0,
            show_progress=False,
        )

        assert [c.basename for c in result.cores] == ["synth_core_02", "synth_core_00"]


# --------------------------------------------------------------------------- #
# 2. emit_new_columns round-trip
# --------------------------------------------------------------------------- #


class TestEmitNewColumns:
    def test_round_trip_synthetic_cohort(self) -> None:
        """All new columns are populated with the right dtype + values."""
        markers: tuple[tuple[str, int], ...] = (("DAPI", 0), ("CD68", 27))
        result = _build_synthetic_cohort_result(markers, n_cores=2, n_cells_per_core=8)

        # Source cells frame in the same row order as the synthetic cores.
        cells_rows: list[dict] = []
        for core in result.cores:
            for _ in range(core.n_cells):
                cells_rows.append(
                    {
                        "basename": core.basename,
                        "ch00_true": 1.5,
                        "ch27_true": 2.5,
                    }
                )
        cells = pd.DataFrame(cells_rows)

        augmented = emit_new_columns(result, cells)

        # The returned frame is a *new* object (we operate on a copy).
        assert augmented is not cells
        # Original columns preserved.
        for c in cells.columns:
            assert c in augmented.columns
        # Cluster ID column populated with int32 and the right per-cell values.
        assert COL_CLUSTER_ID_SPACEC in augmented.columns
        assert augmented[COL_CLUSTER_ID_SPACEC].dtype == np.int32
        # Per-marker columns exist with correct dtypes.
        for _name, channel in markers:
            pos_col = col_pos_spacec(channel)
            zsc_col = col_zscore_spacec(channel)
            assert pos_col in augmented.columns
            assert zsc_col in augmented.columns
            assert augmented[pos_col].dtype == bool
            assert augmented[zsc_col].dtype == np.float32

        # Phenotype columns — one per PHENOTYPE_NAMES.
        for pheno_name in PHENOTYPE_NAMES:
            pheno_col = col_phenotype_spacec(pheno_name)
            assert pheno_col in augmented.columns
            assert augmented[pheno_col].dtype == bool

        # Per-cell values match the source CoreLabelResult exactly.
        row_offset = 0
        for core in result.cores:
            slc = slice(row_offset, row_offset + core.n_cells)
            np.testing.assert_array_equal(
                augmented[COL_CLUSTER_ID_SPACEC].to_numpy()[slc],
                core.cluster_id.astype(np.int32),
            )
            for i, (_name, channel) in enumerate(markers):
                np.testing.assert_array_equal(
                    augmented[col_pos_spacec(channel)].to_numpy()[slc],
                    core.marker_pos[:, i],
                )
                np.testing.assert_array_equal(
                    augmented[col_zscore_spacec(channel)].to_numpy()[slc],
                    core.scaled_intensities[:, i].astype(np.float32),
                )
            row_offset += core.n_cells

    def test_phenotype_columns_match_phenotype_targets_helper(self) -> None:
        """Phenotype columns are computed via ``phenotype_targets_from_marker_pos``.

        We build a synthetic CoreLabelResult that uses the FULL focused-marker
        panel so the helper applies directly with no projection. Each row's
        emitted phenotype must exactly equal the helper's output.
        """
        from hexif.cell_phenotype import (
            FOCUSED_MARKERS,
            MARKER_NAMES,
            phenotype_targets_from_marker_pos,
        )

        markers = tuple(zip(MARKER_NAMES, FOCUSED_MARKERS, strict=True))
        n_cells = 6
        # Deterministic marker_pos: each row toggles a different marker.
        marker_pos = np.zeros((n_cells, len(markers)), dtype=bool)
        # Row 0: CD45 → immune. Row 1: CD45 + CD8 → t_cell + immune.
        marker_pos[0, 1] = True  # CD45
        marker_pos[1, 1] = True
        marker_pos[1, 2] = True  # CD8
        # Row 2: CA9 → tumor. Row 3: CD68 + CD163 → m2_like + macrophage.
        marker_pos[2, 4] = True  # CA9
        marker_pos[3, 6] = True  # CD68
        marker_pos[3, 8] = True  # CD163
        # Row 4: panCK + PDL1 → pdl1_tumor_like + pdl1 + tumor.
        marker_pos[4, 11] = True
        marker_pos[4, 9] = True
        # Row 5: all zeros → no phenotype.

        expected_pheno = phenotype_targets_from_marker_pos(marker_pos)

        core = CoreLabelResult(
            basename="synth_full_panel",
            n_cells=n_cells,
            cluster_id=np.zeros(n_cells, dtype=np.int32),
            scaled_intensities=np.zeros((n_cells, len(markers)), dtype=np.float32),
            marker_pos=marker_pos,
            cluster_zscore_matrix=pd.DataFrame(
                np.zeros((1, len(markers))),
                index=pd.Index(["0"], name="leiden"),
                columns=[n for n, _c in markers],
            ),
            positive_cluster_ids_per_marker={n: () for n, _c in markers},
            n_clusters=1,
        )
        result = CohortLabelResult(
            cores=(core,),
            markers=markers,
            zscore_tau=0.5,
            leiden_resolution=0.6,
            leiden_n_neighbors=20,
            leiden_pca_n_comps=10,
            scale_max_value=5.0,
            random_state=0,
            timestamp="2026-05-14T00:00:00",
            hexif_git_rev="abc1234",
        )
        cells = pd.DataFrame({"basename": [core.basename] * n_cells})

        augmented = emit_new_columns(result, cells)

        for i, pheno_name in enumerate(PHENOTYPE_NAMES):
            np.testing.assert_array_equal(
                augmented[col_phenotype_spacec(pheno_name)].to_numpy(),
                expected_pheno[:, i],
            )

    def test_unpopulated_cells_get_sentinels(self, caplog: pytest.LogCaptureFixture) -> None:
        """Cells whose basename has no CoreLabelResult get NaN / False / -1."""
        markers: tuple[tuple[str, int], ...] = (("DAPI", 0), ("CD68", 27))
        result = _build_synthetic_cohort_result(markers, n_cores=1, n_cells_per_core=4)

        # Mix one populated core + one unknown core in the cells frame.
        cells = pd.DataFrame(
            {
                "basename": ["synth_core_00"] * 4 + ["ghost_core"] * 3,
                "ch00_true": [1.0] * 7,
                "ch27_true": [2.0] * 7,
            }
        )

        with caplog.at_level("WARNING"):
            augmented = emit_new_columns(result, cells)

        # First 4 rows populated.
        assert augmented[COL_CLUSTER_ID_SPACEC].iloc[:4].tolist() == [0, 1, 0, 1]
        # Last 3 rows are sentinel.
        assert (augmented[COL_CLUSTER_ID_SPACEC].iloc[4:] == -1).all()
        assert augmented[col_pos_spacec(0)].iloc[4:].eq(False).all()
        assert augmented[col_zscore_spacec(0)].iloc[4:].isna().all()
        # And a warning was emitted.
        assert any("ghost_core" in r.message for r in caplog.records)

    def test_refuses_to_overwrite_existing_columns(self) -> None:
        """ValueError when a target column is already present."""
        markers: tuple[tuple[str, int], ...] = (("DAPI", 0), ("CD68", 27))
        result = _build_synthetic_cohort_result(markers, n_cores=1, n_cells_per_core=4)

        # ch00_pos_spacec collides — caller forgot to drop a previous run's output.
        cells = pd.DataFrame(
            {
                "basename": ["synth_core_00"] * 4,
                col_pos_spacec(0): [False] * 4,
            }
        )

        with pytest.raises(ValueError, match=col_pos_spacec(0)):
            emit_new_columns(result, cells)


# --------------------------------------------------------------------------- #
# 3. write_spacec_provenance
# --------------------------------------------------------------------------- #


class TestWriteSpacecProvenance:
    def test_round_trip_contains_required_keys(self, tmp_path: Path) -> None:
        markers: tuple[tuple[str, int], ...] = (("DAPI", 0), ("CD68", 27))
        result = _build_synthetic_cohort_result(markers, n_cores=2, n_cells_per_core=8)

        out = write_spacec_provenance(result, tmp_path / "spacec_labels_provenance.json")
        assert out.exists()

        with out.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        # Top-level required keys.
        for key in (
            "schema_version",
            "timestamp",
            "hexif_git_rev",
            "scanpy_version",
            "igraph_version",
            "parameters",
            "markers",
            "n_cores",
            "n_cells_total",
            "per_marker_positive_counts",
        ):
            assert key in payload, f"missing key {key!r} in provenance JSON"

        # Provenance scalars round-trip exactly.
        assert payload["timestamp"] == "2026-05-14T12:34:56"
        assert payload["hexif_git_rev"] == "deadbee"
        assert payload["n_cores"] == 2
        assert payload["n_cells_total"] == 16

        # Per-marker counts cover every marker we passed.
        assert set(payload["per_marker_positive_counts"]) == {"DAPI", "CD68"}
        # The synthetic fixture marks the first cluster positive for marker 0
        # and the last cluster positive for marker 1; both clusters have
        # exactly half the cells per core → 4 cells per core × 2 cores = 8.
        assert payload["per_marker_positive_counts"]["DAPI"] == 8
        assert payload["per_marker_positive_counts"]["CD68"] == 8

        # Parameter block round-trips with the right types.
        params = payload["parameters"]
        assert params["zscore_tau"] == 0.5
        assert params["leiden_resolution"] == 0.6
        assert params["leiden_n_neighbors"] == 20

    def test_atomic_write_leaves_no_tmp_residue(self, tmp_path: Path) -> None:
        markers: tuple[tuple[str, int], ...] = (("DAPI", 0),)
        result = _build_synthetic_cohort_result(markers, n_cores=1, n_cells_per_core=4)

        out_path = tmp_path / "prov.json"
        write_spacec_provenance(result, out_path)

        residues = list(tmp_path.glob("*.tmp"))
        assert residues == [], f"unexpected .tmp residue: {residues}"
        # Final file is present and well-formed JSON.
        with out_path.open("r", encoding="utf-8") as fh:
            json.load(fh)
