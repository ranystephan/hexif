"""Synthetic-data round-trip tests for ``hexif.audit.report``.

We hand-roll a tiny :class:`CohortAuditResult` (2 cores × 2 markers) here
rather than calling ``audit_cohort`` — audit computation owns the compute, report serializer
(this file) owns the serializers. Tests must not import scanpy and must
not need a real bundle on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexif.audit.report import (
    write_cohort_audit_csv,
    write_cohort_audit_json,
    write_cohort_audit_summary,
)
from hexif.audit.types import (
    CohortAuditResult,
    CoreAuditResult,
    MarkerAuditResult,
)

# ---------------------------------------------------------------------------
# Synthetic fixture — 2 cores × 2 markers. Picks one (core, marker) cell
# to fire ``suspicious_zero_consensus`` (consensus = 0, per-core Otsu = 50%).
# ---------------------------------------------------------------------------


_MARKERS = (("DAPI", 0), ("CD68", 27))


def _make_marker(
    *,
    name: str,
    channel: int,
    n_cells: int,
    n_pos_consensus: int,
    n_pos_otsu: int,
    n_pos_leiden: int,
    otsu_thr: float = 1.5,
    cohort_gmm: float = 2.0,
    cohort_otsu: float = 1.8,
    cohort_quantile: float = 2.2,
    leiden_pos_ids: tuple[str, ...] = (),
    max_cluster_zscore: float = 0.3,
) -> MarkerAuditResult:
    return MarkerAuditResult(
        name=name,
        channel=channel,
        n_cells=n_cells,
        n_pos_cohort_consensus=n_pos_consensus,
        n_pos_per_core_otsu=n_pos_otsu,
        n_pos_leiden_zscore=n_pos_leiden,
        per_core_otsu_threshold_log1p=otsu_thr,
        cohort_thresholds_log1p={
            "gmm": cohort_gmm,
            "otsu": cohort_otsu,
            "quantile": cohort_quantile,
            "target_pos": 0.05,
        },
        leiden_positive_cluster_ids=leiden_pos_ids,
        max_leiden_cluster_zscore=max_cluster_zscore,
    )


def _make_core(
    *,
    basename: str,
    cd68_consensus: int,
    cd68_otsu: int,
    cd68_leiden: int,
) -> CoreAuditResult:
    """A 100-cell core with DAPI (always agree) and CD68 (the variable one)."""
    n_cells = 100
    dapi = _make_marker(
        name="DAPI",
        channel=0,
        n_cells=n_cells,
        n_pos_consensus=98,
        n_pos_otsu=99,
        n_pos_leiden=97,
        max_cluster_zscore=2.1,
        leiden_pos_ids=("0", "1"),
    )
    cd68 = _make_marker(
        name="CD68",
        channel=27,
        n_cells=n_cells,
        n_pos_consensus=cd68_consensus,
        n_pos_otsu=cd68_otsu,
        n_pos_leiden=cd68_leiden,
        otsu_thr=0.5,
        max_cluster_zscore=1.4,
        leiden_pos_ids=("2",) if cd68_leiden > 0 else (),
    )
    return CoreAuditResult(
        basename=basename,
        n_cells=n_cells,
        per_marker={"DAPI": dapi, "CD68": cd68},
        leiden_resolution=0.6,
        leiden_n_neighbors=20,
        leiden_pca_n_comps=10,
        leiden_n_clusters=3,
        leiden_cluster_id_per_cell=np.zeros(n_cells, dtype=np.int64),
        leiden_cluster_zscore_matrix=pd.DataFrame(
            np.zeros((3, 2)),
            index=["0", "1", "2"],
            columns=["DAPI", "CD68"],
        ),
        zscore_tau=0.5,
        random_state=0,
    )


@pytest.fixture
def synthetic_result() -> CohortAuditResult:
    """2 cores × 2 markers. Core B is the CD68/A-10 lookalike (suspicious)."""
    core_a = _make_core(
        basename="ccRCC_TMA1__A-10",
        cd68_consensus=0,
        cd68_otsu=50,  # fires suspicious_zero_consensus (50% > 5%)
        cd68_leiden=20,
    )
    core_b = _make_core(
        basename="ccRCC_TMA1__B-3",
        cd68_consensus=15,
        cd68_otsu=18,
        cd68_leiden=12,
    )
    return CohortAuditResult(
        cores=(core_a, core_b),
        markers=_MARKERS,
        zscore_tau=0.5,
        leiden_resolution=0.6,
        leiden_n_neighbors=20,
        leiden_pca_n_comps=10,
        scale_max_value=5.0,
        random_state=0,
        bundle_path=Path("/tmp/fake_bundle"),
        thresholds_path=Path("/tmp/fake_thresholds.json"),
        timestamp="2026-05-14T12:34:56",
        hexif_git_rev="abc1234",
    )


# ---------------------------------------------------------------------------
# CSV tests — 4 rows, pinned columns, boolean round-trip.
# ---------------------------------------------------------------------------


_EXPECTED_CSV_COLUMNS = [
    "basename",
    "marker",
    "channel",
    "n_cells",
    "n_pos_consensus",
    "n_pos_per_core_otsu",
    "n_pos_leiden_zscore",
    "frac_consensus",
    "frac_per_core_otsu",
    "frac_leiden",
    "per_core_otsu_threshold_log1p",
    "cohort_gmm_threshold_log1p",
    "cohort_otsu_threshold_log1p",
    "cohort_quantile_threshold_log1p",
    "max_leiden_cluster_zscore",
    "disagreement_score",
    "suspicious_zero_consensus",
]


def test_csv_round_trip(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    out = tmp_path / "cohort_audit.csv"
    returned = write_cohort_audit_csv(synthetic_result, out)
    assert returned == out.resolve()
    assert out.exists()
    assert not (tmp_path / "cohort_audit.csv.tmp").exists()

    df = pd.read_csv(out)
    # 2 cores × 2 markers = 4 rows.
    assert len(df) == 4
    # Column names and order match the spec verbatim.
    assert list(df.columns) == _EXPECTED_CSV_COLUMNS
    # Suspicious-zero-consensus flag round-trips as boolean truthiness.
    suspicious_rows = df[df["suspicious_zero_consensus"].astype(bool)]
    assert len(suspicious_rows) == 1
    row = suspicious_rows.iloc[0]
    assert row["basename"] == "ccRCC_TMA1__A-10"
    assert row["marker"] == "CD68"
    assert int(row["n_pos_consensus"]) == 0
    assert pytest.approx(float(row["frac_per_core_otsu"]), abs=1e-9) == 0.5

    # Stable sort: rows are ordered by (basename, marker_index_in_MARKERS_DEFAULT).
    # DAPI is index 0, CD68 is index 1, so within each core DAPI comes first.
    assert list(df["basename"]) == [
        "ccRCC_TMA1__A-10",
        "ccRCC_TMA1__A-10",
        "ccRCC_TMA1__B-3",
        "ccRCC_TMA1__B-3",
    ]
    assert list(df["marker"]) == ["DAPI", "CD68", "DAPI", "CD68"]


def test_csv_accepts_str_path(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    out_str = str(tmp_path / "nested" / "cohort_audit.csv")
    returned = write_cohort_audit_csv(synthetic_result, out_str)
    assert returned.exists()
    assert returned.name == "cohort_audit.csv"


# ---------------------------------------------------------------------------
# Markdown tests — four sections present, suspicious row visible.
# ---------------------------------------------------------------------------


def test_markdown_four_sections_and_suspicious_row(
    synthetic_result: CohortAuditResult, tmp_path: Path
) -> None:
    out = tmp_path / "cohort_audit.md"
    returned = write_cohort_audit_summary(synthetic_result, out, top_n_outliers=5)
    assert returned == out.resolve()
    assert out.exists()
    assert not (tmp_path / "cohort_audit.md.tmp").exists()

    text = out.read_text()
    # Section 1 — header.
    assert "# HEXIF cohort label audit (label audit)" in text
    # Section 1 — parameters table.
    assert "## Parameters" in text
    # Section 2 — per-marker cohort agreement.
    assert "## Per-marker cohort agreement" in text
    # Section 3 — top-N suspicious zero-consensus.
    assert "suspicious zero-consensus" in text.lower()
    # Section 4 — top-N highest-disagreement.
    assert "highest-disagreement" in text.lower()

    # The suspicious core/marker is recorded with its per-core Otsu fraction.
    assert "ccRCC_TMA1__A-10" in text
    assert "CD68" in text
    # frac_per_core_otsu = 50/100 = 0.500 — should appear as 0.500 in the
    # suspicious table (default 3-digit float format).
    assert "0.500" in text

    # Provenance bits surface in the header.
    assert "2026-05-14T12:34:56" in text
    assert "abc1234" in text


def test_markdown_top_n_clamps(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    """Passing top_n_outliers larger than data still produces a valid file."""
    out = tmp_path / "summary.md"
    write_cohort_audit_summary(synthetic_result, out, top_n_outliers=100)
    text = out.read_text()
    # Both section headers still appear with the requested N.
    assert "Top-100 suspicious zero-consensus" in text
    assert "Top-100 highest-disagreement" in text


# ---------------------------------------------------------------------------
# JSON tests — round-trip, provenance, no large arrays.
# ---------------------------------------------------------------------------


def test_json_round_trip(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    out = tmp_path / "cohort_audit.json"
    returned = write_cohort_audit_json(synthetic_result, out)
    assert returned == out.resolve()
    assert out.exists()
    assert not (tmp_path / "cohort_audit.json.tmp").exists()

    with open(out) as fh:
        payload = json.load(fh)

    # Provenance fields populated.
    assert payload["timestamp"] == "2026-05-14T12:34:56"
    assert payload["hexif_git_rev"] == "abc1234"
    assert payload["bundle_path"] == "/tmp/fake_bundle"
    assert payload["thresholds_path"] == "/tmp/fake_thresholds.json"
    assert payload["n_cores"] == 2
    assert payload["n_cells_total"] == 200

    # Parameters block matches the dataclass field-by-field.
    p = payload["parameters"]
    assert p["zscore_tau"] == 0.5
    assert p["leiden_resolution"] == 0.6
    assert p["leiden_n_neighbors"] == 20
    assert p["leiden_pca_n_comps"] == 10
    assert p["scale_max_value"] == 5.0
    assert p["random_state"] == 0

    # Markers list matches the cohort tuple.
    assert payload["markers"] == [
        {"name": "DAPI", "channel": 0},
        {"name": "CD68", "channel": 27},
    ]

    # Per-marker summary contains mean / std / median for each fraction.
    pms = payload["per_marker_summary"]
    assert set(pms.keys()) == {"DAPI", "CD68"}
    cd68_stats = pms["CD68"]["frac_per_core_otsu"]
    assert set(cd68_stats.keys()) == {"mean", "std", "median"}
    # Core A otsu frac = 0.5, core B = 0.18 → mean = 0.34.
    assert cd68_stats["mean"] == pytest.approx(0.34, abs=1e-9)

    # The bulky arrays are intentionally absent.
    for core in payload["cores"]:
        assert "leiden_cluster_id_per_cell" not in core
        assert "leiden_cluster_zscore_matrix" not in core

    # The suspicious flag survived.
    core_a = next(c for c in payload["cores"] if c["basename"] == "ccRCC_TMA1__A-10")
    assert core_a["per_marker"]["CD68"]["suspicious_zero_consensus"] is True
    assert core_a["per_marker"]["CD68"]["n_pos_cohort_consensus"] == 0


def test_json_accepts_str_path(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    out_str = str(tmp_path / "deep" / "nested" / "cohort_audit.json")
    returned = write_cohort_audit_json(synthetic_result, out_str)
    assert returned.exists()
    # json.load works — no NaN / inf leaked.
    with open(returned) as fh:
        json.load(fh)


# ---------------------------------------------------------------------------
# Atomic-write sanity — no .tmp residue after any successful writer.
# ---------------------------------------------------------------------------


def test_no_tmp_residue(synthetic_result: CohortAuditResult, tmp_path: Path) -> None:
    csv_path = write_cohort_audit_csv(synthetic_result, tmp_path / "a.csv")
    md_path = write_cohort_audit_summary(synthetic_result, tmp_path / "a.md")
    json_path = write_cohort_audit_json(synthetic_result, tmp_path / "a.json")

    # All three landed.
    assert csv_path.exists()
    assert md_path.exists()
    assert json_path.exists()

    # No .tmp leftover anywhere under tmp_path.
    leftover_tmp = list(tmp_path.rglob("*.tmp"))
    assert leftover_tmp == [], f"unexpected .tmp residue: {leftover_tmp}"
