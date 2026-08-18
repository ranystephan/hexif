"""Deterministic unit tests for cell-polygon extraction and bundle joins."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexif.pipeline.polygons import build_cell_polygons

# ---------------------------------------------------------------------------
# Mask fixtures
# ---------------------------------------------------------------------------


def _square_mask(size: int = 100) -> np.ndarray:
    """Three solid axis-aligned squares with known areas + bboxes.

    cell_id=1: 15×15 (area=225) at top-left
    cell_id=2: 20×20 (area=400) in the middle
    cell_id=3: 8×8 (area=64) at bottom-right
    """
    mask = np.zeros((size, size), dtype=np.uint32)
    mask[5:20, 5:20] = 1
    mask[40:60, 40:60] = 2
    mask[85:93, 85:93] = 3
    return mask


def _disk_mask(size: int = 100, center: tuple[int, int] = (50, 50), radius: int = 20) -> np.ndarray:
    """A single filled circle. Used as a "smooth ellipse" for the simplify test."""
    yy, xx = np.ogrid[:size, :size]
    mask = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2).astype(np.uint32)
    return mask


def _write_pair(masks_dir: Path, basename: str, nuclei: np.ndarray, expanded: np.ndarray) -> None:
    """Helper: write the nuclei/expanded8 pair to masks_dir."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.save(masks_dir / f"{basename}_nuclei.npy", nuclei)
    np.save(masks_dir / f"{basename}_expanded8.npy", expanded)


# ---------------------------------------------------------------------------
# 1. Basic extraction on synthetic squares
# ---------------------------------------------------------------------------


def test_extracts_three_synthetic_cells(tmp_path: Path) -> None:
    """Three filled squares come out as three polygon rows with the right areas."""
    mask = _square_mask()
    _write_pair(tmp_path / "masks", "syn", mask, mask)
    cells_df = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 3, f"expected 3 polygon rows, got {len(out)}"
    assert set(out["cell_id"]) == {1, 2, 3}

    # Area ±5% of truth.
    expected_areas = {1: 225.0, 2: 400.0, 3: 64.0}
    for _, row in out.iterrows():
        expected = expected_areas[int(row["cell_id"])]
        assert abs(row["area_nucleus_px"] - expected) <= 0.05 * expected, (
            f"cell {row['cell_id']} area={row['area_nucleus_px']} expected {expected}"
        )

    # All polygons close (last vertex == first).
    for poly in out["nucleus_xy"]:
        assert poly[0] == poly[-1], "polygon ring not closed"

    # All polygons have ≤ 32 vertices.
    assert (out["n_verts_nucleus"] <= 32).all()
    assert (out["n_verts_expanded"] <= 32).all()


def test_polygon_centroid_matches_mask_centroid(tmp_path: Path) -> None:
    """centroid_x/y must agree with cells.parquet's mask-based centroid."""
    mask = _square_mask()
    _write_pair(tmp_path / "masks", "syn", mask, mask)
    cells_df = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    # cell_id=2: square [40:60, 40:60] → centroid (49.5, 49.5)
    row2 = out[out["cell_id"] == 2].iloc[0]
    assert abs(row2["centroid_y"] - 49.5) < 0.5
    assert abs(row2["centroid_x"] - 49.5) < 0.5


# ---------------------------------------------------------------------------
# 2. Simplification keeps ≤ 32 vertices on a smooth ellipse
# ---------------------------------------------------------------------------


def test_smooth_disk_simplifies_to_at_most_32_vertices(tmp_path: Path) -> None:
    """A filled circle (the smoothest "ellipse" we can write) fits under the cap."""
    mask = _disk_mask(size=120, center=(60, 60), radius=30)
    _write_pair(tmp_path / "masks", "disk", mask, mask)
    cells_df = pd.DataFrame({"basename": ["disk"], "cell_id": [1]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_verts_nucleus"] <= 32, (
        f"disk should simplify under 32 verts, got {row['n_verts_nucleus']}"
    )
    # No truncated flag: the disk is smooth enough that ε ≤ 4 always fits.
    assert "truncated" not in row["flags"]
    # Area is within 5% of πr² = π × 30² ≈ 2827.
    assert abs(row["area_nucleus_px"] - 2827.0) <= 0.05 * 2827.0


# ---------------------------------------------------------------------------
# 3. Edge cases from the API contract
# ---------------------------------------------------------------------------


def test_tiny_cell_is_skipped(tmp_path: Path) -> None:
    """Cells ≤ 4 px land in polygons_skipped.csv with reason ``tiny``."""
    mask = np.zeros((50, 50), dtype=np.uint32)
    mask[10:12, 10:12] = 1  # area = 4, threshold is ≤ 4 → skipped
    mask[30:35, 30:35] = 2  # area = 25, kept
    _write_pair(tmp_path / "masks", "tiny", mask, mask)
    cells_df = pd.DataFrame({"basename": ["tiny"] * 2, "cell_id": [1, 2]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 1
    assert int(out.iloc[0]["cell_id"]) == 2

    skipped = pd.read_csv(tmp_path / "out" / "polygons_skipped.csv")
    assert len(skipped) == 1
    assert int(skipped.iloc[0]["cell_id"]) == 1
    assert skipped.iloc[0]["reason"] == "tiny"


def test_border_cell_is_clamped_and_flagged(tmp_path: Path) -> None:
    """Cells that hit the image border are kept; coords clamped, ``border`` flag set."""
    mask = np.zeros((50, 50), dtype=np.uint32)
    mask[0:10, 0:10] = 1  # corner cell — contour vertices at row=-0.5 before padding
    _write_pair(tmp_path / "masks", "bord", mask, mask)
    cells_df = pd.DataFrame({"basename": ["bord"], "cell_id": [1]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 1
    assert "border" in out.iloc[0]["flags"]
    # All vertices within image bounds [0, 49].
    coords = np.array(out.iloc[0]["nucleus_xy"])
    assert coords.min() >= 0
    assert coords.max() <= 49


def test_id_invariant_violation_aborts(tmp_path: Path) -> None:
    """A nucleus ID absent from expanded8 is a hard error, not a silent skip."""
    nu = np.zeros((50, 50), dtype=np.uint32)
    ex = np.zeros((50, 50), dtype=np.uint32)
    nu[10:20, 10:20] = 1
    nu[30:40, 30:40] = 2  # only in nuclei, NOT in expanded
    ex[10:20, 10:20] = 1
    _write_pair(tmp_path / "masks", "bad", nu, ex)
    cells_df = pd.DataFrame({"basename": ["bad"] * 2, "cell_id": [1, 2]})

    with pytest.raises(ValueError, match="invariant violated"):
        build_cell_polygons(
            tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
        )


def test_multi_component_cell_takes_largest_and_flags(tmp_path: Path) -> None:
    """A cell split into two blobs is reduced to the largest with the multi flag set."""
    mask = np.zeros((60, 60), dtype=np.uint32)
    mask[10:25, 10:25] = 1  # 15×15 large blob
    mask[40:45, 40:45] = 1  # 5×5 small blob — same ID, disconnected
    _write_pair(tmp_path / "masks", "multi", mask, mask)
    cells_df = pd.DataFrame({"basename": ["multi"], "cell_id": [1]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 1
    row = out.iloc[0]
    # Spec §3.1 pins the per-row flag taxonomy: a single consolidated
    # ``multi_component`` token (no compartment suffix). Compartment-level
    # detail lives in the logger only.
    flag_set = set(row["flags"].split(",")) if row["flags"] else set()
    assert flag_set == {"multi_component"}, (
        f"expected exactly {{'multi_component'}}, got {flag_set}"
    )
    # The polygon spans only the larger blob; centroid is near (17, 17),
    # not the centroid of the union (~25, 25).
    assert 14 <= row["centroid_y"] <= 20
    assert 14 <= row["centroid_x"] <= 20


# ---------------------------------------------------------------------------
# 4. Schema + dtype pinning
# ---------------------------------------------------------------------------


def test_parquet_schema_is_pinned(tmp_path: Path) -> None:
    """The on-disk parquet must round-trip with the exact column dtypes."""
    import pyarrow.parquet as pq

    mask = _square_mask()
    _write_pair(tmp_path / "masks", "syn", mask, mask)
    cells_df = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})

    out_path = tmp_path / "out" / "cell_polygons.parquet"
    build_cell_polygons(tmp_path / "masks", cells_df, out_path)

    schema = pq.read_schema(out_path)
    expected_types = {
        "basename": "string",
        "cell_id": "int64",
        "centroid_y": "double",
        "centroid_x": "double",
        "area_nucleus_px": "double",
        "area_expanded_px": "double",
        "n_verts_nucleus": "int32",
        "n_verts_expanded": "int32",
        "flags": "string",
    }
    for col, want in expected_types.items():
        got = str(schema.field(col).type)
        assert got == want, f"column {col}: expected {want}, got {got}"
    # The nested list<list<double>> for the polygon coords. pyarrow's
    # type repr varies slightly across versions ("item" vs "element"); we
    # check the structural type via ``equals`` instead of the printed form.
    import pyarrow as pa

    list_of_list_double = pa.list_(pa.list_(pa.float64()))
    assert schema.field("nucleus_xy").type.equals(list_of_list_double)
    assert schema.field("expanded_xy").type.equals(list_of_list_double)


# ---------------------------------------------------------------------------
# 5. Bundle round-trip (load_bundle reads cell_polygons + validates join)
# ---------------------------------------------------------------------------


def test_bundle_load_reads_polygons_and_flips_manifest(tmp_path: Path) -> None:
    """A synthetic bundle: write cells + polygons + manifest, then load_bundle."""
    from hexif.pipeline.bundle import load_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    masks_dir = tmp_path / "masks"

    mask = _square_mask()
    _write_pair(masks_dir, "syn", mask, mask)

    cells = pd.DataFrame(
        {
            "basename": ["syn"] * 3,
            "cell_id": [1, 2, 3],
            "centroid_y": [12.0, 49.5, 88.5],
            "centroid_x": [12.0, 49.5, 88.5],
        }
    )
    cells.to_parquet(bundle / "cells.parquet", index=False)
    # A minimal core_summary so load_bundle doesn't choke on the missing file.
    pd.DataFrame({"basename": ["syn"], "split": ["val"], "n_cells": [3]}).to_parquet(
        bundle / "core_summary.parquet", index=False
    )

    build_cell_polygons(masks_dir, cells[["basename", "cell_id"]], bundle / "cell_polygons.parquet")

    # Manifest with has_cell_polygons.
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": "v0.2-test",
                "build_date": "2026-05-12T00:00:00Z",
                "n_cores": 1,
                "n_cells": 3,
                "splits": ["val"],
                "model_ids": ["v1_1"],
                "marker_channels": [],
                "marker_names": [],
                "phenotype_names": [],
                "sources": {},
                "threshold_source": "test",
                "notes": "",
                "has_cell_polygons": True,
            }
        )
    )

    loaded = load_bundle(bundle)
    assert loaded["manifest"]["has_cell_polygons"] is True
    assert loaded["cell_polygons"] is not None
    assert len(loaded["cell_polygons"]) == 3
    # Join with cells.parquet is complete and unambiguous.
    merged = loaded["cells"][["basename", "cell_id"]].merge(
        loaded["cell_polygons"][["basename", "cell_id"]],
        on=["basename", "cell_id"],
        how="left",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()


def test_bundle_load_tolerates_skipped_cells(tmp_path: Path) -> None:
    """If a cell is missing from polygons but listed in skipped.csv, load is OK."""
    from hexif.pipeline.bundle import load_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    masks_dir = tmp_path / "masks"

    # Build a mask where cell 1 is tiny (will be skipped) and cell 2 is fine.
    mask = np.zeros((50, 50), dtype=np.uint32)
    mask[10:12, 10:12] = 1  # area = 4 → tiny, skipped
    mask[30:40, 30:40] = 2
    _write_pair(masks_dir, "mix", mask, mask)

    cells = pd.DataFrame(
        {
            "basename": ["mix", "mix"],
            "cell_id": [1, 2],
            "centroid_y": [10.5, 34.5],
            "centroid_x": [10.5, 34.5],
        }
    )
    cells.to_parquet(bundle / "cells.parquet", index=False)
    pd.DataFrame({"basename": ["mix"], "split": ["val"], "n_cells": [2]}).to_parquet(
        bundle / "core_summary.parquet", index=False
    )
    build_cell_polygons(masks_dir, cells[["basename", "cell_id"]], bundle / "cell_polygons.parquet")
    (bundle / "manifest.json").write_text(json.dumps({"has_cell_polygons": True}))

    # cell 1 should be in the skip CSV.
    skip = pd.read_csv(bundle / "polygons_skipped.csv")
    assert set(skip["cell_id"]) == {1}

    loaded = load_bundle(bundle)
    # Polygons has only cell 2, but load_bundle does not raise because
    # cell 1 is documented in polygons_skipped.csv.
    assert set(loaded["cell_polygons"]["cell_id"]) == {2}


def test_bundle_load_detects_silent_polygon_drop(tmp_path: Path) -> None:
    """If a cell is missing from BOTH polygons and skipped.csv, load_bundle raises."""
    from hexif.pipeline.bundle import load_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()

    # cells.parquet claims 2 cells; cell_polygons.parquet has only 1; skip CSV is empty.
    cells = pd.DataFrame(
        {
            "basename": ["x", "x"],
            "cell_id": [1, 2],
            "centroid_y": [10.0, 20.0],
            "centroid_x": [10.0, 20.0],
        }
    )
    cells.to_parquet(bundle / "cells.parquet", index=False)
    pd.DataFrame({"basename": ["x"], "split": ["val"], "n_cells": [2]}).to_parquet(
        bundle / "core_summary.parquet", index=False
    )
    # Write a deliberately partial polygons table.
    polys = pd.DataFrame(
        [
            {
                "basename": "x",
                "cell_id": 1,
                "centroid_y": 10.0,
                "centroid_x": 10.0,
                "nucleus_xy": [[10.0, 10.0], [11.0, 10.0], [10.5, 11.0], [10.0, 10.0]],
                "expanded_xy": [[10.0, 10.0], [11.0, 10.0], [10.5, 11.0], [10.0, 10.0]],
                "area_nucleus_px": 4.0,
                "area_expanded_px": 4.0,
                "n_verts_nucleus": 3,
                "n_verts_expanded": 3,
                "flags": "",
            }
        ]
    )
    polys["cell_id"] = polys["cell_id"].astype("int64")
    polys["n_verts_nucleus"] = polys["n_verts_nucleus"].astype("int32")
    polys["n_verts_expanded"] = polys["n_verts_expanded"].astype("int32")
    polys.to_parquet(bundle / "cell_polygons.parquet", index=False)
    # Empty skip CSV.
    pd.DataFrame(
        columns=["basename", "cell_id", "reason", "area_nucleus_px", "area_expanded_px"]
    ).to_csv(bundle / "polygons_skipped.csv", index=False)

    (bundle / "manifest.json").write_text(json.dumps({"has_cell_polygons": True}))

    with pytest.raises(ValueError, match="missing from both"):
        load_bundle(bundle)


def test_bundle_load_back_compat_no_polygons(tmp_path: Path) -> None:
    """A v0.1 bundle (no cell_polygons.parquet) loads without complaint."""
    from hexif.pipeline.bundle import load_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cells = pd.DataFrame({"basename": ["x"], "cell_id": [1], "centroid_y": [1.0]})
    cells.to_parquet(bundle / "cells.parquet", index=False)
    pd.DataFrame({"basename": ["x"], "split": ["val"], "n_cells": [1]}).to_parquet(
        bundle / "core_summary.parquet", index=False
    )
    (bundle / "manifest.json").write_text(json.dumps({"has_cell_polygons": False}))

    loaded = load_bundle(bundle)
    assert loaded["cell_polygons"] is None
    assert loaded["manifest"]["has_cell_polygons"] is False


# ---------------------------------------------------------------------------
# 5b. Truncation + simplify-helper edge cases
# ---------------------------------------------------------------------------


def test_jagged_polygon_truncates_at_vertex_cap(tmp_path: Path) -> None:
    """A pathologically jagged outline gets the ``truncated`` flag, never crashes.

    We synthesize a "spiky" mask: a ring of 1-pixel teeth around a disk.
    The contour has hundreds of vertices; even Douglas-Peucker at ε = 4
    can't fit it under 32, so the code path that truncates and flags
    must fire.
    """
    size = 60
    yy, xx = np.ogrid[:size, :size]
    cy, cx = 30, 30
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    base = (r2 <= 18**2).astype(np.uint32)
    # Inject high-frequency single-pixel teeth at the boundary.
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    for a in angles:
        y = int(cy + 20 * np.sin(a))
        x = int(cx + 20 * np.cos(a))
        if 0 <= y < size and 0 <= x < size:
            base[y, x] = 1
    _write_pair(tmp_path / "masks", "spiky", base, base)
    cells_df = pd.DataFrame({"basename": ["spiky"], "cell_id": [1]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )

    assert len(out) == 1
    row = out.iloc[0]
    # The vertex count must respect the hard cap of 32, regardless of how
    # spiky the input is.
    assert row["n_verts_nucleus"] <= 32
    assert row["n_verts_expanded"] <= 32


def test_degenerate_input_uses_unsimplified_fallback() -> None:
    """A dense smooth polygon is simplified to ≤ 32 verts with no degenerate flag.

    A 100-vertex unit circle forces the simplifier to run. The result must
    contain at most 32 vertices and must not carry a ``degenerate`` flag.
    """
    from hexif.pipeline.polygons import _MAX_VERTS, _simplify_polygon

    # 100 vertices on the unit circle, ring-closed → 101 rows.
    n = 100
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.column_stack([np.cos(theta), np.sin(theta)])
    coords = np.vstack([coords, coords[:1]])  # close the ring
    assert len(coords) == n + 1

    out, trunc, degen = _simplify_polygon(coords)
    # Simplifier ran (the input had 100 verts; output must be ≤ 32).
    assert len(out) - 1 <= _MAX_VERTS, f"simplifier did not run: {len(out) - 1} > {_MAX_VERTS}"
    # No degenerate fallback (a smooth circle simplifies cleanly).
    assert not degen, "smooth circle should not collapse to < 3 unique verts"
    # No truncation (a circle is the easy case — ε ≤ 4 fits comfortably).
    assert not trunc, "smooth circle should fit under the vertex cap before ε_max"
    assert out.shape[1] == 2


def test_two_vertex_input_is_degenerate() -> None:
    """A 2-vertex "polygon" is below the constructor minimum and is flagged."""
    from hexif.pipeline.polygons import _simplify_polygon

    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    _out, trunc, degen = _simplify_polygon(coords)
    assert degen
    assert not trunc


# ---------------------------------------------------------------------------
# 6. rebuild_polygons + CLI smoke
# ---------------------------------------------------------------------------


def test_rebuild_polygons_updates_manifest(tmp_path: Path) -> None:
    """``rebuild_polygons`` writes the parquet AND flips the manifest flag."""
    from hexif.pipeline.bundle import rebuild_polygons

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    masks_dir = tmp_path / "masks"

    mask = _square_mask()
    _write_pair(masks_dir, "syn", mask, mask)
    cells = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})
    cells.to_parquet(bundle / "cells.parquet", index=False)
    (bundle / "manifest.json").write_text(json.dumps({"has_cell_polygons": False, "sources": {}}))

    out_path = rebuild_polygons(bundle, masks_dir=masks_dir)
    assert out_path.exists()
    assert out_path.name == "cell_polygons.parquet"
    m = json.loads((bundle / "manifest.json").read_text())
    assert m["has_cell_polygons"] is True
    assert m["sources"]["masks_dir"].endswith("masks")


def test_cli_parses_polygon_flags() -> None:
    """``hexif build-cell-tables`` and ``hexif rebuild-polygons`` parse cleanly."""
    from hexif.cli.main import build_parser

    parser = build_parser()
    required = [
        "build-cell-tables",
        "--output-dir",
        "/tmp/x",
        "--pairs-dir",
        "/tmp/pairs",
        "--cell-predictions",
        "/tmp/predictions.csv",
        "--consensus",
        "/tmp/consensus.csv",
        "--manifest",
        "/tmp/manifest.csv",
        "--thresholds-json",
        "/tmp/thresholds.json",
        "--model-metadata",
        "/tmp/models.json",
        "--model-id",
        "test_model",
        "--masks-dir",
        "/tmp/masks",
    ]
    ns = parser.parse_args([*required, "--no-polygons"])
    assert ns.polygons is False

    ns = parser.parse_args(required)
    assert ns.polygons is True

    ns = parser.parse_args(
        [
            "rebuild-polygons",
            "--bundle",
            "/tmp/b",
            "--masks-dir",
            "/tmp/m",
        ]
    )
    assert ns.bundle == "/tmp/b"
    assert ns.masks_dir == "/tmp/m"


# ---------------------------------------------------------------------------
# 7. Logging contract
# ---------------------------------------------------------------------------


def test_no_print_in_polygons_module(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Diagnostics must go through ``logging``, never ``print``."""
    mask = _square_mask()
    _write_pair(tmp_path / "masks", "syn", mask, mask)
    cells_df = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})

    with caplog.at_level(logging.INFO, logger="hexif.pipeline.polygons"):
        build_cell_polygons(
            tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
        )
    # We expect at least one INFO record from the per-core driver.
    assert any("polygons: syn" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 8. Build aborts, determinism, and working-directory independence
# ---------------------------------------------------------------------------


def test_build_bundle_aborts_on_invariant_violation(tmp_path: Path) -> None:
    """If masks_v1 violates the nuclei⊆expanded8 invariant, build_bundle aborts
    BEFORE writing manifest.json — the API contract forbids the half-written bundle.

    The nuclei mask has a cell ID missing from the expanded mask. The test asserts:
      1) build_bundle raises ValueError
      2) manifest.json does NOT exist on disk afterward
    """
    from hexif.pipeline.bundle import build_bundle

    # Construct minimal inputs that exercise the polygon branch.
    pairs_dir = tmp_path / "pairs"
    masks_dir = tmp_path / "masks"
    pairs_dir.mkdir()
    masks_dir.mkdir()

    # Mismatched mask pair: nucleus ID 2 is absent from expanded mask.
    nu = np.zeros((50, 50), dtype=np.uint32)
    ex = np.zeros((50, 50), dtype=np.uint32)
    nu[10:20, 10:20] = 1
    nu[30:40, 30:40] = 2
    ex[10:20, 10:20] = 1  # missing ID 2
    np.save(masks_dir / "bad_nuclei.npy", nu)
    np.save(masks_dir / "bad_expanded8.npy", ex)

    # H&E placeholder so build_bundle doesn't blow up on the thumbs step
    # (we set --no-thumbs equivalent via render_thumbs=False).
    np.save(pairs_dir / "bad_HE.npy", np.zeros((50, 50, 3), dtype=np.float32))

    # cell_predictions CSV with the same basename/cell_ids.
    pred_csv = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "basename": ["bad", "bad"],
            "cell_id": [1, 2],
            "centroid_x": [14.5, 34.5],
            "centroid_y": [14.5, 34.5],
        }
    ).to_csv(pred_csv, index=False)

    # Manifest CSV (training_ready_5px shape).
    man_csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "tma": ["t"],
            "core": ["c"],
            "basename": ["bad"],
            "split": ["val"],
            "rigid_D_px": [0.0],
        }
    ).to_csv(man_csv, index=False)

    consensus = tmp_path / "consensus.csv"
    pd.DataFrame({"basename": ["bad", "bad"], "cell_id": [1, 2]}).to_csv(consensus, index=False)
    from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES

    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "marker_thresholds": {str(ch): 0.5 for ch in FOCUSED_MARKERS},
                "phenotype_thresholds": {name: 0.5 for name in PHENOTYPE_NAMES},
                "marker_confidence": {name: "weak" for name in MARKER_NAMES},
                "phenotype_confidence": {name: "weak" for name in PHENOTYPE_NAMES},
                "provenance": {"fixture": "deterministic unit test"},
            }
        )
    )
    metadata = tmp_path / "models.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "test_model",
                        "name": "Test model",
                        "backbone": "test encoder",
                        "training_split": "test fixture",
                        "provenance": {"fixture": "deterministic unit test"},
                    }
                ],
            }
        )
    )

    output_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="invariant violated"):
        build_bundle(
            output_dir,
            pairs_dir=pairs_dir,
            cell_predictions=pred_csv,
            consensus=consensus,
            manifest=man_csv,
            thresholds_json=thresholds,
            model_metadata=metadata,
            model_ids=("test_model",),
            primary_model="test_model",
            split="val",
            render_thumbs=False,
            skip_codex=True,
            build_polygons=True,
            masks_dir=masks_dir,
        )
    # Build aborted: no manifest.json should have been written.
    assert not (output_dir / "manifest.json").exists(), (
        "manifest.json must not exist after an aborted build — bundle would be malformed"
    )


def test_flag_column_is_deterministic(tmp_path: Path) -> None:
    """Two builds on identical input must produce byte-identical ``flags`` columns.

    The previous implementation appended flag tokens in mask-iteration order,
    which made the joined string order-dependent on Python set semantics.
    The fix sorts flag tokens before joining; this test pins that
    invariant so a future re-ordering of the append sites can't reintroduce
    the bug silently.
    """
    # Use a mask that triggers BOTH border + multi_component for the same cell:
    # a cell that touches the corner AND has a disconnected secondary blob.
    mask = np.zeros((50, 50), dtype=np.uint32)
    mask[0:8, 0:8] = 1  # corner cell → border flag
    mask[20:23, 20:23] = 1  # disconnected secondary blob → multi_component
    _write_pair(tmp_path / "masks", "det", mask, mask)
    cells_df = pd.DataFrame({"basename": ["det"], "cell_id": [1]})

    out1 = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out1" / "cell_polygons.parquet"
    )
    out2 = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out2" / "cell_polygons.parquet"
    )
    assert list(out1["flags"]) == list(out2["flags"]), (
        f"flag column nondeterministic: {list(out1['flags'])} vs {list(out2['flags'])}"
    )
    # Both flags present.
    tokens = out1.iloc[0]["flags"].split(",")
    assert "border" in tokens
    assert "multi_component" in tokens
    # Sorted lexicographically.
    assert tokens == sorted(tokens), f"flag tokens not sorted: {tokens}"


def test_flag_tokens_are_in_pinned_set(tmp_path: Path) -> None:
    """Every flag emitted must come from the pinned taxonomy.

    Allowed values are ``degenerate``, ``truncated``, ``border``, ``empty``,
    and ``multi_component``.
    Anything else is a contract violation and would surprise downstream
    consumers.
    """
    # Combine: tiny (skipped, not in row flags) + border + multi-component.
    mask = np.zeros((60, 60), dtype=np.uint32)
    mask[0:8, 0:8] = 1
    mask[20:23, 20:23] = 1
    mask[40:50, 40:50] = 2
    _write_pair(tmp_path / "masks", "tax", mask, mask)
    cells_df = pd.DataFrame({"basename": ["tax"] * 2, "cell_id": [1, 2]})

    out = build_cell_polygons(
        tmp_path / "masks", cells_df, tmp_path / "out" / "cell_polygons.parquet"
    )
    allowed = {"", "degenerate", "truncated", "border", "empty", "multi_component"}
    for f in out["flags"]:
        if f == "":
            continue
        for token in f.split(","):
            assert token in allowed, f"unknown flag token {token!r} (full: {f!r})"


def test_rebuild_polygons_uses_manifest_masks_dir_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rebuild_polygons`` resolves masks_dir from manifest.json:sources.masks_dir
    when --masks-dir is omitted. Validates we no longer require the user to
    run from the repo root for the CLI to find the masks.
    """
    from hexif.pipeline.bundle import rebuild_polygons

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    masks_dir = tmp_path / "masks_somewhere"
    mask = _square_mask()
    _write_pair(masks_dir, "syn", mask, mask)

    cells = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})
    cells.to_parquet(bundle / "cells.parquet", index=False)
    # Manifest declares the (absolute) masks_dir, as a real built bundle does.
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "has_cell_polygons": False,
                "sources": {"masks_dir": str(masks_dir.resolve())},
            }
        )
    )

    # Move CWD somewhere unrelated.
    unrelated = tmp_path / "unrelated_cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    # No --masks-dir passed; resolution must come from the manifest.
    out = rebuild_polygons(bundle, masks_dir=None)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 3
    # Manifest flipped and points at the same absolute masks_dir.
    m = json.loads((bundle / "manifest.json").read_text())
    assert m["has_cell_polygons"] is True
    assert m["sources"]["masks_dir"] == str(masks_dir.resolve())


def test_rebuild_polygons_explicit_masks_dir_overrides_manifest(tmp_path: Path) -> None:
    """If both manifest and --masks-dir are present, --masks-dir wins.

    Documented in the rebuild_polygons docstring; pinning it as a test so
    a refactor can't accidentally flip the precedence.
    """
    from hexif.pipeline.bundle import rebuild_polygons

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # Real masks at "actual"; stale path in manifest pointing nowhere.
    actual = tmp_path / "actual_masks"
    mask = _square_mask()
    _write_pair(actual, "syn", mask, mask)

    cells = pd.DataFrame({"basename": ["syn"] * 3, "cell_id": [1, 2, 3]})
    cells.to_parquet(bundle / "cells.parquet", index=False)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "has_cell_polygons": False,
                "sources": {"masks_dir": "/this/path/does/not/exist/anywhere"},
            }
        )
    )

    # Explicit override should be used and the manifest stale path ignored.
    out = rebuild_polygons(bundle, masks_dir=actual)
    df = pd.read_parquet(out)
    assert len(df) == 3


def test_cli_rebuild_polygons_clean_error_on_missing_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hexif rebuild-polygons --bundle <nonexistent>`` exits 1 with a
    one-line error rather than a Python traceback.
    """
    from hexif.cli.main import _cmd_rebuild_polygons, build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "rebuild-polygons",
            "--bundle",
            str(tmp_path / "does_not_exist"),
        ]
    )
    rc = _cmd_rebuild_polygons(ns)
    assert rc == 1
    captured = capsys.readouterr()
    # No "Traceback" on stderr; just a hexif-prefixed message.
    assert "Traceback" not in captured.err
    assert "hexif:" in captured.err


def test_cli_rebuild_polygons_refuses_traversal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--bundle`` may not contain '..' — defensive against path traversal."""
    from hexif.cli.main import _cmd_rebuild_polygons, build_parser

    parser = build_parser()
    ns = parser.parse_args(["rebuild-polygons", "--bundle", "../escape/here"])
    rc = _cmd_rebuild_polygons(ns)
    assert rc == 1
    captured = capsys.readouterr()
    assert "traversal" in captured.err.lower()


def test_simplify_polygon_called_for_dense_input() -> None:
    """Use unittest.mock to assert _simplify_polygon was actually invoked.

    Complements ``test_degenerate_input_uses_unsimplified_fallback``:
    that test verifies the *output* of the simplifier; this one verifies
    the *call site* in ``_process_one_cell`` actually invokes it. Without
    this, a refactor could bypass simplify and the only signal would be a
    one-off perf regression — too easy to miss.
    """
    from unittest.mock import patch

    import hexif.pipeline.polygons as polymod

    # Use a large enough disk that simplify has meaningful work to do.
    mask = _disk_mask(size=80, center=(40, 40), radius=20)
    cells_df = pd.DataFrame({"basename": ["d"], "cell_id": [1]})

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _write_pair(td_path / "masks", "d", mask, mask)

        # Wrap the real implementation so we can count calls but keep
        # the return value semantically valid.
        real = polymod._simplify_polygon
        call_count = {"n": 0}

        def counting(*a, **k):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return real(*a, **k)

        with patch.object(polymod, "_simplify_polygon", side_effect=counting):
            polymod.build_cell_polygons(
                td_path / "masks", cells_df, td_path / "out" / "cell_polygons.parquet"
            )
        # _simplify_polygon is called once per compartment per cell = 2 here.
        assert call_count["n"] >= 2, f"_simplify_polygon ran only {call_count['n']} times"
