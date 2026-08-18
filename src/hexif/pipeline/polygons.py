"""Extract per-cell polygons from segmentation masks for the Workbench.

The pipeline emits one polygon per cell for the nucleus and expanded-cell
compartments at bundle-build time, so the frontend does not load
full-resolution masks.

The fast path is bbox-cropped contouring: ``scipy.ndimage.find_objects``
gives us all per-cell bounding boxes in one C-level pass, and
``skimage.measure.find_contours(mask == id, 0.5)`` traces the outline on
the tiny crop (typically 20×20 px) instead of the full 2950×2950 mask.
This drops a 2,000-cell core from O(N · W · H) to O(N · w · h) where
w·h is ~1% of the full image.

Why two polygons per cell, not one? The nucleus polygon localizes DAPI/
Ki67 signal, the expanded polygon localizes membrane markers (CD8, CD45,
…). The frontend renders one of them depending on the active ``mode``
parameter (§3.2.1 of the Workbench contract). Stored as separate columns rather
than a stacked long-format table because the join with cells.parquet is
already one-row-per-cell and we never want both polygons in the same
SQL-style filter.

Sparseness invariants from the spec (§3.1) are *enforced*, not
*hoped-for*:

* ID 0 = background.
* Nucleus IDs ⊆ expanded8 IDs. If a real core ever violates this we
  abort the build rather than emit a malformed bundle — silent
  divergence would corrupt the downstream join key with cells.parquet.
* Every skipped cell lands in ``polygons_skipped.csv`` with a reason.
  ``polygons_skipped.csv`` is the audit log; downstream code that joins
  ``cell_polygons.parquet`` with ``cells.parquet`` checks the missing
  IDs against the skip list and refuses to silently lose rows.

The output schema is pinned (§3.1):

    basename          : string
    cell_id           : int64
    centroid_y        : float64
    centroid_x        : float64
    nucleus_xy        : list<list<float64>>     closed, ≤ 32 pts
    expanded_xy       : list<list<float64>>     closed, ≤ 32 pts
    area_nucleus_px   : float64
    area_expanded_px  : float64
    n_verts_nucleus   : int32
    n_verts_expanded  : int32
    flags             : string
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Cells smaller than this (in nuclei mask) are dropped to ``polygons_skipped``.
# Below 4 px a "polygon" is a single pixel or a 2×2 block — the outline is
# meaningless and shapely's ``Polygon`` constructor needs ≥ 3 distinct points.
_MIN_AREA_PX: float = 4.0

# Polygon-simplification target. The frontend draws ≤ 32-vertex polygons in
# canvas with no perceptual loss for a 2950² core. The exponential ε climb
# (×1.5 per try, capped at 4 px) is the spec's contract from §3.1.
_MAX_VERTS: int = 32
_EPS_START: float = 0.5
_EPS_MAX: float = 4.0
_EPS_GROW: float = 1.5


class _SkipRow(NamedTuple):
    """One row of polygons_skipped.csv. Kept here so the writer is type-safe."""

    basename: str
    cell_id: int
    reason: str
    area_nucleus_px: float
    area_expanded_px: float


# ---------------------------------------------------------------------------
# Polygon extraction primitives
# ---------------------------------------------------------------------------


def _close_ring(coords: np.ndarray) -> np.ndarray:
    """Ensure the first and last vertex coincide (GeoJSON Polygon ring rule)."""
    if len(coords) == 0:
        return coords
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[:1]])
    return coords


def _largest_component_contour(
    mask_crop: np.ndarray, cell_id: int
) -> tuple[np.ndarray | None, bool]:
    """Return the (rows, cols) contour of the largest connected component.

    Returns (contour, had_multiple_components). ``contour`` coordinates
    are in the original crop's frame (the 1-pixel padding we add for
    ``find_contours`` is subtracted off before return). ``contour`` is
    None when no contour can be traced — that happens for pathological
    sub-pixel slivers.

    The dilation step that produced expanded8 can fragment a cell into
    multiple components in pathological cases (a tight U-shape, a cell
    bisected by a strong DAPI valley). We take the biggest blob and flag
    the row; the alternative — multi-polygon GeoJSON ``MultiPolygon`` —
    requires MultiPolygon rendering that the frontend does not support.
    """
    binary = mask_crop == cell_id
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return None, False
    if n_components > 1:
        # Pick the component with the largest pixel count. ``np.bincount``
        # of the labels (ignoring 0 = background) gives us areas in one pass.
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        binary = labeled == keep

    # Pad by 1 px of zeros so cells that fill their bbox still have a
    # 0→1 transition for ``find_contours`` to trace. The offset is
    # subtracted from the result so the caller stays in crop coordinates.
    padded = np.pad(binary.astype(np.float32), pad_width=1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, level=0.5)
    if not contours:
        return None, n_components > 1

    # find_contours may emit multiple polylines for the same component
    # (interior holes). Take the longest — it is the outer ring. We do not
    # represent holes in the current API contract.
    contour = max(contours, key=len)
    # Undo the padding offset (we shifted everything by (+1, +1)).
    contour = contour - 1.0
    return contour, n_components > 1


def _simplify_polygon(
    coords_xy: np.ndarray,
) -> tuple[np.ndarray, bool, bool]:
    """Run Douglas-Peucker with exponential ε climb until ≤ 32 vertices.

    Returns (simplified_xy, was_truncated, was_degenerate).

    ``was_truncated`` means we hit ε ≥ ``_EPS_MAX`` without fitting under
    the vertex cap; we emit at the largest ε ≤ ``_EPS_MAX`` and keep going
    so the frontend never sees a degenerate polygon. ``was_degenerate``
    means the simplification collapsed the polygon to < 3 unique vertices,
    in which case we fall back to the unsimplified contour.
    """
    # ``coords_xy`` is closed (last == first). shapely's Polygon constructor
    # also wants ≥ 3 points; with the closing vertex that means ≥ 4 rows.
    if len(coords_xy) < 4:
        return coords_xy, False, True

    eps = _EPS_START
    truncated = False
    # shapely's simplify needs a valid Polygon to start with. If the raw
    # ring is self-intersecting (rare in practice for cell outlines from
    # find_contours, but possible at border pixels), ``buffer(0)`` repairs
    # it without changing area for valid inputs.
    try:
        poly = Polygon(coords_xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                # Buffer(0) on a self-intersecting ring sometimes returns a
                # MultiPolygon; pick the largest piece.
                if poly.geom_type == "MultiPolygon" and not poly.is_empty:
                    poly = max(poly.geoms, key=lambda g: g.area)
                else:
                    return coords_xy, False, True
    except (ValueError, AttributeError) as e:
        # Why: shapely raises ``ValueError`` for malformed coordinate rings
        # and ``AttributeError`` when its lazy GEOS handle hasn't been
        # populated (rare, but documented). Either way we want to emit the
        # unsimplified contour rather than crash a 2,000-cell build on a
        # single sliver. Broader exception types (e.g. GEOS internal
        # ``TopologyException``) are NOT swallowed — they signal a real bug.
        logger.debug("shapely.Polygon raised %s; using raw contour", type(e).__name__)
        return coords_xy, False, True

    simplified = poly.simplify(eps, preserve_topology=True)
    while len(simplified.exterior.coords) - 1 > _MAX_VERTS and eps < _EPS_MAX:
        eps *= _EPS_GROW
        simplified = poly.simplify(min(eps, _EPS_MAX), preserve_topology=True)

    if len(simplified.exterior.coords) - 1 > _MAX_VERTS:
        # We hit the ε cap but still have too many vertices. Emit at ε_max
        # and flag — the API contract says this is "very convex/irregular" and is
        # rare enough that truncating the head of the list is acceptable.
        truncated = True
        out = np.array(simplified.exterior.coords)
        if len(out) - 1 > _MAX_VERTS:
            # Hard truncation: keep _MAX_VERTS vertices + the closing copy.
            out = np.vstack([out[:_MAX_VERTS], out[:1]])
        return _close_ring(out), truncated, False

    out = np.array(simplified.exterior.coords)
    # Degeneracy check: < 3 distinct vertices after closing.
    unique = np.unique(out, axis=0)
    if len(unique) < 3:
        return coords_xy, False, True
    return _close_ring(out), truncated, False


def _contour_to_xy(
    contour_rc: np.ndarray,
    row_origin: int,
    col_origin: int,
    image_h: int,
    image_w: int,
    *,
    on_border: list[bool] | None = None,
) -> np.ndarray:
    """Convert a (row, col) contour on a crop to global (x, y) image coords.

    Clamps vertices that protrude past the image extent — find_contours can
    place vertices at row = -0.5 when the cell touches the crop edge.
    ``on_border`` is filled in-place with True if any clamping happened.
    """
    rows = contour_rc[:, 0] + row_origin
    cols = contour_rc[:, 1] + col_origin
    border_hit = False
    if rows.min() < 0 or cols.min() < 0 or rows.max() > image_h - 1 or cols.max() > image_w - 1:
        border_hit = True
        rows = np.clip(rows, 0.0, float(image_h - 1))
        cols = np.clip(cols, 0.0, float(image_w - 1))
    if on_border is not None:
        on_border.append(border_hit)
    return np.column_stack([cols, rows])  # (x, y) order


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------


def _process_one_cell(
    cell_id: int,
    nuclei_mask: np.ndarray,
    expanded_mask: np.ndarray,
    nuclei_slice: tuple[slice, slice],
    expanded_slice: tuple[slice, slice],
) -> tuple[dict, list[str]] | tuple[None, list[str]]:
    """Compute both polygons for a single cell.

    Returns ``(row_dict, flags)`` on success or ``(None, [reason])`` if
    the cell should be sent to the skip log. ``flags`` is the
    comma-joined-later list that lands in the row's ``flags`` column.
    """
    h, w = nuclei_mask.shape
    flags: list[str] = []

    # Nucleus area first — tiny-cell rejection happens before we waste
    # work on the expanded contour.
    nu_crop = nuclei_mask[nuclei_slice]
    nu_binary = nu_crop == cell_id
    area_nucleus = float(nu_binary.sum())
    if area_nucleus <= _MIN_AREA_PX:
        return None, ["tiny"]

    nu_contour_rc, nu_multi = _largest_component_contour(nu_crop, cell_id)
    if nu_contour_rc is None:
        return None, ["empty"]
    if nu_multi:
        # Compartment-level detail (nucleus vs expanded) lives in the
        # logger / skip log only — the row flag stays in the pinned set
        # {degenerate, truncated, border, empty}. We dedup multi_component
        # below before the row's flag string is assembled.
        flags.append("multi_component")
        logger.debug("%s cell_id=%d: multi_component split in nucleus mask", "nucleus", cell_id)

    border_hits: list[bool] = []
    nu_xy = _contour_to_xy(
        nu_contour_rc,
        nuclei_slice[0].start,
        nuclei_slice[1].start,
        h,
        w,
        on_border=border_hits,
    )
    nu_xy = _close_ring(nu_xy)
    nu_xy_simp, nu_trunc, nu_degen = _simplify_polygon(nu_xy)
    if nu_degen:
        flags.append("degenerate")
    if nu_trunc:
        flags.append("truncated")

    # Centroid: image-moment of the unsimplified nucleus binary on the
    # crop. We use the binary, not the polygon, so the centroid is
    # identical to what cells.parquet sees (which derives from the mask).
    ys, xs = np.nonzero(nu_binary)
    if len(xs) == 0:
        return None, ["empty"]
    centroid_x = float(xs.mean() + nuclei_slice[1].start)
    centroid_y = float(ys.mean() + nuclei_slice[0].start)

    # Expanded compartment.
    ex_crop = expanded_mask[expanded_slice]
    ex_binary = ex_crop == cell_id
    area_expanded = float(ex_binary.sum())
    if area_expanded <= 0:
        # Spec §3.1: nucleus IDs ⊆ expanded8 IDs is a *core-level* invariant
        # asserted before we get here. If we reach this branch for an
        # individual cell, the mask file disagrees with the ID set we
        # extracted from it — that is a build-time bug, not a data issue.
        # Same semantic class as the global invariant in _assert_id_invariant
        # (ValueError) so callers can catch one exception type for both.
        raise ValueError(
            f"cell_id={cell_id} present in nuclei mask but missing from expanded "
            f"mask crop. Invariant violation: rebuild the masks_v1 outputs."
        )

    ex_contour_rc, ex_multi = _largest_component_contour(ex_crop, cell_id)
    if ex_contour_rc is None:
        return None, ["empty"]
    if ex_multi and "multi_component" not in flags:
        # Same single token regardless of compartment — see the nucleus-side
        # comment above. Compartment info goes to the debug log only.
        flags.append("multi_component")
        logger.debug("%s cell_id=%d: multi_component split in expanded mask", "expanded", cell_id)

    ex_xy = _contour_to_xy(
        ex_contour_rc,
        expanded_slice[0].start,
        expanded_slice[1].start,
        h,
        w,
        on_border=border_hits,
    )
    ex_xy = _close_ring(ex_xy)
    ex_xy_simp, ex_trunc, ex_degen = _simplify_polygon(ex_xy)
    if ex_degen and "degenerate" not in flags:
        flags.append("degenerate")
    if ex_trunc and "truncated" not in flags:
        flags.append("truncated")

    if any(border_hits):
        flags.append("border")

    return {
        "cell_id": int(cell_id),
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "nucleus_xy": nu_xy_simp.tolist(),
        "expanded_xy": ex_xy_simp.tolist(),
        "area_nucleus_px": area_nucleus,
        "area_expanded_px": area_expanded,
        "n_verts_nucleus": int(len(nu_xy_simp) - 1),
        "n_verts_expanded": int(len(ex_xy_simp) - 1),
    }, flags


# ---------------------------------------------------------------------------
# Per-core driver
# ---------------------------------------------------------------------------


def _assert_id_invariant(
    basename: str,
    nuclei_mask: np.ndarray,
    expanded_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify nuclei IDs ⊆ expanded8 IDs; return both ID arrays in one pass.

    Hard abort on violation — silent skipping would corrupt the
    cells.parquet ↔ cell_polygons.parquet join.
    """
    nu_ids = np.unique(nuclei_mask)
    nu_ids = nu_ids[nu_ids != 0]
    ex_ids = np.unique(expanded_mask)
    ex_ids = ex_ids[ex_ids != 0]
    missing = np.setdiff1d(nu_ids, ex_ids, assume_unique=True)
    if missing.size > 0:
        raise ValueError(
            f"[{basename}] mask invariant violated: {missing.size} nuclei IDs are "
            f"absent from the expanded8 mask (e.g., {missing[:5].tolist()}). "
            f"Rebuild masks_v1 or report a bug; do not silently emit a bundle."
        )
    return nu_ids, ex_ids


def _load_pair(masks_dir: Path, basename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load nuclei + expanded8 masks for a single core, with explicit errors.

    Memory-mapped for the full 2950×2950 uint32 arrays so a bundle build
    with 56 cores does not balloon RSS unnecessarily.
    """
    nu_path = masks_dir / f"{basename}_nuclei.npy"
    ex_path = masks_dir / f"{basename}_expanded8.npy"
    if not nu_path.exists():
        raise FileNotFoundError(f"missing nuclei mask: {nu_path}")
    if not ex_path.exists():
        raise FileNotFoundError(f"missing expanded8 mask: {ex_path}")
    nu = np.load(nu_path)
    ex = np.load(ex_path)
    if nu.shape != ex.shape:
        raise ValueError(
            f"[{basename}] nuclei shape {nu.shape} != expanded8 shape {ex.shape}; "
            f"rebuild the masks_v1 outputs."
        )
    return nu, ex


def _build_for_core(
    basename: str,
    masks_dir: Path,
    expected_cell_ids: np.ndarray | None,
) -> tuple[list[dict], list[_SkipRow]]:
    """Extract all polygons for one core. Returns (rows, skip_rows)."""
    logger.info("polygons: %s", basename)
    nuclei_mask, expanded_mask = _load_pair(masks_dir, basename)
    nu_ids, _ = _assert_id_invariant(basename, nuclei_mask, expanded_mask)

    # ``find_objects`` returns a list indexed by label, length = max ID.
    # Bbox tuples are (slice(rows), slice(cols)) or None for absent IDs.
    nu_slices = ndimage.find_objects(nuclei_mask)
    ex_slices = ndimage.find_objects(expanded_mask)
    max_id = max(len(nu_slices), len(ex_slices))

    # Iterate over the union of (expected IDs from cells.parquet, IDs in
    # the mask). The expected set is the contract: every cell in
    # cells.parquet must either land in cell_polygons.parquet or in
    # polygons_skipped.csv with a reason.
    iter_ids: set[int]
    if expected_cell_ids is None:
        iter_ids = set(nu_ids.tolist())
    else:
        iter_ids = set(int(c) for c in expected_cell_ids)
        # Sanity: every expected ID is in the mask. If not, log skip
        # rows; we never silently drop.
        missing = iter_ids - set(nu_ids.tolist())
        if missing:
            logger.warning(
                "%s: %d cell_ids in cells.parquet are absent from the nuclei mask",
                basename,
                len(missing),
            )

    rows: list[dict] = []
    skips: list[_SkipRow] = []
    for cid_int in sorted(iter_ids):
        if cid_int < 1 or cid_int > max_id:
            skips.append(_SkipRow(basename, cid_int, "out_of_mask", 0.0, 0.0))
            continue
        nu_slice = nu_slices[cid_int - 1] if cid_int - 1 < len(nu_slices) else None
        ex_slice = ex_slices[cid_int - 1] if cid_int - 1 < len(ex_slices) else None
        if nu_slice is None or ex_slice is None:
            skips.append(_SkipRow(basename, cid_int, "out_of_mask", 0.0, 0.0))
            continue

        result, flag_tokens = _process_one_cell(
            cid_int, nuclei_mask, expanded_mask, nu_slice, ex_slice
        )
        if result is None:
            reason = flag_tokens[0] if flag_tokens else "unknown"
            # Compute approximate areas for the skip log audit trail.
            area_nu = float((nuclei_mask[nu_slice] == cid_int).sum())
            area_ex = float((expanded_mask[ex_slice] == cid_int).sum())
            skips.append(_SkipRow(basename, cid_int, reason, area_nu, area_ex))
            continue

        result["basename"] = basename
        # Sort lexicographically so the flag column is deterministic given
        # identical input. dedup also: a flag should never appear twice in
        # the same row.
        result["flags"] = ",".join(sorted(set(flag_tokens))) if flag_tokens else ""
        rows.append(result)

    logger.info(
        "%s: kept %d polygons, skipped %d (reasons: %s)",
        basename,
        len(rows),
        len(skips),
        ",".join(sorted({s.reason for s in skips})) or "none",
    )
    return rows, skips


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("basename", pa.string(), nullable=False),
        pa.field("cell_id", pa.int64(), nullable=False),
        pa.field("centroid_y", pa.float64(), nullable=False),
        pa.field("centroid_x", pa.float64(), nullable=False),
        pa.field("nucleus_xy", pa.list_(pa.list_(pa.float64())), nullable=False),
        pa.field("expanded_xy", pa.list_(pa.list_(pa.float64())), nullable=False),
        pa.field("area_nucleus_px", pa.float64(), nullable=False),
        pa.field("area_expanded_px", pa.float64(), nullable=False),
        pa.field("n_verts_nucleus", pa.int32(), nullable=False),
        pa.field("n_verts_expanded", pa.int32(), nullable=False),
        pa.field("flags", pa.string(), nullable=False),
    ]
)


def build_cell_polygons(
    masks_dir: Path,
    cells_df: pd.DataFrame,
    output_path: Path,
    *,
    basenames: Sequence[str] | None = None,
    skipped_csv: Path | None = None,
) -> pd.DataFrame:
    """Build cell_polygons.parquet for every core in ``cells_df``.

    The bundle build calls this after cells.parquet is written. We iterate
    in cells.parquet order so the result is reproducible: same masks +
    same cells_df ⇒ byte-identical parquet (modulo pyarrow version).

    Parameters
    ----------
    masks_dir:
        Directory containing ``<basename>_nuclei.npy`` and
        ``<basename>_expanded8.npy`` for every core in the bundle.
    cells_df:
        The ``cells.parquet`` DataFrame; we only need ``basename`` and
        ``cell_id`` from it. The polygon row order matches the cells.parquet
        order *per core* so client-side joins do not need to sort.
    output_path:
        Destination ``cell_polygons.parquet``. Parent directory created
        if missing.
    basenames:
        Optional explicit core list. Useful for ``rebuild-polygons`` against
        a subset of an existing bundle, or for testing. Defaults to the
        set of unique basenames in ``cells_df``.
    skipped_csv:
        Where to write the skip log. Defaults to a sibling of
        ``output_path`` named ``polygons_skipped.csv``.

    Returns
    -------
    DataFrame with the same rows as the written parquet (caller may
    inspect it without re-reading the file).
    """
    masks_dir = Path(masks_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skipped_csv is None:
        skipped_csv = output_path.parent / "polygons_skipped.csv"
    else:
        skipped_csv = Path(skipped_csv)

    if not {"basename", "cell_id"}.issubset(cells_df.columns):
        raise ValueError(
            "cells_df must have columns ['basename', 'cell_id']; got "
            f"{cells_df.columns.tolist()[:8]}..."
        )

    if basenames is None:
        # ``unique`` preserves first-appearance order, which matches cells.parquet.
        basenames = pd.Index(cells_df["basename"]).unique().tolist()

    all_rows: list[dict] = []
    all_skips: list[_SkipRow] = []
    for base in basenames:
        sub = cells_df[cells_df["basename"] == base]
        expected = sub["cell_id"].to_numpy()
        rows, skips = _build_for_core(base, masks_dir, expected)
        all_rows.extend(rows)
        all_skips.extend(skips)

    out_df = pd.DataFrame(all_rows)
    if out_df.empty:
        # Empty bundles are legitimate during testing; we still write the
        # parquet with the correct schema so downstream readers don't crash.
        out_df = pd.DataFrame(
            {
                "basename": pd.Series(dtype="string"),
                "cell_id": pd.Series(dtype="int64"),
                "centroid_y": pd.Series(dtype="float64"),
                "centroid_x": pd.Series(dtype="float64"),
                "nucleus_xy": pd.Series(dtype="object"),
                "expanded_xy": pd.Series(dtype="object"),
                "area_nucleus_px": pd.Series(dtype="float64"),
                "area_expanded_px": pd.Series(dtype="float64"),
                "n_verts_nucleus": pd.Series(dtype="int32"),
                "n_verts_expanded": pd.Series(dtype="int32"),
                "flags": pd.Series(dtype="string"),
            }
        )
    else:
        out_df = out_df[
            [
                "basename",
                "cell_id",
                "centroid_y",
                "centroid_x",
                "nucleus_xy",
                "expanded_xy",
                "area_nucleus_px",
                "area_expanded_px",
                "n_verts_nucleus",
                "n_verts_expanded",
                "flags",
            ]
        ].copy()
        # Pin dtypes — pyarrow's schema enforcement below would catch mismatches
        # but failing here gives a clearer error message.
        out_df["cell_id"] = out_df["cell_id"].astype("int64")
        out_df["centroid_y"] = out_df["centroid_y"].astype("float64")
        out_df["centroid_x"] = out_df["centroid_x"].astype("float64")
        out_df["area_nucleus_px"] = out_df["area_nucleus_px"].astype("float64")
        out_df["area_expanded_px"] = out_df["area_expanded_px"].astype("float64")
        out_df["n_verts_nucleus"] = out_df["n_verts_nucleus"].astype("int32")
        out_df["n_verts_expanded"] = out_df["n_verts_expanded"].astype("int32")
        out_df["flags"] = out_df["flags"].fillna("").astype("string")
        out_df["basename"] = out_df["basename"].astype("string")

    table = pa.Table.from_pandas(out_df, schema=_PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)
    logger.info("wrote %s (%d polygons)", output_path, len(out_df))

    # Skip log: always write the file (even if empty) so its presence is a
    # provenance signal — "this bundle has run the polygon pipeline".
    skip_df = pd.DataFrame(
        all_skips,
        columns=["basename", "cell_id", "reason", "area_nucleus_px", "area_expanded_px"],
    )
    skip_df.to_csv(skipped_csv, index=False)
    logger.info("wrote %s (%d skipped cells)", skipped_csv, len(skip_df))

    return out_df


__all__ = ["build_cell_polygons"]
