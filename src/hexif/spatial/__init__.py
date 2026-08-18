"""HEXIF spatial-architecture toolkit.

Public entry point: :func:`build_spatial_summary` reads ``cells.parquet``
from a bundle, builds kNN edges per core, computes COZI directional
z-scores + observed/expected baseline + Jaccard hotspot overlap, writes
``spatial_edges.parquet`` (optional) + ``spatial_summary.parquet`` back
into the bundle directory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hexif.cell_phenotype import PHENOTYPE_NAMES
from hexif.spatial.cozi import compute_cozi_all_cores
from hexif.spatial.enrichment import observed_over_expected
from hexif.spatial.hotspots import hotspot_overlap_iou
from hexif.spatial.neighborhoods import (
    NeighborhoodGraph,
    build_neighborhoods,
    build_neighborhoods_all_cores,
)

__all__ = [
    "NeighborhoodGraph",
    "build_neighborhoods",
    "build_neighborhoods_all_cores",
    "build_spatial_summary",
    "compute_cozi_all_cores",
    "hotspot_overlap_iou",
    "observed_over_expected",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cell "dominant phenotype" assignment for the spatial graph
# ---------------------------------------------------------------------------


def assign_dominant_phenotype(
    cells: pd.DataFrame,
    model_id: str = "v1_1",
    phenotype_names: Sequence[str] = PHENOTYPE_NAMES,
    background_label: str = "background",
) -> pd.DataFrame:
    """Pick a single phenotype label per cell based on highest score.

    Spatial statistics need a categorical label, but our phenotype model
    is multi-label (a cell can be both "immune" and "t-cell").  For the
    spatial graph we collapse to the highest-score phenotype above its
    calibrated threshold (the ``_call_`` column).  Cells with no call get
    ``background_label``.
    """
    out = cells.copy()
    call_cols = [f"phenotype_{n}_call_{model_id}" for n in phenotype_names]
    # For v1.1 the score columns may be bare (no model suffix) per the
    # legacy convention in eval_cell_phenotype_v1_1.py; fall back to that
    # when the suffixed form isn't present.
    score_cols = []
    for n in phenotype_names:
        suffixed = f"phenotype_{n}_score_{model_id}"
        bare = f"phenotype_{n}_score"
        if suffixed in out.columns:
            score_cols.append(suffixed)
        elif model_id == "v1_1" and bare in out.columns:
            score_cols.append(bare)
        # else: phenotype missing for this model — skip in scoring
    present_call = [c for c in call_cols if c in out.columns]
    present_score = score_cols
    if not present_score:
        logger.warning("no phenotype score columns for model_id=%s; cannot assign", model_id)
        out["dominant_phenotype"] = background_label
        return out
    scores = out[present_score].to_numpy(dtype=np.float32)

    def _short(col: str) -> str:
        # Strip "phenotype_" prefix and either "_score_<mid>" or "_score" / "_call_<mid>" or "_call"
        s = col.removeprefix("phenotype_")
        for tail in (f"_score_{model_id}", f"_call_{model_id}", "_score", "_call"):
            if s.endswith(tail):
                return s[: -len(tail)]
        return s

    if present_call:
        called = out[present_call].to_numpy(dtype=bool)
        call_names = [_short(c) for c in present_call]
        score_names = [_short(c) for c in present_score]
        mask = np.zeros_like(scores, dtype=bool)
        for i, n in enumerate(score_names):
            if n in call_names:
                j = call_names.index(n)
                mask[:, i] = called[:, j]
        scores_masked = np.where(mask, scores, -np.inf)
    else:
        scores_masked = scores
    has_call = (scores_masked > -np.inf).any(axis=1)
    arg = scores_masked.argmax(axis=1)
    score_short = [_short(c) for c in present_score]
    labels = np.where(
        has_call,
        np.array(score_short)[arg],
        background_label,
    )
    out["dominant_phenotype"] = labels
    return out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_spatial_summary(
    bundle_dir: str | Path,
    *,
    method: str = "knn",
    k: int = 16,
    radius_px: float = 98.0,
    write_edges: bool = False,
    model_id: str = "v1_1",
) -> tuple[Path | None, Path]:
    """Compute spatial-architecture summaries for every core in the bundle."""
    bundle_dir = Path(bundle_dir)
    cells_path = bundle_dir / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"no cells.parquet in {bundle_dir}")
    cells = pd.read_parquet(cells_path)

    cells = assign_dominant_phenotype(cells, model_id=model_id)
    logger.info(
        "loaded %d cells from %s; %d phenotypes (incl background)",
        len(cells),
        cells_path,
        cells["dominant_phenotype"].nunique(),
    )

    edges, metas = build_neighborhoods_all_cores(
        cells,
        method=method,
        k=k,
        radius_px=radius_px,
        phenotype_col="dominant_phenotype",
    )
    logger.info("graphs: %d edges total across %d cores", len(edges), len(metas))

    edges_path = None
    if write_edges and not edges.empty:
        edges_path = bundle_dir / "spatial_edges.parquet"
        edges.to_parquet(edges_path, index=False)
        logger.info("wrote %s", edges_path)

    cozi = compute_cozi_all_cores(cells, edges, phenotype_col="dominant_phenotype")
    ratio = observed_over_expected(cells, edges, phenotype_col="dominant_phenotype")
    if not cozi.empty and not ratio.empty:
        cozi = cozi.merge(
            ratio[["basename", "phenotype_a", "phenotype_b", "obs_over_exp"]],
            on=["basename", "phenotype_a", "phenotype_b"],
            how="left",
        )

    if not edges.empty:
        if "src_phenotype" not in edges.columns:
            ph = cells.set_index(["basename", "cell_id"])["dominant_phenotype"].astype(str)
            edges = edges.copy()
            edges["src_phenotype"] = ph.reindex(
                list(zip(edges["basename"], edges["src_cell_id"], strict=False))
            ).to_numpy()
            edges["dst_phenotype"] = ph.reindex(
                list(zip(edges["basename"], edges["dst_cell_id"], strict=False))
            ).to_numpy()
        pairs = (
            cozi[["phenotype_a", "phenotype_b"]].drop_duplicates()
            if not cozi.empty
            else pd.DataFrame()
        )
        iou_parts: list[pd.DataFrame] = []
        for _, r in pairs.iterrows():
            a, b = str(r["phenotype_a"]), str(r["phenotype_b"])
            iou = hotspot_overlap_iou(edges, a, b).assign(phenotype_a=a, phenotype_b=b)
            iou_parts.append(iou)
        if iou_parts:
            iou_all = pd.concat(iou_parts, ignore_index=True)
            cozi = cozi.merge(iou_all, on=["basename", "phenotype_a", "phenotype_b"], how="left")

    summary_path = bundle_dir / "spatial_summary.parquet"
    if cozi.empty:
        cozi = pd.DataFrame(
            columns=[
                "basename",
                "phenotype_a",
                "phenotype_b",
                "observed",
                "expected",
                "variance",
                "z",
                "p_two_sided",
                "obs_over_exp",
                "hotspot_overlap_iou",
                "n_a",
                "n_b",
                "n_total",
                "n_edges",
            ]
        )
    cozi.to_parquet(summary_path, index=False)
    logger.info("wrote %s (%d rows)", summary_path, len(cozi))
    return edges_path, summary_path
