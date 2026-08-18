"""Postprocess v1.1 cell predictions into the canonical bundle schema.

Functions here are intentionally pure: they take a wide-format
``cell_predictions.csv`` (per
:mod:`hexif.benchmark.cell_predictions`) plus the calibrated thresholds
from :mod:`hexif.pipeline.thresholds` and emit the
``cells.parquet`` rows used by the workbench bundle.

Anything that needs disk access (model loading, mask reading) lives in
``hexif.pipeline.inference_core`` / ``hexif.pipeline.bundle`` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from hexif.cell_phenotype import (
    FOCUSED_MARKERS,
    PHENOTYPE_NAMES,
)
from hexif.pipeline.thresholds import CalibratedThresholds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phenotype call assignment
# ---------------------------------------------------------------------------


def assign_phenotype_calls(
    df: pd.DataFrame,
    thresholds: CalibratedThresholds,
    model_id: str = "v1_1",
) -> pd.DataFrame:
    """Binarize predicted phenotype scores into calls using calibrated thresholds.

    Adds ``phenotype_{name}_call_{model_id}`` columns (int 0/1) for each
    phenotype.  Skips silently if a phenotype's score column isn't in ``df``.
    """
    out = df.copy()
    for name in PHENOTYPE_NAMES:
        score_col = f"phenotype_{name}_score_{model_id}"
        if score_col not in out.columns:
            # Backward-compat: bare phenotype_<name>_score (no model suffix)
            # was the convention before the multi-model schema
            bare = f"phenotype_{name}_score"
            if bare in out.columns and model_id == "v1_1":
                score_col = bare
            else:
                continue
        thr = thresholds.phenotype_threshold(name)
        out[f"phenotype_{name}_call_{model_id}"] = (
            out[score_col].to_numpy(dtype=np.float32) >= thr
        ).astype(np.int8)
    return out


def assign_marker_calls(
    df: pd.DataFrame,
    thresholds: CalibratedThresholds,
    model_id: str = "v1_1",
) -> pd.DataFrame:
    """Binarize predicted marker probabilities into 0/1 calls."""
    out = df.copy()
    for ch in FOCUSED_MARKERS:
        pred_col = f"ch{ch:02d}_pred_{model_id}"
        if pred_col not in out.columns:
            bare = f"ch{ch:02d}_pred"
            if bare in out.columns and model_id == "v1":
                pred_col = bare
            else:
                continue
        thr = thresholds.marker_threshold(int(ch))
        out[f"ch{ch:02d}_call_{model_id}"] = (
            out[pred_col].to_numpy(dtype=np.float32) >= thr
        ).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Per-core composition stats (one row per (basename, model_id))
# ---------------------------------------------------------------------------


def per_core_composition(
    df: pd.DataFrame,
    *,
    model_ids: Iterable[str] = ("v1_1",),
) -> pd.DataFrame:
    """Aggregate cells.parquet rows into per-core composition stats.

    Returns one row per ``basename`` with:
      n_cells, frac_chXX_pos_<model_id> for each model, frac_chXX_pos for
      truth (if available), and frac_phenotype_<name>_call_<model_id> /
      frac_phenotype_<name>_label.
    """
    rows: list[dict] = []
    for basename, grp in df.groupby("basename", sort=False):
        row: dict = {"basename": str(basename), "n_cells": len(grp)}
        # Truth columns
        for ch in FOCUSED_MARKERS:
            col = f"ch{ch:02d}_pos"
            if col in grp.columns:
                row[f"frac_ch{ch:02d}_pos_truth"] = float(np.nanmean(grp[col]))
        for name in PHENOTYPE_NAMES:
            col = f"phenotype_{name}_label"
            if col in grp.columns:
                row[f"frac_phenotype_{name}_truth"] = float(np.nanmean(grp[col]))
        # Per-model calls
        for mid in model_ids:
            for ch in FOCUSED_MARKERS:
                call = f"ch{ch:02d}_call_{mid}"
                pred = f"ch{ch:02d}_pred_{mid}"
                if call in grp.columns:
                    row[f"frac_ch{ch:02d}_pos_{mid}"] = float(np.nanmean(grp[call]))
                if pred in grp.columns:
                    row[f"mean_ch{ch:02d}_pred_{mid}"] = float(np.nanmean(grp[pred]))
            for name in PHENOTYPE_NAMES:
                call = f"phenotype_{name}_call_{mid}"
                if call in grp.columns:
                    row[f"frac_phenotype_{name}_call_{mid}"] = float(np.nanmean(grp[call]))
        rows.append(row)
    return pd.DataFrame(rows)


def attach_tma_tissue(df: pd.DataFrame) -> pd.DataFrame:
    """Add `tma` and `tissue` columns from `basename` (TMA__CORE convention)."""
    out = df.copy()
    if "tma" not in out.columns:
        out["tma"] = out["basename"].astype(str).str.split("__", n=1, expand=True)[0]
    if "tissue" not in out.columns:
        out["tissue"] = out["tma"].astype(str).str.split("_", n=1, expand=True)[0]
    return out
