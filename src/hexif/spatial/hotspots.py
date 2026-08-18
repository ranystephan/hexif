"""Hotspot detection per phenotype (simple Getis-Ord-style local statistic).

For each phenotype, we compute a per-cell local-frequency statistic over
its kNN neighborhood, then z-score against the core mean.  Cells with z
> 1.96 are flagged as belonging to a "hotspot" for that phenotype.

This local-frequency z-score is intentionally not described as Getis-Ord G*:
it uses neither a spatial weight matrix nor the Getis-Ord null model.

Output: per (basename, phenotype) a Jaccard-style hotspot overlap with
another phenotype's hotspots, useful for comparing tumor/immune
co-clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_cell_local_phenotype_count(
    edges: pd.DataFrame,
    phenotype: str,
) -> pd.DataFrame:
    """For each (basename, src_cell), count out-neighbors that have label==phenotype."""
    if edges.empty:
        return pd.DataFrame(columns=["basename", "src_cell_id", "n_neighbors", "n_pheno"])
    e = edges[edges["dst_phenotype"].astype(str) == phenotype]
    if e.empty:
        # still want a row per (basename, src_cell) with zero counts
        all_src = (
            edges.groupby(["basename", "src_cell_id"]).size().rename("n_neighbors").reset_index()
        )
        all_src["n_pheno"] = 0
        return all_src
    n_pheno = e.groupby(["basename", "src_cell_id"]).size().rename("n_pheno").reset_index()
    n_neighbors = (
        edges.groupby(["basename", "src_cell_id"]).size().rename("n_neighbors").reset_index()
    )
    out = n_neighbors.merge(n_pheno, on=["basename", "src_cell_id"], how="left")
    out["n_pheno"] = out["n_pheno"].fillna(0).astype(int)
    return out


def call_hotspots(
    edges: pd.DataFrame,
    phenotype: str,
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """Flag cells whose local phenotype frequency is significantly above the
    core mean.  Returns DataFrame with (basename, src_cell_id, local_frac, z,
    is_hotspot).
    """
    counts = per_cell_local_phenotype_count(edges, phenotype)
    if counts.empty:
        return counts.assign(local_frac=np.nan, z=np.nan, is_hotspot=False)
    counts["local_frac"] = counts["n_pheno"] / counts["n_neighbors"].clip(lower=1)

    # z within each basename
    def _z(group):
        mu, sd = group["local_frac"].mean(), group["local_frac"].std(ddof=0)
        group["z"] = (group["local_frac"] - mu) / (sd + 1e-9)
        return group

    counts = counts.groupby("basename", group_keys=False).apply(_z)
    counts["is_hotspot"] = counts["z"] >= z_threshold
    return counts


def hotspot_overlap_iou(
    edges: pd.DataFrame,
    phenotype_a: str,
    phenotype_b: str,
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """Per-basename Jaccard overlap between phenotype_a's and phenotype_b's
    hotspot cell sets.

    Returns DataFrame: basename, hotspot_overlap_iou.
    """
    if phenotype_a == phenotype_b:
        # trivial: IoU = 1 by definition
        bns = edges["basename"].unique() if not edges.empty else []
        return pd.DataFrame({"basename": bns, "hotspot_overlap_iou": 1.0})
    ha = call_hotspots(edges, phenotype_a, z_threshold)
    hb = call_hotspots(edges, phenotype_b, z_threshold)
    if ha.empty or hb.empty:
        return pd.DataFrame(columns=["basename", "hotspot_overlap_iou"])
    rows = []
    for basename in ha["basename"].unique():
        sa = set(map(int, ha[(ha.basename == basename) & ha.is_hotspot]["src_cell_id"]))
        sb = set(map(int, hb[(hb.basename == basename) & hb.is_hotspot]["src_cell_id"]))
        if not sa and not sb:
            iou = float("nan")
        else:
            inter = len(sa & sb)
            uni = len(sa | sb)
            iou = inter / max(uni, 1)
        rows.append({"basename": basename, "hotspot_overlap_iou": float(iou)})
    return pd.DataFrame(rows)
