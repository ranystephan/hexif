"""Spatial neighborhood graph construction from cell centroids.

All functions are pure NumPy / SciPy.  No I/O.  Used by the spatial
statistics module + the COZI implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class NeighborhoodGraph:
    """Sparse edge list over cells of a single core.

    ``edges`` columns: src_cell_id, dst_cell_id, src_phenotype, dst_phenotype,
    distance_px, edge_type ("knn"|"radius").

    ``meta`` carries the construction parameters so spatial stats can sanity-
    check inputs (e.g. radius_px, k).
    """

    basename: str
    edges: pd.DataFrame
    meta: dict


def build_neighborhoods(
    cells: pd.DataFrame,
    *,
    method: str = "knn",
    k: int = 16,
    radius_px: float = 98.0,  # ~50 µm at MPP 0.51
    phenotype_col: str | None = None,
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> NeighborhoodGraph:
    """Build a kNN or radius graph over the cells of a single core.

    Parameters
    ----------
    cells
        DataFrame for ONE core (single basename).  Must contain ``cell_id``,
        ``centroid_x``, ``centroid_y``.
    method
        ``"knn"`` for k nearest (excluding self), or ``"radius"`` for all
        cells within radius_px (excluding self).
    k
        Number of neighbors (knn only).
    radius_px
        Search radius in HE pixels (radius only).
    phenotype_col
        Optional column to record the source/destination phenotype label on
        each edge.  Pass e.g. ``"phenotype_immune_cd45_call_v1_1"``.

    Returns
    -------
    NeighborhoodGraph with the edges DataFrame.
    """
    if cells["basename"].nunique() > 1:
        raise ValueError("build_neighborhoods expects cells of a single core")
    if len(cells) < 2:
        return NeighborhoodGraph(
            basename=str(cells["basename"].iloc[0]) if len(cells) else "",
            edges=pd.DataFrame(
                columns=[
                    "src_cell_id",
                    "dst_cell_id",
                    "src_phenotype",
                    "dst_phenotype",
                    "distance_px",
                    "edge_type",
                ]
            ),
            meta={"method": method, "k": k, "radius_px": radius_px, "n_cells": len(cells)},
        )

    basename = str(cells["basename"].iloc[0])
    cell_ids = cells[cell_id_col].to_numpy()
    xy = cells[[x_col, y_col]].to_numpy(dtype=np.float32)
    tree = cKDTree(xy)
    phenotypes = (
        cells[phenotype_col].astype(str).to_numpy()
        if phenotype_col and phenotype_col in cells.columns
        else np.array(["_"] * len(cells), dtype=object)
    )

    edges: list[dict] = []
    if method == "knn":
        # +1 to skip self; k+1 neighbors then drop the first column (self at d=0)
        k_eff = min(k + 1, len(cells))
        d, idx = tree.query(xy, k=k_eff)
        if k_eff == 1:
            d = d[:, None]
            idx = idx[:, None]
        for i in range(len(cells)):
            for j_pos in range(1, k_eff):  # skip self
                j = int(idx[i, j_pos])
                edges.append(
                    {
                        "src_cell_id": int(cell_ids[i]),
                        "dst_cell_id": int(cell_ids[j]),
                        "src_phenotype": str(phenotypes[i]),
                        "dst_phenotype": str(phenotypes[j]),
                        "distance_px": float(d[i, j_pos]),
                        "edge_type": "knn",
                    }
                )
    elif method == "radius":
        ix_pairs = tree.query_pairs(r=radius_px, output_type="ndarray")
        # query_pairs is symmetric (i<j); we add both directions for the
        # edge table so per-source statistics see neighbors of every cell
        for a, b in ix_pairs:
            di = float(np.linalg.norm(xy[a] - xy[b]))
            edges.append(
                {
                    "src_cell_id": int(cell_ids[a]),
                    "dst_cell_id": int(cell_ids[b]),
                    "src_phenotype": str(phenotypes[a]),
                    "dst_phenotype": str(phenotypes[b]),
                    "distance_px": di,
                    "edge_type": "radius",
                }
            )
            edges.append(
                {
                    "src_cell_id": int(cell_ids[b]),
                    "dst_cell_id": int(cell_ids[a]),
                    "src_phenotype": str(phenotypes[b]),
                    "dst_phenotype": str(phenotypes[a]),
                    "distance_px": di,
                    "edge_type": "radius",
                }
            )
    else:
        raise ValueError(f"unknown method {method!r}")

    edges_df = pd.DataFrame(edges)
    if not edges_df.empty:
        edges_df.insert(0, "basename", basename)
    meta = {
        "method": method,
        "k": int(k),
        "radius_px": float(radius_px),
        "n_cells": len(cells),
        "n_edges": len(edges_df),
    }
    return NeighborhoodGraph(basename=basename, edges=edges_df, meta=meta)


def build_neighborhoods_all_cores(
    cells: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run build_neighborhoods per core; concatenate edges, collect metas."""
    parts: list[pd.DataFrame] = []
    metas: dict[str, dict] = {}
    for basename, grp in cells.groupby("basename", sort=False):
        g = build_neighborhoods(grp, **kwargs)
        if not g.edges.empty:
            parts.append(g.edges)
        metas[basename] = g.meta
    if not parts:
        return (
            pd.DataFrame(
                columns=[
                    "basename",
                    "src_cell_id",
                    "dst_cell_id",
                    "src_phenotype",
                    "dst_phenotype",
                    "distance_px",
                    "edge_type",
                ]
            ),
            metas,
        )
    return pd.concat(parts, ignore_index=True), metas
