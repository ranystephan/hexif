"""Unit tests for hexif.spatial.cozi.

Strategy: build a tiny core with two phenotypes that are deliberately
clustered (A in upper half, B in lower half).  Build kNN graph.
Expect strong positive COZI z for (A,A) and (B,B) — same-phenotype
clustering — and strong negative z for (A,B) / (B,A) — avoidance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hexif.spatial.cozi import compute_cozi_core
from hexif.spatial.neighborhoods import build_neighborhoods


def _clustered_cells(n_per: int = 20) -> pd.DataFrame:
    """n_per cells of A in (x, y < 0) and n_per cells of B in (x, y > 0)."""
    rng = np.random.default_rng(0)
    xa = rng.uniform(0, 10, n_per)
    ya = rng.uniform(-10, -1, n_per)
    xb = rng.uniform(0, 10, n_per)
    yb = rng.uniform(1, 10, n_per)
    return pd.DataFrame(
        {
            "basename": ["clustered"] * (2 * n_per),
            "cell_id": list(range(1, 2 * n_per + 1)),
            "centroid_x": np.concatenate([xa, xb]),
            "centroid_y": np.concatenate([ya, yb]),
            "label": ["A"] * n_per + ["B"] * n_per,
        }
    )


def test_cozi_same_phenotype_positive_z_on_clustered_layout():
    cells = _clustered_cells(n_per=30)
    g = build_neighborhoods(cells, method="knn", k=6, phenotype_col="label")
    res = compute_cozi_core(cells, g.edges, phenotype_col="label")
    # Same-phenotype z is positive (clustering)
    same = res[res.phenotype_a == res.phenotype_b]
    assert (same["z"] > 5).all(), (
        f"expected strong positive same-phenotype z, got: {same[['phenotype_a', 'z']]}"
    )
    # Cross-phenotype z is negative (avoidance)
    cross = res[res.phenotype_a != res.phenotype_b]
    assert (cross["z"] < -5).all(), (
        f"expected strong negative cross-phenotype z, got: {cross[['phenotype_a', 'phenotype_b', 'z']]}"
    )


def test_cozi_random_layout_z_near_zero():
    rng = np.random.default_rng(0)
    n = 80
    cells = pd.DataFrame(
        {
            "basename": ["rand"] * n,
            "cell_id": list(range(1, n + 1)),
            "centroid_x": rng.uniform(0, 10, n),
            "centroid_y": rng.uniform(0, 10, n),
            "label": rng.choice(["A", "B"], size=n),
        }
    )
    g = build_neighborhoods(cells, method="knn", k=6, phenotype_col="label")
    res = compute_cozi_core(cells, g.edges, phenotype_col="label")
    # With random labels, |z| should be small (sampling, so allow some slack)
    cross = res[res.phenotype_a != res.phenotype_b]
    assert (cross["z"].abs() < 4).all(), (
        f"expected near-zero z under random labels, got: {cross[['phenotype_a', 'phenotype_b', 'z']]}"
    )


def test_cozi_empty_inputs():
    cells = pd.DataFrame(
        {
            "basename": ["empty"],
            "cell_id": [1],
            "centroid_x": [0],
            "centroid_y": [0],
            "label": ["A"],
        }
    )
    edges = pd.DataFrame(
        columns=[
            "basename",
            "src_cell_id",
            "dst_cell_id",
            "src_phenotype",
            "dst_phenotype",
            "distance_px",
            "edge_type",
        ]
    )
    res = compute_cozi_core(cells, edges, phenotype_col="label")
    assert res.empty
