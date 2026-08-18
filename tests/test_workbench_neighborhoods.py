"""Unit tests for hexif.spatial.neighborhoods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hexif.spatial.neighborhoods import build_neighborhoods


def _toy_cells(basename: str = "core_x") -> pd.DataFrame:
    # 4 cells on a unit grid:
    #     (0,0) A       (1,0) A
    #     (0,1) B       (1,1) B
    return pd.DataFrame(
        {
            "basename": [basename] * 4,
            "cell_id": [1, 2, 3, 4],
            "centroid_x": [0, 1, 0, 1],
            "centroid_y": [0, 0, 1, 1],
            "label": ["A", "A", "B", "B"],
        }
    )


def test_knn_graph_k1_excludes_self():
    cells = _toy_cells()
    g = build_neighborhoods(cells, method="knn", k=1, phenotype_col="label")
    # Each cell has exactly 1 neighbor (its nearest, which is at distance 1)
    assert len(g.edges) == 4
    # No self loops
    assert (g.edges["src_cell_id"] != g.edges["dst_cell_id"]).all()
    # All edges have distance 1.0
    assert np.allclose(g.edges["distance_px"].to_numpy(), 1.0)
    # Source/dst phenotypes recorded
    assert set(g.edges["src_phenotype"]) == {"A", "B"}


def test_radius_graph_includes_diagonal():
    cells = _toy_cells()
    # radius 1.5 catches the 4 axis-aligned pairs AND the 2 diagonals (d=√2≈1.41)
    g = build_neighborhoods(cells, method="radius", radius_px=1.5, phenotype_col="label")
    # 6 unique unordered pairs × 2 directions = 12 directed edges
    assert len(g.edges) == 12
    assert (g.edges["edge_type"] == "radius").all()


def test_knn_skips_when_too_few_cells():
    cells = _toy_cells().head(1)
    g = build_neighborhoods(cells, method="knn", k=3, phenotype_col="label")
    assert len(g.edges) == 0
    assert g.meta["n_cells"] == 1


def test_multi_core_rejected():
    df = pd.concat([_toy_cells("a"), _toy_cells("b")], ignore_index=True)
    try:
        build_neighborhoods(df, method="knn", k=1, phenotype_col="label")
    except ValueError:
        return
    raise AssertionError("multi-core input should have raised")
