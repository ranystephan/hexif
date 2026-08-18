"""Simple neighbor-preference baselines (alongside COZI).

These are the "easy" baselines we report next to COZI so the reader can
see how much the conditional z-score is doing.

* Per-cell local composition vector: fraction of each phenotype in the
  cell's neighborhood.
* Per-(A, B) observed-over-expected ratio under the multinomial
  assumption (no permutation): obs / (S_A * n_B / (N - 1)).
* Per-(A, B) Jaccard hotspot overlap (computed in hotspots.py).
"""

from __future__ import annotations

import pandas as pd


def per_cell_local_composition(
    cells: pd.DataFrame,
    edges: pd.DataFrame,
    phenotype_col: str,
) -> pd.DataFrame:
    """For each cell, fraction of its neighbors in each phenotype.

    Returns a DataFrame indexed by (basename, cell_id) with one column
    per phenotype: ``local_frac_{phenotype}``.
    """
    if edges.empty:
        return pd.DataFrame()
    sorted(cells[phenotype_col].astype(str).unique().tolist())
    # Annotate edges with src phenotype if not already present
    if "src_phenotype" not in edges.columns or "dst_phenotype" not in edges.columns:
        ph = cells.set_index(["basename", "cell_id"])[phenotype_col].astype(str)
        edges = edges.copy()
        edges["src_phenotype"] = ph.reindex(
            list(zip(edges["basename"], edges["src_cell_id"], strict=False))
        ).to_numpy()
        edges["dst_phenotype"] = ph.reindex(
            list(zip(edges["basename"], edges["dst_cell_id"], strict=False))
        ).to_numpy()
    # group by (basename, src_cell_id) → count of each dst_phenotype
    g = (
        edges.groupby(["basename", "src_cell_id", "dst_phenotype"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    pivot = g.pivot_table(
        index=["basename", "src_cell_id"],
        columns="dst_phenotype",
        values="n",
        fill_value=0,
    )
    pivot.columns = [f"local_count_{c}" for c in pivot.columns]
    pivot["n_neighbors"] = pivot.sum(axis=1)
    for c in pivot.columns:
        if c.startswith("local_count_"):
            name = c.removeprefix("local_count_")
            pivot[f"local_frac_{name}"] = pivot[c] / pivot["n_neighbors"].clip(lower=1)
    return pivot.reset_index()


def observed_over_expected(
    cells: pd.DataFrame,
    edges: pd.DataFrame,
    phenotype_col: str,
) -> pd.DataFrame:
    """Per-(basename, A, B): observed_AB / expected_AB (multinomial).

    Cheap baseline alongside COZI.  expected_AB = S_A * n_B / (N - 1)
    (no variance, no z-score).
    """
    if edges.empty:
        return pd.DataFrame()
    rows = []
    for basename, grp in cells.groupby("basename", sort=False):
        e = edges[edges["basename"] == basename]
        if e.empty:
            continue
        counts = grp[phenotype_col].astype(str).value_counts()
        n_total = len(grp)
        s_p = e.groupby("src_phenotype").size()
        obs = (
            e.groupby(["src_phenotype", "dst_phenotype"], sort=False)
            .size()
            .rename("observed")
            .reset_index()
        )
        for _, r in obs.iterrows():
            a, b = str(r["src_phenotype"]), str(r["dst_phenotype"])
            n_a = int(counts.get(a, 0))
            n_b = int(counts.get(b, 0))
            if n_a == 0 or n_b == 0:
                continue
            s_a = int(s_p.get(a, 0))
            if a == b:
                exp = s_a * max(n_a - 1, 0) / max(n_total - 1, 1)
            else:
                exp = s_a * n_b / max(n_total - 1, 1)
            ratio = float(r["observed"]) / max(exp, 1e-6)
            rows.append(
                {
                    "basename": basename,
                    "phenotype_a": a,
                    "phenotype_b": b,
                    "observed": int(r["observed"]),
                    "expected": float(exp),
                    "obs_over_exp": float(ratio),
                }
            )
    return pd.DataFrame(rows)
