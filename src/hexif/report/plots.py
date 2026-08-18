"""Plot primitives for the per-core PDF report.

All functions write a PNG to disk and return the path.  They are pure
matplotlib — no Plotly, no headless Chrome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES


def _import_mpl():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


CONFIDENCE_COLORS = {
    "strong": "#1f7a4f",
    "moderate": "#b08800",
    "weak": "#a02022",
    "unknown": "#888888",
}


def composition_donut(
    fractions: dict[str, float],
    out_path: Path,
    *,
    title: str = "Phenotype composition",
    min_show: float = 0.01,
    confidence: dict[str, str] | None = None,
) -> Path:
    """Draw a donut of phenotype fractions.  ``fractions`` maps phenotype → 0..1.

    Phenotypes below ``min_show`` are grouped into "other".
    """
    plt = _import_mpl()
    fractions = {k: float(v) for k, v in fractions.items() if not np.isnan(v)}
    if not fractions:
        # Return an explicit empty-state figure.
        fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
        ax.text(
            0.5,
            0.5,
            "no phenotype calls",
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)
    large = {k: v for k, v in fractions.items() if v >= min_show}
    other = sum(v for k, v in fractions.items() if v < min_show)
    if other > 0:
        large["other"] = other
    labels = list(large.keys())
    sizes = [large[k] for k in labels]
    # Colour by confidence stratum
    confidence = confidence or {}
    colors = [
        CONFIDENCE_COLORS.get(confidence.get(k, "unknown"), "#888888")
        if k != "other"
        else "#cccccc"
        for k in labels
    ]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=140)
    _wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white"),
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def marker_fraction_bars(
    fractions: dict[str, float],
    out_path: Path,
    truth: dict[str, float] | None = None,
    title: str = "Marker-positive fraction",
    confidence: dict[str, str] | None = None,
) -> Path:
    """Horizontal bar plot of per-marker fractions, colored by confidence."""
    plt = _import_mpl()
    names = list(MARKER_NAMES)
    vals = [float(fractions.get(n, 0.0)) for n in names]
    confidence = confidence or {}
    colors = [CONFIDENCE_COLORS.get(confidence.get(n, "unknown"), "#888888") for n in names]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.7, edgecolor="white")
    if truth is not None:
        truth_vals = [float(truth.get(n, np.nan)) for n in names]
        ax.scatter(truth_vals, y, color="black", s=18, zorder=3, label="CODEX truth")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction positive", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def cozi_heatmap(
    summary_for_core: pd.DataFrame,
    out_path: Path,
    title: str = "COZI directional neighbor preference",
    z_cap: float = 30.0,
) -> Path:
    """Heatmap of COZI z(A → B); rows = phenotype_a, cols = phenotype_b."""
    plt = _import_mpl()
    if summary_for_core.empty:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.text(
            0.5,
            0.5,
            "no COZI rows for this core",
            ha="center",
            va="center",
            fontsize=11,
            color="grey",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)
    pivot = summary_for_core.pivot_table(
        index="phenotype_a",
        columns="phenotype_b",
        values="z",
        aggfunc="mean",
    )
    # Sort rows/cols by canonical phenotype order, then any extras alphabetically
    order = [n for n in PHENOTYPE_NAMES if n in pivot.index] + sorted(
        [n for n in pivot.index if n not in PHENOTYPE_NAMES]
    )
    col_order = [n for n in PHENOTYPE_NAMES if n in pivot.columns] + sorted(
        [n for n in pivot.columns if n not in PHENOTYPE_NAMES]
    )
    pivot = pivot.reindex(index=order, columns=col_order)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=140)
    capped = pivot.clip(lower=-z_cap, upper=z_cap)
    im = ax.imshow(capped.to_numpy(), cmap="RdBu_r", vmin=-z_cap, vmax=z_cap, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("neighbor phenotype B", fontsize=9)
    ax.set_ylabel("source phenotype A", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i, _a in enumerate(order):
        for j, _b in enumerate(col_order):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(v) > z_cap * 0.4 else "black",
            )
    fig.colorbar(im, ax=ax, label="COZI z", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def per_marker_score_hist(
    cells: pd.DataFrame,
    out_path: Path,
    model_id: str = "v1_1",
    title: str = "Predicted score distributions",
) -> Path:
    """For each marker, density histogram of predicted scores split by truth."""
    plt = _import_mpl()
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), dpi=140, constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for k, (ch, name) in enumerate(zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False)):
        ax = axes[k // 4, k % 4]
        pred_col = f"ch{ch:02d}_pred_{model_id}"
        truth_col = f"ch{ch:02d}_pos"
        if pred_col not in cells.columns:
            ax.axis("off")
            continue
        s = cells[pred_col].to_numpy(dtype=np.float32)
        if truth_col in cells.columns:
            y = cells[truth_col].astype(bool).to_numpy()
            ax.hist(s[~y], bins=bins, alpha=0.55, color="lightgrey", density=True, label="neg")
            ax.hist(s[y], bins=bins, alpha=0.6, color="crimson", density=True, label="pos")
        else:
            ax.hist(s, bins=bins, alpha=0.6, color="steelblue", density=True)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title(f"ch{ch:02d} {name}", fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
