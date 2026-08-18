"""Render a per-core PDF report from a Workbench bundle.

Inputs: bundle directory (containing core_summary.parquet + cells.parquet
+ spatial_summary.parquet + thumbs/<basename>/), and a target core's
basename.

Outputs: a single PDF file with composition + marker fractions +
phenotype fractions + COZI heatmap + per-marker spatial maps.

Implementation: Jinja2 → HTML string → WeasyPrint → PDF bytes → file.
All thumbnail images are referenced as ``file://`` absolute paths so
WeasyPrint can resolve them without a base URL gymnastics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES
from hexif.pipeline.thresholds import load_thresholds
from hexif.report.plots import (
    composition_donut,
    cozi_heatmap,
    marker_fraction_bars,
)

logger = logging.getLogger(__name__)


TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(),
    )


def _frac(core_row: pd.Series, name: str, default: float | None = None) -> float | None:
    v = core_row.get(name, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return float(v)


def render_core_report(
    bundle_dir: str | Path,
    basename: str,
    *,
    out_path: str | Path | None = None,
    model_id: str = "v1_1",
    include_marker_maps: bool = True,
    n_top_cozi: int = 10,
) -> Path:
    bundle_dir = Path(bundle_dir)
    out_path = Path(out_path) if out_path else (bundle_dir / "reports" / f"{basename}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cores = pd.read_parquet(bundle_dir / "core_summary.parquet")
    cells = pd.read_parquet(bundle_dir / "cells.parquet")
    spatial = None
    if (bundle_dir / "spatial_summary.parquet").exists():
        spatial = pd.read_parquet(bundle_dir / "spatial_summary.parquet")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    threshold_ref = manifest.get("threshold_source")
    if not isinstance(threshold_ref, str) or not threshold_ref:
        raise ValueError("manifest.threshold_source must name a bundle artifact")
    thresholds = load_thresholds(bundle_dir / threshold_ref)

    core_row = cores[cores["basename"] == basename]
    if core_row.empty:
        raise ValueError(f"basename={basename!r} not found in {bundle_dir}/core_summary.parquet")
    core_row = core_row.iloc[0]
    cells_core = cells[cells["basename"] == basename]

    # --- Build per-core composition fractions dict
    composition = {}
    for name in PHENOTYPE_NAMES:
        col = f"frac_phenotype_{name}_call_{model_id}"
        if col in core_row.index:
            v = core_row[col]
            composition[name] = float(v) if not pd.isna(v) else 0.0

    # Detect whether truth columns are present
    any_truth = any(c.startswith("frac_ch") and c.endswith("_pos_truth") for c in core_row.index)

    # --- Generate plots into a per-report scratch dir
    scratch = bundle_dir / "reports" / f".{basename}_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    composition_png = scratch / "composition.png"
    composition_donut(
        composition,
        composition_png,
        title=f"Phenotype composition · {basename}",
        confidence=thresholds.phenotype_confidence,
    )

    marker_pred_fracs: dict[str, float] = {}
    marker_truth_fracs: dict[str, float] = {}
    for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
        pred_col = f"frac_ch{ch:02d}_pos_{model_id}"
        truth_col = f"frac_ch{ch:02d}_pos_truth"
        if pred_col in core_row.index:
            marker_pred_fracs[name] = (
                float(core_row[pred_col]) if not pd.isna(core_row[pred_col]) else 0.0
            )
        if truth_col in core_row.index:
            marker_truth_fracs[name] = (
                float(core_row[truth_col]) if not pd.isna(core_row[truth_col]) else float("nan")
            )
    marker_bars_png = scratch / "marker_bars.png"
    marker_fraction_bars(
        marker_pred_fracs,
        marker_bars_png,
        truth=marker_truth_fracs if marker_truth_fracs else None,
        title=f"Marker-positive fraction · {basename}",
        confidence=thresholds.marker_confidence,
    )

    spatial_for_core = (
        spatial[spatial["basename"] == basename].copy() if spatial is not None else pd.DataFrame()
    )
    cozi_png = scratch / "cozi_heatmap.png"
    cozi_heatmap(spatial_for_core, cozi_png, title=f"COZI z(A→B) · {basename}")

    # --- Marker rows for the table
    marker_rows = []
    for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
        marker_rows.append(
            {
                "channel": int(ch),
                "name": name,
                "frac_pred": marker_pred_fracs.get(name, 0.0),
                "frac_truth": marker_truth_fracs.get(name)
                if name in marker_truth_fracs and not np.isnan(marker_truth_fracs[name])
                else None,
                "confidence": thresholds.marker_confidence[name],
            }
        )

    # --- Phenotype rows
    phenotype_rows = []
    for name in PHENOTYPE_NAMES:
        pred_col = f"frac_phenotype_{name}_call_{model_id}"
        truth_col = f"frac_phenotype_{name}_truth"
        phenotype_rows.append(
            {
                "name": name,
                "frac_pred": float(core_row[pred_col])
                if pred_col in core_row.index and not pd.isna(core_row[pred_col])
                else 0.0,
                "frac_truth": float(core_row[truth_col])
                if truth_col in core_row.index and not pd.isna(core_row[truth_col])
                else None,
                "confidence": thresholds.phenotype_confidence[name],
            }
        )

    # --- Top COZI directional preferences (largest |z|, excluding self/self)
    top_cozi = []
    if not spatial_for_core.empty:
        sc = spatial_for_core.copy()
        sc = sc[sc["phenotype_a"] != sc["phenotype_b"]]
        sc = sc.dropna(subset=["z"])
        sc = sc.reindex(sc["z"].abs().sort_values(ascending=False).index)
        for _, r in sc.head(n_top_cozi).iterrows():
            top_cozi.append(
                {
                    "phenotype_a": str(r["phenotype_a"]),
                    "phenotype_b": str(r["phenotype_b"]),
                    "z": float(r["z"]),
                    "obs_over_exp": float(r.get("obs_over_exp", float("nan"))),
                    "hotspot_overlap_iou": (
                        float(r["hotspot_overlap_iou"])
                        if "hotspot_overlap_iou" in r.index
                        and not pd.isna(r["hotspot_overlap_iou"])
                        else None
                    ),
                }
            )

    # --- Per-marker maps: HE / CODEX / pred for each marker (small set: only "strong" + selected weak)
    pred_maps = []
    if include_marker_maps:
        thumbs_dir = bundle_dir / "thumbs" / basename
        if thumbs_dir.exists():
            he_thumb_uri = (thumbs_dir / "he.png").resolve().as_uri()
            for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
                pred_png = thumbs_dir / f"ch{ch:02d}_pred_{model_id}.png"
                codex_png = thumbs_dir / f"ch{ch:02d}_codex.png"
                if not pred_png.exists():
                    continue
                pred_maps.append(
                    {
                        "channel": int(ch),
                        "name": name,
                        "confidence": thresholds.marker_confidence[name],
                        "he": he_thumb_uri,
                        "codex": codex_png.resolve().as_uri()
                        if codex_png.exists()
                        else he_thumb_uri,
                        "pred": pred_png.resolve().as_uri(),
                    }
                )

    # --- Render HTML
    he_thumb = bundle_dir / "thumbs" / basename / "he.png"
    he_thumb_uri = he_thumb.resolve().as_uri() if he_thumb.exists() else ""
    css_text = (TEMPLATE_DIR / "styles.css").read_text()
    env = _env()
    template = env.get_template("core_report.html.j2")
    html = template.render(
        css=css_text,
        basename=basename,
        tma=str(core_row.get("tma", "")),
        core=str(core_row.get("core", "")),
        tissue=str(core_row.get("tissue", "")),
        split=str(core_row.get("split", "")),
        n_cells=int(core_row.get("n_cells", len(cells_core))),
        bundle_dir=str(bundle_dir.resolve()),
        build_date=manifest.get("build_date", time.strftime("%Y-%m-%d")),
        threshold_source=manifest.get("threshold_source", "(n/a)"),
        model_id=model_id,
        he_thumb=he_thumb_uri,
        composition_png=composition_png.resolve().as_uri(),
        marker_bars_png=marker_bars_png.resolve().as_uri(),
        cozi_png=cozi_png.resolve().as_uri(),
        marker_rows=marker_rows,
        phenotype_rows=phenotype_rows,
        any_truth=any_truth,
        top_cozi=top_cozi,
        pred_maps=pred_maps,
    )

    # --- Render PDF
    from weasyprint import HTML

    HTML(string=html, base_url=str(bundle_dir.resolve())).write_pdf(str(out_path))
    logger.info("wrote %s", out_path)
    return out_path
