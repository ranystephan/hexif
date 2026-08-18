"""Build and load the HEXIF Workbench bundle from real model outputs.

Bundle layout::

    <bundle_dir>/
        manifest.json
        core_summary.parquet          # one row per core
        cells.parquet                 # one row per cell across all cores
        spatial_edges.parquet         # OPTIONAL, written by hexif.spatial
        spatial_summary.parquet       # written by hexif.spatial
        thumbs/<basename>/
            he.png
            ch{NN}_pred.png
            ch{NN}_truth.png          # only when paired CODEX is available
            phenotype_{name}.png

Every input path is supplied explicitly. See ``docs/data-contracts.md``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hexif.cell_phenotype import (
    FOCUSED_MARKERS,
    MARKER_NAMES,
    PHENOTYPE_NAMES,
)
from hexif.pipeline.codex_composite import (
    DEFAULT_PERCENTILE_WINDOW,
)
from hexif.pipeline.codex_composite import (
    build_codex_composite as _build_codex_composite_array,
)
from hexif.pipeline.polygons import build_cell_polygons
from hexif.pipeline.postprocess import (
    assign_marker_calls,
    assign_phenotype_calls,
    attach_tma_tissue,
    per_core_composition,
)
from hexif.pipeline.thresholds import CalibratedThresholds, load_thresholds, save_thresholds_json
from hexif.scaling import QuantileScaler

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write ``json.dumps(payload, indent=2)`` to ``path`` atomically.

    Concurrent readers (e.g., the FastAPI workbench rereading the
    manifest mid-rebuild) should never see a partial file. We write to
    a sibling ``<name>.tmp`` and ``os.replace`` it over the target,
    matching the rename-is-atomic guarantee the parquet writer already
    relies on for ``cells.parquet``.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class BundleManifest:
    bundle_version: str
    build_date: str
    n_cores: int
    n_cells: int
    splits: list[str]
    model_ids: list[str]
    marker_channels: list[int]
    marker_names: list[str]
    phenotype_names: list[str]
    sources: dict
    threshold_source: str
    model_provenance: dict
    image_sizes: dict[str, list[int]]
    notes: str = ""
    has_cell_polygons: bool = False
    models: list[dict] | None = None
    composite_percentiles: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _pick_pred_csv(path: str | Path) -> Path:
    """Validate and return an explicitly supplied predictions CSV."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"cell-predictions CSV does not exist: {resolved}")
    return resolved


def _load_inputs(
    cell_predictions: str | Path,
    consensus: str | Path,
    manifest: str | Path,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the prediction wide-CSV and join the manifest + consensus area
    columns. Returns (cells_df, core_manifest_df).
    """
    pred_path = _pick_pred_csv(cell_predictions)
    df = pd.read_csv(pred_path)
    logger.info("loaded %s  rows=%d  cols=%d", pred_path, len(df), df.shape[1])

    cons_path = Path(consensus)
    if not cons_path.is_file():
        raise FileNotFoundError(f"consensus table does not exist: {cons_path}")
    cons = pd.read_csv(cons_path)
    join_cols = ["basename", "cell_id"]
    keep = [c for c in cons.columns if c not in df.columns or c in join_cols]
    df = df.merge(cons[keep], on=join_cols, how="left", suffixes=("", "_cons"))
    logger.info("joined consensus table (%d rows, %d cols)", len(cons), len(cons.columns))

    man_path = Path(manifest)
    man = pd.read_csv(man_path)
    man = man[["tma", "core", "basename", "split", "rigid_D_px"]].copy()
    if split:
        man_filtered = man[man["split"] == split]
        df = df[df["basename"].isin(man_filtered["basename"])]
        man = man_filtered
        logger.info("filtered to split=%s: %d cores, %d cells", split, len(man), len(df))

    return df, man


# ---------------------------------------------------------------------------
# CODEX composite rendering (CODEX composite)
# ---------------------------------------------------------------------------


def _resolve_codex_scaler(
    scaler_path: str | Path | None,
) -> QuantileScaler:
    """Load an explicitly supplied CODEX quantile-scaler artifact."""
    candidate = Path(scaler_path) if scaler_path is not None else None
    if candidate is None or not candidate.is_file():
        raise FileNotFoundError("pass an existing --codex-scaler JSON artifact")
    return QuantileScaler.load(candidate)


def _render_codex_composite_png(
    basename: str,
    codex_path: Path,
    scaler: QuantileScaler,
    out_dir: Path,
    *,
    percentile_window: tuple[float, float] = DEFAULT_PERCENTILE_WINDOW,
) -> dict | None:
    """Render and persist ``thumbs/<basename>/codex_composite.png``.

    Returns the percentile-cache dict for ``manifest.composite_percentiles``
    on success; ``None`` when the CODEX npy is missing (the caller logs
    a warning — that core's DZI route will 404).

    PNG is the source-of-truth artifact for the DZI lazy-build. We
    write it via :class:`PIL.Image` so the on-disk bytes are
    byte-deterministic for a given input array (libpng's default
    encoder is order-stable; the idempotency test relies on this).
    """
    from PIL import Image as _Image

    if not codex_path.exists():
        return None
    rgb, info = _build_codex_composite_array(
        codex_path,
        scaler,
        percentile_window=percentile_window,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "codex_composite.png"
    _Image.fromarray(rgb, mode="RGB").save(png_path, format="PNG")
    return info


# ---------------------------------------------------------------------------
# Thumbnail rendering (HE + per-marker pred / truth + per-phenotype maps)
# ---------------------------------------------------------------------------


def _render_he_png(he: np.ndarray, path: Path, max_side: int = 1024) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    H, W = he.shape[:2]
    scale = max_side / max(H, W) if max(H, W) > max_side else 1.0
    if scale < 1.0:
        from PIL import Image

        img = Image.fromarray(he).resize((int(W * scale), int(H * scale)))
        he = np.array(img)
    ax.imshow(he)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _render_scatter_png(
    centroids_xy: np.ndarray,  # (N, 2) in (x, y)
    values: np.ndarray,  # (N,) float in [0, 1] (continuous) or 0/1 (binary)
    he_shape: tuple[int, int],
    path: Path,
    *,
    cmap: str = "magma",
    binary: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    point_size: float = 3.0,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    H, W = he_shape
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    if binary:
        v = values.astype(bool)
        ax.scatter(
            centroids_xy[~v, 0],
            centroids_xy[~v, 1],
            s=point_size,
            c="lightgrey",
            linewidths=0,
            alpha=0.45,
        )
        ax.scatter(
            centroids_xy[v, 0],
            centroids_xy[v, 1],
            s=point_size + 1,
            c="crimson",
            linewidths=0,
            alpha=0.9,
        )
    else:
        ax.scatter(
            centroids_xy[:, 0],
            centroids_xy[:, 1],
            s=point_size,
            c=values,
            cmap=cmap,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _render_codex_truth_png_full_res(
    codex_channel: np.ndarray,  # (H, W) uint16
    path: Path,
    *,
    cmap: str = "magma",
) -> None:
    """Render one CODEX channel as an RGB PNG at the input array's resolution.

    Used for the per-marker CODEX base layer in the workbench: cell
    polygons live in the original (e.g. 2950×2950) image space, so the
    base-layer image must match. Writes via PIL directly to keep the
    byte output deterministic (matplotlib's savefig adds white padding
    + DPI-dependent scaling).
    """
    from matplotlib import colormaps as _colormaps
    from PIL import Image as _Image

    arr = codex_channel.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99.5])
    norm = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1)
    rgba = (_colormaps[cmap](norm) * 255).astype(np.uint8)
    rgb = rgba[:, :, :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    _Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def _render_codex_truth_png(
    codex_channel: np.ndarray,  # (H, W) uint16
    path: Path,
    *,
    cmap: str = "magma",
    max_side: int = 1024,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    H, W = codex_channel.shape
    arr = codex_channel.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99.5])
    norm = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1)
    if max(H, W) > max_side:
        from PIL import Image

        img = Image.fromarray((norm * 255).astype(np.uint8))
        scale = max_side / max(H, W)
        img = img.resize((int(W * scale), int(H * scale)))
        norm = np.asarray(img, dtype=np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_core_thumbs(
    basename: str,
    he_path: Path,
    codex_path: Path | None,
    cells_for_core: pd.DataFrame,
    out_dir: Path,
    *,
    model_id: str = "v1_1",
    point_size: float = 3.0,
) -> None:
    """Render he.png + per-marker pred/truth + per-phenotype maps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    he = np.load(he_path)
    he_shape = (he.shape[0], he.shape[1])
    _render_he_png(he, out_dir / "he.png")
    centroids = cells_for_core[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)

    # Per-marker predictions (continuous, magma)
    for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
        pred_col = f"ch{ch:02d}_pred_{model_id}"
        if pred_col not in cells_for_core.columns:
            continue
        vals = cells_for_core[pred_col].to_numpy(dtype=np.float32)
        _render_scatter_png(
            centroids,
            vals,
            he_shape,
            out_dir / f"ch{ch:02d}_pred_{model_id}.png",
            cmap="magma",
            binary=False,
            vmin=0,
            vmax=1,
            point_size=point_size,
            title=f"{name} predicted",
        )

    # Per-marker truth (binary, red/grey) where available
    for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
        truth_col = f"ch{ch:02d}_pos"
        if truth_col not in cells_for_core.columns:
            continue
        v = cells_for_core[truth_col].astype(bool).to_numpy()
        _render_scatter_png(
            centroids,
            v.astype(np.float32),
            he_shape,
            out_dir / f"ch{ch:02d}_truth.png",
            binary=True,
            point_size=point_size,
            title=f"{name} truth",
        )

    # CODEX raw truth images (per-image p1-p99.5 normalized) when paired
    if codex_path is not None and codex_path.exists():
        try:
            codex = np.load(codex_path, mmap_mode="r")
            for ch in FOCUSED_MARKERS:
                if ch >= codex.shape[0]:
                    continue
                _render_codex_truth_png(
                    np.asarray(codex[ch]),
                    out_dir / f"ch{ch:02d}_codex.png",
                )
        except Exception as e:
            logger.warning("CODEX render failed for %s: %s", basename, e)

    # Per-phenotype calls (binary)
    for name in PHENOTYPE_NAMES:
        call_col = f"phenotype_{name}_call_{model_id}"
        score_col = f"phenotype_{name}_score_{model_id}"
        if call_col in cells_for_core.columns:
            v = cells_for_core[call_col].astype(bool).to_numpy()
            _render_scatter_png(
                centroids,
                v.astype(np.float32),
                he_shape,
                out_dir / f"phenotype_{name}_call.png",
                binary=True,
                point_size=point_size,
                title=f"{name} called",
            )
        if score_col in cells_for_core.columns:
            vals = cells_for_core[score_col].to_numpy(dtype=np.float32)
            _render_scatter_png(
                centroids,
                vals,
                he_shape,
                out_dir / f"phenotype_{name}_score.png",
                cmap="magma",
                binary=False,
                vmin=0,
                vmax=1,
                point_size=point_size,
                title=f"{name} score",
            )


# ---------------------------------------------------------------------------
# Bundle building
# ---------------------------------------------------------------------------


def build_bundle(
    output_dir: str | Path,
    *,
    pairs_dir: str | Path,
    cell_predictions: str | Path,
    consensus: str | Path,
    manifest: str | Path,
    thresholds_json: str | Path,
    model_metadata: str | Path,
    split: str | None = "val",
    model_ids: Sequence[str] = ("cell_phenotype",),
    primary_model: str = "cell_phenotype",
    render_thumbs: bool = True,
    max_cores: int | None = None,
    skip_codex: bool = False,
    build_polygons: bool = True,
    masks_dir: str | Path,
    build_codex_composite: bool = True,
    codex_scaler: str | Path | None = None,
) -> Path:
    """Build a Workbench bundle.

    Reads existing outputs only — no GPU, no model loading.  Writes:
        manifest.json, core_summary.parquet, cells.parquet, thumbs/.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(pairs_dir)

    # 1. Load inputs
    df, man = _load_inputs(cell_predictions, consensus, manifest, split)
    if max_cores is not None:
        keep = man.head(max_cores)["basename"].tolist()
        df = df[df["basename"].isin(keep)].copy()
        man = man.head(max_cores).copy()

    # 2. Calibrated calls
    thresholds = load_thresholds(thresholds_json)
    metadata = _load_model_metadata(model_metadata, model_ids)
    for mid in model_ids:
        df = assign_marker_calls(df, thresholds, model_id=mid)
        df = assign_phenotype_calls(df, thresholds, model_id=mid)

    # 3. Per-core composition
    composition = per_core_composition(df, model_ids=model_ids)

    # 4. core_summary = manifest left-join composition (+ tma/tissue)
    core_summary = man.merge(composition, on="basename", how="left")
    core_summary = attach_tma_tissue(core_summary)
    core_summary["model_id"] = primary_model
    cols_lead = ["basename", "tma", "core", "tissue", "split", "rigid_D_px", "n_cells", "model_id"]
    other = [c for c in core_summary.columns if c not in cols_lead]
    core_summary = core_summary[cols_lead + other]

    # 5. cells.parquet
    cells = df.copy()
    cells = attach_tma_tissue(cells)
    cells_path = output_dir / "cells.parquet"
    core_path = output_dir / "core_summary.parquet"
    cells.to_parquet(cells_path, index=False)
    core_summary.to_parquet(core_path, index=False)
    logger.info("wrote %s (%d cells, %d cols)", cells_path, len(cells), len(cells.columns))
    logger.info("wrote %s (%d cores)", core_path, len(core_summary))

    # 5b. Optional cell polygons. When requested, every mask and join
    # invariant is required; partial polygon bundles are not emitted.
    has_polygons = False
    if build_polygons:
        masks_path = Path(masks_dir)
        if not masks_path.is_dir():
            raise FileNotFoundError(f"masks directory does not exist: {masks_path}")
        build_cell_polygons(
            masks_path,
            cells[["basename", "cell_id"]],
            output_dir / "cell_polygons.parquet",
        )
        has_polygons = True

    # 6. Thumbs
    if render_thumbs:
        thumbs_dir = output_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        cells_by_core = {b: g for b, g in cells.groupby("basename", sort=False)}
        t0 = time.time()
        bn_list = man["basename"].tolist()
        for i, base in enumerate(bn_list, 1):
            if base not in cells_by_core:
                raise ValueError(f"manifest core has no cell predictions: {base}")
            he_path = pairs_dir / f"{base}_HE.npy"
            codex_path = pairs_dir / f"{base}_CODEX.npy" if not skip_codex else None
            if not he_path.is_file():
                raise FileNotFoundError(f"registered H&E array does not exist: {he_path}")
            render_core_thumbs(
                base,
                he_path,
                codex_path,
                cells_by_core[base],
                thumbs_dir / base,
                model_id=primary_model,
            )
            if i % 5 == 0 or i == len(bn_list):
                logger.info("rendered %d/%d  wall=%.1fs", i, len(bn_list), time.time() - t0)

    # 6b. Optional CODEX composite and its recorded percentile window.
    composite_percentiles: dict[str, dict] = {}
    composite_scaler_path: Path | None = None
    if build_codex_composite and not skip_codex:
        scaler = _resolve_codex_scaler(codex_scaler)
        composite_scaler_path = Path(codex_scaler) if codex_scaler else None
        thumbs_dir = output_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, base in enumerate(man["basename"].tolist(), 1):
            codex_path = pairs_dir / f"{base}_CODEX.npy"
            info = _render_codex_composite_png(base, codex_path, scaler, thumbs_dir / base)
            if info is None:
                raise FileNotFoundError(f"registered CODEX array does not exist: {codex_path}")
            composite_percentiles[base] = info
            if i % 10 == 0 or i == len(man):
                logger.info(
                    "rendered %d/%d composites  wall=%.1fs",
                    i,
                    len(man),
                    time.time() - t0,
                )

    # 7. Manifest
    pred_path = _pick_pred_csv(cell_predictions)
    sources = {
        "cell_predictions": str(pred_path.resolve()),
        "consensus": str(Path(consensus).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "pairs_dir": str(pairs_dir.resolve()),
        "masks_dir": str(Path(masks_dir).resolve()) if has_polygons else "",
    }
    if composite_scaler_path is not None:
        sources["scaler"] = str(composite_scaler_path.resolve())
    image_sizes: dict[str, list[int]] = {}
    for basename in core_summary["basename"].astype(str):
        he_path = pairs_dir / f"{basename}_HE.npy"
        if not he_path.is_file():
            raise FileNotFoundError(f"registered H&E array does not exist: {he_path}")
        he = np.load(he_path, mmap_mode="r")
        if he.ndim != 3 or he.shape[-1] != 3:
            raise ValueError(f"registered H&E must have shape (H, W, 3): {he_path}")
        image_sizes[basename] = [int(he.shape[1]), int(he.shape[0])]

    models = [
        _model_entry(cells, model_id, pred_path, metadata[model_id]) for model_id in model_ids
    ]
    save_thresholds_json(thresholds, output_dir / "thresholds.json")
    bm = BundleManifest(
        bundle_version="v0.1",
        build_date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_cores=len(core_summary),
        n_cells=len(cells),
        splits=sorted(core_summary["split"].dropna().unique().tolist()),
        model_ids=list(model_ids),
        marker_channels=list(map(int, FOCUSED_MARKERS)),
        marker_names=list(MARKER_NAMES),
        phenotype_names=list(PHENOTYPE_NAMES),
        sources=sources,
        threshold_source="thresholds.json",
        model_provenance={model_id: metadata[model_id]["provenance"] for model_id in model_ids},
        image_sizes=image_sizes,
        notes="Built from explicitly supplied real-data evaluation outputs.",
        has_cell_polygons=has_polygons,
        models=models,
        composite_percentiles=composite_percentiles,
    )
    _atomic_write_json(output_dir / "manifest.json", asdict(bm))
    logger.info("wrote %s", output_dir / "manifest.json")
    return output_dir


# ---------------------------------------------------------------------------
# model selection: multi-model cell-table extension
# ---------------------------------------------------------------------------


def parse_model_csv_pair(spec: str) -> tuple[str, Path]:
    """Parse a ``<model_id>=<csv_path>`` CLI argument.

    The CLI exposes ``--add-models v1=preds.csv --add-models miphei_vit=...``
    so the same source CSV can be reused for multiple models — the
    model id selects which subset of columns is pulled. We surface a
    clear ``ValueError`` here so argparse produces a helpful message
    rather than a confusing FileNotFoundError later.
    """
    if "=" not in spec:
        raise ValueError(f"--add-models expects '<model_id>=<csv_path>', got {spec!r}")
    model_id, csv_path = spec.split("=", 1)
    model_id = model_id.strip()
    csv_path = csv_path.strip()
    if not model_id:
        raise ValueError(f"empty model_id in --add-models {spec!r}")
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"--add-models CSV does not exist: {p}")
    return model_id, p


def _model_pred_cols(df: pd.DataFrame, model_id: str) -> dict[int, str]:
    """Return ``{channel: column_name}`` for a given model id.

    The dense v1 model writes bare ``chXX_pred`` (no suffix);
    everything else uses ``chXX_pred_<model_id>``. We support both so
    the same CSV can supply v1 alongside the suffixed models.
    """
    out: dict[int, str] = {}
    for ch in FOCUSED_MARKERS:
        suffixed = f"ch{ch:02d}_pred_{model_id}"
        if suffixed in df.columns:
            out[int(ch)] = suffixed
            continue
        if model_id == "v1":
            bare = f"ch{ch:02d}_pred"
            if bare in df.columns:
                out[int(ch)] = bare
    return out


def _derive_phenotype_scores(df: pd.DataFrame, model_id: str) -> dict[str, np.ndarray]:
    """Derive per-cell phenotype scores from a model's marker predictions.

    The phenotype hierarchy (see :func:`hexif.cell_phenotype.phenotype_targets_from_marker_pos`)
    composes phenotype labels from binary marker positivity. We mirror
    that composition in probability space:

    * Logical AND  → product of probabilities (``p_a * p_b``)
    * Logical OR   → fuzzy OR (``1 - (1 - p_a)(1 - p_b)``)
    * Single marker → pass-through

    This gives a calibrated-ish probability per phenotype per cell
    that mirrors v1.1's signature without requiring the model to
    actually have a phenotype head. For models with incomplete marker
    coverage (MIPHEI's 9-marker panel), missing markers are treated
    as 0 (negative) which is the conservative default — the workbench
    explicitly visualizes the gap rather than silently filling it in.

    Returns ``{phenotype_name: np.ndarray}`` covering every entry of
    :data:`PHENOTYPE_NAMES`. The arrays are float32 and the length
    matches ``len(df)``.
    """
    cols = _model_pred_cols(df, model_id)

    # Marker index in FOCUSED_MARKERS — same ordering as
    # phenotype_targets_from_marker_pos.
    def prob(ch: int) -> np.ndarray:
        col = cols.get(ch)
        if col is None:
            return np.zeros(len(df), dtype=np.float32)
        # Clip to [0, 1] defensively; sigmoid outputs can drift just
        # outside the unit interval after fp32 → fp64 → fp32 round-trips.
        return np.clip(df[col].to_numpy(dtype=np.float32), 0.0, 1.0)

    p_cd45 = prob(3)
    p_cd8 = prob(7)
    p_ki67 = prob(13)
    p_ca9 = prob(16)
    p_cd68 = prob(27)
    p_fap = prob(31)
    p_cd163 = prob(34)
    p_pdl1 = prob(46)
    p_asma = prob(50)
    p_panck = prob(52)

    def fuzzy_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1.0 - (1.0 - a) * (1.0 - b)

    return {
        "immune_cd45": p_cd45,
        "t_cell_cd45_cd8": p_cd45 * p_cd8,
        "tumor_ca9_or_panck": fuzzy_or(p_ca9, p_panck),
        "macrophage_cd68_or_cd163": fuzzy_or(p_cd68, p_cd163),
        "m2_like_cd68_cd163": p_cd68 * p_cd163,
        "caf_fap_or_asma": fuzzy_or(p_fap, p_asma),
        "proliferating_ki67": p_ki67,
        "pdl1_positive": p_pdl1,
        "pdl1_tumor_like": p_pdl1 * fuzzy_or(p_ca9, p_panck),
    }


def merge_model_predictions(
    cells: pd.DataFrame,
    csv_path: Path,
    model_id: str,
    thresholds: CalibratedThresholds,
) -> pd.DataFrame:
    """Merge a model's predictions into an existing ``cells.parquet`` frame.

    Side-effect-free: returns a new DataFrame with the added columns
    (``chXX_pred_<model_id>``, ``phenotype_<name>_score_<model_id>``,
    ``phenotype_<name>_call_<model_id>``). Existing columns under the
    same names are overwritten — re-running ``rebuild_cell_tables``
    with the same args therefore lands the same bytes byte-for-byte,
    which is the idempotency property the spec mandates.

    The join key is ``(basename, cell_id)``. Rows in ``cells`` that
    don't appear in the CSV land NaN, which the downstream API
    surfaces as "no prediction" — that is the right behavior for a
    model that didn't run on a given core.

    Raises :class:`ValueError` if the CSV has none of the expected
    columns for the requested model — the caller is asking for
    predictions that aren't there.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"add-models CSV missing: {csv_path}")
    src = pd.read_csv(csv_path)
    join_keys = ["basename", "cell_id"]
    for k in join_keys:
        if k not in src.columns or k not in cells.columns:
            raise ValueError(f"{k!r} missing from {'CSV' if k not in src.columns else 'bundle'}")

    pred_cols = _model_pred_cols(src, model_id)
    if not pred_cols:
        raise ValueError(
            f"no ch??_pred_{model_id} (or ch??_pred if model_id=='v1') columns in {csv_path}"
        )
    keep_cols = list(join_keys) + list(pred_cols.values())
    sub = src[keep_cols].copy()

    # Rename bare chXX_pred → chXX_pred_v1 so the bundle column names
    # are uniformly suffixed regardless of source CSV convention.
    rename: dict[str, str] = {}
    for ch, col in pred_cols.items():
        canonical = f"ch{ch:02d}_pred_{model_id}"
        if col != canonical:
            rename[col] = canonical
    if rename:
        sub = sub.rename(columns=rename)

    # Derive phenotype scores from the merged marker probabilities.
    # We do this on ``sub`` (not the joined frame) so the side effect
    # is bounded to rows the new model actually predicts. Cells the
    # model doesn't cover land NaN after the left-merge below.
    pheno_scores = _derive_phenotype_scores(sub, model_id)
    for name, arr in pheno_scores.items():
        sub[f"phenotype_{name}_score_{model_id}"] = arr.astype(np.float32)
        thr = thresholds.phenotype_threshold(name)
        sub[f"phenotype_{name}_call_{model_id}"] = (arr >= thr).astype(np.int8)

    # Derived columns are functions of the suffixed pred columns we
    # just renamed, so the sub frame already has every column we
    # intend to publish. Drop any pre-existing duplicates on the cells
    # side first so the merge's suffix='' doesn't double-up.
    to_add = [c for c in sub.columns if c not in join_keys]
    cells_drop = [c for c in to_add if c in cells.columns]
    if cells_drop:
        cells = cells.drop(columns=cells_drop)

    merged = cells.merge(
        sub[join_keys + to_add],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    return merged


def compute_model_metrics(
    cells: pd.DataFrame,
    model_id: str,
) -> dict[str, object]:
    """Compute per-marker / per-phenotype AP for one model from cells.parquet.

    Reads the per-cell predictions (``ch??_pred_<model_id>``,
    ``phenotype_*_score_<model_id>``) and the consensus truth
    (``ch??_pos``, ``phenotype_*_label``) and returns a dict shaped
    like :class:`webapp.schemas.ModelCardResponse`. Missing channels
    (e.g., MIPHEI's 9-marker panel) are omitted from
    ``per_marker_ap``; the macro number averages only the channels
    that the model actually predicts.

    Sklearn is imported lazily so importing the module on a system
    without sklearn (e.g., a slimmed-down inference image) doesn't
    fail; the metric path is opt-in.
    """
    from sklearn.metrics import average_precision_score

    per_marker_ap: dict[str, float] = {}
    for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
        pred_col = f"ch{ch:02d}_pred_{model_id}"
        truth_col = f"ch{ch:02d}_pos"
        if pred_col not in cells.columns or truth_col not in cells.columns:
            continue
        y_score = cells[pred_col].to_numpy(dtype=np.float64)
        y_true = cells[truth_col].to_numpy()
        mask = ~np.isnan(y_score) & ~pd.isna(y_true)
        # Need at least one positive AND one negative for AP to be defined.
        if mask.sum() < 2 or y_true[mask].astype(int).sum() == 0:
            continue
        per_marker_ap[name] = float(
            average_precision_score(y_true[mask].astype(int), y_score[mask])
        )

    per_phenotype_ap: dict[str, float] = {}
    for name in PHENOTYPE_NAMES:
        score_col = f"phenotype_{name}_score_{model_id}"
        label_col = f"phenotype_{name}_label"
        # Backward-compat: v1.1's per-cell scores in legacy bundles are
        # stored bare. The rebuild path now writes suffixed names, but
        # we accept the bare form so a freshly built bundle
        # against an older cells.parquet doesn't regress its v1_1 AP.
        if score_col not in cells.columns:
            if model_id == "v1_1" and f"phenotype_{name}_score" in cells.columns:
                score_col = f"phenotype_{name}_score"
            else:
                continue
        if label_col not in cells.columns:
            continue
        y_score = cells[score_col].to_numpy(dtype=np.float64)
        y_true = cells[label_col].to_numpy()
        mask = ~np.isnan(y_score) & ~pd.isna(y_true)
        if mask.sum() < 2 or y_true[mask].astype(int).sum() == 0:
            continue
        per_phenotype_ap[name] = float(
            average_precision_score(y_true[mask].astype(int), y_score[mask])
        )

    if not per_marker_ap:
        raise ValueError(f"no evaluable marker predictions for model {model_id!r}")
    if not per_phenotype_ap:
        raise ValueError(f"no evaluable phenotype predictions for model {model_id!r}")
    macro_marker = float(np.mean(list(per_marker_ap.values())))
    macro_pheno = float(np.mean(list(per_phenotype_ap.values())))
    return {
        "macro_marker_ap": macro_marker,
        "macro_phenotype_ap": macro_pheno,
        "per_marker_ap": per_marker_ap,
        "per_phenotype_ap": per_phenotype_ap,
    }


def _load_model_metadata(path: str | Path, model_ids: Sequence[str]) -> dict[str, dict]:
    """Load complete, user-supplied identity and provenance for each model."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model metadata does not exist: {source}")
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("model metadata must have schema_version 1")
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ValueError("model metadata 'models' must be a list")
    required = {"id", "name", "backbone", "training_split", "provenance"}
    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"each model metadata entry requires {sorted(required)}")
        if not all(str(row[key]).strip() for key in required - {"provenance"}):
            raise ValueError("model identity fields must be non-empty")
        if not isinstance(row["provenance"], dict) or not row["provenance"]:
            raise ValueError("each model requires non-empty provenance")
        by_id[str(row["id"])] = dict(row)
    missing = set(model_ids) - set(by_id)
    extra = set(by_id) - set(model_ids)
    if missing or extra:
        raise ValueError(
            f"model metadata ids mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return by_id


def _model_entry(
    cells: pd.DataFrame,
    model_id: str,
    csv_path: Path,
    metadata: dict,
) -> dict[str, object]:
    """Construct the manifest.models entry for one model.

    Computes AP from ``cells.parquet`` so the manifest agrees with the
    exact columns served by the API. Display metadata is supplied explicitly.
    """
    metrics = compute_model_metrics(cells, model_id)
    return {
        "id": model_id,
        "name": str(metadata["name"]),
        "backbone": str(metadata["backbone"]),
        "training_split": str(metadata["training_split"]),
        "macro_marker_ap": float(metrics["macro_marker_ap"]),
        "macro_phenotype_ap": float(metrics["macro_phenotype_ap"]),
        "per_marker_ap": dict(metrics["per_marker_ap"]),
        "per_phenotype_ap": dict(metrics["per_phenotype_ap"]),
        "notes": str(metadata.get("notes", f"Metrics computed from {csv_path.name}.")),
    }


def rebuild_cell_tables(
    bundle_dir: str | Path,
    add_models: list[tuple[str, Path]],
    *,
    model_metadata: str | Path,
    default_model_id: str | None = None,
) -> Path:
    """Extend an existing bundle's ``cells.parquet`` with new model columns.

    Behavior:
      * Reads ``cells.parquet`` from the bundle.
      * For each ``(model_id, csv_path)`` pair, merges in the
        ``ch??_pred_<model_id>`` columns + derives per-cell phenotype
        scores/calls (see :func:`_derive_phenotype_scores`).
      * Atomically replaces ``cells.parquet`` (write to
        ``cells.parquet.new`` then ``os.replace``).
      * Updates ``manifest.json:models`` to include every requested
        model, ordered with the default first.

    Idempotent: re-running with the same args overwrites the same
    columns with the same values, producing the same parquet bytes.

    ``default_model_id`` controls ``manifest.models[0]``. When unset,
    the bundle's declared default remains first.

    Returns the path to the updated ``cells.parquet``.
    """
    bundle_dir = Path(bundle_dir)
    cells_path = bundle_dir / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"no cells.parquet in {bundle_dir}; not a hexif bundle")
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {bundle_dir}")

    cells = pd.read_parquet(cells_path)
    manifest = json.loads(manifest_path.read_text())
    threshold_ref = manifest.get("threshold_source")
    if not isinstance(threshold_ref, str) or not threshold_ref:
        raise ValueError("manifest.threshold_source must name a bundle artifact")
    thresholds = load_thresholds(bundle_dir / threshold_ref)

    # Step 1: merge each requested model's predictions.
    seen_ids: list[str] = []
    csv_by_id: dict[str, Path] = {}
    for model_id, csv_path in add_models:
        if model_id in seen_ids:
            logger.warning("duplicate --add-models entry for %s; later one wins", model_id)
        cells = merge_model_predictions(cells, csv_path, model_id, thresholds)
        csv_by_id[model_id] = csv_path
        if model_id not in seen_ids:
            seen_ids.append(model_id)

    declared_models = manifest.get("models")
    if not isinstance(declared_models, list) or not declared_models:
        raise ValueError("manifest.models must be a non-empty list")
    existing_models = []
    for entry in declared_models:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("every manifest.models entry must have a string id")
        existing_models.append(entry["id"])
    keep_existing = [mid for mid in existing_models if mid not in seen_ids]

    # The default model goes first; the rest follow in argument order.
    if default_model_id is None:
        default_model_id = existing_models[0]

    all_ids: list[str] = []
    if default_model_id in seen_ids:
        all_ids.append(default_model_id)
        all_ids += [mid for mid in seen_ids if mid != default_model_id]
        all_ids += [mid for mid in keep_existing if mid != default_model_id]
    else:
        # The default is already represented by columns in ``cells.parquet``.
        all_ids = [default_model_id, *seen_ids, *keep_existing]
        # Dedupe while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for mid in all_ids:
            if mid not in seen:
                deduped.append(mid)
                seen.add(mid)
        all_ids = deduped

    metadata = _load_model_metadata(model_metadata, all_ids)

    # Build every manifest entry; missing columns or metrics are fatal.
    models_entries: list[dict] = []
    for mid in all_ids:
        csv_for_mid = csv_by_id.get(mid)
        if csv_for_mid is None:
            # Existing models are recomputed from the bundle cell table.
            csv_for_mid = Path(manifest.get("sources", {}).get("cell_predictions", "cells.parquet"))
        models_entries.append(_model_entry(cells, mid, csv_for_mid, metadata[mid]))

    # Write parquet atomically. We use parquet's deterministic
    # serialization (sorted columns + a fixed compression codec) so a
    # rebuild produces the same bytes on every run, which is what the
    # spec's idempotency clause is asking for.
    tmp_path = cells_path.with_suffix(".parquet.new")
    # Sort columns alphabetically for reproducibility; the merge order
    # is otherwise dependent on dict insertion ordering across runs.
    cells_sorted = cells[sorted(cells.columns)]
    cells_sorted.to_parquet(tmp_path, index=False, compression="snappy")
    os.replace(tmp_path, cells_path)
    logger.info(
        "wrote %s (%d rows × %d cols, models=%s)",
        cells_path,
        len(cells_sorted),
        len(cells_sorted.columns),
        [m["id"] for m in models_entries],
    )

    # Preserve every existing key and refresh the model metadata.
    manifest["models"] = models_entries
    manifest["model_ids"] = [m["id"] for m in models_entries]
    _atomic_write_json(manifest_path, manifest)
    logger.info("updated %s (models=%s)", manifest_path, manifest["model_ids"])
    return cells_path


# ---------------------------------------------------------------------------
# In-place polygon rebuild
# ---------------------------------------------------------------------------


def rebuild_polygons(
    bundle_dir: str | Path,
    masks_dir: str | Path | None = None,
) -> Path:
    """Add ``cell_polygons.parquet`` to an existing bundle in place.

    Reads ``cells.parquet`` from the bundle, runs the polygon pipeline,
    writes the parquet + skip CSV next to it, and flips
    ``has_cell_polygons`` in the manifest. The rest of the bundle
    (thumbs, spatial summary, core_summary) is left untouched — this is
    the supported polygon rebuild path.

    The mask directory is supplied explicitly or read from
    ``manifest.json:sources.masks_dir``.
    """
    bundle_dir = Path(bundle_dir)
    cells_path = bundle_dir / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"no cells.parquet in {bundle_dir}; not a hexif bundle")

    manifest_path = bundle_dir / "manifest.json"
    resolved_masks_dir: Path
    if masks_dir is not None:
        resolved_masks_dir = Path(masks_dir)
    elif manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        mfrom = m.get("sources", {}).get("masks_dir") or ""
        if not mfrom:
            raise FileNotFoundError("masks_dir not supplied and absent from manifest.sources")
        resolved_masks_dir = Path(mfrom)
    else:
        raise FileNotFoundError("masks_dir not supplied and bundle manifest is missing")

    cells = pd.read_parquet(cells_path, columns=["basename", "cell_id"])
    out_path = bundle_dir / "cell_polygons.parquet"
    build_cell_polygons(resolved_masks_dir, cells, out_path)

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        m["has_cell_polygons"] = True
        m.setdefault("sources", {})["masks_dir"] = str(resolved_masks_dir.resolve())
        _atomic_write_json(manifest_path, m)
        logger.info("flipped has_cell_polygons=True in %s", manifest_path)
    return out_path


# ---------------------------------------------------------------------------
# In-place CODEX composite rebuild (CODEX composite retrofit)
# ---------------------------------------------------------------------------


def rebuild_codex_composite(
    bundle_dir: str | Path,
    *,
    pairs_dir: str | Path | None = None,
    codex_scaler: str | Path | None = None,
    percentile_window: tuple[float, float] = DEFAULT_PERCENTILE_WINDOW,
) -> dict[str, dict]:
    """Render ``codex_composite.png`` + percentile cache for every core.

    Adds a CODEX composite to a bundle that already contains H&E tiles.
    Rerunning produces
    byte-identical PNGs (PIL's libpng encoder is deterministic for a
    given input array) and overwrites the manifest's
    ``composite_percentiles`` block atomically.

    Resolution order for inputs:

    Returns the ``composite_percentiles`` dict (also written to the
    manifest). Cores without a matching ``_CODEX.npy`` are skipped
    with a logged warning and absent from the returned dict.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {bundle_dir}; not a hexif bundle")
    manifest_doc: dict = json.loads(manifest_path.read_text())
    if pairs_dir is None:
        raise FileNotFoundError("pairs_dir is required")
    resolved_pairs_dir = Path(pairs_dir)
    scaler = _resolve_codex_scaler(codex_scaler)
    assert codex_scaler is not None
    scaler_path_used = Path(codex_scaler)

    core_summary_path = bundle_dir / "core_summary.parquet"
    if not core_summary_path.exists():
        raise FileNotFoundError(f"no core_summary.parquet in {bundle_dir}; cannot enumerate cores")
    cores = pd.read_parquet(core_summary_path, columns=["basename"])
    bn_list: list[str] = cores["basename"].tolist()
    thumbs_dir = bundle_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    composite_percentiles: dict[str, dict] = {}
    t0 = time.time()
    for i, base in enumerate(bn_list, 1):
        codex_path = resolved_pairs_dir / f"{base}_CODEX.npy"
        core_thumb_dir = thumbs_dir / base
        try:
            info = _render_codex_composite_png(
                base,
                codex_path,
                scaler,
                core_thumb_dir,
                percentile_window=percentile_window,
            )
        except (ValueError, OSError) as e:
            logger.error("composite render failed for %s: %s", base, e)
            info = None
        if info is None:
            logger.warning("skip %s (missing CODEX npy or render failed)", base)
            continue
        composite_percentiles[base] = info
        if i % 10 == 0 or i == len(bn_list):
            logger.info(
                "rebuild-codex-composite %d/%d  wall=%.1fs",
                i,
                len(bn_list),
                time.time() - t0,
            )

    # Atomic manifest write: rebuild the dict, write to a temp file in
    # the same directory, rename. The rename is atomic on POSIX which
    # means a reader concurrently opening manifest.json will either see
    # the old or new contents but never a half-written file.
    manifest_doc.setdefault("sources", {})["scaler"] = str(scaler_path_used.resolve())
    manifest_doc["composite_percentiles"] = composite_percentiles
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest_doc, indent=2))
    tmp_path.replace(manifest_path)
    logger.info(
        "rebuild-codex-composite: wrote %d entries to %s (wall=%.1fs)",
        len(composite_percentiles),
        manifest_path,
        time.time() - t0,
    )
    return composite_percentiles


def rebuild_codex_marker_thumbs(
    bundle_dir: str | Path,
    *,
    pairs_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, list[int]]:
    """Render ``ch{NN}_codex.png`` for every focused marker on every core.

    The per-marker CODEX truth PNGs are the source for the workbench's
    per-marker CODEX DZI pyramids — they let the user see raw Ki67 (or
    CD8, etc.) staining as a base layer when the marker dropdown is set.
    They are also the inputs the side-panel marker-bars tooltip already
    cites.

    ``overwrite=False`` (default) skips a PNG that already exists; that
    means the function is cheap to re-run after a partial completion.
    ``overwrite=True`` re-renders every PNG (matplotlib output is
    deterministic for a given input but not bit-identical across
    matplotlib versions; we leave the bytes alone unless asked).

    Returns ``{basename: [channel_idx, ...]}`` listing which channels
    were rendered (or already on disk) for each core. Cores without a
    matching ``<basename>_CODEX.npy`` are skipped with a warning.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {bundle_dir}; not a hexif bundle")
    manifest_doc: dict = json.loads(manifest_path.read_text())
    sources = manifest_doc.get("sources", {}) or {}

    resolved_pairs_dir: Path
    if pairs_dir is not None:
        resolved_pairs_dir = Path(pairs_dir)
    elif sources.get("pairs_dir"):
        resolved_pairs_dir = Path(sources["pairs_dir"])
    else:
        raise FileNotFoundError(
            f"pairs_dir not provided and bundle manifest at {manifest_path} has "
            "no sources.pairs_dir; cannot locate CODEX npy files"
        )

    core_summary_path = bundle_dir / "core_summary.parquet"
    if not core_summary_path.exists():
        raise FileNotFoundError(f"no core_summary.parquet in {bundle_dir}; cannot enumerate cores")
    cores = pd.read_parquet(core_summary_path, columns=["basename"])
    bn_list: list[str] = cores["basename"].tolist()
    thumbs_dir = bundle_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, list[int]] = {}
    t0 = time.time()
    for i, base in enumerate(bn_list, 1):
        codex_path = resolved_pairs_dir / f"{base}_CODEX.npy"
        if not codex_path.exists():
            logger.warning("skip %s (no CODEX npy at %s)", base, codex_path)
            continue
        core_thumb_dir = thumbs_dir / base
        core_thumb_dir.mkdir(parents=True, exist_ok=True)

        produced: list[int] = []
        try:
            codex = np.load(codex_path, mmap_mode="r")
        except (ValueError, OSError) as e:
            logger.error("CODEX load failed for %s: %s", base, e)
            continue

        for ch in FOCUSED_MARKERS:
            if ch >= codex.shape[0]:
                logger.warning(
                    "skip %s ch%02d (codex has only %d channels)",
                    base,
                    ch,
                    codex.shape[0],
                )
                continue
            out_png = core_thumb_dir / f"ch{ch:02d}_codex.png"
            if out_png.exists() and not overwrite:
                produced.append(ch)
                continue
            try:
                _render_codex_truth_png_full_res(np.asarray(codex[ch]), out_png)
                produced.append(ch)
            except (ValueError, OSError) as e:
                logger.error("CODEX truth render failed for %s ch%02d: %s", base, ch, e)

        rendered[base] = produced
        if i % 10 == 0 or i == len(bn_list):
            logger.info(
                "rebuild-codex-marker-thumbs %d/%d  wall=%.1fs",
                i,
                len(bn_list),
                time.time() - t0,
            )

    logger.info(
        "rebuild-codex-marker-thumbs: wrote %d cores in %.1fs",
        len(rendered),
        time.time() - t0,
    )
    return rendered


# ---------------------------------------------------------------------------
# Bundle loading (used by webapp + report)
# ---------------------------------------------------------------------------


def load_bundle(bundle_dir: str | Path) -> dict:
    """Read every parquet in a bundle and return them as a dict.

    ``cell_polygons.parquet`` is read if present, and the
    join with ``cells.parquet`` is validated — every cell_id in cells
    must either appear in cell_polygons or in polygons_skipped.csv.
    Silent loss of cells between the two tables would corrupt the
    frontend's polygon-to-cell mapping; we'd rather fail loudly here than
    let the workbench render the wrong outlines.
    """
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    cells = pd.read_parquet(bundle_dir / "cells.parquet")
    cores = pd.read_parquet(bundle_dir / "core_summary.parquet")
    spatial_edges = None
    spatial_summary = None
    if (bundle_dir / "spatial_edges.parquet").exists():
        spatial_edges = pd.read_parquet(bundle_dir / "spatial_edges.parquet")
    if (bundle_dir / "spatial_summary.parquet").exists():
        spatial_summary = pd.read_parquet(bundle_dir / "spatial_summary.parquet")

    cell_polygons: pd.DataFrame | None = None
    polygons_path = bundle_dir / "cell_polygons.parquet"
    if polygons_path.exists():
        cell_polygons = pd.read_parquet(polygons_path)
        _validate_polygon_join(cells, cell_polygons, bundle_dir)

    return {
        "manifest": manifest,
        "cores": cores,
        "cells": cells,
        "spatial_edges": spatial_edges,
        "spatial_summary": spatial_summary,
        "cell_polygons": cell_polygons,
        "dir": bundle_dir,
    }


def _validate_polygon_join(cells: pd.DataFrame, polygons: pd.DataFrame, bundle_dir: Path) -> None:
    """Assert every cells.parquet row joins to a polygon row OR a skip row.

    The contract from the API contract: ``cell_polygons.parquet`` rows + the
    ``polygons_skipped.csv`` rows cover every (basename, cell_id) pair in
    cells.parquet exactly once. Polygons appearing in neither file are
    silent data loss and a hard fail.
    """
    cells_key = cells[["basename", "cell_id"]].copy()
    poly_key = polygons[["basename", "cell_id"]].copy()
    # Duplicate detection: each (basename, cell_id) must appear at most once
    # in cell_polygons, otherwise client-side joins would multiply rows.
    # Use keep=False so the canonical first occurrence is also included in
    # the diagnostic — easier to see what conflicted.
    dup_mask = poly_key.duplicated(keep=False)
    if dup_mask.any():
        dups = poly_key[dup_mask].head(10)
        raise ValueError(
            f"cell_polygons.parquet contains duplicate (basename, cell_id) keys, e.g.\n{dups}"
        )

    merged = cells_key.merge(poly_key, on=["basename", "cell_id"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"][["basename", "cell_id"]]
    if missing.empty:
        return

    # Look up skipped IDs to confirm the absence is intentional.
    skip_path = bundle_dir / "polygons_skipped.csv"
    if not skip_path.exists():
        raise ValueError(
            f"{len(missing)} cells in cells.parquet have no polygon and no entry "
            f"in polygons_skipped.csv (file missing). Bundle is incomplete."
        )
    skipped = pd.read_csv(skip_path)
    if skipped.empty:
        skipped_set: set[tuple[str, int]] = set()
    else:
        skipped_set = set(
            map(tuple, skipped[["basename", "cell_id"]].itertuples(index=False, name=None))
        )
    leaked = [t for t in missing.itertuples(index=False, name=None) if t not in skipped_set]
    if leaked:
        raise ValueError(
            f"{len(leaked)} cells in cells.parquet are missing from both "
            f"cell_polygons.parquet and polygons_skipped.csv (e.g., {leaked[:5]}). "
            f"This indicates a polygon-build bug; rebuild with rebuild-polygons."
        )
