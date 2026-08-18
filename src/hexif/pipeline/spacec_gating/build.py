"""Cohort orchestrator + I/O for SPACEc gating v1 (SPACEc gating).

This module is the cohort-level entry point. The per-core compute lives in
:mod:`hexif.pipeline.spacec_gating.core` (``label_one_core``); this module
walks every core in a cells DataFrame, bundles per-core
:class:`CoreLabelResult` objects into a single
:class:`CohortLabelResult`, and provides two adapter functions:

* :func:`emit_new_columns` — appends the new ``chXX_pos_spacec`` /
  ``chXX_zscore_spacec`` / ``cluster_id_spacec`` /
  ``phenotype_<name>_call_spacec`` columns to a cells DataFrame.
* :func:`write_spacec_provenance` — dumps the parameter dict + git rev +
  timestamp + per-marker positive-cell counts to JSON (atomic, no NaN).

Algorithms / rationale: ``docs/reproducibility.md``.
Shared contract: :mod:`hexif.pipeline.spacec_gating.types`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hexif.cell_phenotype import PHENOTYPE_NAMES, phenotype_targets_from_marker_pos
from hexif.pipeline.spacec_gating.types import (
    COL_CLUSTER_ID_SPACEC,
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_LEIDEN_PCA_N_COMPS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    MARKERS_DEFAULT,
    CohortLabelResult,
    CoreLabelResult,
    col_phenotype_spacec,
    col_pos_spacec,
    col_zscore_spacec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_spacec_labels",
    "emit_new_columns",
    "write_spacec_provenance",
]


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #


def _short_git_rev(cwd: Path) -> str:
    """Best-effort short git rev for provenance. Returns ``"unknown"`` on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _utc_iso_timestamp() -> str:
    """Return current UTC time as ``YYYY-MM-DDTHH:MM:SS``."""
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


def _package_version(name: str) -> str:
    """Introspect an installed package's version, ``"unknown"`` on failure."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except PackageNotFoundError:
        return "unknown"
    except Exception:  # provenance must never fail the run
        return "unknown"


# --------------------------------------------------------------------------- #
# Cohort orchestrator
# --------------------------------------------------------------------------- #


def build_spacec_labels(
    cells: pd.DataFrame,
    *,
    markers: Sequence[tuple[str, int]] = MARKERS_DEFAULT,
    zscore_tau: float = DEFAULT_ZSCORE_TAU,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_pca_n_comps: int = DEFAULT_LEIDEN_PCA_N_COMPS,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cores: Iterable[str] | None = None,
    show_progress: bool = True,
    git_repo_path: str | Path | None = None,
) -> CohortLabelResult:
    """Run :func:`label_one_core` on every core in ``cells`` and bundle the results.

    Iteration order is ``sorted(cells["basename"].unique())`` (or the
    explicit ``cores`` iterable if provided), so a re-run on the same
    inputs yields the same tuple ordering.

    Args:
        cells: Cell table — must contain a ``basename`` column plus the
            ``chXX_true`` columns referenced by ``markers``.
        markers: ``(display_name, channel_index)`` tuple; defaults to the
            12-marker focused panel pinned in
            :data:`hexif.audit.types.MARKERS_DEFAULT`.
        zscore_tau: Cluster-mean z-score above which a cluster is called
            positive for a marker.
        leiden_resolution: Leiden resolution parameter.
        leiden_n_neighbors: Target neighborhood size for ``sc.pp.neighbors``.
        leiden_pca_n_comps: Target PCA dimensionality.
        scale_max_value: Per-marker z-score clip range (``±scale_max_value``).
        random_state: Seed threaded into Leiden + neighbors.
        cores: Optional iterable of basenames to restrict the rebuild to.
            Default: every distinct ``basename`` in ``cells``, sorted.
        show_progress: If True, wrap the per-core loop in a ``tqdm.auto`` bar.
        git_repo_path: Directory to invoke ``git rev-parse`` from for the
            provenance hexif_git_rev. Default: the repo containing this
            source file.

    Returns:
        :class:`CohortLabelResult` — one :class:`CoreLabelResult` per core,
        in iteration order, plus the hyperparameters, the UTC timestamp,
        and the hexif git rev.

    Raises:
        ValueError: If ``cells`` is missing the ``basename`` column.
    """
    if "basename" not in cells.columns:
        raise ValueError("build_spacec_labels requires a 'basename' column in cells")

    # Lazy import the core impl so import of this module is cheap even on
    # machines without scanpy/igraph (audit extras).
    from hexif.pipeline.spacec_gating.core import label_one_core

    if cores is None:
        core_list: list[str] = sorted(cells["basename"].astype(str).unique().tolist())
    else:
        core_list = [str(c) for c in cores]

    markers_tuple = tuple((str(name), int(channel)) for name, channel in markers)

    iterator: Iterable[str] = core_list
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(core_list, desc="build_spacec_labels", unit="core")
        except ImportError:
            iterator = core_list

    core_results: list[CoreLabelResult] = []
    for basename in iterator:
        sub = cells[cells["basename"].astype(str) == basename].reset_index(drop=True)
        if sub.empty:
            logger.warning("build_spacec_labels: skipping core %s (no rows)", basename)
            continue
        result = label_one_core(
            sub,
            markers=markers_tuple,
            zscore_tau=zscore_tau,
            leiden_resolution=leiden_resolution,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_pca_n_comps=leiden_pca_n_comps,
            scale_max_value=scale_max_value,
            random_state=random_state,
        )
        core_results.append(result)

    git_root = Path(git_repo_path) if git_repo_path is not None else Path(__file__).resolve().parent
    hexif_git_rev = _short_git_rev(git_root)
    timestamp = _utc_iso_timestamp()

    return CohortLabelResult(
        cores=tuple(core_results),
        markers=markers_tuple,
        zscore_tau=float(zscore_tau),
        leiden_resolution=float(leiden_resolution),
        leiden_n_neighbors=int(leiden_n_neighbors),
        leiden_pca_n_comps=int(leiden_pca_n_comps),
        scale_max_value=float(scale_max_value),
        random_state=int(random_state),
        timestamp=timestamp,
        hexif_git_rev=hexif_git_rev,
    )


# --------------------------------------------------------------------------- #
# DataFrame augmentation
# --------------------------------------------------------------------------- #


def _phenotype_columns_from_marker_pos(
    marker_pos: np.ndarray,
    markers: Sequence[tuple[str, int]],
) -> np.ndarray:
    """Apply :func:`phenotype_targets_from_marker_pos` honouring the marker order.

    ``phenotype_targets_from_marker_pos`` is hard-wired to
    :data:`hexif.cell_phenotype.FOCUSED_MARKERS` order; the SPACEc default
    ``MARKERS_DEFAULT`` is constructed from the same panel in the same
    order, so when the caller used the default we can pass ``marker_pos``
    through verbatim. If the caller supplied a custom marker tuple we
    fall back to building a 12-column boolean matrix in the canonical
    FOCUSED_MARKERS order, filling missing markers with False (which
    makes the phenotype call cleanly fall through to "no signal").
    """
    from hexif.cell_phenotype import FOCUSED_MARKERS

    n_cells = marker_pos.shape[0]
    if n_cells == 0:
        return np.zeros((0, len(PHENOTYPE_NAMES)), dtype=bool)

    if tuple(channel for _name, channel in markers) == tuple(FOCUSED_MARKERS):
        return phenotype_targets_from_marker_pos(marker_pos)

    # Custom marker subset / order: reproject into FOCUSED_MARKERS order.
    canonical = np.zeros((n_cells, len(FOCUSED_MARKERS)), dtype=bool)
    channel_to_col: dict[int, int] = {ch: i for i, (_n, ch) in enumerate(markers)}
    for canonical_idx, ch in enumerate(FOCUSED_MARKERS):
        src_col = channel_to_col.get(int(ch))
        if src_col is not None:
            canonical[:, canonical_idx] = marker_pos[:, src_col].astype(bool)
    return phenotype_targets_from_marker_pos(canonical)


def emit_new_columns(
    result: CohortLabelResult,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Return a *copy* of ``cells`` with the new SPACEc columns appended.

    New columns (one each):

    * :func:`col_pos_spacec` for every marker in ``result.markers`` — ``bool``.
    * :func:`col_zscore_spacec` for every marker — ``float32``.
    * :data:`COL_CLUSTER_ID_SPACEC` — ``int32``, per-core Leiden cluster ID
      (cluster 0 in one core is unrelated to cluster 0 in another).
    * :func:`col_phenotype_spacec` for every phenotype in
      :data:`hexif.cell_phenotype.PHENOTYPE_NAMES` — ``bool``.

    Cells whose ``basename`` has no :class:`CoreLabelResult` (e.g. a core
    that was filtered out of the rebuild) receive sentinel values
    (``NaN`` z-scores, ``False`` booleans, ``-1`` cluster ID) and a
    one-shot warning is logged.

    Refuses to overwrite existing columns: if any of the new column names
    is already present in ``cells``, raise :class:`ValueError` listing the
    conflicts. The caller decides whether to rerun on a clean frame or to
    rename the offending column.

    Args:
        result: Output of :func:`build_spacec_labels`.
        cells: Source cell table. Must contain a ``basename`` column.

    Returns:
        A *new* DataFrame (shallow copy plus assignments) with the columns
        appended. Row order is preserved.

    Raises:
        ValueError: If ``cells`` lacks a ``basename`` column, or if any
            of the new column names collide with existing ones.
    """
    if "basename" not in cells.columns:
        raise ValueError("emit_new_columns requires a 'basename' column in cells")

    markers = result.markers

    # ------------------------------------------------------------------ #
    # 1. Validate that none of the new column names already exist.
    # ------------------------------------------------------------------ #
    new_column_names: list[str] = [COL_CLUSTER_ID_SPACEC]
    for _name, channel in markers:
        new_column_names.append(col_pos_spacec(channel))
        new_column_names.append(col_zscore_spacec(channel))
    for pheno_name in PHENOTYPE_NAMES:
        new_column_names.append(col_phenotype_spacec(pheno_name))

    existing = [c for c in new_column_names if c in cells.columns]
    if existing:
        raise ValueError(
            "emit_new_columns refuses to overwrite existing columns: "
            + ", ".join(existing)
            + " — pass a frame without these columns or drop them first."
        )

    # ------------------------------------------------------------------ #
    # 2. Build a per-basename lookup to the CoreLabelResult.
    # ------------------------------------------------------------------ #
    by_basename: dict[str, CoreLabelResult] = {core.basename: core for core in result.cores}

    n_cells = len(cells)
    n_markers = len(markers)

    # Pre-allocate output arrays in the row order of `cells`.
    pos_arrays = np.zeros((n_cells, n_markers), dtype=bool)
    zscore_arrays = np.full((n_cells, n_markers), np.nan, dtype=np.float32)
    cluster_array = np.full(n_cells, -1, dtype=np.int32)
    phenotype_array = np.zeros((n_cells, len(PHENOTYPE_NAMES)), dtype=bool)

    # Track which row indices are populated; the remainder receive
    # sentinel values plus a single warning.
    populated_mask = np.zeros(n_cells, dtype=bool)

    basenames_series = cells["basename"].astype(str)

    for basename, core_result in by_basename.items():
        row_mask = (basenames_series == basename).to_numpy()
        n_in_core = int(row_mask.sum())
        if n_in_core == 0:
            logger.warning(
                "emit_new_columns: CoreLabelResult for %r has no matching rows in cells",
                basename,
            )
            continue
        if n_in_core != core_result.n_cells:
            raise ValueError(
                f"emit_new_columns: core {basename!r} has {n_in_core} rows in cells "
                f"but the CoreLabelResult was computed on {core_result.n_cells} cells. "
                "The CohortLabelResult and the cells frame are out of sync — rebuild."
            )

        pos_arrays[row_mask, :] = core_result.marker_pos.astype(bool)
        zscore_arrays[row_mask, :] = core_result.scaled_intensities.astype(np.float32)
        cluster_array[row_mask] = core_result.cluster_id.astype(np.int32)
        phenotype_array[row_mask, :] = _phenotype_columns_from_marker_pos(
            core_result.marker_pos.astype(bool),
            markers,
        )
        populated_mask |= row_mask

    n_unpopulated = int((~populated_mask).sum())
    if n_unpopulated:
        # One warning per call — never log per-cell.
        unpopulated_basenames = sorted(set(basenames_series[~populated_mask].unique().tolist()))
        logger.warning(
            "emit_new_columns: %d cells across %d cores (%s) have no CoreLabelResult — "
            "filled with sentinel values (NaN z-scores, False booleans, -1 cluster).",
            n_unpopulated,
            len(unpopulated_basenames),
            ", ".join(unpopulated_basenames[:5]) + (" …" if len(unpopulated_basenames) > 5 else ""),
        )

    # ------------------------------------------------------------------ #
    # 3. Compose the augmented DataFrame.
    # ------------------------------------------------------------------ #
    out = cells.copy()
    for i, (_name, channel) in enumerate(markers):
        out[col_pos_spacec(channel)] = pos_arrays[:, i]
        out[col_zscore_spacec(channel)] = zscore_arrays[:, i]
    out[COL_CLUSTER_ID_SPACEC] = cluster_array
    for i, pheno_name in enumerate(PHENOTYPE_NAMES):
        out[col_phenotype_spacec(pheno_name)] = phenotype_array[:, i]

    return out


# --------------------------------------------------------------------------- #
# Provenance JSON
# --------------------------------------------------------------------------- #


def _per_marker_positive_counts(result: CohortLabelResult) -> dict[str, int]:
    """Sum :class:`CoreLabelResult.marker_pos` across all cores per marker."""
    counts: dict[str, int] = {}
    for marker_idx, (name, _channel) in enumerate(result.markers):
        total = 0
        for core in result.cores:
            if core.marker_pos.shape[1] <= marker_idx:
                continue
            total += int(core.marker_pos[:, marker_idx].sum())
        counts[name] = total
    return counts


def _atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a ``<path>.tmp`` swap. Returns resolved final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path.resolve()


def write_spacec_provenance(
    result: CohortLabelResult,
    path: Path | str,
) -> Path:
    """Write the SPACEc-labels provenance JSON to ``path``.

    Schema (see ``docs/reproducibility.md`` § Outputs):

    .. code-block:: json

        {
          "schema_version": 1,
          "timestamp": "YYYY-MM-DDTHH:MM:SS",
          "hexif_git_rev": "abcdef0",
          "scanpy_version": "1.11.5",
          "igraph_version": "0.11.9",
          "parameters": { ... },
          "markers": [{"name": "DAPI", "channel": 0}, ...],
          "n_cores": 56,
          "n_cells_total": 123553,
          "per_marker_positive_counts": {"DAPI": 12345, ...}
        }

    Atomic: writes to ``<path>.tmp`` then ``os.replace`` so a crashed call
    never leaves a half-written file. ``allow_nan=False`` — provenance
    must round-trip cleanly through any JSON parser.

    Args:
        result: Output of :func:`build_spacec_labels`.
        path: Destination JSON path. Parent directories are created.

    Returns:
        The resolved final path written.
    """
    path = Path(path)
    payload: dict = {
        "schema_version": 1,
        "timestamp": result.timestamp,
        "hexif_git_rev": result.hexif_git_rev,
        "scanpy_version": _package_version("scanpy"),
        "igraph_version": _package_version("igraph"),
        "parameters": {
            "zscore_tau": float(result.zscore_tau),
            "leiden_resolution": float(result.leiden_resolution),
            "leiden_n_neighbors": int(result.leiden_n_neighbors),
            "leiden_pca_n_comps": int(result.leiden_pca_n_comps),
            "scale_max_value": float(result.scale_max_value),
            "random_state": int(result.random_state),
        },
        "markers": [
            {"name": str(name), "channel": int(channel)} for name, channel in result.markers
        ],
        "n_cores": int(result.n_cores),
        "n_cells_total": int(result.n_cells_total),
        "per_marker_positive_counts": _per_marker_positive_counts(result),
    }
    # allow_nan=False — a NaN here would mean either a bad timestamp or
    # a corrupted count, both of which we'd rather catch loud than ship.
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False)
    return _atomic_write_text(path, text + "\n")
