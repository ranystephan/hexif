"""label audit cohort label-audit computation.

Implements the three lenses described in
``docs/reproducibility.md`` (cohort consensus / per-core Otsu /
SPACEc-style Leiden z-score) and the per-core / cohort aggregations that
glue them together.

The single-core prototype this generalizes from lives at
``notebooks/audit/cd68_per_core_threshold_audit.py``; the Leiden cluster
assignments produced here on ``ccRCC_TMA1__A-10`` with default seeds match
the notebook numerically.

Public API (re-exported through ``hexif.audit``):
    * :func:`positivity_per_core_otsu`
    * :func:`positivity_leiden_zscore`
    * :func:`audit_core`
    * :func:`audit_cohort`
    * :func:`load_cohort_thresholds`
"""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from hexif.audit.types import (
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_LEIDEN_PCA_N_COMPS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    MARKERS_DEFAULT,
    CohortAuditResult,
    CoreAuditResult,
    MarkerAuditResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "audit_cohort",
    "audit_core",
    "load_cohort_thresholds",
    "positivity_leiden_zscore",
    "positivity_per_core_otsu",
]


# ---------------------------------------------------------------------------
# Lens 2 — per-core Otsu
# ---------------------------------------------------------------------------


def positivity_per_core_otsu(log1p_intensities: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute per-core Otsu threshold and resulting positivity mask.

    Refits :func:`skimage.filters.threshold_otsu` on the per-core
    ``log1p(mean intensity)`` distribution. Cells whose intensity strictly
    exceeds the threshold are called positive.

    Degenerate distributions are handled defensively (the cohort audit calls
    this on >600 (core, marker) pairs and must never abort the run):

    * Empty input -> ``(np.array([], dtype=bool), +inf)``.
    * Fewer than two unique values (e.g. all zero) -> all-False mask with
      threshold ``+inf``. (Otsu's two-class assumption is violated.)

    Args:
        log1p_intensities: 1-D float array of ``log1p(mean intensity)`` for
            every cell in one core, for one marker.

    Returns:
        Tuple of:
            * ``pos``: 1-D bool array, ``len == len(log1p_intensities)``,
              True iff the cell is called positive.
            * ``threshold``: Otsu cutoff in ``log1p`` space, or ``+inf`` for
              degenerate inputs.
    """
    arr = np.asarray(log1p_intensities, dtype=np.float64).ravel()

    if arr.size == 0:
        return np.zeros(0, dtype=bool), float("inf")

    # Degenerate: not enough variation to define a threshold. Otsu would
    # either crash or return an arbitrary value; we explicitly call the core
    # all-negative.
    if np.unique(arr).size < 2:
        return np.zeros(arr.size, dtype=bool), float("inf")

    threshold = float(threshold_otsu(arr))
    pos = arr > threshold
    return pos, threshold


# ---------------------------------------------------------------------------
# Lens 3 — SPACEc-style Leiden clustering with per-cluster z-score
# ---------------------------------------------------------------------------


def positivity_leiden_zscore(
    intensity_matrix: np.ndarray,
    *,
    marker_names: Sequence[str],
    zscore_tau: float = DEFAULT_ZSCORE_TAU,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_pca_n_comps: int = DEFAULT_LEIDEN_PCA_N_COMPS,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Cluster cells with scanpy's Leiden pipeline and assign per-marker calls.

    Pipeline mirrors :file:`notebooks/audit/cd68_per_core_threshold_audit.py`
    sections 4 (per-core Leiden) so the cohort audit reproduces the
    published CD68/A-10 finding:

    1. Wrap the raw intensity matrix in :class:`anndata.AnnData`.
    2. ``sc.pp.log1p`` -> ``sc.pp.scale(max_value=scale_max_value)``
       (per-marker z-score, clipped at ``±scale_max_value``).
    3. ``sc.tl.pca(n_comps=leiden_pca_n_comps)`` (truncated to
       ``min(n_comps, n_cells - 1, n_markers)`` for tiny inputs).
    4. ``sc.pp.neighbors(n_neighbors=leiden_n_neighbors,
       random_state=random_state)``. If the core has fewer cells than the
       requested neighborhood size, fall back to
       ``n_neighbors = max(2, n_cells - 1)``; if even that is impossible
       (``n_cells < 2``), return a single-cluster degenerate result with no
       positive clusters.
    5. ``sc.tl.leiden(resolution=..., flavor="igraph", n_iterations=2,
       directed=False, random_state=random_state)``.
    6. For each marker ``m``, ``cluster_zscore[c, m]`` = mean over cells in
       cluster ``c`` of the scaled-and-clipped marker value. Cluster ``c``
       is *positive* for marker ``m`` iff ``cluster_zscore[c, m] >
       zscore_tau``.

    Args:
        intensity_matrix: ``(n_cells, n_markers)`` float array of raw mean
            intensities (NOT log-transformed — the function applies
            ``log1p`` internally).
        marker_names: Display names for the columns of ``intensity_matrix``.
        zscore_tau: Cluster-z threshold above which a cluster is called
            positive for a marker.
        leiden_resolution: Leiden resolution parameter.
        leiden_n_neighbors: Target neighborhood size for ``sc.pp.neighbors``.
        leiden_pca_n_comps: Target PCA dimensionality.
        scale_max_value: Per-marker z-score clip range (``±scale_max_value``).
        random_state: Seed threaded into both ``sc.pp.neighbors`` and
            ``sc.tl.leiden``. Does not touch global RNG state.

    Returns:
        Tuple of:
            * ``cluster_id_per_cell``: ``(n_cells,)`` int array (cluster
              indices as in the Leiden output, cast to ``int``).
            * ``cluster_zscore_matrix``: ``(n_clusters, n_markers)``
              DataFrame; index = cluster ID as ``str``; columns =
              ``marker_names``.
            * ``positive_cluster_ids_per_marker``: marker name -> tuple of
              cluster ID strings whose mean scaled-z exceeds ``zscore_tau``.

    Raises:
        ValueError: If ``intensity_matrix.shape[1]`` does not equal
            ``len(marker_names)``.
    """
    matrix = np.asarray(intensity_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"intensity_matrix must be 2-D (n_cells, n_markers); got shape {matrix.shape}"
        )
    names = list(marker_names)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"marker_names length {len(names)} does not match intensity_matrix shape {matrix.shape}"
        )

    n_cells = matrix.shape[0]

    # Degenerate: cannot build a kNN graph on <2 cells. Return a single
    # cluster, no positives anywhere — keeps cohort audit running.
    if n_cells < 2:
        cluster_id_per_cell = np.zeros(n_cells, dtype=np.int64)
        zero_z = pd.DataFrame(
            np.zeros((1, len(names)), dtype=np.float64) if n_cells else np.zeros((0, len(names))),
            index=pd.Index(["0"][: 1 if n_cells else 0], name="leiden"),
            columns=list(names),
        )
        return cluster_id_per_cell, zero_z, {n: tuple() for n in names}

    # scanpy import deferred so module import is cheap on machines without
    # the audit extras installed.
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(X=matrix.copy())
    adata.var_names = list(names)

    # Silence scanpy's chatter — the audit logs through our logger.
    prev_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=scale_max_value)

            # PCA via scanpy's default (arpack) requires
            # ``n_comps < min(n_cells, n_features)`` — *strict* inequality.
            # Clamp accordingly; if the data are too small (1 marker, etc.)
            # skip PCA entirely and let neighbors run on the scaled X.
            pca_cap = min(n_cells, matrix.shape[1]) - 1
            pca_comps = min(leiden_pca_n_comps, pca_cap)
            if pca_comps >= 1:
                sc.tl.pca(adata, n_comps=pca_comps)
                neighbors_kwargs: dict = {}
            else:
                # No PCA — neighbors graph is built on the scaled expression.
                neighbors_kwargs = {"use_rep": "X"}

            # Fall back if the core is too small for the requested kNN.
            n_neighbors = leiden_n_neighbors
            if n_cells <= n_neighbors:
                n_neighbors = max(2, n_cells - 1)
                logger.debug(
                    "positivity_leiden_zscore: n_cells=%d <= n_neighbors=%d; "
                    "falling back to n_neighbors=%d",
                    n_cells,
                    leiden_n_neighbors,
                    n_neighbors,
                )

            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                random_state=random_state,
                **neighbors_kwargs,
            )
            sc.tl.leiden(
                adata,
                resolution=leiden_resolution,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=random_state,
            )
    finally:
        sc.settings.verbosity = prev_verbosity

    # Leiden labels are pandas categoricals of string IDs. Keep both a
    # string-keyed cluster-z matrix (matches MarkerAuditResult.
    # leiden_positive_cluster_ids -> tuple[str, ...]) and an int-cast per-cell
    # array (matches CoreAuditResult.leiden_cluster_id_per_cell -> np.ndarray).
    leiden_str = adata.obs["leiden"].astype(str).to_numpy()
    cluster_id_per_cell = leiden_str.astype(np.int64)

    scaled = pd.DataFrame(np.asarray(adata.X), columns=list(names))
    scaled["__leiden__"] = leiden_str
    cluster_zscore_matrix = scaled.groupby("__leiden__", sort=True).mean()
    cluster_zscore_matrix.index = cluster_zscore_matrix.index.astype(str)
    cluster_zscore_matrix.index.name = "leiden"

    positive_cluster_ids_per_marker: dict[str, tuple[str, ...]] = {}
    for marker in names:
        positive_ids = tuple(
            str(cid) for cid, zscore in cluster_zscore_matrix[marker].items() if zscore > zscore_tau
        )
        positive_cluster_ids_per_marker[marker] = positive_ids

    return cluster_id_per_cell, cluster_zscore_matrix, positive_cluster_ids_per_marker


# ---------------------------------------------------------------------------
# Per-core aggregation across all 12 markers
# ---------------------------------------------------------------------------


def audit_core(
    cells_for_core: pd.DataFrame,
    *,
    cohort_thresholds: dict[int, dict[str, float]],
    markers: Sequence[tuple[str, int]] = MARKERS_DEFAULT,
    zscore_tau: float = DEFAULT_ZSCORE_TAU,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_pca_n_comps: int = DEFAULT_LEIDEN_PCA_N_COMPS,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CoreAuditResult:
    """Apply all three lenses to one core, returning a :class:`CoreAuditResult`.

    Inputs come from the bundle's ``cells.parquet`` filtered to a single
    ``basename``. The Leiden lens runs *once* over the 12-marker matrix; all
    markers piggyback on the same cluster assignment.

    Args:
        cells_for_core: Pandas frame for a single core. Must contain columns
            ``ch{ch:02d}_true`` (raw mean intensity, float) and
            ``ch{ch:02d}_pos_consensus`` (bool / 0-1) for every channel in
            ``markers``. The frame's ``basename`` column (or, if all rows
            share one value, any constant ``basename``) labels the core.
        cohort_thresholds: Output of :func:`load_cohort_thresholds`.
            ``cohort_thresholds[channel]`` is a dict with keys ``"gmm"``,
            ``"otsu"``, ``"quantile"`` (and optionally ``"target_pos"``) in
            ``log1p`` space.
        markers: Tuple of ``(display_name, channel_index)`` pairs. Defaults
            to :data:`MARKERS_DEFAULT`.
        zscore_tau: Lens 3 positivity threshold.
        leiden_resolution: See :func:`positivity_leiden_zscore`.
        leiden_n_neighbors: See :func:`positivity_leiden_zscore`.
        leiden_pca_n_comps: See :func:`positivity_leiden_zscore`.
        scale_max_value: See :func:`positivity_leiden_zscore`.
        random_state: Seed threaded into Leiden.

    Returns:
        :class:`CoreAuditResult` with one :class:`MarkerAuditResult` per
        marker.

    Raises:
        ValueError: If the frame is missing the ``chXX_true`` column for any
            requested marker. Missing ``chXX_pos_consensus`` is *warned* (the
            corresponding count is reported as zero) rather than raised, so
            a single missing column doesn't abort the cohort audit.
    """
    markers = tuple(markers)
    marker_names = [m[0] for m in markers]
    marker_channels = [m[1] for m in markers]

    n_cells = len(cells_for_core)

    # Resolve basename. The bundle's cells.parquet has one row per cell, all
    # with the same ``basename`` for a filtered-to-one-core frame; tolerate
    # an empty frame.
    basename_value: str
    if "basename" in cells_for_core.columns and n_cells:
        unique_basenames = cells_for_core["basename"].unique()
        if len(unique_basenames) != 1:
            raise ValueError(
                f"audit_core expected a single-core frame; got {len(unique_basenames)} "
                f"distinct basenames"
            )
        basename_value = str(unique_basenames[0])
    else:
        basename_value = ""

    # Validate the intensity columns up front — easier to debug a misnamed
    # marker than a downstream KeyError inside scanpy.
    true_cols = [f"ch{ch:02d}_true" for ch in marker_channels]
    missing_true = [c for c in true_cols if c not in cells_for_core.columns]
    if missing_true:
        raise ValueError(f"cells_for_core is missing required intensity columns: {missing_true}")

    intensity_matrix = cells_for_core[true_cols].to_numpy(dtype=np.float32)

    # --- Lens 3 (one Leiden run per core) ---
    cluster_id_per_cell, cluster_zscore_matrix, positive_cluster_ids = positivity_leiden_zscore(
        intensity_matrix,
        marker_names=marker_names,
        zscore_tau=zscore_tau,
        leiden_resolution=leiden_resolution,
        leiden_n_neighbors=leiden_n_neighbors,
        leiden_pca_n_comps=leiden_pca_n_comps,
        scale_max_value=scale_max_value,
        random_state=random_state,
    )
    leiden_str_per_cell = (
        cluster_id_per_cell.astype(str) if cluster_id_per_cell.size else np.zeros(0, dtype="U1")
    )
    n_clusters = int(cluster_zscore_matrix.shape[0]) if n_cells else 0

    per_marker: dict[str, MarkerAuditResult] = {}
    for name, channel in markers:
        true_col = f"ch{channel:02d}_true"
        cons_col = f"ch{channel:02d}_pos_consensus"

        log1p_intensities = (
            np.log1p(cells_for_core[true_col].to_numpy(dtype=np.float64))
            if n_cells
            else np.zeros(0, dtype=np.float64)
        )

        # --- Lens 1 — cohort consensus ---
        if cons_col in cells_for_core.columns:
            n_pos_cohort_consensus = int(cells_for_core[cons_col].astype(bool).to_numpy().sum())
        else:
            logger.warning(
                "audit_core: missing column %s for marker %s on core %s; "
                "reporting n_pos_cohort_consensus=0",
                cons_col,
                name,
                basename_value,
            )
            n_pos_cohort_consensus = 0

        # --- Lens 2 — per-core Otsu ---
        otsu_pos, otsu_thr = positivity_per_core_otsu(log1p_intensities)
        n_pos_per_core_otsu = int(otsu_pos.sum())

        # --- Lens 3 — Leiden cluster z-score ---
        positive_cluster_ids_for_marker = positive_cluster_ids.get(name, tuple())
        if positive_cluster_ids_for_marker and leiden_str_per_cell.size:
            leiden_pos_mask = np.isin(leiden_str_per_cell, positive_cluster_ids_for_marker)
            n_pos_leiden = int(leiden_pos_mask.sum())
        else:
            n_pos_leiden = 0

        max_zscore = (
            float(cluster_zscore_matrix[name].max())
            if n_cells and name in cluster_zscore_matrix.columns
            else float("-inf")
        )

        # Cohort thresholds — defensively fill missing entries with NaN so
        # downstream consumers (CSV, report) see "value present, undefined"
        # rather than crashing on a missing key.
        ch_thr = dict(cohort_thresholds.get(channel, {}))
        cohort_thresholds_log1p: dict[str, float] = {
            "gmm": float(ch_thr.get("gmm", float("nan"))),
            "otsu": float(ch_thr.get("otsu", float("nan"))),
            "quantile": float(ch_thr.get("quantile", float("nan"))),
        }
        if "target_pos" in ch_thr:
            cohort_thresholds_log1p["target_pos"] = float(ch_thr["target_pos"])

        per_marker[name] = MarkerAuditResult(
            name=name,
            channel=channel,
            n_cells=n_cells,
            n_pos_cohort_consensus=n_pos_cohort_consensus,
            n_pos_per_core_otsu=n_pos_per_core_otsu,
            n_pos_leiden_zscore=n_pos_leiden,
            per_core_otsu_threshold_log1p=float(otsu_thr),
            cohort_thresholds_log1p=cohort_thresholds_log1p,
            leiden_positive_cluster_ids=tuple(positive_cluster_ids_for_marker),
            max_leiden_cluster_zscore=max_zscore,
        )

    return CoreAuditResult(
        basename=basename_value,
        n_cells=n_cells,
        per_marker=per_marker,
        leiden_resolution=leiden_resolution,
        leiden_n_neighbors=leiden_n_neighbors,
        leiden_pca_n_comps=leiden_pca_n_comps,
        leiden_n_clusters=n_clusters,
        leiden_cluster_id_per_cell=cluster_id_per_cell,
        leiden_cluster_zscore_matrix=cluster_zscore_matrix,
        zscore_tau=zscore_tau,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Cohort-level driver
# ---------------------------------------------------------------------------


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


def audit_cohort(
    cells: pd.DataFrame,
    *,
    cohort_thresholds: dict[int, dict[str, float]],
    bundle_path: str | Path,
    thresholds_path: str | Path,
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
) -> CohortAuditResult:
    """Run :func:`audit_core` on every core in ``cells`` and bundle results.

    Args:
        cells: Bundle cell table — ``basename`` column plus the
            ``chXX_true`` / ``chXX_pos_consensus`` columns the marker tuple
            references.
        cohort_thresholds: Output of :func:`load_cohort_thresholds`.
        bundle_path: Path to the bundle directory (or ``cells.parquet`` —
            recorded verbatim for provenance only).
        thresholds_path: Path to the thresholds JSON (recorded verbatim).
        markers: Tuple of ``(display_name, channel_index)`` pairs.
        zscore_tau: See :func:`audit_core`.
        leiden_resolution: See :func:`audit_core`.
        leiden_n_neighbors: See :func:`audit_core`.
        leiden_pca_n_comps: See :func:`audit_core`.
        scale_max_value: See :func:`audit_core`.
        random_state: See :func:`audit_core`.
        cores: Optional iterable of core basenames to restrict the audit
            to. Default: every distinct ``basename`` in ``cells``, sorted.
        show_progress: If True, wrap the per-core loop in a ``tqdm`` bar.
        git_repo_path: Directory to invoke ``git rev-parse`` from. Default:
            the repo containing this source file.

    Returns:
        :class:`CohortAuditResult` with one :class:`CoreAuditResult` per
        core (in iteration order).

    Raises:
        ValueError: If ``cells`` is missing the ``basename`` column.
    """
    if "basename" not in cells.columns:
        raise ValueError("audit_cohort requires a 'basename' column in cells")

    if cores is None:
        core_list: list[str] = sorted(cells["basename"].astype(str).unique().tolist())
    else:
        core_list = [str(c) for c in cores]

    iterator: Iterable[str] = core_list
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(core_list, desc="audit_cohort", unit="core")
        except ImportError:
            iterator = core_list

    core_results: list[CoreAuditResult] = []
    for basename in iterator:
        sub = cells[cells["basename"].astype(str) == basename].reset_index(drop=True)
        if sub.empty:
            logger.warning("audit_cohort: skipping core %s (no rows)", basename)
            continue
        result = audit_core(
            sub,
            cohort_thresholds=cohort_thresholds,
            markers=markers,
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

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

    return CohortAuditResult(
        cores=tuple(core_results),
        markers=tuple(markers),
        zscore_tau=zscore_tau,
        leiden_resolution=leiden_resolution,
        leiden_n_neighbors=leiden_n_neighbors,
        leiden_pca_n_comps=leiden_pca_n_comps,
        scale_max_value=scale_max_value,
        random_state=random_state,
        bundle_path=Path(bundle_path),
        thresholds_path=Path(thresholds_path),
        timestamp=timestamp,
        hexif_git_rev=hexif_git_rev,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_cohort_thresholds(path: str | Path) -> dict[int, dict[str, float]]:
    """Load the cohort thresholds JSON, normalizing channel keys to ``int``.

    The on-disk format mirrors the pipeline's
    an explicitly supplied thresholds JSON:
    a top-level mapping ``channel_str -> {"gmm": ..., "otsu": ...,
    "quantile": ..., "target_pos": ...}`` in ``log1p`` space. JSON object
    keys are always strings; this helper casts them to ``int`` so downstream
    code can look up by channel index.

    Args:
        path: Path to the thresholds JSON.

    Returns:
        Mapping ``channel_index -> {"gmm": float, "otsu": float,
        "quantile": float, "target_pos": float}``. Inner dict keys are
        passed through verbatim (only the outer key is normalized).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a top-level key is not convertible to ``int``.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    out: dict[int, dict[str, float]] = {}
    for k, v in raw.items():
        try:
            channel = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"load_cohort_thresholds: non-integer channel key {k!r} in {p}"
            ) from exc
        out[channel] = {kk: float(vv) for kk, vv in v.items()}
    return out
