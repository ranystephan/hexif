"""Pure-function core of the SPACEc gating v1 (SPACEc gating) pipeline.

Implements the four label-generation primitives the rest of SPACEc gating
composes:

* :func:`normalize_per_core_log1p_zscore` — ``log1p`` + per-marker
  z-score + clip ``±max_value`` (the same transform label audit's audit
  applies inside ``sc.pp.scale``).
* :func:`cluster_leiden_in_marker_space` — scanpy PCA + kNN + Leiden,
  with auto-clamping for small cores so the cohort orchestrator never
  trips on a sparsely-segmented core.
* :func:`assign_marker_positivity_from_clusters` — cluster mean-z >
  ``zscore_tau`` → cluster is positive for that marker → broadcast back
  to every cell.
* :func:`label_one_core` — composes the three over a single-core cell
  table and returns a :class:`CoreLabelResult`.

Algorithm and rationale: ``docs/reproducibility.md``.

This module deliberately **does not import** from
``hexif.audit.label_audit``. SPACEc gating owns its own production code so
the audit and the label-generation pipeline can evolve independently
even though they share the same Leiden math today.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hexif.pipeline.spacec_gating.types import (
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_LEIDEN_PCA_N_COMPS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    MARKERS_DEFAULT,
    CoreLabelResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "assign_marker_positivity_from_clusters",
    "cluster_leiden_in_marker_space",
    "label_one_core",
    "normalize_per_core_log1p_zscore",
]


# --------------------------------------------------------------------------- #
# Step 1 — normalize
# --------------------------------------------------------------------------- #


def normalize_per_core_log1p_zscore(
    intensity_matrix: np.ndarray,
    *,
    marker_names: Sequence[str],
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
) -> np.ndarray:
    """Apply ``log1p`` + per-marker z-score + symmetric clip to a core.

    Mirrors ``sc.pp.log1p`` followed by ``sc.pp.scale(max_value=...)``,
    fitted **per core** (mean and standard deviation are computed over
    the rows of ``intensity_matrix`` only — never across the cohort).

    Constant-variance marker columns (``std == 0``) would otherwise
    produce NaNs through division by zero; this function returns a
    zero column for those markers and emits a warning naming the
    affected marker.

    Args:
        intensity_matrix: ``(n_cells, n_markers)`` array of raw mean
            intensities. Any numeric dtype is accepted; the function
            casts to ``float32``.
        marker_names: Display names matching the columns of
            ``intensity_matrix``; length must equal
            ``intensity_matrix.shape[1]``.
        scale_max_value: Symmetric clip cap applied after z-scoring.
            Values outside ``[-scale_max_value, +scale_max_value]`` are
            clipped to the endpoints.

    Returns:
        ``(n_cells, n_markers)`` float32 array of scaled intensities.

    Raises:
        ValueError: If ``intensity_matrix`` is not 2-D, if
            ``len(marker_names)`` does not match its second axis, or if
            it contains NaN / negative values (``log1p`` is only
            defined on non-negative reals and missing intensities are
            a bug at the caller).
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

    if matrix.size and not np.isfinite(matrix).all():
        raise ValueError(
            "intensity_matrix contains NaN or inf values; raw intensities must be finite"
        )
    if matrix.size and (matrix < 0).any():
        raise ValueError(
            "intensity_matrix contains negative values; raw intensities must be >= 0 (log1p domain)"
        )

    n_cells = matrix.shape[0]
    if n_cells == 0:
        return np.zeros((0, len(names)), dtype=np.float32)

    # Promote to float64 for the mean / std computation — float32
    # accumulates rounding error on the order of 1e-7, which would
    # make truly-constant columns look like they have tiny non-zero
    # variance and yield nonsense z-scores after division.
    logged = np.log1p(matrix.astype(np.float64))

    # Per-marker z-score. We avoid `ddof=1` to match scanpy's
    # ``sc.pp.scale`` (population std, not sample std).
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)

    # A column is "constant" if its standard deviation is at or below
    # the float64 rounding floor of the per-column mean magnitude.
    # This catches both truly-constant inputs (std == 0) and inputs
    # where every cell shares a value that produces a float-rounding
    # ripple after log1p.
    eps = np.finfo(np.float64).eps
    constant_mask = std <= np.maximum(np.abs(mean) * eps * 16.0, eps)

    safe_std = np.where(constant_mask, 1.0, std)
    scaled = (logged - mean) / safe_std

    if constant_mask.any():
        for j, is_const in enumerate(constant_mask):
            if is_const:
                logger.warning(
                    "normalize_per_core_log1p_zscore: marker %r has zero variance in "
                    "this core; returning a zero z-score column",
                    names[j],
                )
        scaled[:, constant_mask] = 0.0

    np.clip(scaled, -scale_max_value, scale_max_value, out=scaled)
    return scaled.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Step 2 — cluster
# --------------------------------------------------------------------------- #


def cluster_leiden_in_marker_space(
    scaled_intensities: np.ndarray,
    *,
    marker_names: Sequence[str],
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_pca_n_comps: int = DEFAULT_LEIDEN_PCA_N_COMPS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster cells in marker space with scanpy's PCA + kNN + Leiden.

    Pipeline (matches the label audit audit so cluster IDs are bit-equal at
    matched seeds):

    1. Wrap the scaled intensity matrix in :class:`anndata.AnnData`.
    2. ``sc.tl.pca(n_comps=leiden_pca_n_comps)`` — clamped to
       ``min(n_cells, n_markers) - 1``. If the data are too small for
       PCA (1 cell, 1 marker, etc.) the neighbors graph is built on
       the scaled expression directly via ``use_rep="X"``.
    3. ``sc.pp.neighbors(n_neighbors=..., random_state=...)``. When
       the core has fewer cells than the requested neighborhood, fall
       back to ``n_neighbors = max(2, n_cells - 1)``.
    4. ``sc.tl.leiden(resolution=..., flavor="igraph",
       n_iterations=2, directed=False, random_state=...)``.

    Degenerate edge: ``n_cells < 2`` short-circuits to a single
    cluster covering every (zero or one) cell, with the corresponding
    one-row z-score matrix taken directly from
    ``scaled_intensities``.

    Args:
        scaled_intensities: ``(n_cells, n_markers)`` float array
            already normalized via
            :func:`normalize_per_core_log1p_zscore`. Any numeric
            dtype is accepted; the function casts to ``float32``.
        marker_names: Display names matching the columns of
            ``scaled_intensities``.
        leiden_resolution: Leiden resolution parameter.
        leiden_n_neighbors: Target neighborhood size for
            ``sc.pp.neighbors``; auto-clamps for small cores.
        leiden_pca_n_comps: Target PCA dimensionality; auto-clamps
            for small cores.
        random_state: Seed threaded into both ``sc.pp.neighbors`` and
            ``sc.tl.leiden``. Output is deterministic given this seed.

    Returns:
        Tuple of:
            * ``cluster_id_per_cell``: ``(n_cells,)`` int32 array of
              per-cell Leiden cluster IDs (cluster indices cast to
              ``int32``).
            * ``cluster_zscore_matrix``: ``(n_clusters, n_markers)``
              DataFrame; index is cluster ID as ``str`` with name
              ``"leiden"``; columns are ``marker_names``.

    Raises:
        ValueError: If ``scaled_intensities`` is not 2-D or if
            ``len(marker_names)`` does not match its second axis.
    """
    matrix = np.asarray(scaled_intensities, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"scaled_intensities must be 2-D (n_cells, n_markers); got shape {matrix.shape}"
        )
    names = list(marker_names)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"marker_names length {len(names)} does not match scaled_intensities shape "
            f"{matrix.shape}"
        )

    n_cells, n_markers = matrix.shape

    # Degenerate: can't build a kNN graph on <2 cells. Single cluster,
    # no positives. Mirrors the label audit audit's fallback so the
    # cohort orchestrator never crashes on a near-empty core.
    if n_cells < 2:
        cluster_id_per_cell = np.zeros(n_cells, dtype=np.int32)
        if n_cells == 1:
            zmatrix = pd.DataFrame(
                matrix.astype(np.float64),
                index=pd.Index(["0"], name="leiden"),
                columns=list(names),
            )
        else:
            zmatrix = pd.DataFrame(
                np.zeros((0, n_markers), dtype=np.float64),
                index=pd.Index([], dtype=object, name="leiden"),
                columns=list(names),
            )
        return cluster_id_per_cell, zmatrix

    # scanpy import deferred so this module is cheap to import on
    # machines without the audit/spacec extras installed.
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(X=matrix.copy())
    adata.var_names = list(names)

    prev_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # PCA via scanpy's default (arpack) requires
            # ``n_comps < min(n_cells, n_features)`` strictly. Clamp
            # and skip PCA entirely if the data are too small.
            pca_cap = min(n_cells, n_markers) - 1
            pca_comps = min(leiden_pca_n_comps, pca_cap)
            if pca_comps >= 1:
                sc.tl.pca(adata, n_comps=pca_comps)
                neighbors_kwargs: dict = {}
            else:
                neighbors_kwargs = {"use_rep": "X"}

            n_neighbors = leiden_n_neighbors
            if n_cells <= n_neighbors:
                n_neighbors = max(2, n_cells - 1)
                logger.debug(
                    "cluster_leiden_in_marker_space: n_cells=%d <= n_neighbors=%d; "
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

    leiden_str = adata.obs["leiden"].astype(str).to_numpy()
    cluster_id_per_cell = leiden_str.astype(np.int32)

    scaled_df = pd.DataFrame(np.asarray(adata.X), columns=list(names))
    scaled_df["__leiden__"] = leiden_str
    cluster_zscore_matrix = scaled_df.groupby("__leiden__", sort=True).mean()
    cluster_zscore_matrix.index = cluster_zscore_matrix.index.astype(str)
    cluster_zscore_matrix.index.name = "leiden"

    return cluster_id_per_cell, cluster_zscore_matrix


# --------------------------------------------------------------------------- #
# Step 3 — cluster → marker positivity
# --------------------------------------------------------------------------- #


def assign_marker_positivity_from_clusters(
    cluster_id_per_cell: np.ndarray,
    cluster_zscore_matrix: pd.DataFrame,
    *,
    marker_names: Sequence[str],
    zscore_tau: float = DEFAULT_ZSCORE_TAU,
) -> tuple[np.ndarray, dict[str, tuple[str, ...]]]:
    """Broadcast cluster-level marker calls down to per-cell positivity.

    For each marker ``m``, a cluster ``c`` is **positive** iff
    ``cluster_zscore_matrix.loc[c, m] > zscore_tau`` (strict
    inequality — a cluster whose mean-z is exactly ``tau`` is *not*
    positive). A cell is positive for marker ``m`` iff its cluster is
    in the positive set for ``m``.

    Args:
        cluster_id_per_cell: ``(n_cells,)`` integer array of per-cell
            cluster IDs as produced by
            :func:`cluster_leiden_in_marker_space`.
        cluster_zscore_matrix: ``(n_clusters, n_markers)`` DataFrame
            of cluster mean-z; index entries are cluster IDs as
            strings.
        marker_names: Display names whose order defines the column
            order of the returned ``marker_pos`` matrix. Must be a
            subset of ``cluster_zscore_matrix.columns``.
        zscore_tau: Cluster mean-z cutoff. Clusters with mean z >
            ``zscore_tau`` for a marker are called positive for that
            marker.

    Returns:
        Tuple of:
            * ``marker_pos``: ``(n_cells, n_markers)`` bool array.
              Column order matches ``marker_names``.
            * ``positive_cluster_ids_per_marker``: mapping
              ``marker_name -> tuple of cluster_id_str`` listing the
              clusters that exceed ``zscore_tau`` for each marker.
              Markers with no positive cluster map to ``()``.

    Raises:
        ValueError: If ``marker_names`` contains an entry not in
            ``cluster_zscore_matrix.columns``.
    """
    names = list(marker_names)
    missing = [n for n in names if n not in cluster_zscore_matrix.columns]
    if missing:
        raise ValueError(
            f"marker_names entries missing from cluster_zscore_matrix columns: {missing}"
        )

    cluster_ids = np.asarray(cluster_id_per_cell)
    n_cells = int(cluster_ids.shape[0])
    n_markers = len(names)

    marker_pos = np.zeros((n_cells, n_markers), dtype=bool)
    positive_cluster_ids_per_marker: dict[str, tuple[str, ...]] = {}

    if cluster_zscore_matrix.shape[0] == 0:
        # No clusters → no positives. Still return the empty mapping
        # so downstream consumers can iterate uniformly.
        for name in names:
            positive_cluster_ids_per_marker[name] = ()
        return marker_pos, positive_cluster_ids_per_marker

    # Per-cell cluster ID as string for set membership against the
    # cluster_zscore_matrix index.
    cluster_ids_str = cluster_ids.astype(str) if n_cells else np.zeros(0, dtype="U1")

    for j, marker in enumerate(names):
        positive_ids = tuple(
            str(cid) for cid, zscore in cluster_zscore_matrix[marker].items() if zscore > zscore_tau
        )
        positive_cluster_ids_per_marker[marker] = positive_ids
        if positive_ids and n_cells:
            marker_pos[:, j] = np.isin(cluster_ids_str, positive_ids)

    return marker_pos, positive_cluster_ids_per_marker


# --------------------------------------------------------------------------- #
# Step 4 — composition: one core end-to-end
# --------------------------------------------------------------------------- #


def label_one_core(
    cells_for_core: pd.DataFrame,
    *,
    markers: Sequence[tuple[str, int]] = MARKERS_DEFAULT,
    zscore_tau: float = DEFAULT_ZSCORE_TAU,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_pca_n_comps: int = DEFAULT_LEIDEN_PCA_N_COMPS,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CoreLabelResult:
    """Run the full SPACEc-style label pipeline on one core's cells.

    Composes :func:`normalize_per_core_log1p_zscore`,
    :func:`cluster_leiden_in_marker_space`, and
    :func:`assign_marker_positivity_from_clusters` into the per-core
    :class:`CoreLabelResult` the cohort orchestrator consumes.

    Args:
        cells_for_core: Cell table filtered to one core. Must contain:

            * a ``basename`` column with a single unique value
              identifying the core, and
            * a ``chXX_true`` column (raw mean intensity) for every
              channel in ``markers``.

            Any other columns are ignored.
        markers: Tuple of ``(display_name, channel_index)`` pairs.
            Defaults to :data:`MARKERS_DEFAULT` (the focused 12-marker
            panel).
        zscore_tau: Cluster mean-z cutoff for positivity (see
            :func:`assign_marker_positivity_from_clusters`).
        leiden_resolution: Leiden resolution parameter.
        leiden_n_neighbors: Target neighborhood size; auto-clamps for
            small cores.
        leiden_pca_n_comps: Target PCA dimensionality; auto-clamps
            for small cores.
        scale_max_value: Symmetric clip cap applied after z-scoring.
        random_state: Seed threaded into PCA / neighbors / Leiden.
            Determinism guarantee: identical inputs and seed produce
            identical :class:`CoreLabelResult` outputs.

    Returns:
        :class:`CoreLabelResult` bundling the cluster assignment,
        scaled intensities, per-cell marker positivity matrix, and
        cluster-level z-score matrix.

    Raises:
        ValueError: If ``cells_for_core`` is empty, if ``basename`` is
            missing or has more than one unique value, or if any
            ``chXX_true`` column is missing.
    """
    markers = tuple(markers)
    marker_names = [m[0] for m in markers]
    marker_channels = [m[1] for m in markers]

    n_cells = len(cells_for_core)
    if n_cells == 0:
        raise ValueError("label_one_core: no cells in core (cells_for_core is empty)")

    if "basename" not in cells_for_core.columns:
        raise ValueError("label_one_core: cells_for_core must contain 'basename' column")
    unique_basenames = pd.unique(cells_for_core["basename"])
    if len(unique_basenames) != 1:
        raise ValueError(
            "label_one_core: cells_for_core must contain exactly one unique 'basename' "
            f"value; got {len(unique_basenames)}"
        )
    basename_value = str(unique_basenames[0])

    true_cols = [f"ch{ch:02d}_true" for ch in marker_channels]
    missing_true = [c for c in true_cols if c not in cells_for_core.columns]
    if missing_true:
        raise ValueError(
            f"label_one_core: cells_for_core is missing required intensity columns: {missing_true}"
        )

    intensity_matrix = cells_for_core[true_cols].to_numpy(dtype=np.float32)

    # Step 1 — normalize. Surfaces NaN / negative values at the
    # caller with a clear message.
    scaled_intensities = normalize_per_core_log1p_zscore(
        intensity_matrix,
        marker_names=marker_names,
        scale_max_value=scale_max_value,
    )

    # Step 2 — cluster.
    cluster_id_per_cell, cluster_zscore_matrix = cluster_leiden_in_marker_space(
        scaled_intensities,
        marker_names=marker_names,
        leiden_resolution=leiden_resolution,
        leiden_n_neighbors=leiden_n_neighbors,
        leiden_pca_n_comps=leiden_pca_n_comps,
        random_state=random_state,
    )

    # Step 3 — broadcast cluster calls to cells.
    marker_pos, positive_cluster_ids_per_marker = assign_marker_positivity_from_clusters(
        cluster_id_per_cell,
        cluster_zscore_matrix,
        marker_names=marker_names,
        zscore_tau=zscore_tau,
    )

    n_clusters = int(cluster_zscore_matrix.shape[0])

    return CoreLabelResult(
        basename=basename_value,
        n_cells=n_cells,
        cluster_id=cluster_id_per_cell.astype(np.int32, copy=False),
        scaled_intensities=scaled_intensities.astype(np.float32, copy=False),
        marker_pos=marker_pos.astype(bool, copy=False),
        cluster_zscore_matrix=cluster_zscore_matrix,
        positive_cluster_ids_per_marker=positive_cluster_ids_per_marker,
        n_clusters=n_clusters,
    )
