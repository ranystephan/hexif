"""Frozen dataclasses + column-naming helpers for SPACEc gating v1 (SPACEc gating).

This file is the **shared contract** between the three SPACEc gating
implementation modules and their tests. Treat it as read-only after the
agents start work — changing a field here cascades into all of them.

Algorithms and rationale live in
``docs/reproducibility.md``. Read that file before
changing defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse label audit's panel + hyperparameter defaults — by design, SPACEc gating
# uses the exact same Leiden state we already validated in the cohort
# audit, so the new labels match what the audit saw.
from hexif.audit.types import (
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_LEIDEN_PCA_N_COMPS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    MARKERS_DEFAULT,
)

__all__ = [
    "COL_CLUSTER_ID_SPACEC",
    "DEFAULT_LEIDEN_N_NEIGHBORS",
    "DEFAULT_LEIDEN_PCA_N_COMPS",
    "DEFAULT_LEIDEN_RESOLUTION",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_SCALE_MAX_VALUE",
    "DEFAULT_ZSCORE_TAU",
    "MARKERS_DEFAULT",
    "CohortLabelResult",
    "CoreLabelResult",
    "col_phenotype_spacec",
    "col_pos_spacec",
    "col_zscore_spacec",
]


# --------------------------------------------------------------------------- #
# Column-naming helpers — single source of truth for the new schema so the
# library, the CLI, the trainer, and the tests never drift.
# --------------------------------------------------------------------------- #

COL_CLUSTER_ID_SPACEC: str = "cluster_id_spacec"
"""Per-cell Leiden cluster ID, scoped per core (int32)."""


def col_pos_spacec(channel: int) -> str:
    """Boolean per-cell positivity column name for a marker channel."""
    return f"ch{channel:02d}_pos_spacec"


def col_zscore_spacec(channel: int) -> str:
    """Per-cell scaled (log1p → z-score → clip ±max) intensity for a marker."""
    return f"ch{channel:02d}_zscore_spacec"


def col_phenotype_spacec(phenotype_name: str) -> str:
    """Per-cell phenotype-derived call column name. Mirrors v1.1's
    ``phenotype_<name>_call_v1_1`` convention so workbench consumers can
    pivot on the suffix."""
    return f"phenotype_{phenotype_name}_call_spacec"


# --------------------------------------------------------------------------- #
# Per-core result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CoreLabelResult:
    """SPACEc-style labelling outcome for one core.

    The clustering happens once per core in 12-marker space; this
    bundle captures everything the cohort writer needs to emit new
    columns plus what the workbench needs to visualize cluster identity.
    """

    basename: str
    n_cells: int

    cluster_id: np.ndarray
    """``(n_cells,)`` int32 — Leiden cluster ID per cell. Cluster IDs are
    only meaningful within this core (cluster 0 in core A and cluster 0
    in core B are unrelated)."""

    scaled_intensities: np.ndarray
    """``(n_cells, n_markers)`` float32 — log1p → per-marker z-score →
    clip ±``scale_max_value``. Column order matches ``MARKERS_DEFAULT``.
    Used by the workbench probability slider in Marker focus."""

    marker_pos: np.ndarray
    """``(n_cells, n_markers)`` bool — final per-cell-per-marker positivity
    call from the cluster-then-label rule. Column order matches
    ``MARKERS_DEFAULT``."""

    cluster_zscore_matrix: pd.DataFrame
    """``(n_clusters, n_markers)`` — mean scaled-z per cluster × marker.
    Index = cluster ID as string; columns = marker display names in
    ``MARKERS_DEFAULT`` order. The workbench heatmap reads this."""

    positive_cluster_ids_per_marker: dict[str, tuple[str, ...]]
    """``marker_name -> tuple of cluster_id_str`` — clusters whose mean z
    exceeds ``zscore_tau``."""

    n_clusters: int


# --------------------------------------------------------------------------- #
# Cohort-level result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CohortLabelResult:
    """Top-level result of a SPACEc-gating rebuild over all cores.

    Holds the per-core results plus the parameters used. The augmented
    cell DataFrame is materialized on demand via :meth:`emit_new_columns`
    so this dataclass stays small enough to pickle / pass around.
    """

    cores: tuple[CoreLabelResult, ...]
    markers: tuple[tuple[str, int], ...]
    zscore_tau: float
    leiden_resolution: float
    leiden_n_neighbors: int
    leiden_pca_n_comps: int
    scale_max_value: float
    random_state: int
    timestamp: str
    """ISO 8601 UTC ``YYYY-MM-DDTHH:MM:SS`` — when ``build_spacec_labels`` ran."""
    hexif_git_rev: str
    """Short git rev of the repo at build time, or ``"unknown"`` if not a checkout."""

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def n_cells_total(self) -> int:
        return sum(c.n_cells for c in self.cores)
