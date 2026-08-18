"""Frozen dataclasses + module-level constants for the label audit label audit.

This file is the **shared contract** between
``src/hexif/audit/label_audit.py`` (computation), ``src/hexif/audit/report.py``
(serialization), and ``tests/test_label_audit*.py`` (validation). Treat it as
read-only after the agents start work — changing a field here cascades into
all three.

Algorithms and rationale live in
``docs/reproducibility.md``. Read that file before changing
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The 12-marker focused panel — pinned to ``src/hexif/cell_phenotype.py``.
# Kept here as a tuple of ``(display_name, channel_index)`` so the audit
# module is self-contained and importable without pulling in torch.
MARKERS_DEFAULT: tuple[tuple[str, int], ...] = (
    ("DAPI", 0),
    ("CD45", 3),
    ("CD8", 7),
    ("Ki67", 13),
    ("CA9", 16),
    ("Vimentin", 17),
    ("CD68", 27),
    ("FAP", 31),
    ("CD163", 34),
    ("PDL1", 46),
    ("aSMA", 50),
    ("panCK", 52),
)

# Default hyperparameters — pinned to
# ``notebooks/audit/cd68_per_core_threshold_audit.py`` so the cohort
# audit reproduces the published CD68/A-10 finding. Exposed as CLI
# flags so future sweeps can vary them without editing this file.
DEFAULT_ZSCORE_TAU: float = 0.5
DEFAULT_LEIDEN_RESOLUTION: float = 0.6
DEFAULT_LEIDEN_N_NEIGHBORS: int = 20
DEFAULT_LEIDEN_PCA_N_COMPS: int = 10
DEFAULT_SCALE_MAX_VALUE: float = 5.0
DEFAULT_RANDOM_STATE: int = 0

# Outlier ranking threshold — a core is "suspiciously zero-consensus" iff
# the current pipeline calls 0% and an alternative lens calls more than this
# fraction. 5% reflects the smallest immune compartment we'd care about
# missing in a TMA core.
SUSPICIOUS_ALT_FRAC_THRESHOLD: float = 0.05


@dataclass(frozen=True, slots=True)
class MarkerAuditResult:
    """Per-(core, marker) audit outcome.

    All counts are integer cell counts; fractions are recomputed properties.
    Thresholds are in ``log1p(mean intensity)`` space (the space the consensus
    pipeline thresholds were fit in).
    """

    name: str
    channel: int
    n_cells: int
    n_pos_cohort_consensus: int
    n_pos_per_core_otsu: int
    n_pos_leiden_zscore: int
    per_core_otsu_threshold_log1p: float
    cohort_thresholds_log1p: dict[str, float]
    """Keys: ``"gmm"``, ``"otsu"``, ``"quantile"``, optionally ``"target_pos"``."""
    leiden_positive_cluster_ids: tuple[str, ...]
    max_leiden_cluster_zscore: float
    """Highest scaled-z across all Leiden clusters for this marker."""

    # --- derived properties (kept out of the storable fields) -------------

    @property
    def frac_cohort_consensus(self) -> float:
        return self.n_pos_cohort_consensus / self.n_cells if self.n_cells else 0.0

    @property
    def frac_per_core_otsu(self) -> float:
        return self.n_pos_per_core_otsu / self.n_cells if self.n_cells else 0.0

    @property
    def frac_leiden_zscore(self) -> float:
        return self.n_pos_leiden_zscore / self.n_cells if self.n_cells else 0.0

    @property
    def disagreement_score(self) -> float:
        """Sum of absolute gaps from the consensus fraction. The outlier-rank metric.

        Symmetric in sign — a core where the alternatives find *more* positives
        and one where they find *fewer* are both flagged.
        """
        f0 = self.frac_cohort_consensus
        return abs(self.frac_per_core_otsu - f0) + abs(self.frac_leiden_zscore - f0)

    @property
    def suspicious_zero_consensus(self) -> bool:
        """The CD68/A-10 pattern: consensus = 0, alternatives non-trivial."""
        return self.n_pos_cohort_consensus == 0 and (
            self.frac_per_core_otsu > SUSPICIOUS_ALT_FRAC_THRESHOLD
            or self.frac_leiden_zscore > SUSPICIOUS_ALT_FRAC_THRESHOLD
        )


@dataclass(frozen=True, slots=True)
class CoreAuditResult:
    """Audit outcome for one core. Bundles all 12 markers + the shared Leiden state."""

    basename: str
    n_cells: int
    per_marker: dict[str, MarkerAuditResult]
    """Keyed by marker display name (e.g. ``"CD68"``)."""
    leiden_resolution: float
    leiden_n_neighbors: int
    leiden_pca_n_comps: int
    leiden_n_clusters: int
    leiden_cluster_id_per_cell: np.ndarray
    """``(n_cells,)`` int array — cluster index per cell."""
    leiden_cluster_zscore_matrix: pd.DataFrame
    """``(n_clusters, n_markers)`` — mean scaled-z per cluster × marker.

    Index = cluster ID as string (matching ``leiden_positive_cluster_ids``).
    Columns = marker display names in ``MARKERS_DEFAULT`` order.
    """
    zscore_tau: float
    random_state: int


@dataclass(frozen=True, slots=True)
class CohortAuditResult:
    """Top-level audit result over all cores in a cohort."""

    cores: tuple[CoreAuditResult, ...]
    markers: tuple[tuple[str, int], ...]
    zscore_tau: float
    leiden_resolution: float
    leiden_n_neighbors: int
    leiden_pca_n_comps: int
    scale_max_value: float
    random_state: int
    bundle_path: Path
    thresholds_path: Path
    timestamp: str
    """ISO 8601 ``YYYY-MM-DDTHH:MM:SS`` (UTC) — when ``audit_cohort`` ran."""
    hexif_git_rev: str
    """Short git rev of the repo at audit time, or ``"unknown"`` if not a git checkout."""

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def n_cells_total(self) -> int:
        return sum(c.n_cells for c in self.cores)
