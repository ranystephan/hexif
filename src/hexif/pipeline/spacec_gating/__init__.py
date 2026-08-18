"""HEXIF SPACEc-style cluster-then-label gating module (SPACEc gating).

Replaces the cohort-fit threshold pipeline that produces
``chXX_pos_consensus`` with cluster-derived per-cell positivity columns
``chXX_pos_spacec`` (plus phenotype, cluster ID, and scaled-intensity
companions). The label-generation algorithm is locked to the same
Leiden state is validated by the label audit.

See ``docs/reproducibility.md`` for the design doc.

The public surface below is implemented across two sibling modules:

  - ``hexif.pipeline.spacec_gating.core`` — pure functions
  - ``hexif.pipeline.spacec_gating.build`` — cohort orchestrator + I/O
  - ``hexif.pipeline.spacec_gating.types`` — frozen dataclasses + constants
"""

from __future__ import annotations

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


def __getattr__(name: str):
    """Lazy re-export from the impl modules.

    Keeps ``import hexif.pipeline.spacec_gating`` cheap and tolerant of a
    partially-built module tree during the optional-dependency import
    (SPACEc gating). Once ``core.py`` and ``build.py`` land the attributes
    resolve normally and ``__all__`` is extended to include them at
    integration time.
    """
    if name in {
        "assign_marker_positivity_from_clusters",
        "cluster_leiden_in_marker_space",
        "label_one_core",
        "normalize_per_core_log1p_zscore",
    }:
        from hexif.pipeline.spacec_gating import core as _impl

        return getattr(_impl, name)
    if name in {
        "build_spacec_labels",
        "emit_new_columns",
        "write_spacec_provenance",
    }:
        from hexif.pipeline.spacec_gating import build as _impl

        return getattr(_impl, name)
    raise AttributeError(f"module 'hexif.pipeline.spacec_gating' has no attribute {name!r}")
