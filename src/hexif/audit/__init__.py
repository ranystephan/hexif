"""HEXIF cohort label-audit module (label audit).

Diagnoses where the current cohort-fit threshold pipeline disagrees with
two alternative cell-typing lenses (per-core Otsu, SPACEc-style Leiden
clustering). Read-only — does not modify training labels.

See ``docs/reproducibility.md`` for the design doc and
``notebooks/audit/cd68_per_core_threshold_audit.py`` for the single-core
prototype this generalizes from.

Public API
==========
Constants
    ``MARKERS_DEFAULT``, ``DEFAULT_ZSCORE_TAU``, ``DEFAULT_LEIDEN_RESOLUTION``,
    ``DEFAULT_LEIDEN_N_NEIGHBORS``, ``DEFAULT_LEIDEN_PCA_N_COMPS``,
    ``DEFAULT_SCALE_MAX_VALUE``, ``DEFAULT_RANDOM_STATE``,
    ``SUSPICIOUS_ALT_FRAC_THRESHOLD``.

Dataclasses
    ``MarkerAuditResult``, ``CoreAuditResult``, ``CohortAuditResult``.

Computation (``hexif.audit.label_audit``)
    ``positivity_per_core_otsu(log1p_intensities) -> (mask, threshold)``
    ``positivity_leiden_zscore(intensity_matrix, marker_names, ...) ->
        (cluster_id_per_cell, cluster_zscore_matrix, positive_cluster_ids_per_marker)``
    ``audit_core(cells_for_core, *, cohort_thresholds, ...) -> CoreAuditResult``
    ``audit_cohort(cells, *, cohort_thresholds, ...) -> CohortAuditResult``
    ``load_cohort_thresholds(path) -> dict[int, dict[str, float]]``

Reporting (``hexif.audit.report``)
    ``write_cohort_audit_csv(result, path) -> Path``
    ``write_cohort_audit_summary(result, path, *, top_n_outliers=20) -> Path``
    ``write_cohort_audit_json(result, path) -> Path``
"""

from __future__ import annotations

from hexif.audit.label_audit import (
    audit_cohort,
    audit_core,
    load_cohort_thresholds,
    positivity_leiden_zscore,
    positivity_per_core_otsu,
)
from hexif.audit.report import (
    write_cohort_audit_csv,
    write_cohort_audit_json,
    write_cohort_audit_summary,
)
from hexif.audit.types import (
    DEFAULT_LEIDEN_N_NEIGHBORS,
    DEFAULT_LEIDEN_PCA_N_COMPS,
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALE_MAX_VALUE,
    DEFAULT_ZSCORE_TAU,
    MARKERS_DEFAULT,
    SUSPICIOUS_ALT_FRAC_THRESHOLD,
    CohortAuditResult,
    CoreAuditResult,
    MarkerAuditResult,
)

__all__ = [
    "DEFAULT_LEIDEN_N_NEIGHBORS",
    "DEFAULT_LEIDEN_PCA_N_COMPS",
    "DEFAULT_LEIDEN_RESOLUTION",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_SCALE_MAX_VALUE",
    "DEFAULT_ZSCORE_TAU",
    "MARKERS_DEFAULT",
    "SUSPICIOUS_ALT_FRAC_THRESHOLD",
    "CohortAuditResult",
    "CoreAuditResult",
    "MarkerAuditResult",
    "audit_cohort",
    "audit_core",
    "load_cohort_thresholds",
    "positivity_leiden_zscore",
    "positivity_per_core_otsu",
    "write_cohort_audit_csv",
    "write_cohort_audit_json",
    "write_cohort_audit_summary",
]
