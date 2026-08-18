"""Serializers for the label audit label audit (CSV / Markdown / JSON).

These writers consume a :class:`hexif.audit.types.CohortAuditResult` and
produce the three artifacts that ship under
a user-selected audit directory:

- ``cohort_audit.csv``  — one row per (core, marker); columns pinned in the
  spec (``docs/reproducibility.md``).
- ``cohort_audit.md``   — human-readable summary (four sections).
- ``cohort_audit.json`` — provenance + summary stats (no large arrays).

All three writers are atomic: they materialize to ``<path>.tmp`` first and
then ``os.replace`` to the final name so a crashed call never leaves a
half-written file in the published benchmarks tree.

These writers are silent — the CLI driver
(``hexif.cli.main._cmd_audit_labels``) owns user-facing prints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hexif.audit.types import (
    MARKERS_DEFAULT,
    CohortAuditResult,
    CoreAuditResult,
    MarkerAuditResult,
)

# ---------------------------------------------------------------------------
# CSV — one row per (core, marker), columns pinned by the spec.
# ---------------------------------------------------------------------------

# Pinned column order — matches ``docs/reproducibility.md``
# "Outputs / cohort_audit.csv" verbatim. Do not reorder without bumping the
# spec — downstream SPACEc gating tooling reads by name but humans diff by column.
_CSV_COLUMNS: tuple[str, ...] = (
    "basename",
    "marker",
    "channel",
    "n_cells",
    "n_pos_consensus",
    "n_pos_per_core_otsu",
    "n_pos_leiden_zscore",
    "frac_consensus",
    "frac_per_core_otsu",
    "frac_leiden",
    "per_core_otsu_threshold_log1p",
    "cohort_gmm_threshold_log1p",
    "cohort_otsu_threshold_log1p",
    "cohort_quantile_threshold_log1p",
    "max_leiden_cluster_zscore",
    "disagreement_score",
    "suspicious_zero_consensus",
)


def _marker_row(core: CoreAuditResult, marker_name: str, marker_idx: int) -> dict[str, Any]:
    """Flatten one ``MarkerAuditResult`` into the CSV column shape.

    ``marker_idx`` is the marker's position in :data:`MARKERS_DEFAULT` and
    is used only for stable sorting; it is not emitted as a column.
    """
    m: MarkerAuditResult = core.per_marker[marker_name]
    cohort = m.cohort_thresholds_log1p
    return {
        "basename": core.basename,
        "marker": m.name,
        "channel": int(m.channel),
        "n_cells": int(m.n_cells),
        "n_pos_consensus": int(m.n_pos_cohort_consensus),
        "n_pos_per_core_otsu": int(m.n_pos_per_core_otsu),
        "n_pos_leiden_zscore": int(m.n_pos_leiden_zscore),
        "frac_consensus": float(m.frac_cohort_consensus),
        "frac_per_core_otsu": float(m.frac_per_core_otsu),
        "frac_leiden": float(m.frac_leiden_zscore),
        "per_core_otsu_threshold_log1p": float(m.per_core_otsu_threshold_log1p),
        "cohort_gmm_threshold_log1p": float(cohort.get("gmm", float("nan"))),
        "cohort_otsu_threshold_log1p": float(cohort.get("otsu", float("nan"))),
        "cohort_quantile_threshold_log1p": float(cohort.get("quantile", float("nan"))),
        "max_leiden_cluster_zscore": float(m.max_leiden_cluster_zscore),
        "disagreement_score": float(m.disagreement_score),
        "suspicious_zero_consensus": bool(m.suspicious_zero_consensus),
        # private sort key, stripped before write
        "_marker_idx": marker_idx,
    }


def _build_long_frame(result: CohortAuditResult) -> pd.DataFrame:
    """Compose the long (core × marker) DataFrame, sorted reproducibly.

    Sort key: ``(basename, marker_index_in_MARKERS_DEFAULT)`` — so a re-run
    on the same cohort yields the same row order regardless of dict
    iteration order.
    """
    rows: list[dict[str, Any]] = []
    # We honour the cohort's declared marker order if it differs from the
    # default — audit computation may have audited a subset. The marker index used
    # for sorting is the position in that declared tuple.
    markers = result.markers if result.markers else MARKERS_DEFAULT
    for core in result.cores:
        for marker_idx, (marker_name, _channel) in enumerate(markers):
            if marker_name not in core.per_marker:
                # Defensive: a missing marker on a core shouldn't crash the
                # serializer. Skip it — audit computation's audit_cohort is expected
                # to fill every (core, marker) cell, but a partial result is
                # still useful for debugging.
                continue
            rows.append(_marker_row(core, marker_name, marker_idx))
    if not rows:
        return pd.DataFrame(columns=list(_CSV_COLUMNS))
    df = pd.DataFrame(rows)
    df = df.sort_values(["basename", "_marker_idx"], kind="stable").reset_index(drop=True)
    df = df.drop(columns=["_marker_idx"])
    # Pin column order; presence of every column is required by the spec.
    return df[list(_CSV_COLUMNS)]


def _atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a ``<path>.tmp`` swap. Returns resolved final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # If a stale .tmp exists from a previous crashed run, overwrite it.
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path.resolve()


def write_cohort_audit_csv(result: CohortAuditResult, path: Path | str) -> Path:
    """Write the per-(core, marker) audit CSV.

    One row per (core, marker) sorted by ``(basename,
    marker_index_in_MARKERS_DEFAULT)`` for reproducibility. The 17-column
    schema is pinned by ``docs/reproducibility.md``.

    Atomic: writes to ``<path>.tmp`` and then ``os.replace`` so a crashed
    call never leaves a half-written file. Creates parent directories.

    Parameters
    ----------
    result:
        The cohort audit produced by :func:`hexif.audit.audit_cohort`.
    path:
        Output path. ``str`` is accepted as a convenience.

    Returns
    -------
    Path
        The resolved final path written.
    """
    path = Path(path)
    df = _build_long_frame(result)
    # Build the CSV body via DataFrame.to_csv to honour pandas' quoting /
    # NaN handling, then commit atomically.
    csv_text = df.to_csv(index=False)
    return _atomic_write_text(path, csv_text)


# ---------------------------------------------------------------------------
# Markdown — header + per-marker table + top-N outlier tables.
# ---------------------------------------------------------------------------


def _format_float(x: float, digits: int = 3) -> str:
    """Render a float for a Markdown table cell — handles NaN / inf gracefully."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(xf):
        return "n/a"
    if np.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    return f"{xf:.{digits}f}"


def _per_marker_summary(
    result: CohortAuditResult,
) -> dict[str, dict[str, float]]:
    """Aggregate frac-* across cores per marker. Used by both the markdown table
    and the JSON ``per_marker_summary`` block so the two views can't drift.
    """
    markers = result.markers if result.markers else MARKERS_DEFAULT
    out: dict[str, dict[str, float]] = {}
    for marker_name, _channel in markers:
        consensus_fracs: list[float] = []
        otsu_fracs: list[float] = []
        leiden_fracs: list[float] = []
        disagreement: list[float] = []
        for core in result.cores:
            m = core.per_marker.get(marker_name)
            if m is None:
                continue
            consensus_fracs.append(m.frac_cohort_consensus)
            otsu_fracs.append(m.frac_per_core_otsu)
            leiden_fracs.append(m.frac_leiden_zscore)
            disagreement.append(m.disagreement_score)
        if not consensus_fracs:
            continue

        def _stats(xs: list[float]) -> dict[str, float]:
            arr = np.asarray(xs, dtype=np.float64)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
            }

        out[marker_name] = {
            "frac_consensus": _stats(consensus_fracs),
            "frac_per_core_otsu": _stats(otsu_fracs),
            "frac_leiden": _stats(leiden_fracs),
            "disagreement_score": _stats(disagreement),
        }
    return out


def _render_header(result: CohortAuditResult) -> list[str]:
    return [
        "# HEXIF cohort label audit (label audit)",
        "",
        f"- Timestamp (UTC): `{result.timestamp}`",
        f"- Bundle: `{result.bundle_path}`",
        f"- Thresholds: `{result.thresholds_path}`",
        f"- hexif git rev: `{result.hexif_git_rev}`",
        f"- Cores: **{result.n_cores}** ({result.n_cells_total:,} cells total)",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| `zscore_tau` | {_format_float(result.zscore_tau)} |",
        f"| `leiden_resolution` | {_format_float(result.leiden_resolution)} |",
        f"| `leiden_n_neighbors` | {int(result.leiden_n_neighbors)} |",
        f"| `leiden_pca_n_comps` | {int(result.leiden_pca_n_comps)} |",
        f"| `scale_max_value` | {_format_float(result.scale_max_value)} |",
        f"| `random_state` | {int(result.random_state)} |",
        "",
    ]


def _render_per_marker_section(result: CohortAuditResult) -> list[str]:
    summary = _per_marker_summary(result)
    lines = [
        "## Per-marker cohort agreement",
        "",
        (
            "Mean ± std of each lens's per-core positive fraction, plus mean "
            "disagreement score. Disagreement = "
            "`|frac_per_core_otsu − frac_consensus| + |frac_leiden − frac_consensus|`."
        ),
        "",
        "| Marker | frac_consensus | frac_per_core_otsu | frac_leiden | mean disagreement |",
        "|---|---|---|---|---|",
    ]
    markers = result.markers if result.markers else MARKERS_DEFAULT
    for marker_name, _channel in markers:
        s = summary.get(marker_name)
        if s is None:
            lines.append(f"| {marker_name} | n/a | n/a | n/a | n/a |")
            continue
        c = s["frac_consensus"]
        o = s["frac_per_core_otsu"]
        ll = s["frac_leiden"]
        d = s["disagreement_score"]
        lines.append(
            f"| {marker_name} "
            f"| {_format_float(c['mean'])} ± {_format_float(c['std'])} "
            f"| {_format_float(o['mean'])} ± {_format_float(o['std'])} "
            f"| {_format_float(ll['mean'])} ± {_format_float(ll['std'])} "
            f"| {_format_float(d['mean'])} |"
        )
    lines.append("")
    return lines


def _render_suspicious_section(df: pd.DataFrame, top_n: int) -> list[str]:
    suspicious = df[df["suspicious_zero_consensus"]].copy()
    # Within suspicious rows, rank by max(alt_fracs) so the worst comes first.
    suspicious["alt_max"] = suspicious[["frac_per_core_otsu", "frac_leiden"]].max(axis=1)
    suspicious = suspicious.sort_values(
        ["alt_max", "basename", "marker"], ascending=[False, True, True]
    ).head(top_n)
    lines = [
        f"## Top-{top_n} suspicious zero-consensus (core, marker) pairs",
        "",
        (
            "Cores where the current pipeline calls **0** positive cells but at "
            "least one alternative lens calls > 5%. This is the CD68/A-10 "
            "anomaly pattern — these are the cores most likely mis-labelled."
        ),
        "",
        "| Core | Marker | n_cells | frac_per_core_otsu | frac_leiden | per_core_otsu_thr | max_cluster_zscore |",
        "|---|---|---|---|---|---|---|",
    ]
    if suspicious.empty:
        lines.append("| _(none)_ | | | | | | |")
    else:
        for _, row in suspicious.iterrows():
            lines.append(
                f"| {row['basename']} "
                f"| {row['marker']} "
                f"| {int(row['n_cells'])} "
                f"| {_format_float(row['frac_per_core_otsu'])} "
                f"| {_format_float(row['frac_leiden'])} "
                f"| {_format_float(row['per_core_otsu_threshold_log1p'])} "
                f"| {_format_float(row['max_leiden_cluster_zscore'])} |"
            )
    lines.append("")
    return lines


def _render_disagreement_section(df: pd.DataFrame, top_n: int) -> list[str]:
    top = df.sort_values(
        ["disagreement_score", "basename", "marker"], ascending=[False, True, True]
    ).head(top_n)
    lines = [
        f"## Top-{top_n} highest-disagreement (core, marker) pairs",
        "",
        (
            "Ranked by `disagreement_score`. Includes cases where the "
            "alternatives find *more* positives and where they find *fewer* — "
            "both directions are interesting."
        ),
        "",
        "| Core | Marker | frac_consensus | frac_per_core_otsu | frac_leiden | disagreement |",
        "|---|---|---|---|---|---|",
    ]
    if top.empty:
        lines.append("| _(none)_ | | | | | |")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"| {row['basename']} "
                f"| {row['marker']} "
                f"| {_format_float(row['frac_consensus'])} "
                f"| {_format_float(row['frac_per_core_otsu'])} "
                f"| {_format_float(row['frac_leiden'])} "
                f"| {_format_float(row['disagreement_score'])} |"
            )
    lines.append("")
    return lines


def write_cohort_audit_summary(
    result: CohortAuditResult,
    path: Path | str,
    *,
    top_n_outliers: int = 20,
) -> Path:
    """Write the human-readable Markdown summary.

    Four sections in order (per ``docs/reproducibility.md``):

    1. **Header** — timestamp, bundle / thresholds paths, parameter table.
    2. **Per-marker cohort agreement** — mean / std of each lens's per-core
       positive fraction; mean disagreement.
    3. **Top-N suspicious zero-consensus** — the CD68/A-10 pattern.
    4. **Top-N highest-disagreement (marker, core) pairs**.

    Atomic write (``<path>.tmp`` → ``os.replace``). Creates parent dirs.
    """
    path = Path(path)
    df = _build_long_frame(result)
    lines: list[str] = []
    lines.extend(_render_header(result))
    lines.extend(_render_per_marker_section(result))
    lines.extend(_render_suspicious_section(df, top_n_outliers))
    lines.extend(_render_disagreement_section(df, top_n_outliers))
    text = "\n".join(lines).rstrip() + "\n"
    return _atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# JSON — provenance + summary stats (no large arrays).
# ---------------------------------------------------------------------------


def _marker_to_json(m: MarkerAuditResult) -> dict[str, Any]:
    return {
        "name": m.name,
        "channel": int(m.channel),
        "n_cells": int(m.n_cells),
        "n_pos_cohort_consensus": int(m.n_pos_cohort_consensus),
        "n_pos_per_core_otsu": int(m.n_pos_per_core_otsu),
        "n_pos_leiden_zscore": int(m.n_pos_leiden_zscore),
        "per_core_otsu_threshold_log1p": float(m.per_core_otsu_threshold_log1p),
        "cohort_thresholds_log1p": {k: float(v) for k, v in m.cohort_thresholds_log1p.items()},
        "leiden_positive_cluster_ids": list(m.leiden_positive_cluster_ids),
        "max_leiden_cluster_zscore": float(m.max_leiden_cluster_zscore),
        "frac_cohort_consensus": float(m.frac_cohort_consensus),
        "frac_per_core_otsu": float(m.frac_per_core_otsu),
        "frac_leiden_zscore": float(m.frac_leiden_zscore),
        "disagreement_score": float(m.disagreement_score),
        "suspicious_zero_consensus": bool(m.suspicious_zero_consensus),
    }


def _core_to_json(core: CoreAuditResult) -> dict[str, Any]:
    """Serialize a CoreAuditResult without the bulky numpy / pandas blobs.

    Per the spec, we intentionally drop ``leiden_cluster_id_per_cell`` and
    ``leiden_cluster_zscore_matrix`` — they balloon the JSON for no
    reproducibility gain (the full audit can be re-run from inputs).
    """
    return {
        "basename": core.basename,
        "n_cells": int(core.n_cells),
        "leiden_resolution": float(core.leiden_resolution),
        "leiden_n_neighbors": int(core.leiden_n_neighbors),
        "leiden_pca_n_comps": int(core.leiden_pca_n_comps),
        "leiden_n_clusters": int(core.leiden_n_clusters),
        "zscore_tau": float(core.zscore_tau),
        "random_state": int(core.random_state),
        "per_marker": {name: _marker_to_json(m) for name, m in core.per_marker.items()},
    }


def write_cohort_audit_json(result: CohortAuditResult, path: Path | str) -> Path:
    """Write the provenance + summary-stats JSON.

    Includes every scalar field of :class:`CohortAuditResult`, per-core
    summaries (no per-cell arrays), and a ``per_marker_summary`` block
    aggregating each fraction across cores. Designed to be SPACEc gating's
    "which cores are worst" input file.

    Atomic write (``<path>.tmp`` → ``os.replace``). Creates parent dirs.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": result.timestamp,
        "hexif_git_rev": result.hexif_git_rev,
        "bundle_path": str(result.bundle_path),
        "thresholds_path": str(result.thresholds_path),
        "parameters": {
            "zscore_tau": float(result.zscore_tau),
            "leiden_resolution": float(result.leiden_resolution),
            "leiden_n_neighbors": int(result.leiden_n_neighbors),
            "leiden_pca_n_comps": int(result.leiden_pca_n_comps),
            "scale_max_value": float(result.scale_max_value),
            "random_state": int(result.random_state),
        },
        "markers": [{"name": name, "channel": int(channel)} for name, channel in result.markers],
        "n_cores": int(result.n_cores),
        "n_cells_total": int(result.n_cells_total),
        "cores": [_core_to_json(core) for core in result.cores],
        "per_marker_summary": _per_marker_summary(result),
    }
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False, default=str)
    return _atomic_write_text(path, text + "\n")


__all__ = [
    "write_cohort_audit_csv",
    "write_cohort_audit_json",
    "write_cohort_audit_summary",
]
