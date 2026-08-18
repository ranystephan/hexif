#!/usr/bin/env python3
"""Calibrate CODEX cell-level marker predictions and score phenotypes.

This script operates on cell tables produced by
``experiments/eval/eval_codex_cell_level.py``.  It deliberately separates:

1. marker positivity labels, derived from train-fitted true-CODEX thresholds;
2. marker calibration, fitted on train-cell predictions only;
3. phenotype labels, deterministic Boolean combinations of marker labels;
4. phenotype scores, deterministic combinations of calibrated marker
   probabilities.

No validation/test labels are used to fit thresholds, calibrators, or operating
points.  This gives a fair readout of whether the existing virtual-CODEX model
contains biologically useful cell-level signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from hexif.utils import CODEX_CHANNEL_NAMES

PHENOTYPES: dict[str, dict[str, Any]] = {
    "immune_cd45": {
        "label": lambda p: p[3],
        "score": lambda q: q[3],
        "description": "CD45+ immune-like cells",
    },
    "t_cell_cd45_cd8": {
        "label": lambda p: p[3] & p[7],
        "score": lambda q: q[3] * q[7],
        "description": "CD45+ CD8+ T-cell-like cells",
    },
    "tumor_ca9_or_panck": {
        "label": lambda p: p[16] | p[52],
        "score": lambda q: _prob_or(q[16], q[52]),
        "description": "CA9+ or panCK+ tumor/epithelial-like cells",
    },
    "macrophage_cd68_or_cd163": {
        "label": lambda p: p[27] | p[34],
        "score": lambda q: _prob_or(q[27], q[34]),
        "description": "CD68+ or CD163+ macrophage-like cells",
    },
    "m2_like_cd68_cd163": {
        "label": lambda p: p[27] & p[34],
        "score": lambda q: q[27] * q[34],
        "description": "CD68+ CD163+ macrophage-like cells",
    },
    "caf_fap_or_asma": {
        "label": lambda p: p[31] | p[50],
        "score": lambda q: _prob_or(q[31], q[50]),
        "description": "FAP+ or aSMA+ stromal/CAF-like cells",
    },
    "proliferating_ki67": {
        "label": lambda p: p[13],
        "score": lambda q: q[13],
        "description": "Ki67+ proliferating cells",
    },
    "pdl1_positive": {
        "label": lambda p: p[46],
        "score": lambda q: q[46],
        "description": "PDL1+ cells",
    },
    "pdl1_tumor_like": {
        "label": lambda p: p[46] & (p[16] | p[52]),
        "score": lambda q: q[46] * _prob_or(q[16], q[52]),
        "description": "PDL1+ CA9/panCK+ tumor-like cells",
    },
}


def _prob_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - a) * (1.0 - b)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate marker predictions and score CODEX phenotypes"
    )
    p.add_argument("--train_cell_table", required=True)
    p.add_argument("--eval_cell_table", required=True)
    p.add_argument("--thresholds_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--marker_channels", required=True)
    p.add_argument(
        "--phenotypes",
        default=",".join(PHENOTYPES),
        help="Comma-separated phenotype names or 'all'",
    )
    p.add_argument(
        "--max_train_cells",
        type=int,
        default=300_000,
        help="Subsample train cells for logistic fitting; 0 uses all",
    )
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--write_eval_table", action="store_true")
    p.add_argument(
        "--neighborhood_k",
        default="",
        help="Comma-separated list of neighborhood sizes (e.g. '8,16,32'). "
        "Empty disables neighborhood features (D3 baseline).",
    )
    p.add_argument(
        "--neighborhood_features",
        default="mean,density,posfrac",
        help="Which neighborhood aggregates to compute. Comma-separated subset of "
        "'mean','std','density','posfrac'. Default keeps the diagnostic compact.",
    )
    p.add_argument(
        "--shuffle_within_core",
        action="store_true",
        help="D3 control: permute chXX_pred values across cells within each core "
        "before computing neighborhood features. Tests whether spatial structure "
        "carries information beyond marginal feature distributions.",
    )
    p.add_argument(
        "--core_mean_features",
        action="store_true",
        help="D3 tighter control: add per-core mean of chXX_pred as a feature. "
        "If most neighborhood lift comes from this single per-core feature, "
        "the neighborhood is leaking core-identity, not spatial structure.",
    )
    return p.parse_args()


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_names(s: str) -> list[str]:
    names = (
        list(PHENOTYPES)
        if s.strip().lower() == "all"
        else [x.strip() for x in s.split(",") if x.strip()]
    )
    missing = [x for x in names if x not in PHENOTYPES]
    if missing:
        raise ValueError(f"Unknown phenotypes {missing}. Valid: {sorted(PHENOTYPES)}")
    return names


def _load_thresholds(path: Path, channels: list[int]) -> dict[int, float]:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict) and "thresholds" in rows:
        rows = rows["thresholds"]
    out = {int(row["channel"]): float(row["threshold"]) for row in rows}
    missing = [ch for ch in channels if ch not in out]
    if missing:
        raise ValueError(f"Threshold file is missing channels {missing}")
    return out


def _ensure_labels(
    df: pd.DataFrame, channels: list[int], thresholds: dict[int, float]
) -> pd.DataFrame:
    out = df.copy()
    for ch in channels:
        pos_col = f"ch{ch:02d}_pos"
        true_col = f"ch{ch:02d}_true"
        if pos_col not in out.columns:
            out[pos_col] = out[true_col].to_numpy(dtype=np.float32) > thresholds[ch]
        out[pos_col] = out[pos_col].astype(bool)
    return out


def _safe_metric(fn: Callable, y_true: np.ndarray, y_score_or_pred: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(fn(y_true, y_score_or_pred))
    except Exception:
        return float("nan")


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1))
    tau = float(thresholds[min(idx, len(thresholds) - 1)]) if len(thresholds) else 0.5
    return tau, float(f1[idx])


def _fit_train_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    tau, _ = _best_f1_threshold(y_true, score)
    if not np.isfinite(tau):
        raise ValueError("cannot fit an operating threshold without both label classes")
    return float(tau)


def _confidence_from_ap(value: float) -> str:
    """Map measured average precision to the documented display stratum."""
    if not np.isfinite(value):
        raise ValueError("cannot assign confidence from a non-finite AP")
    if value >= 0.70:
        return "strong"
    if value >= 0.55:
        return "moderate"
    return "weak"


def _sample_train(df: pd.DataFrame, max_cells: int, random_state: int) -> pd.DataFrame:
    if max_cells <= 0 or len(df) <= max_cells:
        return df
    return df.sample(n=int(max_cells), random_state=int(random_state))


def _parse_k_list(s: str) -> list[int]:
    if not s.strip():
        return []
    out = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if any(k < 1 for k in out):
        raise ValueError("neighborhood_k must be positive")
    return out


def _shuffle_within_core(
    df: pd.DataFrame, channels: list[int], thresholds: dict[int, float], random_state: int
) -> pd.DataFrame:
    """D3 control: permute chXX_pred and chXX_pos within each core.

    Centroids stay fixed; neighborhood relationships are preserved.  Only the
    pred values attached to each (cell_id, x, y) row are scrambled inside the
    core, so any neighborhood lift that depends on spatial co-occurrence of
    marker values disappears here while marginal distributions per core are
    preserved.
    """
    rng = np.random.default_rng(int(random_state))
    out = df.copy()
    pred_cols = [f"ch{ch:02d}_pred" for ch in channels]
    pos_cols = [f"ch{ch:02d}_pos" for ch in channels]
    available_pos = [c for c in pos_cols if c in out.columns]
    target_cols = pred_cols + available_pos
    for _, group in out.groupby("basename", sort=False):
        idx = group.index.to_numpy()
        if idx.size <= 1:
            continue
        perm = rng.permutation(idx)
        out.loc[idx, target_cols] = out.loc[perm, target_cols].to_numpy()
    return out


def _compute_core_mean_features(
    df: pd.DataFrame,
    channels: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Add per-core mean of chXX_pred as 12 extra features.

    Sanity / leakage probe.  If this captures most of what k-NN neighborhood
    features add, the lift is core-identity, not spatial structure.
    """
    out = df.copy()
    new_cols: list[str] = []
    for ch in channels:
        col = f"core_mean_ch{ch:02d}_pred"
        out[col] = out.groupby("basename")[f"ch{ch:02d}_pred"].transform("mean").astype(np.float32)
        new_cols.append(col)
    return out, new_cols


def _compute_neighborhood_features(
    df: pd.DataFrame,
    channels: list[int],
    k_list: list[int],
    feature_set: set[str],
    thresholds: dict[int, float],
) -> tuple[pd.DataFrame, list[str]]:
    """Add k-NN aggregate features per cell, grouped by core.

    Coordinates are taken from ``centroid_y``/``centroid_x`` in pixel units.
    For each k in ``k_list``, the cell itself is excluded from its neighborhood.
    Cores with fewer than k+1 cells skip that k (those cells get NaN, filled
    with the per-core mean as a graceful fallback).
    """
    if not k_list:
        return df, []
    out = df.copy()
    pred_cols = [f"ch{ch:02d}_pred" for ch in channels]
    pred_arr_full = out[pred_cols].to_numpy(dtype=np.float32)
    new_cols: list[str] = []

    feature_buffers: dict[str, np.ndarray] = {}
    for k in k_list:
        if "mean" in feature_set:
            for ch in channels:
                col = f"nbr{k}_ch{ch:02d}_pred_mean"
                feature_buffers[col] = np.full(len(out), np.nan, dtype=np.float32)
                new_cols.append(col)
        if "std" in feature_set:
            for ch in channels:
                col = f"nbr{k}_ch{ch:02d}_pred_std"
                feature_buffers[col] = np.full(len(out), np.nan, dtype=np.float32)
                new_cols.append(col)
        if "posfrac" in feature_set:
            for ch in channels:
                col = f"nbr{k}_ch{ch:02d}_posfrac"
                feature_buffers[col] = np.full(len(out), np.nan, dtype=np.float32)
                new_cols.append(col)
        if "density" in feature_set:
            col = f"nbr{k}_density"
            feature_buffers[col] = np.full(len(out), np.nan, dtype=np.float32)
            new_cols.append(col)

    for _, group in out.groupby("basename", sort=False):
        idx = group.index.to_numpy()
        if idx.size < 2:
            continue
        coords = group[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
        tree = cKDTree(coords)
        local_pred = pred_arr_full[idx]
        for k in k_list:
            k_eff = min(int(k), len(group) - 1)
            if k_eff < 1:
                continue
            dists, neigh_idx = tree.query(coords, k=k_eff + 1)
            if k_eff == 1:
                dists = dists[:, None]
                neigh_idx = neigh_idx[:, None]
            dists = dists[:, 1:]
            neigh_idx = neigh_idx[:, 1:]
            neighbor_pred = local_pred[neigh_idx]
            if "mean" in feature_set:
                m = neighbor_pred.mean(axis=1)
                for ci, ch in enumerate(channels):
                    feature_buffers[f"nbr{k}_ch{ch:02d}_pred_mean"][idx] = m[:, ci]
            if "std" in feature_set:
                s = neighbor_pred.std(axis=1)
                for ci, ch in enumerate(channels):
                    feature_buffers[f"nbr{k}_ch{ch:02d}_pred_std"][idx] = s[:, ci]
            if "posfrac" in feature_set:
                taus = np.array([thresholds[ch] for ch in channels], dtype=np.float32)
                pos = (neighbor_pred > taus).astype(np.float32)
                pf = pos.mean(axis=1)
                for ci, ch in enumerate(channels):
                    feature_buffers[f"nbr{k}_ch{ch:02d}_posfrac"][idx] = pf[:, ci]
            if "density" in feature_set:
                feature_buffers[f"nbr{k}_density"][idx] = (1.0 / (dists + 1.0)).mean(axis=1)

    for col, buf in feature_buffers.items():
        if np.isnan(buf).any():
            mean_val = float(np.nanmean(buf)) if np.isfinite(buf).any() else 0.0
            buf = np.where(np.isnan(buf), mean_val, buf)
        out[col] = buf

    return out, new_cols


def _calibrate_markers(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    channels: list[int],
    max_train_cells: int,
    random_state: int,
    extra_pred_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    train_fit = _sample_train(train_df, max_train_cells, random_state)
    pred_cols = [f"ch{ch:02d}_pred" for ch in channels]
    if extra_pred_cols:
        pred_cols = pred_cols + list(extra_pred_cols)
    pos_cols = [f"ch{ch:02d}_pos" for ch in channels]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_fit[pred_cols].to_numpy(dtype=np.float32))
    y_train = train_fit[pos_cols].to_numpy(dtype=bool)
    x_train_all = scaler.transform(train_df[pred_cols].to_numpy(dtype=np.float32))
    x_eval = scaler.transform(eval_df[pred_cols].to_numpy(dtype=np.float32))
    y_eval = eval_df[pos_cols].to_numpy(dtype=bool)

    clf = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(random_state))
    )
    clf.fit(x_train, y_train)
    train_probs = np.asarray(clf.predict_proba(x_train_all), dtype=np.float32)
    probs = np.asarray(clf.predict_proba(x_eval), dtype=np.float32)

    train_out = train_df.copy()
    out = eval_df.copy()
    rows: list[dict[str, Any]] = []
    for i, ch in enumerate(channels):
        name = CODEX_CHANNEL_NAMES[ch] if ch < len(CODEX_CHANNEL_NAMES) else f"ch{ch}"
        y = y_eval[:, i]
        raw_score = eval_df[f"ch{ch:02d}_pred"].to_numpy(dtype=np.float32)
        prob = probs[:, i]
        train_out[f"ch{ch:02d}_prob_logreg"] = train_probs[:, i]
        train_tau_raw = _fit_train_f1_threshold(
            train_fit[f"ch{ch:02d}_pos"].to_numpy(dtype=bool),
            train_fit[f"ch{ch:02d}_pred"].to_numpy(dtype=np.float32),
        )
        pred_train_tau = raw_score > train_tau_raw
        pred_logreg = prob > 0.5
        best_tau_prob, best_f1_prob = _best_f1_threshold(y, prob)
        out[f"ch{ch:02d}_prob_logreg"] = prob
        rows.append(
            {
                "channel": int(ch),
                "name": name,
                "n_eval_cells": len(y),
                "positive_fraction": float(np.mean(y)) if len(y) else float("nan"),
                "auc_raw": _safe_metric(roc_auc_score, y, raw_score),
                "ap_raw": _safe_metric(average_precision_score, y, raw_score),
                "auc_logreg": _safe_metric(roc_auc_score, y, prob),
                "ap_logreg": _safe_metric(average_precision_score, y, prob),
                "train_best_raw_threshold": train_tau_raw,
                "balanced_acc_train_best_raw_threshold": _safe_metric(
                    balanced_accuracy_score, y, pred_train_tau
                ),
                "f1_train_best_raw_threshold": _safe_metric(f1_score, y, pred_train_tau),
                "balanced_acc_logreg_0p5": _safe_metric(balanced_accuracy_score, y, pred_logreg),
                "f1_logreg_0p5": _safe_metric(f1_score, y, pred_logreg),
                "best_possible_f1_logreg": best_f1_prob,
                "best_logreg_threshold_eval_only": best_tau_prob,
            }
        )

    meta = {
        "train_cells_input": len(train_df),
        "train_cells_fit": len(train_fit),
        "eval_cells": len(eval_df),
        "pred_cols": pred_cols,
        "pos_cols": pos_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return train_out, out, rows, meta


def _marker_bool_dict(df: pd.DataFrame, channels: list[int]) -> dict[int, np.ndarray]:
    return {ch: df[f"ch{ch:02d}_pos"].to_numpy(dtype=bool) for ch in channels}


def _marker_prob_dict(df: pd.DataFrame, channels: list[int]) -> dict[int, np.ndarray]:
    return {ch: df[f"ch{ch:02d}_prob_logreg"].to_numpy(dtype=np.float32) for ch in channels}


def _score_phenotypes(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    channels: list[int],
    phenotype_names: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_pos = _marker_bool_dict(train_df, channels)
    eval_pos = _marker_bool_dict(eval_df, channels)
    eval_prob = _marker_prob_dict(eval_df, channels)
    out = eval_df.copy()
    rows: list[dict[str, Any]] = []
    for name in phenotype_names:
        spec = PHENOTYPES[name]
        train_y = np.asarray(spec["label"](train_pos), dtype=bool)
        y = np.asarray(spec["label"](eval_pos), dtype=bool)
        score = np.asarray(spec["score"](eval_prob), dtype=np.float32)
        tau = _fit_train_f1_threshold(
            train_y,
            np.asarray(spec["score"](_marker_prob_dict(train_df, channels)), dtype=np.float32),
        )
        pred = score > tau
        best_tau, best_f1 = _best_f1_threshold(y, score)
        out[f"phenotype_{name}_label"] = y
        out[f"phenotype_{name}_score"] = score
        rows.append(
            {
                "phenotype": name,
                "description": str(spec["description"]),
                "n_eval_cells": len(y),
                "positive_fraction": float(np.mean(y)) if len(y) else float("nan"),
                "auc": _safe_metric(roc_auc_score, y, score),
                "ap": _safe_metric(average_precision_score, y, score),
                "train_best_threshold": tau,
                "balanced_acc_train_threshold": _safe_metric(balanced_accuracy_score, y, pred),
                "f1_train_threshold": _safe_metric(f1_score, y, pred),
                "best_possible_f1": best_f1,
                "best_threshold_eval_only": best_tau,
            }
        )
    return out, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    channels = _parse_ints(args.marker_channels)
    phenotype_names = _parse_names(args.phenotypes)
    thresholds = _load_thresholds(Path(args.thresholds_json), channels)

    train_df = _ensure_labels(pd.read_csv(args.train_cell_table), channels, thresholds)
    eval_df = _ensure_labels(pd.read_csv(args.eval_cell_table), channels, thresholds)

    k_list = _parse_k_list(args.neighborhood_k)
    feature_set = {x.strip() for x in args.neighborhood_features.split(",") if x.strip()}
    extra_pred_cols: list[str] = []
    neighborhood_meta: dict[str, Any] = {
        "k_list": k_list,
        "feature_set": sorted(feature_set),
        "shuffle_within_core": bool(args.shuffle_within_core),
        "core_mean_features": bool(args.core_mean_features),
    }
    if args.core_mean_features:
        train_df, train_cm = _compute_core_mean_features(train_df, channels)
        eval_df, eval_cm = _compute_core_mean_features(eval_df, channels)
        if train_cm != eval_cm:
            raise RuntimeError("core-mean feature columns diverged train vs eval")
        extra_pred_cols.extend(train_cm)
        neighborhood_meta["core_mean_columns"] = train_cm
    if k_list:
        train_for_nbr = train_df
        eval_for_nbr = eval_df
        if args.shuffle_within_core:
            train_for_nbr = _shuffle_within_core(
                train_df, channels, thresholds, int(args.random_state)
            )
            eval_for_nbr = _shuffle_within_core(
                eval_df, channels, thresholds, int(args.random_state) + 1
            )
        train_df, train_new = _compute_neighborhood_features(
            train_for_nbr, channels, k_list, feature_set, thresholds
        )
        eval_df, eval_new = _compute_neighborhood_features(
            eval_for_nbr, channels, k_list, feature_set, thresholds
        )
        if train_new != eval_new:
            raise RuntimeError(
                f"Neighborhood feature columns diverged between train/eval: "
                f"{set(train_new) ^ set(eval_new)}"
            )
        extra_pred_cols.extend(train_new)
        neighborhood_meta["new_columns"] = train_new
        neighborhood_meta["train_cells_after_nbr"] = len(train_df)
        neighborhood_meta["eval_cells_after_nbr"] = len(eval_df)

    calibrated_train, calibrated_eval, marker_rows, calibration_meta = _calibrate_markers(
        train_df=train_df,
        eval_df=eval_df,
        channels=channels,
        max_train_cells=int(args.max_train_cells),
        random_state=int(args.random_state),
        extra_pred_cols=extra_pred_cols,
    )
    phenotype_eval, phenotype_rows = _score_phenotypes(
        train_df=calibrated_train,
        eval_df=calibrated_eval,
        channels=channels,
        phenotype_names=phenotype_names,
    )

    _write_csv(outdir / "marker_calibration_metrics.csv", marker_rows)
    _write_csv(outdir / "phenotype_metrics.csv", phenotype_rows)
    threshold_artifact = {
        "schema_version": 1,
        "marker_thresholds": {
            str(row["channel"]): float(row["train_best_raw_threshold"]) for row in marker_rows
        },
        "phenotype_thresholds": {
            str(row["phenotype"]): float(row["train_best_threshold"]) for row in phenotype_rows
        },
        "marker_confidence": {
            str(row["name"]): _confidence_from_ap(float(row["ap_raw"])) for row in marker_rows
        },
        "phenotype_confidence": {
            str(row["phenotype"]): _confidence_from_ap(float(row["ap"])) for row in phenotype_rows
        },
        "provenance": {
            "fit_partition": "train",
            "evaluation_partition": "eval",
            "selection_metric": "F1 on training partition",
            "confidence_rule": "strong: AP >= 0.70; moderate: 0.55 <= AP < 0.70; weak: AP < 0.55",
            "train_cell_table": str(Path(args.train_cell_table).resolve()),
            "eval_cell_table": str(Path(args.eval_cell_table).resolve()),
            "label_thresholds": str(Path(args.thresholds_json).resolve()),
            "random_state": int(args.random_state),
        },
    }
    (outdir / "workbench_thresholds.json").write_text(
        json.dumps(threshold_artifact, indent=2) + "\n"
    )
    if args.write_eval_table:
        phenotype_eval.to_csv(outdir / "cell_table_eval_calibrated_phenotypes.csv", index=False)

    summary = {
        "train_cell_table": str(Path(args.train_cell_table)),
        "eval_cell_table": str(Path(args.eval_cell_table)),
        "thresholds_json": str(Path(args.thresholds_json)),
        "channels": channels,
        "phenotypes": phenotype_names,
        "marker_metrics": marker_rows,
        "phenotype_metrics": phenotype_rows,
        "calibration_meta": calibration_meta,
        "neighborhood_meta": neighborhood_meta,
        "args": vars(args),
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {outdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
