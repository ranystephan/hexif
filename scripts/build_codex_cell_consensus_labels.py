#!/usr/bin/env python3
"""Build multi-method consensus marker labels for CODEX cells.

Per the D1 finding (some markers have Cohen's κ < 0.55 between threshold
methods), single-method positivity is unstable on the immune markers we care
about most.  This script produces:

- ``chXX_pos_consensus``: True iff at least N of {GMM, Otsu, train-quantile
  matched to GMM positivity rate} call the cell positive on log1p
  intensities.  Default N = 2 of 3.
- ``chXX_pos_disagree``: True if methods disagree (a "split decision" cell).
  v1 will exclude these from per-cell loss for unstable markers.
- ``chXX_pos_majority``: True if ≥ 2 methods call positive (alias of
  ``chXX_pos_consensus`` when N=2).

Thresholds are fit on TRAIN cells only.  No val/test data leaks into
threshold fitting.

Output: writes new train + eval cell tables alongside the existing ones,
with the original ``chXX_pos`` column preserved for backwards comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from hexif.utils import CODEX_CHANNEL_NAMES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build consensus marker labels")
    p.add_argument("--train_cell_table", required=True)
    p.add_argument(
        "--eval_cell_tables",
        nargs="+",
        required=True,
        help="One or more eval/val/test cell tables to label.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--marker_channels", default="0,3,7,13,16,17,27,31,34,46,50,52")
    p.add_argument(
        "--gmm_thresholds_json",
        required=True,
        help="Existing GMM thresholds; reused so the consensus inherits the calibrated GMM gate",
    )
    p.add_argument("--consensus_n", type=int, default=2)
    p.add_argument("--write_diagnostics", action="store_true")
    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _fit_gmm_intersection(values: np.ndarray) -> float:
    if values.size < 50:
        return float(np.quantile(values, 0.9))
    g = GaussianMixture(n_components=2, random_state=0, max_iter=200, reg_covar=1e-4).fit(
        values.reshape(-1, 1)
    )
    means = g.means_.flatten()
    stds = np.sqrt(g.covariances_.flatten())
    weights = g.weights_
    order = np.argsort(means)
    m1, m2 = means[order]
    s1, s2 = stds[order]
    w1, w2 = weights[order]
    grid = np.linspace(m1, m2, 1024)
    p1 = w1 * np.exp(-0.5 * ((grid - m1) / max(s1, 1e-3)) ** 2) / max(s1, 1e-3)
    p2 = w2 * np.exp(-0.5 * ((grid - m2) / max(s2, 1e-3)) ** 2) / max(s2, 1e-3)
    crossings = np.where(np.diff(np.sign(p2 - p1)) != 0)[0]
    return float(grid[crossings[-1]]) if crossings.size else float((m1 + m2) / 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    channels = _parse_int_list(args.marker_channels)
    consensus_n = max(1, min(3, int(args.consensus_n)))

    train = pd.read_csv(args.train_cell_table)
    n_train = len(train)
    logging.info("loaded %d train cells", n_train)

    # If the GMM thresholds JSON exists, prefer those tau_gmm so the consensus
    # is anchored on the same gate the existing baseline uses.  Otherwise fit
    # GMM here.
    existing_gmm: dict[int, float] = {}
    existing_pos: dict[int, float] = {}
    gmm_path = Path(args.gmm_thresholds_json)
    if gmm_path.exists():
        rows = json.loads(gmm_path.read_text())
        if isinstance(rows, dict) and "thresholds" in rows:
            rows = rows["thresholds"]
        for r in rows:
            existing_gmm[int(r["channel"])] = float(r["threshold"])
            existing_pos[int(r["channel"])] = float(r.get("train_positive_fraction", 0.1))

    diag_rows: list[dict] = []
    method_thresholds: dict[int, dict[str, float]] = {}
    for ch in channels:
        col = f"ch{ch:02d}_true"
        v_train = train[col].to_numpy(dtype=np.float32)
        tau_gmm = existing_gmm.get(ch, _fit_gmm_intersection(v_train))
        tau_otsu = (
            float(threshold_otsu(v_train))
            if v_train.size > 50 and not np.allclose(v_train.min(), v_train.max())
            else float(np.quantile(v_train, 0.9))
        )
        target_p = existing_pos.get(ch, float(np.mean(v_train > tau_gmm)))
        q = float(np.clip(1.0 - target_p, 0.01, 0.999))
        tau_q = float(np.quantile(v_train, q))
        method_thresholds[ch] = {
            "gmm": tau_gmm,
            "otsu": tau_otsu,
            "quantile": tau_q,
            "target_pos": target_p,
        }
        name = CODEX_CHANNEL_NAMES[ch] if ch < len(CODEX_CHANNEL_NAMES) else f"ch{ch}"
        diag_rows.append(
            {
                "channel": ch,
                "name": name,
                "tau_gmm": tau_gmm,
                "tau_otsu": tau_otsu,
                "tau_quantile": tau_q,
                "train_pos_gmm": float(np.mean(v_train > tau_gmm)),
                "train_pos_otsu": float(np.mean(v_train > tau_otsu)),
                "train_pos_quantile": float(np.mean(v_train > tau_q)),
            }
        )
    pd.DataFrame(diag_rows).to_csv(out / "consensus_thresholds.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(method_thresholds, indent=2, default=float))

    def label(df: pd.DataFrame, name: str) -> pd.DataFrame:
        out_df = df.copy()
        for ch in channels:
            v = df[f"ch{ch:02d}_true"].to_numpy(dtype=np.float32)
            taus = method_thresholds[ch]
            gmm_pos = v > taus["gmm"]
            otsu_pos = v > taus["otsu"]
            q_pos = v > taus["quantile"]
            sum_pos = gmm_pos.astype(np.int32) + otsu_pos.astype(np.int32) + q_pos.astype(np.int32)
            consensus = sum_pos >= consensus_n
            disagree = (sum_pos != 0) & (sum_pos != 3)
            # preserve original chXX_pos as chXX_pos_gmm for traceability
            if f"ch{ch:02d}_pos" in out_df.columns:
                out_df[f"ch{ch:02d}_pos_gmm_orig"] = out_df[f"ch{ch:02d}_pos"]
            out_df[f"ch{ch:02d}_pos_gmm"] = gmm_pos
            out_df[f"ch{ch:02d}_pos_otsu"] = otsu_pos
            out_df[f"ch{ch:02d}_pos_quantile"] = q_pos
            out_df[f"ch{ch:02d}_pos_consensus"] = consensus
            out_df[f"ch{ch:02d}_pos_disagree"] = disagree
            # Replace the canonical chXX_pos with consensus (so downstream
            # scripts use the new label by default; preserved orig as
            # *_pos_gmm_orig for comparability).
            out_df[f"ch{ch:02d}_pos"] = consensus
        return out_df

    # disambiguate train vs eval by parent dir, since both are typically named
    # "cell_table_eval.csv" inside their respective output dirs.
    train_path = Path(args.train_cell_table)
    train_out = label(train, "train")
    train_tag = train_path.parent.name or "train"
    train_dst = out / f"{train_tag}__{train_path.stem}_consensus.csv"
    train_out.to_csv(train_dst, index=False)
    logging.info("wrote consensus train table: %s (%d cells)", train_dst, len(train_out))

    for ev in args.eval_cell_tables:
        ev_path = Path(ev)
        df = pd.read_csv(ev_path)
        out_df = label(df, ev_path.stem)
        ev_tag = ev_path.parent.name or "eval"
        dst = out / f"{ev_tag}__{ev_path.stem}_consensus.csv"
        out_df.to_csv(dst, index=False)
        logging.info("wrote consensus eval table: %s (%d cells)", dst, len(out_df))

        # diagnostic: how many cells changed positivity per marker?
        if args.write_diagnostics:
            rows = []
            for ch in channels:
                name = CODEX_CHANNEL_NAMES[ch] if ch < len(CODEX_CHANNEL_NAMES) else f"ch{ch}"
                if f"ch{ch:02d}_pos_gmm_orig" in out_df.columns:
                    o = out_df[f"ch{ch:02d}_pos_gmm_orig"].astype(bool)
                    c = out_df[f"ch{ch:02d}_pos_consensus"].astype(bool)
                    rows.append(
                        {
                            "channel": ch,
                            "name": name,
                            "n_cells": len(out_df),
                            "frac_pos_gmm": float(o.mean()),
                            "frac_pos_consensus": float(c.mean()),
                            "frac_disagree_among_methods": float(
                                out_df[f"ch{ch:02d}_pos_disagree"].mean()
                            ),
                            "frac_changed_label": float((o != c).mean()),
                        }
                    )
            pd.DataFrame(rows).to_csv(out / f"{ev_path.stem}_label_change.csv", index=False)


if __name__ == "__main__":
    main()
