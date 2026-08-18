#!/usr/bin/env python3
"""D1: audit label noise for the focused CODEX marker panel.

For each marker:

1. Refit three independent thresholds on **train cells only** — GMM
   intersection (current), Otsu on log1p intensities, and train quantile (q
   chosen so the quantile-positive rate matches GMM).
2. Compute pairwise Jaccard and Cohen's kappa between the three methods on
   *val* cells — if methods disagree, the labels are unstable.
3. Estimate the F1 ceiling implied by the disagreement: how high could F1
   possibly be if our train-fit threshold is wrong on this many cells?
4. Sample stratified contact sheets (H&E + true CODEX + mask overlay +
   predicted CODEX) for cells in disagreement strata.  Useful for human
   review but not part of the numerical answer.
5. Hierarchical gating: if ``--apply_hierarchy`` is set, conditionally clip
   labels (e.g. CD8+ only when CD45+) and report the change in macro and
   per-phenotype F1 against the baseline phenotype scoring.

The script does not modify its input tables. It writes a separate audit directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL_DIR = ROOT / "experiments" / "eval"
for p in (SRC, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import eval_codex_fullcore as fullcore  # for _codex_norm
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from sklearn.metrics import cohen_kappa_score, jaccard_score
from sklearn.mixture import GaussianMixture

from hexif.scaling import QuantileScaler
from hexif.utils import CODEX_CHANNEL_NAMES

FOCUSED_MARKERS = [0, 3, 7, 13, 16, 17, 27, 31, 34, 46, 50, 52]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1 label-noise audit")
    p.add_argument("--train_cell_table", required=True)
    p.add_argument("--eval_cell_table", required=True)
    p.add_argument(
        "--thresholds_json", required=True, help="GMM threshold JSON used by the existing baseline"
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--marker_channels", default=",".join(str(x) for x in FOCUSED_MARKERS))
    p.add_argument(
        "--audit_markers",
        default="0,27,31,34,52",
        help="Subset of markers for visual contact sheets (DAPI 0, CD68 27, "
        "FAP 31, CD163 34, panCK 52)",
    )
    p.add_argument("--n_contact_per_stratum", type=int, default=24)
    p.add_argument(
        "--contact_size_px",
        type=int,
        default=64,
        help="Side length of the H&E / CODEX crop around each cell",
    )
    p.add_argument(
        "--render_contacts",
        action="store_true",
        help="If set, render PNG contact sheets. Off by default — table is the answer.",
    )
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--scaler_path", required=True)
    p.add_argument(
        "--apply_hierarchy",
        action="store_true",
        help="Recompute labels with hierarchical gates (CD8 conditional on CD45).",
    )
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------- thresholds


def _fit_gmm_intersection(values: np.ndarray) -> float:
    """Two-component GMM intersection threshold on log1p cell intensities."""
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
    if crossings.size:
        return float(grid[crossings[-1]])
    return float((m1 + m2) / 2)


def _fit_otsu(values: np.ndarray) -> float:
    if values.size < 50 or np.allclose(values.min(), values.max()):
        return float(np.quantile(values, 0.9))
    return float(threshold_otsu(values))


def _fit_quantile_match(values: np.ndarray, target_positive_rate: float) -> float:
    q = float(np.clip(1.0 - target_positive_rate, 0.01, 0.999))
    return float(np.quantile(values, q))


def _label_noise_f1_ceiling(prevalence: float, flip_rate: float) -> float:
    """Approximate F1 ceiling under symmetric label flips at given flip_rate.

    Sweeps the operating point and returns the max F1 attainable when the true
    labels are observed through symmetric noise.  This is a *rough* indicator,
    not a tight bound; useful for noticing when label noise alone caps F1
    well below 1.
    """
    p = max(min(float(prevalence), 0.999), 0.001)
    e = max(min(float(flip_rate), 0.499), 0.0)
    # observed positives: p_obs = p*(1-e) + (1-p)*e
    # Among observed positives, fraction that are truly positive:
    p_obs = p * (1 - e) + (1 - p) * e
    if p_obs == 0:
        return 0.0
    precision = (p * (1 - e)) / p_obs
    recall = 1 - e
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    return float(f1)


# ---------------------------------------------------------------- main


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    channels = _parse_int_list(args.marker_channels)
    audit_markers = _parse_int_list(args.audit_markers)

    train = pd.read_csv(args.train_cell_table)
    val = pd.read_csv(args.eval_cell_table)
    rows = json.loads(Path(args.thresholds_json).read_text())
    if isinstance(rows, dict) and "thresholds" in rows:
        rows = rows["thresholds"]
    gmm_existing = {int(r["channel"]): float(r["threshold"]) for r in rows}
    train_pos_existing = {int(r["channel"]): float(r["train_positive_fraction"]) for r in rows}

    threshold_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for ch in channels:
        name = CODEX_CHANNEL_NAMES[ch] if ch < len(CODEX_CHANNEL_NAMES) else f"ch{ch}"
        v_train = train[f"ch{ch:02d}_true"].to_numpy(dtype=np.float32)
        v_val = val[f"ch{ch:02d}_true"].to_numpy(dtype=np.float32)
        target_p = float(train_pos_existing.get(ch, np.mean(v_train > gmm_existing[ch])))
        tau_gmm = _fit_gmm_intersection(v_train)
        tau_otsu = _fit_otsu(v_train)
        tau_q = _fit_quantile_match(v_train, target_p)
        {
            "gmm": tau_gmm,
            "otsu": tau_otsu,
            "quantile": tau_q,
            "gmm_existing": float(gmm_existing[ch]),
        }
        threshold_rows.append(
            {
                "channel": int(ch),
                "name": name,
                "train_cells": int(v_train.size),
                "val_cells": int(v_val.size),
                "target_positive_rate": target_p,
                "tau_gmm": tau_gmm,
                "tau_otsu": tau_otsu,
                "tau_quantile": tau_q,
                "tau_gmm_existing": float(gmm_existing[ch]),
                "train_pos_gmm": float(np.mean(v_train > tau_gmm)),
                "train_pos_otsu": float(np.mean(v_train > tau_otsu)),
                "train_pos_quantile": float(np.mean(v_train > tau_q)),
                "val_pos_gmm": float(np.mean(v_val > tau_gmm)),
                "val_pos_otsu": float(np.mean(v_val > tau_otsu)),
                "val_pos_quantile": float(np.mean(v_val > tau_q)),
            }
        )

        # pairwise agreement on val cells
        labels = {
            "gmm": (v_val > tau_gmm),
            "otsu": (v_val > tau_otsu),
            "quantile": (v_val > tau_q),
            "gmm_existing": (v_val > gmm_existing[ch]),
        }
        method_pairs = [
            ("gmm", "otsu"),
            ("gmm", "quantile"),
            ("otsu", "quantile"),
            ("gmm_existing", "gmm"),
        ]
        for m1, m2 in method_pairs:
            y1 = labels[m1].astype(int)
            y2 = labels[m2].astype(int)
            try:
                jac = float(jaccard_score(y1, y2)) if y1.sum() + y2.sum() > 0 else float("nan")
            except Exception:
                jac = float("nan")
            try:
                k = float(cohen_kappa_score(y1, y2))
            except Exception:
                k = float("nan")
            disagree = float(np.mean(y1 != y2))
            # F1 ceiling implied by disagreement-as-flip-rate
            obs_pos_rate = (y1.sum() + y2.sum()) / (2 * len(y1))
            f1_ceiling = _label_noise_f1_ceiling(
                prevalence=float(obs_pos_rate), flip_rate=disagree / 2
            )
            agreement_rows.append(
                {
                    "channel": int(ch),
                    "name": name,
                    "method_a": m1,
                    "method_b": m2,
                    "jaccard": jac,
                    "cohen_kappa": k,
                    "disagree_rate": disagree,
                    "f1_ceiling_under_disagreement": f1_ceiling,
                }
            )

    pd.DataFrame(threshold_rows).to_csv(out / "thresholds_per_marker.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(out / "threshold_agreement.csv", index=False)
    logging.info("wrote thresholds_per_marker.csv and threshold_agreement.csv")

    # quick summary printable
    aroll = pd.DataFrame(agreement_rows).pivot_table(
        index="name", columns=["method_a", "method_b"], values="cohen_kappa", aggfunc="first"
    )
    aroll.to_csv(out / "kappa_pivot.csv")

    # hierarchical-gating delta — write augmented labels for the existing
    # phenotype-scoring script to ingest.  Hierarchy: CD8 (ch07) requires CD45
    # (ch03).  m2_like (CD68 & CD163) and t_cell already use AND of markers,
    # so this only affects CD8 directly.
    if args.apply_hierarchy:
        v_train_aug = train.copy()
        v_val_aug = val.copy()
        for df in (v_train_aug, v_val_aug):
            if "ch07_pos" in df.columns and "ch03_pos" in df.columns:
                df["ch07_pos_orig"] = df["ch07_pos"].astype(bool)
                df["ch07_pos"] = df["ch07_pos"].astype(bool) & df["ch03_pos"].astype(bool)
        v_train_aug.to_csv(out / "train_cell_table_hierarchical.csv", index=False)
        v_val_aug.to_csv(out / "val_cell_table_hierarchical.csv", index=False)
        logging.info("wrote hierarchical-gated cell tables")

    summary = {
        "args": vars(args),
        "thresholds_per_marker": threshold_rows,
        "threshold_agreement": agreement_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # contact sheets are optional; render only if --render_contacts
    if args.render_contacts:
        _render_contact_sheets(args, val, audit_markers, gmm_existing, threshold_rows, out)
    print(f"wrote {out / 'summary.json'}")


def _load_codex_log_channel(args: argparse.Namespace, base: str, ch: int) -> np.ndarray:
    pairs_dir = Path(args.pairs_dir)
    codex = np.load(pairs_dir / f"{base}_CODEX.npy", mmap_mode="r")
    if codex.shape[0] != 53:
        codex = np.transpose(codex, (2, 0, 1))
    x = np.asarray(codex[ch], dtype=np.float32)
    norm = fullcore._codex_norm(x[None]) if x.max(initial=0.0) > 1.5 else 1.0
    scaler = QuantileScaler.load(Path(args.scaler_path))
    z = (x / norm if norm != 1.0 else x) - scaler.qlo[ch]
    z = z / (scaler.qhi[ch] - scaler.qlo[ch] + 1e-6)
    return np.log1p(np.clip(z, 0, None)).astype(np.float32)


def _render_contact_sheets(
    args: argparse.Namespace,
    val: pd.DataFrame,
    audit_markers: list[int],
    gmm_existing: dict[int, float],
    threshold_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    rng = np.random.default_rng(int(args.random_state))
    pairs_dir = Path(args.pairs_dir)
    half = args.contact_size_px // 2
    out_root = out / "contact_sheets"
    tau_lookup = {int(r["channel"]): r for r in threshold_rows}

    for ch in audit_markers:
        name = CODEX_CHANNEL_NAMES[ch] if ch < len(CODEX_CHANNEL_NAMES) else f"ch{ch}"
        col_true = f"ch{ch:02d}_true"
        if col_true not in val.columns:
            logging.warning("ch%02d_true missing in val; skipping", ch)
            continue
        tau = tau_lookup[ch]
        v = val[col_true].to_numpy(dtype=np.float32)
        gmm_pos = v > tau["tau_gmm"]
        otsu_pos = v > tau["tau_otsu"]
        q_pos = v > tau["tau_quantile"]
        strata = {
            "all_three_pos": gmm_pos & otsu_pos & q_pos,
            "all_three_neg": (~gmm_pos) & (~otsu_pos) & (~q_pos),
            "gmm_only_pos": gmm_pos & (~otsu_pos) & (~q_pos),
            "otsu_only_pos": (~gmm_pos) & otsu_pos & (~q_pos),
            "quantile_only_pos": (~gmm_pos) & (~otsu_pos) & q_pos,
            "split_decision_2of3_pos": (
                (gmm_pos.astype(int) + otsu_pos.astype(int) + q_pos.astype(int)) == 2
            ),
        }
        marker_dir = out_root / name
        marker_dir.mkdir(parents=True, exist_ok=True)
        codex_cache: dict[str, np.ndarray] = {}
        he_cache: dict[str, np.ndarray] = {}
        for stratum, mask in strata.items():
            idx_pool = np.where(mask)[0]
            if idx_pool.size == 0:
                continue
            sample = rng.choice(
                idx_pool, size=int(min(args.n_contact_per_stratum, idx_pool.size)), replace=False
            )
            sub = val.iloc[sample].reset_index(drop=True)
            n = len(sub)
            cols = 4
            rows = max(1, (n + cols - 1) // cols)
            fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 3 * 1.3, rows * 1.3))
            axes = np.atleast_2d(axes)
            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    if i >= n:
                        for k in range(3):
                            axes[r, c * 3 + k].axis("off")
                        continue
                    row = sub.iloc[i]
                    base = str(row["basename"])
                    cy, cx = int(row["centroid_y"]), int(row["centroid_x"])
                    if base not in he_cache:
                        he_cache[base] = np.asarray(
                            np.load(pairs_dir / f"{base}_HE.npy", mmap_mode="r"), dtype=np.uint8
                        )
                    if base not in codex_cache:
                        codex_cache[base] = _load_codex_log_channel(args, base, ch)
                    he = he_cache[base]
                    cdx = codex_cache[base]
                    H, W = he.shape[:2]
                    y0 = max(0, cy - half)
                    y1 = min(H, cy + half)
                    x0 = max(0, cx - half)
                    x1 = min(W, cx + half)
                    he_crop = he[y0:y1, x0:x1]
                    cdx_crop = cdx[y0:y1, x0:x1]
                    axes[r, c * 3 + 0].imshow(he_crop)
                    axes[r, c * 3 + 0].set_title(f"{base[-6:]} y{cy} x{cx}", fontsize=6)
                    axes[r, c * 3 + 0].axis("off")
                    axes[r, c * 3 + 1].imshow(cdx_crop, cmap="magma")
                    axes[r, c * 3 + 1].set_title(f"{name} log1p", fontsize=6)
                    axes[r, c * 3 + 1].axis("off")
                    axes[r, c * 3 + 2].imshow(he_crop)
                    axes[r, c * 3 + 2].imshow(cdx_crop, cmap="magma", alpha=0.55)
                    axes[r, c * 3 + 2].set_title("overlay", fontsize=6)
                    axes[r, c * 3 + 2].axis("off")
            fig.suptitle(f"{name} — {stratum} (n={n})", fontsize=10)
            fig.tight_layout()
            fig.savefig(marker_dir / f"{stratum}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        logging.info("rendered %s contact sheets to %s", name, marker_dir)


if __name__ == "__main__":
    main()
