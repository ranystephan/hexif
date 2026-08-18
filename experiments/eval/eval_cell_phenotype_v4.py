#!/usr/bin/env python3
"""Evaluate within-core AP lift for a cell-phenotype checkpoint.

The primary metric is the gap between the trained model's per-cell AP and the
per-core-mean-baseline AP. If the model has no per-cell signal — i.e. all
its predictive power comes from the encoder's ability to identify what
TYPE of core this is — the gap collapses to zero. The size of the gap
tells you how much the model sees beyond core composition. This is the
metric measures cell-level signal beyond core composition.

Output:
    <output_dir>/within_core_metrics.csv  — per-marker model AP vs baseline
                                            AP vs lift
    <output_dir>/within_core_metrics.json — same data plus macro summary

Example::

    python experiments/eval/eval_cell_phenotype_v4.py \\
      --checkpoint /path/to/frozen_checkpoint.pt \\
      --cell_table /path/to/held_out_cells.csv \\
      --pairs_dir /path/to/registered_pairs \\
      --output_dir /path/to/evaluation_output \\
      --label_set spacec
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from hexif.cell_model import CellPhenotypeModel, apply_lora_to_vit, build_encoder
from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES
from hexif.training.cell_phenotype import (
    CellPatchDataset,
    cell_patch_collate,
    evaluate,
    parse_int_list,
    safe_metric,
    set_seeds,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to runs/<run>/best_model.pt produced by train_cell_phenotype_v4.py",
    )
    p.add_argument(
        "--cell_table",
        required=True,
        help="Val cell table with chXX_pred + chXX_pos_spacec columns",
    )
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--label_set", default="spacec", choices=["gmm", "consensus", "spacec"])
    p.add_argument("--marker_channels", default=",".join(str(x) for x in FOCUSED_MARKERS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _filter_cells_by_border(df: pd.DataFrame, pairs_dir: Path, patch_size: int) -> np.ndarray:
    """Replicate CellPatchDataset's border filter without loading every patch.

    Vectorized — looks up each core's H&E shape once via mmap, then masks
    the whole df at once. Without this we get a length mismatch between
    the model's predictions (which skip border cells) and the df rows.
    """
    half = patch_size // 2
    # Per-core shape lookup. Memory-map the H&E so we read only the header.
    shapes: dict[str, tuple[int, int]] = {}
    for base in df["basename"].unique():
        he = np.load(pairs_dir / f"{base}_HE.npy", mmap_mode="r")
        shapes[base] = he.shape[:2]
    H = df["basename"].map(lambda b: shapes[b][0]).to_numpy()
    W = df["basename"].map(lambda b: shapes[b][1]).to_numpy()
    cy = df["centroid_y"].to_numpy()
    cx = df["centroid_x"].to_numpy()
    return (cy - half >= 0) & (cy + half <= H) & (cx - half >= 0) & (cx + half <= W)


def _build_v4_from_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct CellPhenotypeModel from a checkpoint produced by train_cell_phenotype_v4.

    Reads saved CLI args from the checkpoint to pick the right encoder
    (uni2 / h_optimus_0 / vit_base_smoke) and head shape. We build the
    encoder with ``pretrained=False`` since the checkpoint will overwrite
    every weight a moment later — no point downloading the foundation
    model again.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    encoder_name = saved_args["encoder"]
    if encoder_name not in {"uni2", "h_optimus_0", "vit_base_smoke"}:
        raise ValueError(
            f"This eval is for v4 encoders; got encoder={encoder_name!r}. "
            "Use eval_cell_phenotype_v1_1.py for ctranspath checkpoints."
        )

    encoder = build_encoder(encoder_name, pretrained=False)
    apply_lora_to_vit(
        encoder,
        rank=int(saved_args["lora_rank"]),
        alpha=float(saved_args["lora_alpha"]),
    )
    head_hidden = tuple(parse_int_list(saved_args["mlp_hidden"]))
    model = CellPhenotypeModel(
        encoder=encoder,
        embed_dim=int(encoder.embed_dim),
        n_pred=12,
        n_core_mean=12,
        head_hidden=head_hidden,
        dropout=float(saved_args["dropout"]),
        use_gradient_checkpointing=False,  # eval is pure forward
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    return model, saved_args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    args = _parse_args()
    set_seeds(int(args.seed))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    marker_channels = parse_int_list(args.marker_channels)

    logging.info("loading checkpoint: %s", args.checkpoint)
    model, saved_args = _build_v4_from_checkpoint(Path(args.checkpoint), device)
    logging.info(
        "model loaded: encoder=%s embed_dim=%d trainable=%d",
        saved_args["encoder"],
        int(model.embed_dim),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    logging.info("reading cell table: %s", args.cell_table)
    df = pd.read_csv(args.cell_table)
    logging.info("raw rows: %d", len(df))

    valid_mask = _filter_cells_by_border(df, Path(args.pairs_dir), int(args.patch_size))
    df = df.loc[valid_mask].reset_index(drop=True)
    logging.info("rows after border filter: %d", len(df))

    ds = CellPatchDataset(
        cell_table=df,
        pairs_dir=Path(args.pairs_dir),
        marker_channels=marker_channels,
        patch_size=int(args.patch_size),
        train=False,
        label_set=str(args.label_set),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=cell_patch_collate,
        pin_memory=True,
    )

    logging.info("running inference on %d cells", len(ds))
    model_marker_probs, model_pheno_probs, marker_y, pheno_y = evaluate(model, loader, device)
    if model_marker_probs.shape[0] != len(df):
        raise RuntimeError(
            f"inference returned {model_marker_probs.shape[0]} rows but df has "
            f"{len(df)} — border filter mismatch?"
        )
    logging.info(
        "inference complete: marker_probs %s phenotype_probs %s",
        model_marker_probs.shape,
        model_pheno_probs.shape,
    )

    # ---------- per-cell prediction CSV (workbench input) ----------
    # The workbench-add script (`scripts/add_model_to_workbench_bundle.py`)
    # consumes a CSV with `chXX_pred_v1_1` columns and renames them to the
    # caller's model id. We mirror eval_cell_phenotype_v1_1.py's layout
    # verbatim so the same downstream script works unchanged for v4/v6.
    # `df` here is already border-filtered, so model rows align 1:1.
    keep_cols = ["basename", "cell_id", "centroid_y", "centroid_x"]
    pred_table = df[keep_cols].copy()
    for ci, ch in enumerate(marker_channels):
        # The "_v1_1" suffix is deliberately retained — the workbench-add
        # script's renamer keys off it. See comment in
        # `scripts/add_model_to_workbench_bundle.py:_rename_eval_csv`.
        pred_table[f"ch{ch:02d}_pred_v1_1"] = model_marker_probs[:, ci]
        pred_table[f"ch{ch:02d}_pos"] = marker_y[:, ci]
        if f"ch{ch:02d}_pred" in df.columns:
            pred_table[f"ch{ch:02d}_pred"] = df[f"ch{ch:02d}_pred"].to_numpy(dtype=np.float32)
    for pi, pname in enumerate(PHENOTYPE_NAMES):
        if pi < model_pheno_probs.shape[1]:
            pred_table[f"phenotype_{pname}_score"] = model_pheno_probs[:, pi]
        if pi < pheno_y.shape[1]:
            pred_table[f"phenotype_{pname}_label"] = pheno_y[:, pi]
    pred_table.to_csv(out / "cell_predictions.csv", index=False)
    logging.info(
        "wrote %s (%d rows × %d cols)",
        out / "cell_predictions.csv",
        len(pred_table),
        pred_table.shape[1],
    )

    # Baseline: per-core mean of chXX_pred (the dense ORION model's prediction)
    # broadcast back to each cell. This is what the trainer's
    # ``core_mean_features`` computes for the model's fusion input; we
    # recompute it here so the eval is self-contained.
    pred_cols = [f"ch{ch:02d}_pred" for ch in marker_channels]
    baseline_scores = df.groupby("basename")[pred_cols].transform("mean").to_numpy(dtype=np.float32)

    # Per-marker eval — three numbers per channel.
    rows = []
    for ci, ch in enumerate(marker_channels):
        name = MARKER_NAMES[ci] if ci < len(MARKER_NAMES) else f"ch{ch:02d}"
        y = marker_y[:, ci]
        model_ap = safe_metric(average_precision_score, y, model_marker_probs[:, ci])
        baseline_ap = safe_metric(average_precision_score, y, baseline_scores[:, ci])
        lift = (
            float("nan")
            if (np.isnan(model_ap) or np.isnan(baseline_ap))
            else model_ap - baseline_ap
        )
        rows.append(
            {
                "channel": int(ch),
                "name": name,
                "n_eval_cells": len(y),
                "positive_fraction": float(np.mean(y)),
                "model_ap": float(model_ap),
                "baseline_ap_core_mean": float(baseline_ap),
                "lift": float(lift),
            }
        )

    # Macro summaries — nanmean so a marker with zero positives doesn't
    # corrupt the aggregate.
    macro_model_ap = float(np.nanmean([r["model_ap"] for r in rows]))
    macro_baseline_ap = float(np.nanmean([r["baseline_ap_core_mean"] for r in rows]))
    macro_lift = float(np.nanmean([r["lift"] for r in rows]))

    pd.DataFrame(rows).to_csv(out / "within_core_metrics.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "cell_table": str(args.cell_table),
        "label_set": args.label_set,
        "n_eval_cells": len(df),
        "encoder": saved_args["encoder"],
        "macro_model_ap": macro_model_ap,
        "macro_baseline_ap_core_mean": macro_baseline_ap,
        "macro_within_core_lift": macro_lift,
        "per_marker": rows,
    }
    (out / "within_core_metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"=== v4 within-core eval ({saved_args['encoder']}) ===")
    print(f"n_eval_cells          = {len(df)}")
    print(f"macro model AP        = {macro_model_ap:.4f}")
    print(f"macro baseline AP     = {macro_baseline_ap:.4f}  (per-core mean of chXX_pred)")
    print(f"macro within-core LIFT = {macro_lift:+.4f}")
    print()
    print("per-marker (sorted by lift):")
    for r in sorted(rows, key=lambda r: -r["lift"]):
        print(
            f"  ch{r['channel']:02d} {r['name']:<10}  model={r['model_ap']:.3f} "
            f"baseline={r['baseline_ap_core_mean']:.3f}  lift={r['lift']:+.3f}  "
            f"(pos_frac={r['positive_fraction']:.3f})"
        )
    print()
    print(f"wrote {out / 'within_core_metrics.csv'}")
    print(f"wrote {out / 'within_core_metrics.json'}")


if __name__ == "__main__":
    main()
