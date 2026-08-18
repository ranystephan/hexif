"""Shared training infrastructure for cell-phenotype models.

Trainer scripts under ``experiments/train/`` build their own model
(``V1_1Model`` for the CTransPath baseline, ``V4Model`` for the UNI2 /
H-optimus-0 path) and hand it to :func:`train` here. Everything that
is the same across model versions — dataset construction, focal-BCE /
ASL marker loss, focal-BCE phenotype loss, hierarchy loss, AMP scaler,
gradient accumulation, validation, early stop, best-checkpoint saving,
and the final per-marker / per-phenotype metric tables — lives here so
each trainer stays a thin (~150-line) script.

Model contract:
    ``model(rgb, pred, core_mean) -> (marker_logits, phenotype_logits)``

Both ``V1_1Model`` and ``V4Model`` satisfy this contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from hexif.cell_phenotype import (
    FOCUSED_MARKERS,
    MARKER_NAMES,
    PHENOTYPE_NAMES,
    UNSTABLE_MARKERS_BY_INDEX,
    VALID_LABEL_SETS,
    asymmetric_loss_with_logits,
    cell_table_to_targets,
    focal_bce_with_logits,
    hierarchy_loss,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================ #
# Dataset
# ============================================================================ #


class CellPatchDataset(torch.utils.data.Dataset):
    """Yields per-cell H&E patches with augmentation + targets + fusion features.

    Avoids loading full TMA cores per ``__getitem__`` by caching the H&E array
    once per core in a worker-local dict. ``__getitem__`` crops a
    ``patch_size × patch_size`` window around the cell centroid, applies
    augmentation (train only), and returns a tensor + label tuple. Cells whose
    centroid is too close to a core border for the patch to fit return
    ``None`` and are skipped by :func:`cell_patch_collate`.
    """

    def __init__(
        self,
        cell_table: pd.DataFrame,
        pairs_dir: Path,
        marker_channels: list[int],
        patch_size: int = 224,
        train: bool = True,
        label_set: str = "consensus",
    ) -> None:
        self.df = cell_table.reset_index(drop=True).copy()
        self.pairs_dir = Path(pairs_dir)
        self.marker_channels = list(marker_channels)
        self.patch_size = int(patch_size)
        self.half = self.patch_size // 2
        self.train = bool(train)
        self.he_cache: dict[str, np.ndarray] = {}
        self.label_set = str(label_set)

        marker_y, disagree, phenotype_y, _ = cell_table_to_targets(
            self.df, marker_channels=tuple(marker_channels), label_set=self.label_set
        )
        self.marker_y = marker_y.astype(np.float32)
        self.disagree = disagree.astype(np.bool_)
        self.phenotype_y = phenotype_y.astype(np.float32)
        pred_cols = [f"ch{ch:02d}_pred" for ch in marker_channels]
        if not all(c in self.df.columns for c in pred_cols):
            raise ValueError(f"cell_table missing pred columns: {pred_cols}")
        self.pred_features = self.df[pred_cols].to_numpy(dtype=np.float32)
        means = self.df.groupby("basename")[pred_cols].transform("mean").to_numpy(dtype=np.float32)
        self.core_mean_features = means
        self.basenames = self.df["basename"].to_numpy()
        self.cy = self.df["centroid_y"].to_numpy(dtype=np.float32)
        self.cx = self.df["centroid_x"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _load_he(self, base: str) -> np.ndarray:
        he = self.he_cache.get(base)
        if he is None:
            he = np.asarray(
                np.load(self.pairs_dir / f"{base}_HE.npy", mmap_mode="r"), dtype=np.uint8
            )
            # Bound the cache so DataLoader workers do not OOM on big TMAs.
            if len(self.he_cache) > 16:
                self.he_cache.pop(next(iter(self.he_cache)))
            self.he_cache[base] = he
        return he

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        base = self.basenames[idx]
        he = self._load_he(str(base))
        H, W = he.shape[:2]
        cy = int(self.cy[idx])
        cx = int(self.cx[idx])
        if cy - self.half < 0 or cy + self.half > H or cx - self.half < 0 or cx + self.half > W:
            return None
        crop = he[cy - self.half : cy + self.half, cx - self.half : cx + self.half, :]
        if self.train:
            r = torch.randint(0, 4, (1,)).item()
            if r:
                crop = np.rot90(crop, k=int(r), axes=(0, 1)).copy()
            if torch.rand(1).item() < 0.5:
                crop = crop[:, ::-1, :].copy()
            if torch.rand(1).item() < 0.5:
                crop = crop[::-1, :, :].copy()
            if torch.rand(1).item() < 0.7:
                # Color jitter (brightness/contrast/saturation each ±0.1).
                crop_f = crop.astype(np.float32) / 255.0
                bf = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                cf = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                sf = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                crop_f = np.clip(crop_f * bf, 0.0, 1.0)
                mean = crop_f.mean(axis=(0, 1), keepdims=True)
                crop_f = np.clip((crop_f - mean) * cf + mean, 0.0, 1.0)
                gray = crop_f.mean(axis=2, keepdims=True)
                crop_f = np.clip((crop_f - gray) * sf + gray, 0.0, 1.0)
                rgb = (crop_f - IMAGENET_MEAN) / IMAGENET_STD
                rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
                return {
                    "rgb": rgb,
                    "marker_y": self.marker_y[idx],
                    "phenotype_y": self.phenotype_y[idx],
                    "disagree": self.disagree[idx],
                    "pred": self.pred_features[idx],
                    "core_mean": self.core_mean_features[idx],
                }
        rgb = crop.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
        return {
            "rgb": rgb,
            "marker_y": self.marker_y[idx],
            "phenotype_y": self.phenotype_y[idx],
            "disagree": self.disagree[idx],
            "pred": self.pred_features[idx],
            "core_mean": self.core_mean_features[idx],
        }


def cell_patch_collate(
    batch: list[dict[str, Any] | None],
) -> dict[str, torch.Tensor]:
    """Stack dict samples into batched tensors. Skips ``None`` samples
    (centroids too close to a border for the patch to fit)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    rgb = torch.from_numpy(np.stack([b["rgb"] for b in batch], axis=0))
    marker_y = torch.from_numpy(np.stack([b["marker_y"] for b in batch], axis=0))
    phenotype_y = torch.from_numpy(np.stack([b["phenotype_y"] for b in batch], axis=0))
    disagree = torch.from_numpy(np.stack([b["disagree"] for b in batch], axis=0))
    pred = torch.from_numpy(np.stack([b["pred"] for b in batch], axis=0))
    core_mean = torch.from_numpy(np.stack([b["core_mean"] for b in batch], axis=0))
    return {
        "rgb": rgb,
        "marker_y": marker_y,
        "phenotype_y": phenotype_y,
        "disagree": disagree,
        "pred": pred,
        "core_mean": core_mean,
    }


# ============================================================================ #
# Metric helpers
# ============================================================================ #


def safe_metric(fn, y: np.ndarray, s: np.ndarray) -> float:
    """Compute ``fn(y, s)`` returning NaN if y has fewer than two unique
    values (e.g., a marker with zero positives in the eval set)."""
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(fn(y, s))
    except Exception:
        return float("nan")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return ``(threshold, F1)`` at the operating point that maximizes F1
    on the *eval* set. This is a per-marker upper-bound, not a calibrated
    threshold — useful as a ceiling, not as a deployment threshold."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = (2 * p * r) / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1))
    return float(t[min(i, len(t) - 1)]) if len(t) else 0.5, float(f1[i])


def parse_int_list(s: str) -> list[int]:
    """Parse a comma-separated list of integers — used for ``--marker_channels``."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================================ #
# Argparse — the args every cell-phenotype trainer shares
# ============================================================================ #


def add_shared_args(
    parser: argparse.ArgumentParser,
    *,
    default_batch: int = 128,
    default_grad_accum: int = 1,
) -> None:
    """Add CLI args common to all cell-phenotype trainers.

    Args:
        parser: an ``argparse.ArgumentParser`` to mutate.
        default_batch: micro-batch size.
        default_grad_accum: gradient-accumulation step count.
    """
    parser.add_argument("--train_cell_table", required=True)
    parser.add_argument("--val_cell_table", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--mlp_hidden", default="768,384")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--lr_lora", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--w_phenotype", type=float, default=0.5)
    parser.add_argument("--w_hierarchy", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    # --- Marker-loss selector (v4 = focal, v6 = ASL). -------------------------
    # ``focal`` is the v1.1 / v2 / v4 default; ``asl`` is Asymmetric Loss
    # (Ridnik et al. 2021, arXiv 2009.14119) — designed for rare-positive
    # multi-label binary classification, state of the art on MS-COCO / NUS-WIDE /
    # Open Images. v6 swaps the marker loss to ASL on top of the v4
    # encoder + parallel head; everything else is unchanged.
    parser.add_argument(
        "--marker_loss",
        choices=["focal", "asl"],
        default="focal",
        help=(
            "Loss family for the marker head. 'focal' = class-balanced focal "
            "BCE (v1.1/v2/v4 default). 'asl' = Asymmetric Loss "
            "(Ridnik et al. 2021); decoupled γ⁺/γ⁻ + probability-shift `m`. "
            "When 'asl' is selected, class-balanced α weighting is "
            "automatically disabled for the marker loss per the paper "
            "(the focusing exponents subsume α). Phenotype loss is "
            "unaffected by this flag."
        ),
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=0.0,
        help=(
            "ASL γ⁺ — focusing on positives. Paper default 0; preserves "
            "full gradient on rare-positive cells."
        ),
    )
    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help=(
            "ASL γ⁻ — focusing on negatives. Paper GitHub default 4 "
            "(γ⁻=2 with clip=0.2 is the ICCV camera-ready main config; "
            "ablation is non-monotone, γ⁻=8 underperforms γ⁻=2)."
        ),
    )
    parser.add_argument(
        "--asl_clip",
        type=float,
        default=0.05,
        help=(
            "ASL probability-shift margin `m`. Paper GitHub default 0.05 "
            "(the ICCV main uses 0.2; the ablation saturates past 0.05)."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_cores", type=int, default=0)
    parser.add_argument("--max_val_cores", type=int, default=0)
    parser.add_argument(
        "--max_train_cells_per_epoch",
        type=int,
        default=120000,
        help="Random-sample this many cells per epoch from train; 0 = use all (much slower).",
    )
    parser.add_argument(
        "--label_set",
        choices=list(VALID_LABEL_SETS),
        required=True,
        help=(
            "Which per-marker positivity column family to read as training "
            "targets: 'gmm' (chXX_pos_gmm_orig / chXX_pos, v1.0 default), "
            "'consensus' (chXX_pos_consensus, v1.1 / v2 default), or 'spacec' "
            "(chXX_pos_spacec, v4 default — see "
            "docs/reproducibility.md)."
        ),
    )
    parser.add_argument("--apply_disagree_mask", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--marker_channels", default=",".join(str(x) for x in FOCUSED_MARKERS))
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=default_grad_accum,
        help=(
            "Number of micro-batches to accumulate gradients across before "
            "calling opt.step(). Effective batch = --batch × --grad_accum_steps. "
            "Default 1 = no accumulation (v1.1 / v2). v4 launches at 4 to fit "
            "ViT-Huge / Giant on a 12-24 GB GPU at micro-batch 32."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: tiny config (small batch, 2 epochs, few cores)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore any existing last_state.pt in --output_dir and start "
            "training from epoch 0. Without this flag, training auto-resumes "
            "from last_state.pt if it exists — this is what makes SLURM "
            "--requeue work after preemption."
        ),
    )


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    """Apply --smoke overrides in-place: 2 epochs, 4 train / 2 val cores,
    2000 train cells/epoch, batch 32, num_workers 0. Idempotent."""
    args.epochs = 2
    args.batch = 32
    args.max_train_cores = 4
    args.max_val_cores = 2
    args.max_train_cells_per_epoch = 2000
    args.num_workers = 0


def set_seeds(seed: int) -> None:
    """Seed all RNGs the trainer touches. Call this BEFORE building the
    model — LoRA A/B matrices are randomly initialized, so seeding after
    construction silently gives non-reproducible weights across runs."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


# ============================================================================ #
# Validation loop
# ============================================================================ #


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run validation over ``loader`` and return
    ``(marker_probs, phenotype_probs, marker_y, phenotype_y)``.

    All arrays have shape ``(n_valid_cells, n_classes)``. Empty tuples are
    returned when the loader yielded no valid batches (e.g., all cells were
    filtered out by the border check).
    """
    model.eval()
    mp_all: list[np.ndarray] = []
    pp_all: list[np.ndarray] = []
    ym_all: list[np.ndarray] = []
    yp_all: list[np.ndarray] = []
    for batch in loader:
        if not batch:
            continue
        rgb = batch["rgb"].to(device, non_blocking=True)
        pred = batch["pred"].to(device, non_blocking=True)
        core_mean = batch["core_mean"].to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                mlogits, plogits = model(rgb, pred, core_mean)
            mp = torch.sigmoid(mlogits).float().cpu().numpy()
            pp = torch.sigmoid(plogits).float().cpu().numpy()
        mp_all.append(mp)
        pp_all.append(pp)
        ym_all.append(batch["marker_y"].numpy())
        yp_all.append(batch["phenotype_y"].numpy())
    if not mp_all:
        return (np.zeros((0, 12)), np.zeros((0, 9)), np.zeros((0, 12)), np.zeros((0, 9)))
    return (
        np.concatenate(mp_all, axis=0),
        np.concatenate(pp_all, axis=0),
        np.concatenate(ym_all, axis=0),
        np.concatenate(yp_all, axis=0),
    )


# ============================================================================ #
# Checkpoint save / load — per-epoch state for SLURM --requeue
# ============================================================================ #


def _save_training_state(
    path: Path,
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: dict[str, Any],
    no_improve: int,
    epoch_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    info: dict[str, Any],
) -> None:
    """Atomically dump full training state to ``path``.

    The state is sufficient to resume training bit-for-bit (modulo cuDNN
    nondeterminism, which we don't try to control here) from the next
    epoch. Write goes to ``path.tmp`` first then ``Path.replace`` swaps it
    in atomically, so a process kill mid-save can corrupt the tmp file
    but never the canonical state.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best": best,
        "no_improve": int(no_improve),
        "epoch_rows": epoch_rows,
        "torch_rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
        "args": vars(args),
        "info": info,
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, tmp)
    tmp.replace(path)


def _load_training_state(
    path: Path,
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, dict[str, Any], int, list[dict[str, Any]]]:
    """Restore training state from ``path`` into the supplied objects in-place.

    Returns ``(next_epoch, best, no_improve, epoch_rows)`` for the caller
    to plug into the training loop. ``next_epoch`` is ``saved_epoch + 1``
    (training resumes at the next un-trained epoch).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("torch_cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])
    np.random.set_state(ckpt["np_rng_state"])
    return (
        int(ckpt["epoch"]) + 1,
        ckpt["best"],
        int(ckpt["no_improve"]),
        list(ckpt.get("epoch_rows", [])),
    )


# ============================================================================ #
# Training loop
# ============================================================================ #


def _build_param_groups(model: nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Group trainable params by role so head + LoRA can use different LRs.

    Returns three param groups: LoRA (A.weight + B.weight), head, and any
    other trainable params (e.g., feat_norm if affine).
    """
    lora_params = [
        p for n, p in model.named_parameters() if (".A." in n or ".B." in n) and p.requires_grad
    ]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and ".A." not in n and ".B." not in n and "head" not in n
    ]
    if other_params:
        logging.info("other trainable params: %d", sum(x.numel() for x in other_params))
    return [
        {
            "params": lora_params,
            "lr": float(args.lr_lora),
            "weight_decay": float(args.lora_weight_decay),
        },
        {
            "params": head_params,
            "lr": float(args.lr_head),
            "weight_decay": float(args.weight_decay),
        },
        {
            "params": other_params,
            "lr": float(args.lr_head),
            "weight_decay": float(args.weight_decay),
        },
    ]


def _filter_dataframe_by_max_cores(df: pd.DataFrame, max_cores: int) -> pd.DataFrame:
    if max_cores <= 0:
        return df
    keep = sorted(df["basename"].unique())[:max_cores]
    return df[df.basename.isin(keep)].reset_index(drop=True)


def _build_train_loader(
    train_ds: CellPatchDataset,
    args: argparse.Namespace,
    epoch: int,
) -> torch.utils.data.DataLoader:
    """Build the per-epoch train DataLoader.

    Either samples ``max_train_cells_per_epoch`` cells without replacement
    (deterministic per epoch via ``seed + epoch``) or shuffles all cells.
    """
    if args.max_train_cells_per_epoch > 0 and len(train_ds) > args.max_train_cells_per_epoch:
        sel = np.random.default_rng(int(args.seed) + epoch).choice(
            len(train_ds), size=int(args.max_train_cells_per_epoch), replace=False
        )
        sampler: Any = torch.utils.data.SubsetRandomSampler(sel.tolist())
        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=int(args.batch),
            sampler=sampler,
            num_workers=int(args.num_workers),
            collate_fn=cell_patch_collate,
            pin_memory=True,
            persistent_workers=int(args.num_workers) > 0,
        )
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=cell_patch_collate,
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )


def _run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    args: argparse.Namespace,
    device: torch.device,
    marker_channels: list[int],
    alpha_m: torch.Tensor,
    alpha_p: torch.Tensor,
) -> tuple[float, int]:
    """Train one epoch with focal-BCE + hierarchy loss + gradient accumulation.

    Returns ``(loss_sum, n_seen)`` so the caller can compute mean loss.
    """
    model.train()
    loss_sum = 0.0
    n_seen = 0
    # Gradient accumulation. We divide the per-micro-batch loss by
    # grad_accum_steps so the accumulated gradients have the same magnitude
    # as a single forward+backward at the effective batch size.
    # opt.step() / scaler.update() / opt.zero_grad() run only at the end of
    # each accumulation cycle. For grad_accum_steps=1 this collapses to the
    # standard one-micro-batch-per-step loop.
    accum = max(1, int(args.grad_accum_steps))
    opt.zero_grad(set_to_none=True)
    n_accumulated = 0
    for batch in loader:
        if not batch:
            continue
        rgb = batch["rgb"].to(device, non_blocking=True)
        ym = batch["marker_y"].to(device, non_blocking=True)
        yp = batch["phenotype_y"].to(device, non_blocking=True)
        pred = batch["pred"].to(device, non_blocking=True)
        core_mean = batch["core_mean"].to(device, non_blocking=True)
        disagree = batch["disagree"].to(device, non_blocking=True).bool()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            mlogits, plogits = model(rgb, pred, core_mean)
            if args.apply_disagree_mask:
                mask = torch.zeros(len(marker_channels), dtype=torch.bool, device=device)
                for i in UNSTABLE_MARKERS_BY_INDEX:
                    mask[i] = True
                ig = disagree & mask.view(1, -1)
            else:
                ig = None
            # Marker-loss dispatch. v1.1 / v2 / v4 use class-balanced focal BCE
            # (the ``alpha_m`` path); v6 switches to Asymmetric Loss (Ridnik
            # 2021) — the paper argues against combining ASL with α, so we
            # drop α on the ASL branch.
            marker_loss_kind = getattr(args, "marker_loss", "focal")
            if marker_loss_kind == "asl":
                l_m = asymmetric_loss_with_logits(
                    mlogits,
                    ym,
                    gamma_pos=float(args.asl_gamma_pos),
                    gamma_neg=float(args.asl_gamma_neg),
                    clip=float(args.asl_clip),
                    ignore_mask=ig,
                )
            else:
                l_m = focal_bce_with_logits(mlogits, ym, alpha_m, float(args.focal_gamma), ig)
            # Phenotype + hierarchy losses stay unchanged across all variants —
            # they operate on a separate parallel head whose imbalance regime
            # is mild enough that ASL hasn't been ablated for it. (Future
            # v6.1 may switch the phenotype head to ASL too; deferred.)
            l_p = focal_bce_with_logits(plogits, yp, alpha_p, float(args.focal_gamma))
            l_h = hierarchy_loss(plogits)
            loss = (l_m + float(args.w_phenotype) * l_p + float(args.w_hierarchy) * l_h) / accum
        scaler.scale(loss).backward()
        # Track un-scaled loss for reporting (matches the loss landscape
        # regardless of accumulation factor).
        loss_sum += float(loss.item()) * rgb.shape[0] * accum
        n_seen += rgb.shape[0]
        n_accumulated += 1
        if n_accumulated == accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            n_accumulated = 0
    # Flush any remaining accumulated gradients at end-of-epoch. With
    # grad_accum_steps=1 this is a no-op. With larger accumulation it lets
    # the final partial cycle still contribute an update; the effective
    # batch for that final step is smaller but the convergence impact is
    # negligible.
    if n_accumulated > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    return loss_sum, n_seen


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_final_metrics(
    out: Path,
    *,
    args: argparse.Namespace,
    info: dict[str, Any],
    marker_channels: list[int],
    best: dict[str, Any],
) -> dict[str, Any]:
    """Write marker_metrics.csv + phenotype_metrics.csv + metrics.json at
    the best-epoch operating point. Returns the summary dict for tests."""
    marker_rows: list[dict[str, Any]] = []
    for ci, ch in enumerate(marker_channels):
        name = MARKER_NAMES[ci] if ci < len(MARKER_NAMES) else f"ch{ch:02d}"
        y_e = best["marker_y"][:, ci]
        p_e = best["marker_probs"][:, ci]
        best_tau, best_f1 = best_f1_threshold(y_e, p_e)
        marker_rows.append(
            {
                "channel": int(ch),
                "name": name,
                "n_eval_cells": len(y_e),
                "positive_fraction": float(np.mean(y_e)),
                "auc": safe_metric(roc_auc_score, y_e, p_e),
                "ap": safe_metric(average_precision_score, y_e, p_e),
                "f1_at_0p5": safe_metric(f1_score, y_e, p_e > 0.5),
                "f1_train_threshold": safe_metric(f1_score, y_e, p_e > 0.5),
                "best_possible_f1": best_f1,
                "best_threshold_eval_only": best_tau,
            }
        )
    pheno_rows: list[dict[str, Any]] = []
    for pi, pname in enumerate(PHENOTYPE_NAMES):
        y_e = best["phenotype_y"][:, pi]
        s_e = best["phenotype_probs"][:, pi]
        best_tau, best_f1 = best_f1_threshold(y_e, s_e)
        pheno_rows.append(
            {
                "phenotype": pname,
                "n_eval_cells": len(y_e),
                "positive_fraction": float(np.mean(y_e)),
                "auc": safe_metric(roc_auc_score, y_e, s_e),
                "ap": safe_metric(average_precision_score, y_e, s_e),
                "f1_at_0p5": safe_metric(f1_score, y_e, s_e > 0.5),
                "f1_train_threshold": safe_metric(f1_score, y_e, s_e > 0.5),
                "best_possible_f1": best_f1,
                "best_threshold_eval_only": best_tau,
            }
        )

    _write_csv(out / "marker_metrics.csv", marker_rows)
    _write_csv(out / "phenotype_metrics.csv", pheno_rows)
    summary = {
        "args": vars(args),
        "best_epoch": int(best["epoch"]),
        "macro_marker_ap": float(best["macro_marker_ap"]),
        "macro_phenotype_ap": float(np.nanmean([r["ap"] for r in pheno_rows])),
        "marker_metrics": marker_rows,
        "phenotype_metrics": pheno_rows,
        "info": info,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def train(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    info: dict[str, Any],
    marker_channels: list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Full training loop: load data → optimize → eval → early stop → write metrics.

    Caller responsibilities (handled in each trainer script):
      - Build ``model`` (with LoRA applied + freeze policy set).
      - Provide ``info`` (a logging-only dict with model construction details:
        ``n_lora_modules``, ``n_lora_params``, ``n_head_params``,
        ``n_trainable``, ``encoder_chs``). Persisted to ``metrics.json`` at the
        end so eval scripts can reconstruct the encoder.
      - Pre-create ``args.output_dir`` if necessary.

    Returns the best-epoch summary written to ``metrics.json``.
    """
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info("loading cell tables")
    train_df = pd.read_csv(args.train_cell_table)
    val_df = pd.read_csv(args.val_cell_table)
    train_df = _filter_dataframe_by_max_cores(train_df, int(args.max_train_cores))
    val_df = _filter_dataframe_by_max_cores(val_df, int(args.max_val_cores))
    logging.info("train cells=%d, val cells=%d", len(train_df), len(val_df))

    logging.info("training targets: label_set=%s", args.label_set)
    train_ds = CellPatchDataset(
        cell_table=train_df,
        pairs_dir=Path(args.pairs_dir),
        marker_channels=marker_channels,
        patch_size=args.patch_size,
        train=True,
        label_set=str(args.label_set),
    )
    val_ds = CellPatchDataset(
        cell_table=val_df,
        pairs_dir=Path(args.pairs_dir),
        marker_channels=marker_channels,
        patch_size=args.patch_size,
        train=False,
        label_set=str(args.label_set),
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(args.batch) * 2,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=cell_patch_collate,
        pin_memory=True,
    )

    logging.info(
        "model built: lora_modules=%d lora_params=%d head_params=%d trainable=%d enc_chs=%s",
        info["n_lora_modules"],
        info["n_lora_params"],
        info["n_head_params"],
        info["n_trainable"],
        info["encoder_chs"],
    )

    opt = torch.optim.AdamW(_build_param_groups(model, args))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Class-balanced focal alphas from train labels.
    pos_m = train_ds.marker_y.sum(axis=0) + 1
    neg_m = (1 - train_ds.marker_y).sum(axis=0) + 1
    alpha_m = torch.tensor(neg_m / (pos_m + neg_m), dtype=torch.float32, device=device)
    pos_p = train_ds.phenotype_y.sum(axis=0) + 1
    neg_p = (1 - train_ds.phenotype_y).sum(axis=0) + 1
    alpha_p = torch.tensor(neg_p / (pos_p + neg_p), dtype=torch.float32, device=device)

    best: dict[str, Any] = {
        "epoch": -1,
        "macro_marker_ap": -1.0,
        "marker_probs": None,
        "phenotype_probs": None,
        "marker_y": None,
        "phenotype_y": None,
    }
    no_improve = 0
    epoch_rows: list[dict[str, Any]] = []

    # Auto-resume from last_state.pt unless --restart was passed. This is
    # what makes SLURM --requeue work: after preemption, the re-launched
    # job uses the same --output_dir, sees last_state.pt, and picks up at
    # the next un-trained epoch. --restart forces a fresh start.
    start_epoch = 0
    resume_path = out / "last_state.pt"
    if resume_path.exists() and not getattr(args, "restart", False):
        logging.info("resuming from %s", resume_path)
        start_epoch, best, no_improve, epoch_rows = _load_training_state(
            resume_path,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
        )
        logging.info(
            "resumed: next epoch=%d (of %d), best_marker_ap=%.4f, no_improve=%d",
            start_epoch,
            int(args.epochs),
            float(best.get("macro_marker_ap", -1.0)),
            no_improve,
        )
        if start_epoch >= int(args.epochs):
            logging.info(
                "saved state is already at args.epochs=%d; writing final metrics and exiting",
                int(args.epochs),
            )
            if best["marker_probs"] is None:
                raise RuntimeError("resumed but best has no predictions — checkpoint corrupt?")
            return _write_final_metrics(
                out,
                args=args,
                info=info,
                marker_channels=marker_channels,
                best=best,
            )

    for ep in range(start_epoch, int(args.epochs)):
        train_loader = _build_train_loader(train_ds, args, epoch=ep)
        t0 = time.time()
        loss_sum, n_seen = _run_one_epoch(
            model,
            train_loader,
            opt,
            scaler,
            args=args,
            device=device,
            marker_channels=marker_channels,
            alpha_m=alpha_m,
            alpha_p=alpha_p,
        )
        sched.step()
        train_dt = time.time() - t0

        mp, pp, ym_full, yp_full = evaluate(model, val_loader, device)
        macro_ap_marker = (
            float(
                np.nanmean(
                    [
                        safe_metric(average_precision_score, ym_full[:, j], mp[:, j])
                        for j in range(ym_full.shape[1])
                    ]
                )
            )
            if mp.size
            else float("nan")
        )
        macro_ap_pheno = (
            float(
                np.nanmean(
                    [
                        safe_metric(average_precision_score, yp_full[:, j], pp[:, j])
                        for j in range(yp_full.shape[1])
                    ]
                )
            )
            if pp.size
            else float("nan")
        )

        improved = macro_ap_marker > best["macro_marker_ap"]
        if improved:
            best.update(
                epoch=ep,
                macro_marker_ap=macro_ap_marker,
                marker_probs=mp,
                phenotype_probs=pp,
                marker_y=ym_full,
                phenotype_y=yp_full,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": ep,
                    "macro_marker_ap": macro_ap_marker,
                },
                out / "best_model.pt",
            )
            no_improve = 0
        else:
            no_improve += 1
        epoch_rows.append(
            {
                "epoch": ep,
                "loss": loss_sum / max(n_seen, 1),
                "val_macro_ap_marker": macro_ap_marker,
                "val_macro_ap_phenotype": macro_ap_pheno,
                "train_seconds": train_dt,
            }
        )
        logging.info(
            "epoch %02d  loss=%.4f  val_marker_AP=%.4f  val_pheno_AP=%.4f  best=%.4f%s  (%.1fs train)",
            ep,
            loss_sum / max(n_seen, 1),
            macro_ap_marker,
            macro_ap_pheno,
            best["macro_marker_ap"],
            " *" if improved else "",
            train_dt,
        )
        # Persist full training state every epoch (atomic write). Combined
        # with SLURM ``--requeue``, this lets a preempted job pick up
        # exactly where it left off when re-launched. ``epoch_log.csv`` is
        # also rewritten each epoch so a preempted run still leaves a
        # usable on-disk history.
        _save_training_state(
            out / "last_state.pt",
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            epoch=ep,
            best=best,
            no_improve=no_improve,
            epoch_rows=epoch_rows,
            args=args,
            info=info,
        )
        pd.DataFrame(epoch_rows).to_csv(out / "epoch_log.csv", index=False)
        if no_improve >= int(args.early_stop_patience):
            logging.info("early stop at epoch %d", ep)
            break

    if best["marker_probs"] is None:
        raise RuntimeError("training produced no predictions")

    summary = _write_final_metrics(
        out,
        args=args,
        info=info,
        marker_channels=marker_channels,
        best=best,
    )
    print(f"wrote {out / 'metrics.json'}  best marker AP={best['macro_marker_ap']:.4f}")
    return summary
