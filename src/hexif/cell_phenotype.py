"""HEXIF cell-phenotype v1.

Cell-level phenotype prediction from H&E.  Uses pre-extracted, frozen
features from a raw pretrained CTransPath encoder (D2 found that
virtual-mIF fine-tuning *degrades* cell-level signal).  Multi-task
heads predict per-cell marker positivity and phenotype labels with a
soft hierarchy regulariser.

See ``docs/reproducibility.md`` for the full spec.
"""

from __future__ import annotations

import warnings
from typing import Final, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Marker channels (focused panel) — fixed by data/panels/orion.yaml
FOCUSED_MARKERS = (0, 3, 7, 13, 16, 17, 27, 31, 34, 46, 50, 52)
MARKER_NAMES = (
    "DAPI",
    "CD45",
    "CD8",
    "Ki67",
    "CA9",
    "Vimentin",
    "CD68",
    "FAP",
    "CD163",
    "PDL1",
    "aSMA",
    "panCK",
)


# Phenotype names — must match scripts/calibrate_codex_cell_phenotypes.py
PHENOTYPE_NAMES = (
    "immune_cd45",
    "t_cell_cd45_cd8",
    "tumor_ca9_or_panck",
    "macrophage_cd68_or_cd163",
    "m2_like_cd68_cd163",
    "caf_fap_or_asma",
    "proliferating_ki67",
    "pdl1_positive",
    "pdl1_tumor_like",
)

# Hierarchy: child phenotype → parent phenotype index in PHENOTYPE_NAMES
# Each entry says: child_idx must imply parent_idx.
PHENOTYPE_HIERARCHY: tuple[tuple[int, int], ...] = (
    (1, 0),  # t_cell_cd45_cd8 implies immune_cd45
    (4, 3),  # m2_like implies macrophage
    (8, 7),  # pdl1_tumor_like implies pdl1_positive
    (8, 2),  # pdl1_tumor_like implies tumor_ca9_or_panck
)

# Markers whose train-fit thresholds disagree most across methods (D1
# Cohen's κ < 0.55 between GMM and Otsu).  For these markers, cells in
# the "disagree" stratum (1 or 2 of 3 methods positive) are excluded
# from the per-marker loss (`ignore_mask = pos_disagree`).
UNSTABLE_MARKERS_BY_INDEX: tuple[int, ...] = (
    1,  # CD45
    2,  # CD8
    3,  # Ki67
    9,  # PDL1
    10,  # aSMA
)

# Legal values for the ``label_set`` parameter of ``cell_table_to_targets`` and
# the ``--label_set`` flag of ``train_cell_phenotype_v1_1.py``. Centralized so
# the trainer, the function, and the tests never drift.
VALID_LABEL_SETS: Final[tuple[str, ...]] = ("gmm", "consensus", "spacec")


def phenotype_targets_from_marker_pos(marker_pos: np.ndarray) -> np.ndarray:
    """Compute consensus phenotype labels from marker positivity booleans.

    ``marker_pos`` shape: (N, 12) with columns matching FOCUSED_MARKERS order.
    Returns (N, 9) boolean array matching PHENOTYPE_NAMES order.
    """
    p = marker_pos.astype(bool)
    # marker index in FOCUSED_MARKERS:
    # 0 DAPI, 1 CD45, 2 CD8, 3 Ki67, 4 CA9, 5 Vimentin, 6 CD68, 7 FAP,
    # 8 CD163, 9 PDL1, 10 aSMA, 11 panCK
    out = np.empty((p.shape[0], len(PHENOTYPE_NAMES)), dtype=bool)
    out[:, 0] = p[:, 1]  # immune_cd45
    out[:, 1] = p[:, 1] & p[:, 2]  # t_cell_cd45_cd8
    out[:, 2] = p[:, 4] | p[:, 11]  # tumor_ca9_or_panck
    out[:, 3] = p[:, 6] | p[:, 8]  # macrophage
    out[:, 4] = p[:, 6] & p[:, 8]  # m2_like
    out[:, 5] = p[:, 7] | p[:, 10]  # caf
    out[:, 6] = p[:, 3]  # ki67
    out[:, 7] = p[:, 9]  # pdl1
    out[:, 8] = p[:, 9] & (p[:, 4] | p[:, 11])  # pdl1_tumor_like
    return out


class CellPhenotypeV1Head(nn.Module):
    """Multi-task head over pre-extracted cell features.

    Input: D-dim cell feature vector (e.g. 1656 for raw encoder + chXX_pred + per-core mean).
    Output: 12 marker logits + 9 phenotype logits.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (768, 384),
        dropout: float = 0.1,
        n_markers: int = 12,
        n_phenotypes: int = 9,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(in_dim)
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.marker_head = nn.Linear(last, n_markers)
        self.phenotype_head = nn.Linear(last, n_phenotypes)
        self.in_dim = int(in_dim)
        self.n_markers = int(n_markers)
        self.n_phenotypes = int(n_phenotypes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.marker_head(h), self.phenotype_head(h)


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
    ignore_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-balanced focal BCE with optional per-cell-per-class ignore mask.

    Shapes: logits / target / ignore_mask broadcastable to (B, C).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    a = torch.where(target > 0.5, alpha, 1 - alpha)
    loss = a * (1 - pt).pow(gamma) * bce
    if ignore_mask is not None:
        keep = (~ignore_mask).to(loss.dtype)
        denom = keep.sum().clamp_min(1.0)
        return (loss * keep).sum() / denom
    return loss.mean()


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    ignore_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Asymmetric Loss for multi-label classification (Ridnik et al. 2021).

    Paper: https://arxiv.org/abs/2009.14119 — ICCV 2021. state of the art on MS-COCO /
    NUS-WIDE / Open Images multi-label benchmarks; designed specifically
    for the positive-negative imbalance regime where rare positives
    coexist with abundant easy negatives — exactly our CD8 / Ki67 /
    CD163 regime (7-13% positive prevalence on the focused panel).

    Two changes vs ``focal_bce_with_logits``:

    1. **Decoupled focusing** (γ⁺, γ⁻). Standard focal BCE uses one γ for
       both classes; ASL decouples them so easy negatives can be
       suppressed hard (large γ⁻) while gradient signal on rare
       positives is preserved (γ⁺ → 0).
    2. **Probability shifting** (``clip`` / ``m``). The negative-side
       probability is shifted ``p_m = max(p - m, 0)``, which forces the
       loss to **exactly zero** for negatives the model already predicts
       below the threshold (vs focal-BCE's "approximately zero but
       still differentiable"). This is the headline trick — it discards
       easy-negative gradients entirely rather than just down-weighting
       them.

    Math (paper Eq. 4)::

        L = -[ y · (1 - p)^γ⁺ · log(p)
              + (1 - y) · p_m^γ⁻ · log(1 - p_m) ]
        where  p   = sigmoid(logits),  p_m = max(p - clip, 0).

    Args:
        logits: ``(B, C)`` raw logits.
        target: ``(B, C)`` binary targets in ``{0, 1}`` (floats).
        gamma_pos: focusing exponent for **positives**. Default ``0``
            (paper main: γ⁺=0) means no down-weighting of positives →
            preserves gradient on rare-positive cells.
        gamma_neg: focusing exponent for **negatives**. Default ``4``
            matches the Alibaba-MIIL reference implementation and the
            paper's preprint defaults; the ICCV camera-ready also tested
            ``2`` (with ``clip=0.2``) and reports a non-monotone
            sensitivity — ``γ⁻=8`` underperforms ``γ⁻=2`` by 1.1 mAP on
            MS-COCO. Worth ablating on real data.
        clip: probability-shift margin ``m``. Default ``0.05`` matches
            the official implementation; the paper's main config uses
            ``0.2``. The ablation in the paper saturates after ``0.05``
            (gain beyond that is < 0.3 mAP).
        eps: numerical floor for the ``log(1 - p_m)`` term — prevents
            ``log(0)`` for very confident predictions in fp16 AMP.
        ignore_mask: optional ``(B, C)`` bool tensor; True entries are
            excluded from the mean. Same semantics as
            :func:`focal_bce_with_logits`'s ``ignore_mask``.

    Returns:
        Scalar loss (mean over non-masked entries).

    Notes:
        - The focusing weights ``(1-p)^γ⁺`` and ``p_m^γ⁻`` are computed
          under :func:`torch.no_grad` — matches the paper's
          ``disable_torch_grad_focal_loss=True`` default, which the
          authors note is critical for training stability (it prevents
          the focusing terms from contributing high-variance second-
          order gradients to the backward pass).
        - The function does **not** combine ASL with class-balanced
          α weighting. The paper explicitly argues against this — "Simple
          linear weighting is insufficient to tackle negative-positive
          imbalance"; the focusing exponents are meant to subsume α.
          Use this loss as a drop-in replacement for
          ``focal_bce_with_logits`` and drop the α parameter.
        - In the BCE limit ``γ⁺ = γ⁻ = 0, clip = 0`` this function
          reduces exactly to :func:`F.binary_cross_entropy_with_logits`
          (mean reduction). Tested in
          ``tests/test_asymmetric_loss.py``.
    """
    if gamma_pos < 0 or gamma_neg < 0:
        raise ValueError(
            f"asymmetric_loss_with_logits: focusing γ must be >= 0; "
            f"got γ⁺={gamma_pos}, γ⁻={gamma_neg}"
        )
    if clip < 0 or clip >= 1:
        raise ValueError(f"asymmetric_loss_with_logits: clip ``m`` must be in [0, 1); got {clip}")

    # Numerically stable log-sigmoid for the positive (BCE) term:
    # log(p) = log(sigmoid(logits)) = -softplus(-logits) = F.logsigmoid(logits).
    log_p = F.logsigmoid(logits)

    # Probability of the positive class (used for negative-side focusing
    # and for the probability-shift). Computing sigmoid once is cheaper
    # than logsigmoid + exp.
    p = torch.sigmoid(logits)
    p_m = (p - clip).clamp_min(0.0) if clip > 0 else p

    # Negative-side log term: log(1 - p_m). Uses .clamp_min(eps) to keep
    # the gradient defined when p_m → 1 (very confident wrong predictions
    # for a true negative). In fp16, eps=1e-8 is below the smallest
    # representable normal — torch promotes to fp32 inside log silently,
    # so this is safe under AMP.
    log_1m_pm = torch.log((1.0 - p_m).clamp_min(eps))

    # Focusing weights — detached so they act as fixed scaling factors
    # (paper's ``disable_torch_grad_focal_loss=True``). Without this, the
    # focusing-weight gradients add high-variance second-order terms that
    # destabilize training; the authors document this as a critical
    # implementation detail.
    with torch.no_grad():
        w_pos = (1.0 - p).pow(gamma_pos) if gamma_pos > 0 else torch.ones_like(p)
        w_neg = p_m.pow(gamma_neg) if gamma_neg > 0 else torch.ones_like(p_m)

    # Per-element loss = -(positive_term + negative_term).
    loss = -(target * w_pos * log_p + (1.0 - target) * w_neg * log_1m_pm)

    if ignore_mask is not None:
        keep = (~ignore_mask).to(loss.dtype)
        denom = keep.sum().clamp_min(1.0)
        return (loss * keep).sum() / denom
    return loss.mean()


def hierarchy_loss(
    phenotype_logits: torch.Tensor,
    pairs: tuple[tuple[int, int], ...] = PHENOTYPE_HIERARCHY,
) -> torch.Tensor:
    """Soft penalty for child > parent in logit space."""
    if not pairs:
        return phenotype_logits.new_zeros(())
    total = phenotype_logits.new_zeros(())
    for child, parent in pairs:
        diff = F.relu(phenotype_logits[:, child] - phenotype_logits[:, parent])
        total = total + diff.mean()
    return total / float(len(pairs))


# ----------------------------------------------------------------- features


def stack_v1_features(
    feature_dir: str,
    cell_table: pd.DataFrame,
    *,
    feature_set: tuple[str, ...] = ("global", "deep", "fine"),
    include_predictions: bool = True,
    include_core_means: bool = True,
    marker_channels: tuple[int, ...] = FOCUSED_MARKERS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached encoder features and concatenate v1's full feature stack.

    Returns ``(X, kept_rows)`` where ``kept_rows`` is the subset of
    ``cell_table`` whose features were available in the cache.
    """
    from pathlib import Path

    parts_X: list[np.ndarray] = []
    parts_rows: list[pd.DataFrame] = []
    bn_groups = cell_table.groupby("basename", sort=False)
    fdir = Path(feature_dir)
    for base, group in bn_groups:
        f = fdir / f"{base}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cid_arr = z["cell_id"].astype(np.int64)
        if cid_arr.size == 0:
            continue
        cid_index = pd.Index(cid_arr)
        sel = cid_index.get_indexer(group["cell_id"].to_numpy(dtype=np.int64))
        keep = sel >= 0
        if not keep.any():
            continue
        sel_valid = sel[keep]
        pieces: list[np.ndarray] = []
        if "global" in feature_set:
            pieces.append(np.asarray(z["feat_global"][sel_valid], dtype=np.float32))
        if "deep" in feature_set:
            pieces.append(np.asarray(z["feat_cell_deep"][sel_valid], dtype=np.float32))
        if "fine" in feature_set:
            pieces.append(np.asarray(z["feat_cell_fine"][sel_valid], dtype=np.float32))
        X_core = np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
        parts_X.append(X_core)
        parts_rows.append(group.iloc[keep].reset_index(drop=True))
    if not parts_X:
        return np.zeros((0, 0), dtype=np.float32), cell_table.iloc[0:0]
    X = np.concatenate(parts_X, axis=0)
    kept = pd.concat(parts_rows, ignore_index=True)
    if include_predictions:
        cols = [f"ch{ch:02d}_pred" for ch in marker_channels]
        if not all(c in kept.columns for c in cols):
            raise ValueError(f"Cell table missing pred columns: {cols}")
        X = np.concatenate([X, kept[cols].to_numpy(dtype=np.float32)], axis=1)
    if include_core_means:
        cols = [f"ch{ch:02d}_pred" for ch in marker_channels]
        cm = kept.groupby("basename")[cols].transform("mean").to_numpy(dtype=np.float32)
        X = np.concatenate([X, cm], axis=1)
    return X, kept


def cell_table_to_targets(
    df: pd.DataFrame,
    *,
    marker_channels: tuple[int, ...] = FOCUSED_MARKERS,
    label_set: Literal["gmm", "consensus", "spacec"] = "consensus",
    use_consensus: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (marker_y, marker_disagree_mask, phenotype_y, marker_y_orig).

    Args:
        df: cell table with per-marker positivity columns.
        marker_channels: focused panel channel indices.
        label_set: which positivity column family to read:

            - ``"gmm"``: ``chXX_pos_gmm_orig`` if present, else ``chXX_pos``
              (the original v1.0 default).
            - ``"consensus"``: ``chXX_pos_consensus`` (v1.1 default, the
              2026-05-10 calibrated targets).
            - ``"spacec"``: ``chXX_pos_spacec`` (v2 default, the 2026-05-14
              cluster-then-label targets — see
              ``docs/reproducibility.md``).

        use_consensus: deprecated. ``True`` → ``label_set="consensus"``;
            ``False`` → ``label_set="gmm"``. Emits ``DeprecationWarning``.
            Passing both arguments (non-None ``use_consensus`` AND a
            non-default ``label_set``) raises ``ValueError``.

    Returns:
        ``(marker_y, marker_disagree_mask, phenotype_y, marker_y_orig)``:

        - ``marker_y``: per-cell positivity from the requested
          ``label_set``, shape ``(N, n_markers)`` bool.
        - ``marker_disagree_mask``: per-cell-per-marker mask of entries
          to ignore in the loss; ``True`` = ignore. Read from
          ``chXX_pos_disagree`` under ``"gmm"`` / ``"consensus"``; forced
          to all-False under ``"spacec"`` (cluster-derived labels have no
          "disagreement" notion). Shape ``(N, n_markers)`` bool.
        - ``phenotype_y``: derived from ``marker_y`` via
          ``phenotype_targets_from_marker_pos``. Shape
          ``(N, n_phenotypes)`` bool.
        - ``marker_y_orig``: original GMM-only labels for reporting
          backward-comparable metrics. Read from ``chXX_pos_gmm_orig``
          regardless of ``label_set``; falls back to ``marker_y`` if
          that column family is absent.

    Raises:
        ValueError: if ``label_set="spacec"`` and any required
            ``chXX_pos_spacec`` column is missing; or if both
            ``use_consensus`` and ``label_set`` are explicitly set; or if
            ``label_set`` is not in :data:`VALID_LABEL_SETS`.
    """
    # --- Resolve label_set vs the deprecated use_consensus alias ---
    if use_consensus is not None:
        if label_set != "consensus":
            raise ValueError(
                "Pass either `label_set=...` or the deprecated `use_consensus=...`, "
                f"not both. Got `label_set={label_set!r}` and "
                f"`use_consensus={use_consensus!r}`."
            )
        warnings.warn(
            "`use_consensus` is deprecated; use `label_set='consensus'` (True) or "
            "`label_set='gmm'` (False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        label_set = "consensus" if use_consensus else "gmm"

    if label_set not in VALID_LABEL_SETS:
        raise ValueError(f"Invalid label_set={label_set!r}; must be one of {VALID_LABEL_SETS}.")

    # --- Read marker_y from the column family selected by label_set ---
    if label_set == "spacec":
        missing = [
            f"ch{ch:02d}_pos_spacec"
            for ch in marker_channels
            if f"ch{ch:02d}_pos_spacec" not in df.columns
        ]
        if missing:
            raise ValueError(
                "label_set='spacec' requires chXX_pos_spacec columns for every "
                f"focused marker; first missing column: {missing[0]!r} "
                f"({len(missing)} missing in total). Build them with "
                "`hexif build-spacec-labels`."
            )
        marker_y = np.stack(
            [df[f"ch{ch:02d}_pos_spacec"].to_numpy(dtype=bool) for ch in marker_channels],
            axis=1,
        )
    elif label_set == "consensus":
        if all(f"ch{ch:02d}_pos_consensus" in df.columns for ch in marker_channels):
            marker_y = np.stack(
                [df[f"ch{ch:02d}_pos_consensus"].to_numpy(dtype=bool) for ch in marker_channels],
                axis=1,
            )
        else:
            # consensus columns absent → fall back to the original GMM positivity,
            # matching the v1.1 behaviour before label_set existed.
            marker_y = np.stack(
                [df[f"ch{ch:02d}_pos"].to_numpy(dtype=bool) for ch in marker_channels],
                axis=1,
            )
    else:  # label_set == "gmm"
        if all(f"ch{ch:02d}_pos_gmm_orig" in df.columns for ch in marker_channels):
            marker_y = np.stack(
                [df[f"ch{ch:02d}_pos_gmm_orig"].to_numpy(dtype=bool) for ch in marker_channels],
                axis=1,
            )
        else:
            marker_y = np.stack(
                [df[f"ch{ch:02d}_pos"].to_numpy(dtype=bool) for ch in marker_channels],
                axis=1,
            )

    # --- Disagreement mask: not meaningful under spacec (cluster-derived) ---
    if label_set == "spacec":
        disagree = np.zeros_like(marker_y)
    elif all(f"ch{ch:02d}_pos_disagree" in df.columns for ch in marker_channels):
        disagree = np.stack(
            [df[f"ch{ch:02d}_pos_disagree"].to_numpy(dtype=bool) for ch in marker_channels],
            axis=1,
        )
    else:
        disagree = np.zeros_like(marker_y)

    # --- marker_y_orig is always the GMM-original column (reporting artifact) ---
    if all(f"ch{ch:02d}_pos_gmm_orig" in df.columns for ch in marker_channels):
        marker_y_orig = np.stack(
            [df[f"ch{ch:02d}_pos_gmm_orig"].to_numpy(dtype=bool) for ch in marker_channels],
            axis=1,
        )
    else:
        marker_y_orig = marker_y.copy()

    phenotype_y = phenotype_targets_from_marker_pos(marker_y)
    return marker_y, disagree, phenotype_y, marker_y_orig
