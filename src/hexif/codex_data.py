"""Dataset utilities for H&E -> CODEX training pairs.

The registered CODEX dataset uses ``*_CODEX.npy`` targets with
53 channels, so it needs a separate dataset class and basename discovery path.

The sample contract intentionally mirrors ``HE2OrionDataset``:

    {
        "he":      (3, patch_size, patch_size) normalized tensor,
        "tgt_log": (C, patch_size, patch_size) log-scaled target tensor,
        "info":    dict with coordinates and basename
    }

This lets training code share most downstream loss/evaluation mechanics once
the model head is made CODEX-aware.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from hexif.image_arrays import to_float01
from hexif.scaling import QuantileScaler

try:
    from scipy.ndimage import gaussian_filter, map_coordinates

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def build_codex_basenames(pairs_dir: str | Path) -> list[str]:
    """Return basenames that have both ``*_HE.npy`` and ``*_CODEX.npy`` files."""
    root = Path(pairs_dir)
    basenames: list[str] = []
    for he_path in sorted(root.glob("*_HE.npy")):
        base = he_path.name[: -len("_HE.npy")]
        if (root / f"{base}_CODEX.npy").exists():
            basenames.append(base)
    return basenames


def load_codex_manifest_basenames(
    manifest_csv: str | Path,
    *,
    split: str | None = None,
) -> list[str]:
    """Load basenames from ``manifest_trainable_5px.csv``.

    Args:
        manifest_csv: CSV produced by ``scripts/finalize_codex_training_dataset.py``.
        split: Optional split filter: ``"train"``, ``"val"``, or ``"test"``.
    """
    path = Path(manifest_csv)
    out: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") != split:
                continue
            base = row.get("basename", "")
            if base:
                out.append(base)
    return out


class HE2CodexDataset(Dataset):
    """Patch dataset for registered H&E -> CODEX core pairs.

    CODEX arrays may be channel-first ``(C, H, W)`` or channel-last
    ``(H, W, C)``. Internally, targets are exposed channel-last for slicing and
    returned channel-first as PyTorch tensors.
    """

    def __init__(
        self,
        pairs_dir: str | Path,
        basenames: list[str],
        scaler: QuantileScaler,
        patch_size: int = 224,
        mode: str = "train",
        grid_stride: int = 112,
        augment: bool = True,
        center_window: int = 12,
        pos_frac: float = 0.6,
        pos_threshold: float = 0.10,
        channel_thresholds: np.ndarray | None = None,
        resample_tries: int = 8,
        samples_per_core: int = 64,
        channel_sampling_weights: np.ndarray | None = None,
        min_pos_fraction: float = 0.01,
        channel_min_pos_fraction: float = 0.002,
        positive_region: str = "full",
        center_target_size: int = 32,
    ):
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        if positive_region not in ("full", "center"):
            raise ValueError("positive_region must be 'full' or 'center'")
        self.dir = Path(pairs_dir)
        self.basenames = list(basenames)
        self.scaler = scaler
        self.C = int(scaler.C)
        self.ps = int(patch_size)
        self.mode = mode
        self.grid_stride = int(grid_stride)
        self.augment = bool(augment and mode == "train")
        self.center_window = max(0, int(center_window))
        self.pos_frac = float(pos_frac)
        self.pos_threshold = float(pos_threshold)
        self.resample_tries = int(resample_tries)
        self.samples_per_core = int(samples_per_core)
        self.min_pos_fraction = float(min_pos_fraction)
        self.channel_min_pos_fraction = float(channel_min_pos_fraction)
        self.positive_region = positive_region
        self.center_target_size = int(center_target_size)

        if channel_thresholds is not None:
            ch_thresh = np.asarray(channel_thresholds, dtype=np.float32)
            if ch_thresh.shape != (self.C,):
                raise ValueError("channel_thresholds must have one value per CODEX channel")
            self.channel_thresholds = ch_thresh
        else:
            self.channel_thresholds = None

        if channel_sampling_weights is not None:
            csw = np.asarray(channel_sampling_weights, dtype=np.float64)
            if csw.shape[0] != self.C:
                raise ValueError("channel_sampling_weights must match CODEX channel count")
            if csw.sum() <= 0:
                self.channel_sampling_weights = None
            else:
                csw = np.clip(csw, 1e-8, None)
                self.channel_sampling_weights = (csw / csw.sum()).astype(np.float32)
        else:
            self.channel_sampling_weights = None

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in self.basenames]
        self.codex_paths = [self.dir / f"{b}_CODEX.npy" for b in self.basenames]
        for hp, cp in zip(self.he_paths, self.codex_paths, strict=False):
            if not hp.exists() or not cp.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {cp}")

        self._pair_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._norm_cache: dict[int, float] = {}
        self.shapes: list[tuple[int, int]] = []
        for cp in self.codex_paths:
            arr = np.load(cp, mmap_mode="r")
            if arr.ndim != 3:
                raise RuntimeError(f"Unexpected CODEX shape {arr.shape} for {cp}")
            if arr.shape[0] == self.C:
                h, w = int(arr.shape[1]), int(arr.shape[2])
            elif arr.shape[2] == self.C:
                h, w = int(arr.shape[0]), int(arr.shape[1])
            else:
                raise RuntimeError(
                    f"Expected {self.C} CODEX channels, got shape {arr.shape} for {cp}"
                )
            self.shapes.append((h, w))

        if self.mode == "val":
            grid: list[tuple[int, int, int]] = []
            for i, (h, w) in enumerate(self.shapes):
                ys = (
                    [0]
                    if h <= self.ps
                    else list(range(0, max(1, h - self.ps) + 1, self.grid_stride))
                )
                xs = (
                    [0]
                    if w <= self.ps
                    else list(range(0, max(1, w - self.ps) + 1, self.grid_stride))
                )
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self.grid = grid
            self._len = len(grid)
        else:
            self.grid = None
            self._len = len(self.basenames) * self.samples_per_core

        self._tf_he = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.ps, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tf_train = self._tf_he
        self.tf_eval = self._tf_he

    def __len__(self) -> int:
        return self._len

    def _open_pair(self, idx_core: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._pair_cache.get(idx_core)
        if cached is not None:
            return cached
        he = np.load(self.he_paths[idx_core], mmap_mode="r")
        codex = np.load(self.codex_paths[idx_core], mmap_mode="r")
        if codex.ndim == 3 and codex.shape[0] == self.C:
            codex = np.transpose(codex, (1, 2, 0))
        self._pair_cache[idx_core] = (he, codex)
        return he, codex

    def _load_pair(self, idx_core: int) -> tuple[np.ndarray, np.ndarray]:
        he, codex = self._open_pair(idx_core)
        he = to_float01(np.asarray(he))
        codex = to_float01(np.asarray(codex))
        return he, codex

    def _core_codex_norm(self, idx_core: int) -> float:
        cached = self._norm_cache.get(idx_core)
        if cached is not None:
            return cached
        arr = np.load(self.codex_paths[idx_core], mmap_mode="r")
        if arr.shape[0] == self.C:
            sample = np.asarray(arr[:, ::64, ::64])
        else:
            sample = np.asarray(arr[::64, ::64, :])
        norm = max(float(np.percentile(sample, 99.9)), 1e-6)
        self._norm_cache[idx_core] = norm
        return norm

    def _scale_to_log(self, codex_patch: np.ndarray, norm: float | None = None) -> np.ndarray:
        x = np.asarray(codex_patch).astype(np.float32, copy=False)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D CODEX patch, got {x.shape}")
        channel_first = x.shape[0] == self.C and x.shape[-1] != self.C
        if channel_first:
            x_cf = x
        elif x.shape[-1] == self.C:
            x_cf = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Expected {self.C} CODEX channels, got {x.shape}")
        if x_cf.max(initial=0.0) > 1.5:
            if norm is None:
                norm = max(float(np.percentile(x_cf, 99.9)), 1e-6)
            x_cf = x_cf / max(float(norm), 1e-6)
        out_cf = np.empty_like(x_cf, dtype=np.float32)
        for c in range(self.C):
            z = (x_cf[c] - self.scaler.qlo[c]) / (self.scaler.qhi[c] - self.scaler.qlo[c] + 1e-6)
            out_cf[c] = np.log1p(np.clip(z, 0, None))
        return np.transpose(out_cf, (1, 2, 0)) if not channel_first else out_cf

    @staticmethod
    def _rand_coords(h: int, w: int, ps: int) -> tuple[int, int]:
        if h <= ps or w <= ps:
            return 0, 0
        y0 = np.random.randint(0, h - ps + 1)
        x0 = np.random.randint(0, w - ps + 1)
        return int(y0), int(x0)

    def _is_positive_patch(self, target_log: np.ndarray, channel: int | None = None) -> bool:
        c_total = target_log.shape[2]
        total_px = target_log.shape[0] * target_log.shape[1]
        if channel is not None and (channel < 0 or channel >= c_total):
            channel = None
        channels = [channel] if channel is not None else range(c_total)
        frac_thresh = (
            self.channel_min_pos_fraction if channel is not None else self.min_pos_fraction
        )
        for c in channels:
            thresh = (
                float(self.channel_thresholds[c])
                if self.channel_thresholds is not None
                else self.pos_threshold
            )
            frac = float((target_log[..., c] > thresh).sum()) / max(1, total_px)
            if frac >= frac_thresh:
                return True
        return False

    def _positive_region_view(self, target_log: np.ndarray) -> np.ndarray:
        if self.positive_region != "center":
            return target_log
        h, w = target_log.shape[:2]
        size = min(self.center_target_size, h, w)
        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        return target_log[y0 : y0 + size, x0 : x0 + size, :]

    @staticmethod
    def _paired_geometric_augment(
        he: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(0, 4)
        if k > 0:
            he = np.rot90(he, k, axes=(0, 1)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
        if np.random.rand() < 0.5:
            he = np.flip(he, axis=1).copy()
            target = np.flip(target, axis=1).copy()
        if np.random.rand() < 0.5:
            he = np.flip(he, axis=0).copy()
            target = np.flip(target, axis=0).copy()
        if HAS_SCIPY and np.random.rand() < 0.3:
            he, target = HE2CodexDataset._elastic_deform(he, target)
        return he, target

    @staticmethod
    def _elastic_deform(
        he: np.ndarray, target: np.ndarray, alpha: float = 80.0, sigma: float = 8.0
    ):
        h, w = he.shape[:2]
        dx = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
        dy = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = [np.clip(y_grid + dy, 0, h - 1), np.clip(x_grid + dx, 0, w - 1)]

        he_out = np.stack(
            [
                map_coordinates(he[..., c], coords, order=1, mode="reflect")
                for c in range(he.shape[2])
            ],
            axis=-1,
        ).astype(he.dtype)
        target_out = np.stack(
            [
                map_coordinates(target[..., c], coords, order=1, mode="reflect")
                for c in range(target.shape[2])
            ],
            axis=-1,
        ).astype(target.dtype)
        return he_out, target_out

    @staticmethod
    def _stain_augment(he_rgb: np.ndarray, alpha_range: float = 0.2, beta_range: float = 0.05):
        od = -np.log(np.clip(he_rgb, 1e-6, 1.0))
        alpha = 1.0 + np.random.uniform(-alpha_range, alpha_range, size=3).astype(np.float32)
        beta = np.random.uniform(-beta_range, beta_range, size=3).astype(np.float32)
        od = od * alpha.reshape(1, 1, 3) + beta.reshape(1, 1, 3)
        return np.clip(np.exp(-od), 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int):
        ps = self.ps
        if self.mode == "val":
            if self.grid is None:
                raise RuntimeError("Validation grid was not initialized")
            core_idx, y0, x0 = self.grid[idx]
            he, codex = self._open_pair(core_idx)
            he_crop = to_float01(np.asarray(he[y0 : y0 + ps, x0 : x0 + ps, :]).copy())
            codex_crop = np.asarray(codex[y0 : y0 + ps, x0 : x0 + ps, :]).copy()
            target_log = self._scale_to_log(codex_crop, norm=self._core_codex_norm(core_idx))
            he_img = (np.clip(he_crop, 0.0, 1.0) * 255).astype(np.uint8)
            info = {
                "y0": y0,
                "x0": x0,
                "core_idx": core_idx,
                "basename": self.basenames[core_idx],
                "target_channel": -1,
            }
            return {
                "he": self._tf_he(he_img),
                "tgt_log": torch.from_numpy(target_log.transpose(2, 0, 1)),
                "info": info,
            }

        core_idx = np.random.randint(0, len(self.basenames))
        he, codex = self._open_pair(core_idx)
        want_pos = np.random.rand() < self.pos_frac
        target_channel: int | None = None
        tries = self.resample_tries if want_pos else 1
        if want_pos and self.channel_sampling_weights is not None:
            target_channel = int(np.random.choice(self.C, p=self.channel_sampling_weights))

        for _ in range(tries):
            y0, x0 = self._rand_coords(*self.shapes[core_idx], ps)
            he_crop = to_float01(np.asarray(he[y0 : y0 + ps, x0 : x0 + ps, :]).copy())
            codex_crop = np.asarray(codex[y0 : y0 + ps, x0 : x0 + ps, :]).copy()
            target_log = self._scale_to_log(codex_crop, norm=self._core_codex_norm(core_idx))
            if (not want_pos) or self._is_positive_patch(
                self._positive_region_view(target_log), channel=target_channel
            ):
                break

        if self.augment:
            he_crop, target_log = self._paired_geometric_augment(he_crop, target_log)
            he_crop = self._stain_augment(he_crop)

        he_img = (np.clip(he_crop, 0.0, 1.0) * 255).astype(np.uint8)
        info = {
            "y0": y0,
            "x0": x0,
            "core_idx": core_idx,
            "basename": self.basenames[core_idx],
            "target_channel": int(target_channel) if target_channel is not None else -1,
        }
        return {
            "he": self._tf_he(he_img),
            "tgt_log": torch.from_numpy(target_log.transpose(2, 0, 1)),
            "info": info,
        }


class HE2CodexPatchCacheDataset(Dataset):
    """Fast dataset backed by pre-extracted memory-mapped CODEX patches.

    The cache stores raw H&E patches as uint8 and pre-scaled CODEX targets as
    float16 log-space tensors. This avoids per-sample full-core npy reads,
    percentile normalization, quantile scaling, and positive-patch rejection
    during training.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        patch_size: int = 224,
        augment: bool = True,
    ):
        self.dir = Path(cache_dir)
        self.split = split
        self.ps = int(patch_size)
        self.augment = bool(augment and split == "train")

        he_path = self.dir / f"{split}_he_uint8.npy"
        tgt_path = self.dir / f"{split}_tgt_log_f16.npy"
        meta_path = self.dir / f"{split}_patches.csv"
        if not he_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(f"Missing patch cache arrays for split {split!r} in {self.dir}")

        self.he = np.load(he_path, mmap_mode="r")
        self.tgt = np.load(tgt_path, mmap_mode="r")
        if self.he.ndim != 4 or self.tgt.ndim != 4:
            raise RuntimeError(
                f"Unexpected patch cache shapes: he={self.he.shape}, tgt={self.tgt.shape}"
            )
        if len(self.he) != len(self.tgt):
            raise RuntimeError(
                f"Patch cache length mismatch: he={len(self.he)}, tgt={len(self.tgt)}"
            )
        if self.he.shape[1:3] != (self.ps, self.ps) or self.tgt.shape[2:4] != (self.ps, self.ps):
            raise RuntimeError(
                f"Patch cache patch_size mismatch: expected {self.ps}, "
                f"got he={self.he.shape}, tgt={self.tgt.shape}"
            )

        self.C = int(self.tgt.shape[1])
        self.meta: list[dict[str, str]] = []
        if meta_path.exists():
            with meta_path.open(newline="") as f:
                self.meta = list(csv.DictReader(f))
        if len(self.meta) != len(self.he):
            self.meta = [{} for _ in range(len(self.he))]

        self.basenames = sorted({m.get("basename", "") for m in self.meta if m.get("basename")})
        self.tf_train = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tf_eval = self.tf_train

    def __len__(self) -> int:
        return len(self.he)

    def get_sanity_patch(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        idx = int(idx)
        he = np.asarray(self.he[idx]).copy()
        tgt = np.asarray(self.tgt[idx]).astype(np.float32, copy=True)
        meta = self.meta[idx] if idx < len(self.meta) else {}
        base = meta.get("basename", self.split)
        y0 = meta.get("y0", "0")
        x0 = meta.get("x0", "0")
        return he, tgt, f"{base}_y{y0}_x{x0}"

    @staticmethod
    def _paired_augment(he: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(0, 4)
        if k:
            he = np.rot90(he, k, axes=(0, 1)).copy()
            tgt = np.rot90(tgt, k, axes=(1, 2)).copy()
        if np.random.rand() < 0.5:
            he = np.flip(he, axis=1).copy()
            tgt = np.flip(tgt, axis=2).copy()
        if np.random.rand() < 0.5:
            he = np.flip(he, axis=0).copy()
            tgt = np.flip(tgt, axis=1).copy()
        return he, tgt

    def __getitem__(self, idx: int):
        he = np.asarray(self.he[idx]).copy()
        tgt = np.asarray(self.tgt[idx]).astype(np.float32, copy=True)
        if self.augment:
            he, tgt = self._paired_augment(he, tgt)

        meta = self.meta[idx]
        info = {
            "y0": int(meta.get("y0", 0) or 0),
            "x0": int(meta.get("x0", 0) or 0),
            "core_idx": int(meta.get("core_idx", -1) or -1),
            "basename": meta.get("basename", ""),
            "target_channel": int(meta.get("target_channel", -1) or -1),
        }
        return {"he": self.tf_train(he), "tgt_log": torch.from_numpy(tgt), "info": info}


__all__ = [
    "HE2CodexDataset",
    "HE2CodexPatchCacheDataset",
    "build_codex_basenames",
    "load_codex_manifest_basenames",
]
