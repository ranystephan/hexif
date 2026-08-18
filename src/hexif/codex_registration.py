"""CODEX TMA core registration pipeline.

The workflow is explicit:

1. discover CODEX core files for a TMA,
2. detect H&E core bounding boxes on the matching SVS slide,
3. crop H&E and rigidly align each CODEX core to that crop with VALIS,
4. gate bad registrations by rigid residual,
5. write incremental per-TMA and combined manifests.

Heavy preprocessing dependencies (pyvips, cellpose, VALIS, OpenCV) are imported lazily
inside the functions that need them, so the package remains importable without the
``hexif[preprocess]`` stack.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class TmaSpec:
    """Files and orientation metadata for one CODEX/H&E TMA pair."""

    he_slide: str
    codex_subdir: str
    swap_xy: bool = True
    flip_rows: bool = True
    flip_cols: bool = True


def load_tma_specs(path: str | Path) -> dict[str, TmaSpec]:
    """Load a private TMA-to-file mapping from YAML."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TMA mapping does not exist: {source}")
    document = yaml.safe_load(source.read_text())
    if not isinstance(document, dict) or not document:
        raise ValueError("TMA mapping must be a non-empty mapping")
    specs: dict[str, TmaSpec] = {}
    for name, values in document.items():
        if not isinstance(values, dict):
            raise ValueError(f"TMA entry {name!r} must be a mapping")
        try:
            specs[str(name)] = TmaSpec(**values)
        except TypeError as exc:
            raise ValueError(f"invalid TMA entry {name!r}: {exc}") from exc
    return specs


@dataclass
class CodexRegistrationConfig:
    """Registration constants and I/O locations."""

    codex_root: Path
    out_dir: Path
    tmas: dict[str, TmaSpec]
    he_mpp: float = 0.13773
    codex_mpp: float = 0.5085
    pitch_um: float = 1500.0
    well_um: float = 1000.0
    bbox_pad: float = 0.50
    empty_he_sat: float = 2.0
    empty_dapi_mean: float = 30.0
    bad_align_px: float = 30.0
    valis_max_processed_image_dim_px: int = 1500

    def resolve(self, root: Path | None = None) -> CodexRegistrationConfig:
        """Return a copy with relative paths resolved against ``root``."""

        root = Path.cwd() if root is None else root
        return CodexRegistrationConfig(
            codex_root=_resolve_path(self.codex_root, root),
            out_dir=_resolve_path(self.out_dir, root),
            tmas=dict(self.tmas),
            he_mpp=self.he_mpp,
            codex_mpp=self.codex_mpp,
            pitch_um=self.pitch_um,
            well_um=self.well_um,
            bbox_pad=self.bbox_pad,
            empty_he_sat=self.empty_he_sat,
            empty_dapi_mean=self.empty_dapi_mean,
            bad_align_px=self.bad_align_px,
            valis_max_processed_image_dim_px=self.valis_max_processed_image_dim_px,
        )


def _resolve_path(path: Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _build_valis_registrar(
    he_path: Path,
    dapi_path: Path,
    output_dir: Path,
    *,
    max_processed_image_dim_px: int,
):
    """Create a rigid VALIS registrar for one H&E/DAPI image pair."""
    from valis import registration

    return registration.Valis(
        src_dir=str(he_path.parent),
        dst_dir=str(output_dir),
        img_list=[str(he_path), str(dapi_path)],
        imgs_ordered=True,
        reference_img_f=str(he_path),
        align_to_reference=True,
        max_processed_image_dim_px=max_processed_image_dim_px,
        create_masks=True,
        non_rigid_registrar_cls=None,
    )


def discover_cells(best_focus_dir: Path) -> list[tuple[str, int]]:
    """Return sorted ``(row_letter, column_number)`` cells from bestFocus TIF files."""

    cells: list[tuple[str, int]] = []
    for tif in sorted(Path(best_focus_dir).glob("*.tif")):
        if "-" not in tif.stem:
            continue
        row, col = tif.stem.split("-", 1)
        try:
            cells.append((row, int(col)))
        except ValueError:
            continue
    return sorted(cells)


def lattice_basis(cells: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
    """Return the filled row/column lattice basis spanning the observed cells."""

    if not cells:
        raise ValueError("no CODEX core files found")
    all_rows = sorted({r for r, _ in cells})
    all_cols = sorted({c for _, c in cells})
    rows = [chr(c) for c in range(ord(all_rows[0]), ord(all_rows[-1]) + 1)]
    cols = list(range(min(all_cols), max(all_cols) + 1))
    return rows, cols


def core_id(row: str, col: int) -> str:
    return f"{row}-{col}"


def core_sort_key(core: str) -> tuple[str, int]:
    row, col = core.split("-", 1)
    return row, int(col)


def load_meta_residual(meta_path: Path) -> float | None:
    """Read rigid residual from a core ``meta.json`` if present."""

    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    value = meta.get("mean_rigid_D", meta.get("rigid_D"))
    return None if value is None else float(value)


def _build_expected_mask(
    cells: list[tuple[str, int]],
    rows: list[str],
    cols: list[int],
    *,
    swap_xy: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> np.ndarray:
    n_rows, n_cols = len(rows), len(cols)
    n_x = n_rows if swap_xy else n_cols
    n_y = n_cols if swap_xy else n_rows
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: j for j, c in enumerate(cols)}
    i_for_row = np.arange(n_rows)[::-1] if flip_rows else np.arange(n_rows)
    j_for_col = np.arange(n_cols)[::-1] if flip_cols else np.arange(n_cols)
    expected = np.zeros((n_y, n_x), dtype=bool)
    for r, c in cells:
        ri, ci = row_idx[r], col_idx[c]
        if swap_xy:
            i_g, j_g = j_for_col[ci], i_for_row[ri]
        else:
            i_g, j_g = i_for_row[ri], j_for_col[ci]
        expected[i_g, j_g] = True
    return expected


def _fit_origin_brute(
    cents: np.ndarray,
    pitch: float,
    n_x: int,
    n_y: int,
    *,
    expected_mask: np.ndarray | None = None,
    search: float = 0.8,
    cap: float = 0.4,
) -> tuple[float, float]:
    cap2 = (cap * pitch) ** 2
    cx_lo, cx_hi = float(np.min(cents[:, 0])), float(np.max(cents[:, 0]))
    cy_lo, cy_hi = float(np.min(cents[:, 1])), float(np.max(cents[:, 1]))
    x0_init = (cx_lo + cx_hi) / 2 - (n_x - 1) * pitch / 2
    y0_init = (cy_lo + cy_hi) / 2 - (n_y - 1) * pitch / 2
    cents_n = len(cents)
    best: tuple[tuple[float, float] | None, float] = (None, float("inf"))
    for dx in np.linspace(-search * pitch, search * pitch, 61):
        for dy in np.linspace(-search * pitch, search * pitch, 61):
            x0 = x0_init + dx
            y0 = y0_init + dy
            gx = x0 + np.arange(n_x) * pitch
            gy = y0 + np.arange(n_y) * pitch
            d2x = (cents[:, 0:1] - gx[None, :]) ** 2
            d2y = (cents[:, 1:2] - gy[None, :]) ** 2
            j_idx = np.argmin(d2x, axis=1)
            i_idx = np.argmin(d2y, axis=1)
            d2 = d2x[np.arange(cents_n), j_idx] + d2y[np.arange(cents_n), i_idx]
            loss = float(np.minimum(d2, cap2).sum())
            if expected_mask is not None:
                close = d2 < (0.35 * pitch) ** 2
                detected = np.zeros_like(expected_mask)
                detected[i_idx[close], j_idx[close]] = True
                fn = int(np.logical_and(expected_mask, ~detected).sum())
                fp = int(np.logical_and(~expected_mask, detected).sum())
                loss += (fn + fp) * 5.0 * cap2
            if loss < best[1]:
                best = ((x0, y0), loss)
    if best[0] is None:
        raise RuntimeError("could not fit TMA lattice origin")
    return best[0]


def _icp_refine(
    cents: np.ndarray,
    grid_xy: np.ndarray,
    pitch: float,
    m_init: np.ndarray,
    n_iter: int = 14,
) -> np.ndarray:
    import cv2

    m = m_init.copy()
    gates = np.geomspace(0.80, 0.20, n_iter)
    for gate in gates:
        pred = (m[:, :2] @ grid_xy.T + m[:, 2:3]).T
        d2 = ((cents[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2)
        nn = d2.argmin(axis=1)
        nn_d = np.sqrt(d2[np.arange(len(cents)), nn])
        keep = nn_d < gate * pitch
        if keep.sum() < 8:
            break
        m_new, _ = cv2.estimateAffinePartial2D(
            grid_xy[nn[keep]].astype(np.float32),
            cents[keep].astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=0.12 * pitch,
        )
        if m_new is None:
            break
        if np.allclose(m_new, m, atol=0.1):
            m = m_new
            break
        m = m_new
    return m


def _refine_local(
    thumb_gray_inv: np.ndarray, px: float, py: float, rad: float, search: float = 0.5
) -> tuple[float, float]:
    import cv2

    h, w = thumb_gray_inv.shape
    win = int(rad * (1 + search))
    x0 = max(0, int(px) - win)
    x1 = min(w, int(px) + win + 1)
    y0 = max(0, int(py) - win)
    y1 = min(h, int(py) + win + 1)
    patch = thumb_gray_inv[y0:y1, x0:x1]
    if patch.size == 0:
        return px, py

    thr_val, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = max(int(thr_val), 35)
    mask = (patch > thr_val).astype(np.uint8)
    k = max(3, int(rad * 0.12)) | 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if nlabels < 2:
        return px, py

    px_p, py_p = px - x0, py - y0
    min_area = 0.10 * np.pi * rad**2
    max_area = 4.0 * np.pi * rad**2
    max_drift = rad

    best_lbl, best_d = -1, float("inf")
    for lbl in range(1, nlabels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx_l = stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH] / 2
        cy_l = stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT] / 2
        d = ((cx_l - px_p) ** 2 + (cy_l - py_p) ** 2) ** 0.5
        if d <= max_drift and d < best_d:
            best_d = d
            best_lbl = lbl
    if best_lbl < 0:
        return px, py

    ys, xs = np.where(labels == best_lbl)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx_new_p, cy_new_p), r_enc = cv2.minEnclosingCircle(pts)
    if r_enc < 0.4 * rad or r_enc > 1.7 * rad:
        return px, py
    return float(cx_new_p + x0), float(cy_new_p + y0)


def detect_bboxes(
    he_path: Path,
    rows: list[str],
    cols: list[int],
    *,
    cells: list[tuple[str, int]],
    spec: TmaSpec,
    config: CodexRegistrationConfig,
    ds_extra: int = 4,
) -> tuple[
    dict[tuple[str, int], tuple[int, int, int, int]],
    dict[tuple[str, int], tuple[float, float, float, float]],
]:
    """Detect H&E bounding boxes for the requested CODEX lattice cells."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import cv2
    import pyvips
    from cellpose import models as cp_models
    from scipy.optimize import linear_sum_assignment

    cellpose = cp_models.Cellpose(model_type="cyto3", gpu=False)

    img2 = pyvips.Image.new_from_file(str(he_path), level=2, access="sequential")
    thumb = np.ndarray(
        buffer=img2.write_to_memory(), dtype=np.uint8, shape=[img2.height, img2.width, img2.bands]
    )
    if thumb.shape[-1] >= 4:
        thumb = thumb[..., :3]
    img0 = pyvips.Image.new_from_file(str(he_path), level=0, access="random")
    full_w, full_h = img0.width, img0.height
    ds = max(full_w, full_h) / max(thumb.shape[:2])
    pitch = config.pitch_um / config.he_mpp / ds
    rad = config.well_um / 2 / config.he_mpp / ds

    small = cv2.resize(
        thumb,
        (thumb.shape[1] // ds_extra, thumb.shape[0] // ds_extra),
        interpolation=cv2.INTER_AREA,
    )
    gray = 255 - cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    diam = int(config.well_um / config.he_mpp / ds / ds_extra)
    masks, *_ = cellpose.eval(
        gray, diameter=diam, channels=[0, 0], flow_threshold=0.0, cellprob_threshold=-1.0
    )

    expected_area = np.pi * (diam / 2) ** 2
    expected_r = diam / 2
    cents = []
    for lbl in np.unique(masks):
        if lbl == 0:
            continue
        ys, xs = np.where(masks == lbl)
        n = len(xs)
        if n < 0.10 * expected_area or n > 4.0 * expected_area:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_cx, _cy), r_enc = cv2.minEnclosingCircle(pts)
        if r_enc <= 0:
            continue
        if n / (np.pi * r_enc**2) < 0.30:
            continue
        if r_enc < 0.35 * expected_r or r_enc > 1.7 * expected_r:
            continue
        cents.append((_cx * ds_extra, _cy * ds_extra))
    cents_arr = np.asarray(cents, dtype=np.float32)
    if len(cents_arr) == 0:
        raise RuntimeError("cellpose produced no plausible H&E core detections")
    print(f"cellpose: {len(cents_arr)} cores after shape filter")

    n_rows, n_cols = len(rows), len(cols)
    n_x = n_rows if spec.swap_xy else n_cols
    n_y = n_cols if spec.swap_xy else n_rows
    grid_xy = np.array(
        [[j * pitch, i * pitch] for i in range(n_y) for j in range(n_x)], dtype=np.float32
    )

    expected_mask = _build_expected_mask(
        cells,
        rows,
        cols,
        swap_xy=spec.swap_xy,
        flip_rows=spec.flip_rows,
        flip_cols=spec.flip_cols,
    )
    print(f"expected: {expected_mask.sum()} known core slots / {expected_mask.size} grid points")

    x0, y0 = _fit_origin_brute(cents_arr, pitch, n_x, n_y, expected_mask=expected_mask)
    m_init = np.array([[1, 0, x0], [0, 1, y0]], dtype=np.float32)
    m = _icp_refine(cents_arr, grid_xy, pitch, m_init)

    pred = (m[:, :2] @ grid_xy.T + m[:, 2:3]).T
    d2 = ((cents_arr[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2)
    nn_d = np.sqrt(d2.min(axis=1))
    print(
        f"icp: {(nn_d < 0.30 * pitch).sum()}/{len(cents_arr)} matched within 0.30*pitch "
        f"(median {np.median(nn_d) / pitch:.3f}*pitch, "
        f"p95 {np.percentile(nn_d, 95) / pitch:.3f}*pitch)"
    )

    he_pts = pred.reshape(n_y, n_x, 2)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: j for j, c in enumerate(cols)}
    i_for_row = np.arange(n_rows)[::-1] if spec.flip_rows else np.arange(n_rows)
    j_for_col = np.arange(n_cols)[::-1] if spec.flip_cols else np.arange(n_cols)

    half_thumb = rad * (1 + config.bbox_pad)
    half = int(half_thumb * ds)
    thumb_gray_inv = 255 - cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)

    pred_xy = np.zeros((len(cells), 2), dtype=np.float32)
    for k, (r, c) in enumerate(cells):
        ri = row_idx[r]
        ci = col_idx[c]
        if spec.swap_xy:
            pred_xy[k] = he_pts[j_for_col[ci], i_for_row[ri]]
        else:
            pred_xy[k] = he_pts[i_for_row[ri], j_for_col[ci]]

    seed_xy = pred_xy.copy()
    seed_used_centroid = np.zeros(len(cells), dtype=bool)
    cost = np.sqrt(((cents_arr[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(cost)
    distances = np.array([cost[ki, rci] for ki, rci in zip(row_ind, col_ind, strict=False)])
    median_d = float(np.median(distances))
    gate = float(np.clip(3.0 * median_d, 0.4 * pitch, 0.7 * pitch))
    rejected = 0
    kept_distances: list[float] = []
    for ki, rci, dist in zip(row_ind, col_ind, distances, strict=False):
        if dist < gate:
            seed_xy[rci] = cents_arr[ki]
            seed_used_centroid[rci] = True
            kept_distances.append(float(dist))
        else:
            rejected += 1
    if kept_distances:
        print(
            f"hungarian: {seed_used_centroid.sum()}/{len(cells)} (r,c) seeded by centroid  "
            f"gate={gate / pitch:.2f}*pitch  median={median_d / pitch:.3f}*pitch  "
            f"max_kept={max(kept_distances) / pitch:.3f}*pitch  rejected={rejected}"
        )
    else:
        print(f"hungarian: 0/{len(cells)} seeded (all {rejected} rejected)")

    bboxes: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    refined: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for k, (r, c) in enumerate(cells):
        px, py = float(pred_xy[k, 0]), float(pred_xy[k, 1])
        seed_x, seed_y = float(seed_xy[k, 0]), float(seed_xy[k, 1])
        cx, cy = _refine_local(thumb_gray_inv, seed_x, seed_y, rad)
        refined[(r, c)] = (cx, cy, px, py)
        cxf, cyf = int(cx * ds), int(cy * ds)
        left = max(0, cxf - half)
        top = max(0, cyf - half)
        right = min(full_w, cxf + half)
        bottom = min(full_h, cyf + half)
        bboxes[(r, c)] = (left, top, max(1, right - left), max(1, bottom - top))
    return bboxes, refined


class CodexRegistrationPipeline:
    """End-to-end CODEX/H&E core registration runner."""

    def __init__(self, config: CodexRegistrationConfig) -> None:
        self.config = config.resolve()

    def paths_for_tma(self, tma: str) -> tuple[TmaSpec, Path, Path, Path]:
        spec = self.config.tmas[tma]
        he_path = self.config.codex_root / "HandE" / spec.he_slide
        bf_dir = self.config.codex_root / spec.codex_subdir / "bestFocus"
        out_dir = self.config.out_dir / tma
        return spec, he_path, bf_dir, out_dir

    def crop_he(self, svs: Path, bbox: tuple[int, int, int, int]) -> np.ndarray:
        import pyvips

        left, top, width, height = bbox
        img = pyvips.Image.new_from_file(str(svs), level=0, access="sequential")
        crop = img.crop(left, top, width, height).resize(self.config.he_mpp / self.config.codex_mpp)
        arr = np.ndarray(
            buffer=crop.write_to_memory(),
            dtype=np.uint8,
            shape=[crop.height, crop.width, crop.bands],
        )
        return arr[..., :3] if arr.shape[-1] >= 4 else arr

    def register_one(
        self,
        he_path: Path,
        bf_dir: Path,
        row: str,
        col: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        import cv2
        import tifffile
        from valis import slide_io

        he_arr = self.crop_he(he_path, bbox)
        if cv2.cvtColor(he_arr, cv2.COLOR_RGB2HSV)[..., 1].mean() < self.config.empty_he_sat:
            raise RuntimeError("empty_he")

        codex_full = tifffile.imread(str(bf_dir / f"{row}-{col}.tif"))
        codex_full = np.rot90(codex_full, 2, axes=(-2, -1))
        if codex_full[0].mean() < self.config.empty_dapi_mean:
            raise RuntimeError("empty_codex")

        work = Path(tempfile.mkdtemp(prefix="valis_codex_"))
        try:
            tifffile.imwrite(str(work / "HE.tif"), he_arr, photometric="rgb")
            tifffile.imwrite(str(work / "DAPI.tif"), codex_full[0], photometric="minisblack")
            reg = _build_valis_registrar(
                work / "HE.tif",
                work / "DAPI.tif",
                work / "reg",
                max_processed_image_dim_px=self.config.valis_max_processed_image_dim_px,
            )
            _, _, err = reg.register(reader_cls=slide_io.BioFormatsSlideReader)
            codex_slide = reg.get_slide(str(work / "DAPI.tif"))
            he_slide = reg.get_slide(str(work / "HE.tif"))

            h_codex, w_codex = codex_full.shape[1:]
            src = np.array(
                [
                    [w_codex * 0.1, h_codex * 0.1],
                    [w_codex * 0.9, h_codex * 0.1],
                    [w_codex * 0.5, h_codex * 0.9],
                ],
                dtype=np.float32,
            )
            dst = np.array(
                codex_slide.warp_xy_from_to(src, he_slide, non_rigid=False), dtype=np.float32
            )
            affine = cv2.getAffineTransform(src, dst)

            he_h, he_w = he_arr.shape[:2]
            codex_aligned = np.empty((codex_full.shape[0], he_h, he_w), dtype=codex_full.dtype)
            for channel in range(codex_full.shape[0]):
                codex_aligned[channel] = cv2.warpAffine(
                    codex_full[channel], affine, (he_w, he_h), flags=cv2.INTER_LINEAR
                )
            residuals = err.iloc[0].to_dict() if err is not None and len(err) else {}
            return he_arr, codex_aligned, residuals
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def cached_row(self, tma: str, core: str, out_core: Path) -> dict[str, Any]:
        row: dict[str, Any] = {"tma": tma, "core": core, "status": "cached"}
        try:
            d_rigid = load_meta_residual(out_core / "meta.json")
        except Exception as exc:
            row["cache_meta_error"] = str(exc)
            return row
        if d_rigid is not None:
            row["rigid_D_px"] = d_rigid
            if np.isfinite(d_rigid) and d_rigid > self.config.bad_align_px:
                row["status"] = "bad_alignment"
                row["error"] = f"rigid_D={d_rigid:.1f}px > {self.config.bad_align_px}"
        return row

    def wipe_retry_rows(self, out_dir: Path) -> int:
        manifest = out_dir / "manifest.csv"
        if not manifest.exists():
            return 0
        prev = pd.read_csv(manifest)
        bad_status = prev["status"].isin(["bad_alignment", "fail", "empty_he", "empty_codex"])
        bad_resid = (
            prev.get("rigid_D_px", pd.Series(float("nan"), index=prev.index))
            .astype(float)
            .gt(self.config.bad_align_px)
        )
        bad = prev[bad_status | bad_resid]
        wiped = 0
        for _, row in bad.iterrows():
            core_path = out_dir / row["core"]
            for filename in ("he.npy", "codex.npy", "meta.json"):
                path = core_path / filename
                if path.exists():
                    path.unlink()
                    wiped += 1
        if wiped:
            print(f"retry_bad: cleared {wiped} files for {len(bad)} prior-bad cores")
        return wiped

    def process_tma(
        self, tma: str, *, force: bool = False, retry_bad: bool = False
    ) -> pd.DataFrame:
        if tma not in self.config.tmas:
            raise KeyError(f"unknown TMA {tma!r}; known TMAs: {sorted(self.config.tmas)}")

        spec, he_path, bf_dir, out_dir = self.paths_for_tma(tma)
        cells = discover_cells(bf_dir)
        rows, cols = lattice_basis(cells)
        bboxes, _ = detect_bboxes(he_path, rows, cols, cells=cells, spec=spec, config=self.config)

        out_dir.mkdir(parents=True, exist_ok=True)
        if retry_bad:
            self.wipe_retry_rows(out_dir)

        results: list[dict[str, Any]] = []
        manifest = out_dir / "manifest.csv"
        for (row, col), bbox in bboxes.items():
            cid = core_id(row, col)
            out_core = out_dir / cid
            out_core.mkdir(exist_ok=True)

            if not force and (out_core / "he.npy").exists() and (out_core / "codex.npy").exists():
                results.append(self.cached_row(tma, cid, out_core))
                pd.DataFrame(results).to_csv(manifest, index=False)
                continue

            t0 = time.time()
            try:
                he_arr, codex_aligned, residuals = self.register_one(
                    he_path, bf_dir, row, col, bbox
                )
                d_rigid = residuals.get("mean_rigid_D", residuals.get("rigid_D", float("nan")))
                d_rigid = float(d_rigid) if d_rigid is not None else float("nan")
                if not np.isfinite(d_rigid) or d_rigid > self.config.bad_align_px:
                    result = {
                        "tma": tma,
                        "core": cid,
                        "status": "bad_alignment",
                        "rigid_D_px": d_rigid,
                        "error": f"rigid_D={d_rigid:.1f}px > {self.config.bad_align_px}",
                    }
                    print(
                        f"  {cid}: bad_alignment  rigid_D={d_rigid:.1f}px  ({time.time() - t0:.1f}s)"
                    )
                else:
                    np.save(out_core / "he.npy", he_arr)
                    np.save(out_core / "codex.npy", codex_aligned)
                    meta = {
                        "core": cid,
                        "tma": tma,
                        "he_shape": list(he_arr.shape),
                        "codex_shape": list(codex_aligned.shape),
                        **{
                            k: float(v)
                            for k, v in residuals.items()
                            if isinstance(v, int | float | np.number) and not np.isnan(v)
                        },
                    }
                    (out_core / "meta.json").write_text(json.dumps(meta, indent=2))
                    result = {
                        "tma": tma,
                        "core": cid,
                        "status": "ok",
                        "t_sec": time.time() - t0,
                        "rigid_D_px": d_rigid,
                    }
                    print(f"  {cid}: ok  rigid_D={d_rigid:.2f}px  ({time.time() - t0:.1f}s)")
                results.append(result)
            except RuntimeError as exc:
                msg = str(exc)
                status = msg if msg in ("empty_he", "empty_codex") else "fail"
                results.append({"tma": tma, "core": cid, "status": status, "error": msg})
                print(f"  {cid}: skip ({msg})")
            except Exception as exc:
                results.append({"tma": tma, "core": cid, "status": "fail", "error": str(exc)})
                print(f"  {cid}: FAIL {exc}")

            pd.DataFrame(results).to_csv(manifest, index=False)

        return pd.DataFrame(results)

    def process_all(
        self, *, force: bool = False, retry_bad: bool = True, kill_jvm: bool = True
    ) -> pd.DataFrame:
        all_dfs = []
        for tma in self.config.tmas:
            print(f"\n=== {tma} ===")
            all_dfs.append(self.process_tma(tma, force=force, retry_bad=retry_bad))
        combined = pd.concat(all_dfs, ignore_index=True)
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self.config.out_dir / "manifest_all.csv", index=False)
        if kill_jvm:
            try:
                from valis import registration as valis_registration

                valis_registration.kill_jvm()
            except Exception:
                pass
        return combined

    def rebuild_manifest(self) -> pd.DataFrame:
        frames = []
        for tma_dir in sorted(p for p in self.config.out_dir.iterdir() if p.is_dir()):
            manifest = tma_dir / "manifest.csv"
            if manifest.exists():
                df = pd.read_csv(manifest)
                rows = []
                for _, row in df.iterrows():
                    rec = row.to_dict()
                    core_path = tma_dir / str(row["core"])
                    has_arrays = (core_path / "he.npy").exists() and (
                        core_path / "codex.npy"
                    ).exists()
                    try:
                        d_rigid = load_meta_residual(core_path / "meta.json")
                    except Exception as exc:
                        rec["cache_meta_error"] = str(exc)
                        d_rigid = None
                    if d_rigid is not None:
                        rec["rigid_D_px"] = d_rigid
                    if has_arrays and d_rigid is not None and np.isfinite(d_rigid):
                        if d_rigid > self.config.bad_align_px:
                            rec["status"] = "bad_alignment"
                            rec["error"] = f"rigid_D={d_rigid:.1f}px > {self.config.bad_align_px}"
                        elif rec.get("status") not in ("ok", "cached"):
                            rec["status"] = "cached"
                            rec["error"] = ""
                    rows.append(rec)
                repaired = pd.DataFrame(rows)
                repaired.to_csv(manifest, index=False)
                frames.append(repaired)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(self.config.out_dir / "manifest_all.csv", index=False)
        return combined

    def purge_rejected_arrays(self, manifest: pd.DataFrame | None = None) -> int:
        if manifest is None:
            manifest = pd.read_csv(self.config.out_dir / "manifest_all.csv")
        removed = 0
        for _, row in manifest.iterrows():
            if row["status"] in ("ok", "cached"):
                continue
            core_path = self.config.out_dir / row["tma"] / row["core"]
            for filename in ("he.npy", "codex.npy"):
                path = core_path / filename
                if path.exists():
                    path.unlink()
                    removed += 1
        return removed


def summarize_manifest(manifest: pd.DataFrame) -> str:
    """Return a concise human-readable status and residual summary."""

    lines = [str(manifest.groupby(["tma", "status"]).size().unstack(fill_value=0))]
    usable = manifest[manifest["status"].isin(["ok", "cached"])].copy()
    if "rigid_D_px" in usable and usable["rigid_D_px"].notna().any():
        residuals = usable["rigid_D_px"].dropna()
        lines.append(
            "usable residuals: "
            f"n={len(residuals)} mean={residuals.mean():.2f}px "
            f"median={residuals.median():.2f}px p95={residuals.quantile(0.95):.2f}px "
            f"max={residuals.max():.2f}px"
        )
    return "\n".join(lines)
