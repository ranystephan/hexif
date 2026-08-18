#!/usr/bin/env python3
"""Build cached cell-compartment masks for registered CODEX cores.

The cell-level benchmark should use one stable segmentation per core, derived
from the measured CODEX DAPI channel rather than from model predictions.  This
script creates those label masks and lightweight QC overlays.

The primary segmentation is a DAPI-derived nuclei mask.  Optional derived
compartments keep the same instance labels:

- expanded{N}: nuclei labels expanded by N pixels without overlap
- ring{N}: expanded{N} minus the original nucleus, useful as a perinuclear
  cytoplasm / membrane proxy when true membrane segmentation is unavailable
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, segmentation, transform


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache Cellpose nuclei masks for CODEX cores")
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--manifest_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--splits", default="train,val,test", help="Comma-separated split names to process"
    )
    p.add_argument(
        "--eval_cores", default="", help="Optional comma-separated basenames; overrides --splits"
    )
    p.add_argument("--channel_count", type=int, default=53)
    p.add_argument("--dapi_channel", type=int, default=0)
    p.add_argument("--model_type", default="nuclei", help="Cellpose model_type, usually 'nuclei'")
    p.add_argument(
        "--diameter",
        type=float,
        default=30.0,
        help="Approximate nucleus diameter in full-resolution pixels",
    )
    p.add_argument(
        "--downsample",
        type=float,
        default=2.0,
        help="Run Cellpose at this downsample factor, then resize labels back",
    )
    p.add_argument("--cellprob_threshold", type=float, default=-1.0)
    p.add_argument(
        "--flow_threshold", type=float, default=0.0, help="Keep 0.0 for Cellpose 3.1 compatibility"
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--min_area", type=int, default=40)
    p.add_argument("--max_area", type=int, default=20000)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--compartments",
        default="nuclei",
        help="Comma-separated compartments to write: nuclei, expanded, ring",
    )
    p.add_argument(
        "--expand_px",
        type=int,
        default=8,
        help="Full-resolution expansion radius for expanded/ring compartments",
    )
    p.add_argument(
        "--qc_every",
        type=int,
        default=25,
        help="Write QC overlay for every Nth core; 0 disables periodic QC",
    )
    p.add_argument("--qc_max_side", type=int, default=1400)
    return p.parse_args()


def _load_basenames(manifest_csv: Path, splits: set[str]) -> list[str]:
    out: list[str] = []
    with manifest_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") in splits and row.get("basename"):
                out.append(row["basename"])
    return out


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _as_channel_first(codex: np.ndarray, channel_count: int) -> np.ndarray:
    if codex.ndim != 3:
        raise ValueError(f"Expected 3D CODEX array, got {codex.shape}")
    if codex.shape[0] == channel_count:
        return codex
    if codex.shape[-1] == channel_count:
        return np.transpose(codex, (2, 0, 1))
    raise ValueError(f"Expected {channel_count} channels, got {codex.shape}")


def _normalize_for_cellpose(dapi: np.ndarray) -> np.ndarray:
    x = np.asarray(dapi, dtype=np.float32)
    lo, hi = np.percentile(x, [0.5, 99.8])
    if not np.isfinite(hi) or hi <= lo:
        hi = float(x.max(initial=1.0))
        lo = float(x.min(initial=0.0))
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return x


def _resize_image(x: np.ndarray, downsample: float) -> np.ndarray:
    if downsample <= 1.0:
        return x
    h, w = x.shape
    out_shape = (max(1, round(h / downsample)), max(1, round(w / downsample)))
    return transform.resize(
        x,
        out_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == shape:
        return labels
    return transform.resize(
        labels,
        shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.int32)


def _filter_labels(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int32)
    keep = np.zeros(mask.max(initial=0) + 1, dtype=bool)
    keep[0] = False
    for region in measure.regionprops(mask):
        if min_area <= int(region.area) <= max_area:
            keep[int(region.label)] = True
    filtered = np.where(keep[mask], mask, 0)
    relabeled, _, _ = segmentation.relabel_sequential(filtered)
    return relabeled.astype(np.uint32, copy=False)


def _save_qc(path: Path, dapi: np.ndarray, mask: np.ndarray, max_side: int) -> None:
    scale = min(1.0, max_side / max(dapi.shape))
    dapi_show = dapi
    mask_show = mask
    if scale < 1.0:
        out_shape = (round(dapi.shape[0] * scale), round(dapi.shape[1] * scale))
        dapi_show = transform.resize(
            dapi, out_shape, order=1, anti_aliasing=True, preserve_range=True
        )
        mask_show = transform.resize(
            mask, out_shape, order=0, anti_aliasing=False, preserve_range=True
        ).astype(mask.dtype)
    dapi_norm = _normalize_for_cellpose(dapi_show)
    boundaries = segmentation.find_boundaries(mask_show, mode="outer")
    rgb = np.stack([dapi_norm, dapi_norm, dapi_norm], axis=-1)
    rgb[boundaries] = (1.0, 0.0, 0.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title(f"{path.stem}: {int(mask.max())} nuclei")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _compartment_name(kind: str, expand_px: int) -> str:
    kind = kind.strip().lower()
    if kind == "nuclei":
        return "nuclei"
    if kind in {"expanded", "ring"}:
        return f"{kind}{int(expand_px)}"
    raise ValueError(f"Unknown compartment {kind!r}; use nuclei, expanded, or ring")


def _derive_compartment(nuclei: np.ndarray, kind: str, expand_px: int) -> np.ndarray:
    kind = kind.strip().lower()
    if kind == "nuclei":
        return nuclei.astype(np.uint32, copy=False)
    expanded = segmentation.expand_labels(nuclei, distance=int(expand_px)).astype(
        np.uint32, copy=False
    )
    if kind == "expanded":
        return expanded
    if kind == "ring":
        ring = expanded.copy()
        ring[nuclei > 0] = 0
        return ring
    raise ValueError(f"Unknown compartment {kind!r}")


def main() -> None:
    args = _parse_args()
    from cellpose import models as cp_models

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.output_dir)
    mask_dir = out_dir / "masks"
    qc_dir = out_dir / "qc_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    requested_compartments = [
        _compartment_name(x, int(args.expand_px)) for x in _parse_csv_list(args.compartments)
    ]
    requested_kinds = [x.strip().lower() for x in _parse_csv_list(args.compartments)]
    if "nuclei" not in requested_compartments:
        requested_compartments.insert(0, "nuclei")
        requested_kinds.insert(0, "nuclei")

    if args.eval_cores:
        basenames = _parse_csv_list(args.eval_cores)
    else:
        basenames = _load_basenames(Path(args.manifest_csv), set(_parse_csv_list(args.splits)))
    if not basenames:
        raise RuntimeError("No cores selected for mask building")

    model = cp_models.Cellpose(model_type=args.model_type, gpu=bool(args.gpu))
    rows: list[dict[str, Any]] = []
    effective_diameter = max(1.0, float(args.diameter) / max(float(args.downsample), 1.0))

    for i, base in enumerate(basenames, start=1):
        mask_path = mask_dir / f"{base}_nuclei.npy"
        summary_path = mask_dir / f"{base}_summary.json"
        compartment_paths = {
            name: mask_dir / f"{base}_{name}.npy" for name in requested_compartments
        }
        have_all_compartments = all(p.exists() for p in compartment_paths.values())
        if (
            mask_path.exists()
            and summary_path.exists()
            and have_all_compartments
            and not args.force
        ):
            summary = json.loads(summary_path.read_text())
            rows.append(summary)
            if i % 25 == 0:
                print(f"cached {i}/{len(basenames)}", flush=True)
            continue

        codex = np.load(pairs_dir / f"{base}_CODEX.npy", mmap_mode="r")
        codex_cf = _as_channel_first(codex, args.channel_count)
        dapi = np.asarray(codex_cf[args.dapi_channel])
        if mask_path.exists() and not args.force:
            masks = np.load(mask_path)
        else:
            x = _normalize_for_cellpose(dapi)
            x_small = _resize_image(x, float(args.downsample))
            masks, *_ = model.eval(
                x_small,
                diameter=effective_diameter,
                channels=[0, 0],
                flow_threshold=float(args.flow_threshold),
                cellprob_threshold=float(args.cellprob_threshold),
            )
            masks = _resize_labels(masks, dapi.shape)
            masks = _filter_labels(masks, int(args.min_area), int(args.max_area))
            np.save(mask_path, masks)

        written_compartments: dict[str, str] = {}
        for kind, name in zip(requested_kinds, requested_compartments, strict=False):
            out_path = compartment_paths[name]
            if out_path.exists() and not args.force:
                written_compartments[name] = str(out_path)
                continue
            comp = _derive_compartment(masks, kind, int(args.expand_px))
            np.save(out_path, comp)
            written_compartments[name] = str(out_path)

        areas = np.bincount(masks.ravel())[1:]
        summary = {
            "basename": base,
            "mask_path": str(mask_path),
            "height": int(masks.shape[0]),
            "width": int(masks.shape[1]),
            "n_cells": int(masks.max(initial=0)),
            "area_mean": float(np.mean(areas)) if len(areas) else 0.0,
            "area_median": float(np.median(areas)) if len(areas) else 0.0,
            "area_p05": float(np.percentile(areas, 5)) if len(areas) else 0.0,
            "area_p95": float(np.percentile(areas, 95)) if len(areas) else 0.0,
            "diameter": float(args.diameter),
            "downsample": float(args.downsample),
            "model_type": str(args.model_type),
            "expand_px": int(args.expand_px),
            "compartments": ",".join(requested_compartments),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        rows.append(summary)

        if args.qc_every and (i == 1 or i % int(args.qc_every) == 0):
            _save_qc(qc_dir / f"{base}_nuclei_qc.png", dapi, masks, int(args.qc_max_side))
            for name in requested_compartments:
                if name == "nuclei":
                    continue
                comp = np.load(compartment_paths[name], mmap_mode="r")
                _save_qc(qc_dir / f"{base}_{name}_qc.png", dapi, comp, int(args.qc_max_side))
        print(f"{base}: {summary['n_cells']} nuclei ({i}/{len(basenames)})", flush=True)

    summary_csv = out_dir / "mask_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["basename"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "mask_build_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
