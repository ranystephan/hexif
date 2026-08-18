#!/usr/bin/env python3
"""Strictly validate real cell-training inputs before GPU allocation.

This command only reads data. It never creates, repairs, or substitutes
scientific inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hexif.cell_phenotype import FOCUSED_MARKERS, PHENOTYPE_NAMES, cell_table_to_targets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_path(raw_path: str, *, directory: bool) -> Path:
    path = Path(raw_path).expanduser().resolve(strict=False)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise argparse.ArgumentTypeError(f"not a readable {kind}: {path}")
    return path


def existing_file(raw_path: str) -> Path:
    return existing_path(raw_path, directory=False)


def existing_directory(raw_path: str) -> Path:
    return existing_path(raw_path, directory=True)


def parse_channels(value: str) -> tuple[int, ...]:
    try:
        channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not channels or len(channels) != len(set(channels)):
        raise argparse.ArgumentTypeError("channels must be non-empty and unique")
    return channels


def load_table(path: Path, channels: tuple[int, ...], label_set: str) -> pd.DataFrame:
    suffix = {"gmm": "pos_gmm_orig", "consensus": "pos_consensus", "spacec": "pos_spacec"}[
        label_set
    ]
    required = ["basename", "cell_id", "centroid_y", "centroid_x"]
    required += [f"ch{channel:02d}_pred" for channel in channels]
    required += [f"ch{channel:02d}_{suffix}" for channel in channels]
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return pd.read_csv(path, usecols=required)


def validate_table(
    name: str, table: pd.DataFrame, channels: tuple[int, ...], label_set: str
) -> tuple[np.ndarray, np.ndarray]:
    if table.empty:
        raise ValueError(f"{name} table is empty")
    if table[["basename", "cell_id"]].isna().any().any():
        raise ValueError(f"{name} table contains missing cell identities")
    duplicate = table.duplicated(["basename", "cell_id"], keep=False)
    if duplicate.any():
        raise ValueError(f"{name} table contains {int(duplicate.sum())} duplicate cells")
    numeric_columns = ["centroid_y", "centroid_x"] + [
        f"ch{channel:02d}_pred" for channel in channels
    ]
    numeric = table[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} table contains non-finite coordinates or features")
    if (table[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError(f"{name} table contains negative centroid coordinates")
    marker_y, _, phenotype_y, _ = cell_table_to_targets(
        table, marker_channels=channels, label_set=label_set
    )
    for index, channel in enumerate(channels):
        if np.unique(marker_y[:, index]).size != 2:
            raise ValueError(f"{name} marker ch{channel:02d} does not contain both classes")
    for index, phenotype in enumerate(PHENOTYPE_NAMES):
        if np.unique(phenotype_y[:, index]).size != 2:
            raise ValueError(f"{name} phenotype {phenotype!r} does not contain both classes")
    return marker_y, phenotype_y


def validate_images(
    tables: list[pd.DataFrame], pairs_dir: Path, patch_size: int
) -> dict[str, dict[str, object]]:
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch size must be a positive even integer")
    half = patch_size // 2
    combined = pd.concat(tables, ignore_index=True)
    summary: dict[str, dict[str, object]] = {}
    invalid_cells = 0
    for basename, rows in combined.groupby("basename", sort=True):
        path = pairs_dir / f"{basename}_HE.npy"
        if not path.is_file():
            raise ValueError(f"missing H&E array (including broken symlink): {path}")
        image = np.load(path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 HxWx3, found {image.dtype} {image.shape}")
        height, width = image.shape[:2]
        y = rows["centroid_y"].to_numpy(dtype=np.float64)
        x = rows["centroid_x"].to_numpy(dtype=np.float64)
        fits = (y >= half) & (y + half <= height) & (x >= half) & (x + half <= width)
        invalid_cells += int((~fits).sum())
        summary[str(basename)] = {
            "path": str(path.resolve()),
            "shape": list(map(int, image.shape)),
            "dtype": str(image.dtype),
            "cells": len(rows),
            "cells_with_full_patch": int(fits.sum()),
        }
    if invalid_cells:
        raise ValueError(f"{invalid_cells} cells cannot supply a complete {patch_size}px patch")
    return summary


def split_summary(
    path: Path,
    table: pd.DataFrame,
    marker_y: np.ndarray,
    phenotype_y: np.ndarray,
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(table),
        "cores": int(table["basename"].nunique()),
        "marker_positives": marker_y.sum(axis=0).astype(int).tolist(),
        "phenotype_positives": phenotype_y.sum(axis=0).astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", required=True, type=existing_file)
    parser.add_argument("--val-table", required=True, type=existing_file)
    parser.add_argument("--pairs-dir", required=True, type=existing_directory)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label-set", required=True, choices=["gmm", "consensus", "spacec"])
    parser.add_argument("--marker-channels", type=parse_channels, default=FOCUSED_MARKERS)
    parser.add_argument("--patch-size", type=int, default=224)
    args = parser.parse_args()

    train = load_table(args.train_table, args.marker_channels, args.label_set)
    val = load_table(args.val_table, args.marker_channels, args.label_set)
    train_targets = validate_table("train", train, args.marker_channels, args.label_set)
    val_targets = validate_table("validation", val, args.marker_channels, args.label_set)
    overlap = sorted(set(train["basename"]) & set(val["basename"]))
    if overlap:
        raise ValueError(f"train/validation core overlap: {', '.join(map(str, overlap))}")
    images = validate_images([train, val], args.pairs_dir, args.patch_size)
    payload = {
        "schema_version": 1,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "label_set": args.label_set,
        "marker_channels": list(args.marker_channels),
        "patch_size": args.patch_size,
        "train": split_summary(args.train_table, train, *train_targets),
        "validation": split_summary(args.val_table, val, *val_targets),
        "pairs_dir": str(args.pairs_dir),
        "images": images,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"validated {len(train):,} train and {len(val):,} validation cells")
    print(f"validated {len(images):,} real H&E arrays")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
