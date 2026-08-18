# Data contracts

HEXIF operates on locally supplied research data. No cohort is bundled with
the source distribution, and no command is permitted to replace missing data
with generated samples.

## Privacy boundary

The following belong outside Git:

- source or derived microscopy images;
- patient, specimen, TMA, or core lookup tables;
- cell-level tables and segmentation masks;
- train, validation, and test assignments;
- checkpoints, optimizer states, logs, reports, and figures;
- credentials and machine-specific paths.

Use coded identifiers in analysis artifacts. Confirm that release of any
aggregate table or figure is allowed by the governing protocol before
publication.

## Registered image pairs

The registration pipeline produces one H&E array and one CODEX array per core,
plus a manifest. Arrays are NumPy files:

- `<core_id>_HE.npy`: `(height, width, 3)`, RGB, finite numeric values;
- `<core_id>_CODEX.npy`: `(height, width, channels)`, finite numeric values;
- manifest: one row per core with a stable coded ID, split, source cohort, and
  registration quality fields.

Channel order is an input property and must be recorded with the dataset. It
must not be inferred from filenames or copied from a different cohort.

## Cell tables

Training and evaluation consume one row per segmented cell. Required identity
and geometry columns are:

- `basename`: coded core identifier matching the image pair;
- `cell_id`: unique within `basename`;
- `centroid_x`, `centroid_y`: pixel coordinates in registered H&E space.

Marker prediction and label columns are named by channel index. The exact
columns required by a command are validated by that command before work
begins. Split membership is core-level: cells from a core may occur in exactly
one split.

## Workbench bundle

A bundle is derived from real model outputs and contains:

- `manifest.json`;
- `thresholds.json`, containing complete train-fitted operating thresholds,
  evidence-derived confidence strata, and calibration provenance;
- `cells.parquet`;
- `core_summary.parquet`;
- optional `cell_polygons.parquet`, `spatial_edges.parquet`, and
  `spatial_summary.parquet`;
- optional image pyramids and reports derived from the same registered inputs.

The manifest records source identifiers, model identity, run provenance,
per-core image dimensions, marker and phenotype names, build time, and counts.
Model metadata must provide a non-empty name, backbone, training split, and
provenance record for every model ID. The builder computes displayed metrics
from the supplied cell table; it never accepts hand-entered metric values.

Bundle construction requires explicit paths for predictions, consensus labels,
the core manifest, thresholds, model metadata, image pairs, and masks. Requested
polygons, thumbnails, or CODEX composites are all-or-nothing: a missing or
malformed source aborts the build. A public bundle additionally needs documented
authorization and SHA-256 hashes for every file.

## Test fixtures

Unit tests may create small deterministic arrays and tables in a temporary
directory to verify software behavior. Such fixtures are not scientific data,
must never be used for reported performance, and must be visibly scoped to
tests. Integration tests use an explicitly configured real bundle and skip
when it is absent.
