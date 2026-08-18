# Spatial Cytometry Workbench

The Workbench is a read-only FastAPI application over a bundle of real
cell-level predictions. It does not run model inference and does not ship with
a sample bundle.

## Build

Inspect every command before running it against the frozen rerun artifacts:

```bash
hexif build-cell-tables --help
hexif rebuild-polygons --help
hexif rebuild-codex-composite --help
hexif rebuild-codex-marker-thumbs --help
hexif spatial-summary --help
hexif report --help
```

All source paths must point outside the Git checkout or to ignored local
directories. The bundle contract is summarized in
[`data-contracts.md`](data-contracts.md).

The initial build requires `--thresholds-json`, `--model-metadata`, and
`--model-id`. Use the `workbench_thresholds.json` written by the calibration
step and metadata tied to the same frozen training run. Missing requested masks,
images, scaler parameters, or model outputs abort the build.

## Serve

```bash
hexif serve-workbench --bundle /absolute/path/to/validated_bundle --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>. API documentation is available at
`/api/docs`, and `/api/health` provides a health probe.

Binding to a non-loopback address may expose sensitive derived data. Do so only
in an access-controlled environment approved for the cohort.

## Interpretation

Workbench outputs are research estimates. Marker calls depend on the frozen
calibration procedure, and phenotypes are logical combinations of marker
calls. Spatial summaries inherit errors from registration, segmentation,
marker prediction, calibration, and phenotype assignment. They are not
clinical measurements.
