# Reproducibility and release procedure

This is the mandatory procedure for the next real-data rerun. A result is not
release-qualified unless every item below is complete.

## 1. Freeze inputs

1. Verify authorization to use the source cohort.
2. Record coded cohort identifiers, assay panel, channel order, image
   resolution, registration method, and all exclusions.
3. Define train, validation, and test membership at the patient or highest
   relevant biological grouping. Freeze the split before fitting thresholds or
   models.
4. Hash the split manifest and every configuration file with SHA-256.
5. Keep raw and derived data outside the Git checkout.

## 2. Prepare labels

Run registration, segmentation, consensus-label construction, calibration, and
label audit only on the permitted partition. Save command lines, logs, package
versions, random seeds, and output hashes. Review audit failures before
training; do not silently drop failed cores.

The relevant entry points are:

```bash
hexif codex-register --help
python scripts/build_codex_cell_masks.py --help
python scripts/build_codex_cell_consensus_labels.py --help
python scripts/calibrate_codex_cell_phenotypes.py --help
hexif audit-labels --help
```

## 3. Train

Choose either the focal-loss v4 entry point or the asymmetric-loss v6 entry
point before examining test results. Pass all data and output paths explicitly.
Capture:

- Git commit and a clean/dirty status;
- complete parsed arguments;
- Python, operating system, CUDA, driver, PyTorch, and dependency versions;
- encoder identifier and upstream weight revision;
- random seed and deterministic-algorithm settings;
- dataset, label-table, split-manifest, and configuration SHA-256 hashes;
- epoch metrics, final checkpoint hash, and termination reason.

After each stage, capture the exact command and hashes of its real inputs and
outputs:

```bash
python scripts/capture_run_provenance.py --help
```

Never select a checkpoint using the test split.

### Sherlock Slurm workflow

Use `slurm/train_cell_phenotype.sbatch` for focal-loss v4 or
asymmetric-loss v6 training. The script requests one GPU with at least 80 GB
of memory on `owners`, eight CPU cores, 64 GB of host memory, and eight hours.
Keep cohort paths and run-specific settings outside Git:

```bash
cp slurm/run.env.example /absolute/private/path/v4-smoke.env
chmod 600 /absolute/private/path/v4-smoke.env
# Replace every example value with a reviewed real path or value.
scripts/submit_cell_training.sh /absolute/private/path/v4-smoke.env
```

This submits a CPU preflight followed by a GPU job with an `afterok`
dependency. The GPU cannot start if real-input validation fails. The jobs
refuse a dirty or unexpected Git commit, relative paths, outputs
inside the checkout, broken data links, malformed labels, duplicate cells,
train/validation core overlap, incompatible H&E arrays, and cells without a
complete patch. It records the resolved command, hardware, logs, GPU
utilization, input validation, checkpoint, metrics, and artifact hashes under
`HEXIF_OUTPUT_DIR`.

Run with `HEXIF_SMOKE=1` first. Review the log, validation JSON, GPU memory,
loss curves, metrics schema, and checkpoint before creating a new production
environment with `HEXIF_SMOKE=0`. Do not reuse a smoke output directory for
a production run.

## 4. Evaluate once on the held-out test split

The retained evaluator reports within-core average-precision lift over a
per-core-mean baseline. Before publishing, also report per-label sample counts,
confidence intervals, missing-label handling, and all prespecified aggregate
metrics. Preserve machine-readable outputs and their hashes.

Do not copy metrics from terminal output into a manuscript by hand. Generate
tables and figures from the frozen machine-readable evaluation outputs.

## 5. Build the Workbench

Build the bundle from the frozen predictions and source manifest, compute
spatial summaries, and render reports. Confirm that all displayed model names,
metrics, confidence strata, and limitations come from the same validated run.
The Workbench must not imply that CODEX truth is available for H&E-only cases.

The calibration command writes `workbench_thresholds.json`. Supply that file,
the model ID, and a schema-version-1 model metadata JSON to
`hexif build-cell-tables`; do not edit thresholds, confidence strata, or metric
fields by hand. Preserve the provenance capture and hashes alongside the
bundle.

## 6. Independent release review

Before adding any result, checkpoint, figure, or manuscript:

- reproduce the environment from a lock file;
- rerun unit tests and real-data integration tests;
- run `python scripts/check_publication_hygiene.py`;
- verify all hashes and split-disjointness checks;
- inspect a stratified sample of registrations, masks, labels, and predictions;
- obtain scientific, privacy, and licensing review;
- document deviations from the prespecified procedure.

Only then update the README, citation metadata, model card, and manuscript.
Until that review is complete, this repository intentionally makes no
performance claim.
