# HEXIF

HEXIF is a research codebase for predicting cell-level marker positivity and
cell phenotypes from H&E image patches, then exploring those predictions in a
spatial cytometry Workbench.

This repository is being prepared for a new, fully reproducible validation.
It intentionally contains no patient data, trained checkpoints, benchmark
results, or demonstration bundle. Results will be published only after the
real-data pipeline has been rerun and its artifacts have passed the checks in
[`docs/reproducibility.md`](docs/reproducibility.md).


## Data availability

The scientific dataset is **not included in this repository**. Raw and derived
microscopy data are stored outside Git because they are large, access-controlled
research files; `data/`, `datasets/`, bundles, runs, logs, and checkpoints are
therefore ignored. A source checkout alone cannot run the scientific pipeline,
and HEXIF never downloads, generates, or substitutes missing cohort data.

The current rerun is blocked because the source H&E slides and registered image
arrays have not been recovered. The required external data inventory is:

| Stage | Required external data |
| --- | --- |
| Source acquisition | Five H&E whole-slide images: `CCoC 1 TMA.svs`, `CCoC 2 TMA.svs`, `CCoC 3 TMA.svs`, `CCRCC TMA Region 1.svs`, and `CCRCC TMA Region 2.svs` |
| Source acquisition | Matching raw 53-channel CODEX acquisitions for `ccOC_TMA1`, `ccOC_TMA2`, `ccOC_TMA3`, `ccRCC_TMA1`, and `ccRCC_TMA2` |
| Assay metadata | Marker/channel list, TMA maps, coded core identifiers, image resolution, orientation, exclusions, and a private TMA-to-file mapping |
| Registered images | One `<core_id>_HE.npy` and one `<core_id>_CODEX.npy` per accepted core, plus the registration/QC manifest; the frozen train/validation tables currently reference 323 H&E arrays |
| Cell data | Stable CODEX-derived instance/compartment masks and cell tables containing coded identity, registered-space centroids, marker measurements, and reviewed labels |
| Splits and calibration | Frozen patient/core-level train, validation, and test assignments; label thresholds, consensus-label provenance, phenotype rules, and calibration artifacts fitted on training data only |

Supplying only the 323 real `*_HE.npy` arrays and the existing frozen cell
tables is sufficient to resume the guarded cell-model training run. Recovering
the five source slide/acquisition pairs is preferred because it permits the
registration, masks, labels, and exclusions to be rebuilt and independently
validated. Exact schemas and release requirements are defined in
[`docs/data-contracts.md`](docs/data-contracts.md) and
[`docs/reproducibility.md`](docs/reproducibility.md).


## What is included

- A multi-task cell model with pathology foundation-model encoders, LoRA
  adaptation, marker heads, phenotype heads, and hierarchy-aware losses.
- Training and held-out evaluation entry points for the focal-loss and
  asymmetric-loss configurations.
- Real-data preparation, label-audit, calibration, and registration tools.
- A FastAPI Workbench for cell tables, spatial summaries, image overlays, and
  per-core reports.
- Unit tests that use small deterministic fixtures created only inside tests.

No synthetic scientific data are generated or presented as evidence.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/hexif/`](src/hexif/) | Models, training, registration, bundle construction, spatial analysis, and reports |
| [`experiments/train/`](experiments/train/) | Canonical real-data training entry points |
| [`experiments/eval/`](experiments/eval/) | Canonical held-out evaluation entry point |
| [`scripts/`](scripts/) | Real-data masks, labels, calibration, and audit utilities |
| [`requirements/train-cu121-py312.lock`](requirements/train-cu121-py312.lock) | Exact CUDA 12.1 training environment for the real-data rerun |
| [`webapp/`](webapp/) | Spatial Cytometry Workbench backend and static frontend |
| [`tests/`](tests/) | Unit and integration tests |
| [`docs/data-contracts.md`](docs/data-contracts.md) | Required input schemas and privacy boundary |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Required rerun, provenance, and release procedure |
| [`docs/workbench.md`](docs/workbench.md) | Bundle contract and Workbench launch instructions |

Historical experiments, presentations, papers, logs, arrays, and lost-run
outputs are not part of the publication tree. They are retained privately for
provenance review.

### Canonical entry points

| Task | File |
| --- | --- |
| Register CODEX/H&E cores | [`src/hexif/codex_registration.py`](src/hexif/codex_registration.py) |
| Build cell masks | [`scripts/build_codex_cell_masks.py`](scripts/build_codex_cell_masks.py) |
| Build consensus labels | [`scripts/build_codex_cell_consensus_labels.py`](scripts/build_codex_cell_consensus_labels.py) |
| Audit labels | [`scripts/audit_codex_cell_labels.py`](scripts/audit_codex_cell_labels.py) |
| Calibrate thresholds | [`scripts/calibrate_codex_cell_phenotypes.py`](scripts/calibrate_codex_cell_phenotypes.py) |
| Validate cell-training inputs | [`scripts/validate_cell_training_inputs.py`](scripts/validate_cell_training_inputs.py) |
| Train focal-loss model | [`experiments/train/train_cell_phenotype_v4.py`](experiments/train/train_cell_phenotype_v4.py) |
| Train asymmetric-loss model | [`experiments/train/train_cell_phenotype_v6.py`](experiments/train/train_cell_phenotype_v6.py) |
| Submit guarded Sherlock jobs | [`scripts/submit_cell_training.sh`](scripts/submit_cell_training.sh) |
| Evaluate a checkpoint | [`experiments/eval/eval_cell_phenotype_v4.py`](experiments/eval/eval_cell_phenotype_v4.py) |
| Build Workbench artifacts | [`src/hexif/pipeline/bundle.py`](src/hexif/pipeline/bundle.py) |
| Serve the Workbench | [`webapp/workbench_app.py`](webapp/workbench_app.py) |
| Capture run provenance | [`scripts/capture_run_provenance.py`](scripts/capture_run_provenance.py) |
| Enforce publication hygiene | [`scripts/check_publication_hygiene.py`](scripts/check_publication_hygiene.py) |

The command-line interface is defined in
[`src/hexif/cli/main.py`](src/hexif/cli/main.py). Data and artifact schemas are
normative in [`docs/data-contracts.md`](docs/data-contracts.md) and
[`docs/workbench.md`](docs/workbench.md); the rerun gates are normative in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,web]"
```

For development:

```bash
python -m pip install -e ".[train,web,dev]"
```

Some encoders are gated by their upstream provider. Accept the applicable
model terms and authenticate with Hugging Face before training. HEXIF does not
redistribute those weights.

## Real-data workflow

The commands below require real, locally supplied inputs conforming to
[`docs/data-contracts.md`](docs/data-contracts.md). Paths are explicit: the
software does not download a cohort or silently substitute sample data.

1. Register paired CODEX and H&E cores:

   ```bash
   hexif codex-register --help
   ```

2. Build and audit cell labels:

   ```bash
   python scripts/build_codex_cell_masks.py --help
   python scripts/build_codex_cell_consensus_labels.py --help
   python scripts/calibrate_codex_cell_phenotypes.py --help
   hexif audit-labels --help
   ```

3. Train one declared configuration:

   ```bash
   python experiments/train/train_cell_phenotype_v4.py --help
   python experiments/train/train_cell_phenotype_v6.py --help
   ```

4. Evaluate a frozen checkpoint on a held-out split:

   ```bash
   python experiments/eval/eval_cell_phenotype_v4.py --help
   ```

5. Build and serve a Workbench bundle:

   ```bash
   hexif build-cell-tables --help
   hexif spatial-summary --help
   hexif report --help
   hexif serve-workbench --help
   ```

The complete order of operations, artifact hashes, split freezing, and release
gates is defined in [`docs/reproducibility.md`](docs/reproducibility.md).

## Verification

```bash
python scripts/check_publication_hygiene.py
ruff check .
ruff format --check .
pytest -q
python -m build
```

Tests use deterministic fixtures and must never fetch data or model weights.
Real-data revalidation is a separate release gate and must never be represented
by a skipped or fabricated test.

## Results and citation

There are currently no release-qualified results or public checkpoints. Do not
cite metrics from repository history. The citation metadata in
[`CITATION.cff`](CITATION.cff) covers the software only; article metadata will
be added after the manuscript and real-data validation are final.

## License and contributing

Code is licensed under Apache-2.0; see [`LICENSE`](LICENSE). Contributions are
described in [`CONTRIBUTING.md`](CONTRIBUTING.md), and participation is governed
by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
