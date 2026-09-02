# HEXIF

HEXIF predicts cell-level marker positivity and cell phenotypes from H&E image
patches and exposes predictions through a spatial cytometry Workbench.

## Data

Scientific data are not included in this repository because microscopy files
and derived cell data are large and access-controlled. `data/`, `datasets/`,
`bundles/`, `runs/`, `logs/`, and `checkpoints/` are ignored. HEXIF does not
download, generate, or substitute missing cohort data.

The pipeline requires:

- five H&E whole-slide images: `CCoC 1 TMA.svs`, `CCoC 2 TMA.svs`,
  `CCoC 3 TMA.svs`, `CCRCC TMA Region 1.svs`, and
  `CCRCC TMA Region 2.svs`;
- matching 53-channel CODEX acquisitions for `ccOC_TMA1`, `ccOC_TMA2`,
  `ccOC_TMA3`, `ccRCC_TMA1`, and `ccRCC_TMA2`;
- marker/channel metadata, TMA maps, coded identifiers, orientations, and
  exclusions;
- registered `<core_id>_HE.npy` and `<core_id>_CODEX.npy` arrays, registration
  manifests, cell masks, and cell tables;
- frozen train, validation, and test assignments, reviewed labels, thresholds,
  phenotype rules, and calibration artifacts.

The frozen train and validation tables reference 323 H&E arrays. Those arrays
and their source slides are not currently available, so the training preflight
stops before allocating a GPU. See [`docs/data-contracts.md`](docs/data-contracts.md)
for schemas and [`docs/reproducibility.md`](docs/reproducibility.md) for the
required validation procedure.

No synthetic scientific data or placeholder results are used.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/hexif/`](src/hexif/) | Models, training, registration, bundle construction, spatial analysis, and reports |
| [`experiments/`](experiments/) | Training and held-out evaluation entry points |
| [`scripts/`](scripts/) | Data validation, masks, labels, calibration, provenance, and audits |
| [`slurm/`](slurm/) | Guarded Sherlock preflight and training jobs |
| [`webapp/`](webapp/) | Workbench backend and frontend |
| [`tests/`](tests/) | Unit and integration tests |
| [`requirements/train-cu121-py312.lock`](requirements/train-cu121-py312.lock) | Pinned CUDA 12.1 training environment |
| [`docs/data-contracts.md`](docs/data-contracts.md) | Input and artifact schemas |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Validation and release procedure |
| [`docs/workbench.md`](docs/workbench.md) | Workbench bundle and launch instructions |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Development and review requirements |
| [`CITATION.cff`](CITATION.cff) | Software citation metadata |

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

UNI2 and H-optimus-0 are gated upstream. Accept their terms and authenticate
with Hugging Face before training. HEXIF does not redistribute model weights.

## Workflow

All commands require explicit real-data paths.

1. Register CODEX and H&E cores:

   ```bash
   hexif codex-register --help
   ```

2. Build and audit labels:

   ```bash
   python scripts/build_codex_cell_masks.py --help
   python scripts/build_codex_cell_consensus_labels.py --help
   python scripts/calibrate_codex_cell_phenotypes.py --help
   hexif audit-labels --help
   ```

3. Validate inputs and submit a guarded training run:

   ```bash
   python scripts/validate_cell_training_inputs.py --help
   scripts/submit_cell_training.sh /absolute/private/path/run.env
   ```

4. Evaluate a frozen checkpoint:

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

## Verification

```bash
python scripts/check_publication_hygiene.py
ruff check .
ruff format --check .
pytest -q
python -m build
```

Code is licensed under Apache-2.0; see [`LICENSE`](LICENSE). The license applies
to the code only and grants no rights to research data.
