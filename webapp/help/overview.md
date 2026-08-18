# HEXIF Spatial Cytometry Workbench

The Workbench is a read-only browser for a validated bundle of cell-level
predictions. It shows H&E imagery, cell polygons, marker probabilities,
phenotype calls, spatial summaries, and—only when the bundle contains paired
measurements—CODEX truth.

## Screen layout

- **Controls** select the core, model, image layer, marker, and phenotype
  filters.
- **Viewer** displays the registered image and interactive cell polygons.
- **Legend** describes the active palette and confidence categories.
- **Cell detail** shows model outputs, available truth, and morphology for the
  selected cell.
- **Stats** summarizes the active core and any validation values stored in the
  bundle manifest.

The marker and phenotype panels are defined by the bundle manifest. Values in
the interface are never substituted when a source is missing.

## Interpretation

Marker probabilities are model estimates. Phenotypes are thresholded logical
combinations of marker outputs. Spatial results inherit uncertainty from image
registration, segmentation, prediction, calibration, and cell typing.

The Workbench is research software, not a medical device. Its outputs must not
be used for diagnosis or treatment decisions. Performance on one cohort does
not establish validity for another tissue, panel, scanner, or population.
