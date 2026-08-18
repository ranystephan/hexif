# Frequently asked questions

## Why is a cell grey?

In phenotype mode, grey means no phenotype cleared its configured threshold.
In truth mode, grey can also mean that paired truth is unavailable. Inspect the
cell detail and bundle manifest before interpreting the color.

## Why is CODEX disabled?

The active core has no paired CODEX image in the bundle. The Workbench does not
invent or estimate a truth layer.

## What is average precision?

Average precision summarizes a precision-recall curve for a binary target.
Macro average precision is the unweighted mean across labels. Values shown by
the Workbench are read from the active bundle's validated model entry; they are
not recomputed by the browser.

## Why are predictions constant within each polygon?

HEXIF predicts cell-level probabilities. The viewer assigns one value to each
cell polygon, whereas a measured CODEX image contains pixel-level intensity.
They are different data types and should not be compared as image fidelity.

## How do I export results?

`GET /api/export/{core}` returns the available cell and spatial tables for one
core. `hexif report --bundle <bundle> <core>` renders a per-core report. Treat
exports as derived research data subject to the same privacy controls as the
source bundle.

## How should I cite HEXIF?

Use the software metadata in `CITATION.cff`. Article metadata and performance
claims will be added only after the new real-data validation is complete.
