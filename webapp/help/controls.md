# Workbench controls

## Model and core

The model selector lists entries recorded in `manifest.json`. The model card
shows provenance and validation fields from that manifest. The core selector
changes the active registered image and cell table.

## Image and cell display

- **H&E / CODEX** changes the base image. CODEX is disabled when no paired
  measurement exists.
- **Phenotype** colors cells by their thresholded phenotype call.
- **Marker pred** colors cells by the selected marker probability.
- **Marker truth** uses the paired measured label when available.
- **Marker error** displays prediction minus truth when both exist.
- Phenotype chips hide or show calls without changing the underlying data.

Click a polygon to inspect that cell. Marker and phenotype values are shown as
unavailable when the active model or bundle does not provide them.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `[` / `]` | Toggle the left / right sidebar |
| `H` | Toggle help |
| `?` | Show shortcuts |
| `1`–`4` | Select phenotype, prediction, truth, or error coloring |
| `M` | Toggle nucleus / expanded polygon |
| `C` | Toggle prediction / truth comparison |
| `Esc` | Clear cell selection |

Controls are encoded in the URL where practical so a view can be shared among
users who have access to the same bundle. A shared URL does not contain or
grant access to the underlying data.

The shareable state uses these parameters when applicable: `core`, `model`,
`color_by`, `marker`, `base`, and `compare`. Unknown values are rejected or
removed during URL normalization.
