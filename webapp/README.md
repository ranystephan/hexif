# Spatial Cytometry Workbench

This directory contains the FastAPI backend and static frontend for exploring
a validated HEXIF Workbench bundle.

- `workbench_app.py`: application factory, routes, and server entry point;
- `cell_service.py`: validated bundle access and query operations;
- `dzi.py`: Deep Zoom image serving;
- `schemas.py`: API response and request schemas;
- `help.py` and `help/`: in-application documentation;
- `static/workbench/`: dependency-free browser client.

The application is read-only and does not include data, checkpoints, or a demo
bundle. See [`../docs/workbench.md`](../docs/workbench.md) for the bundle
contract, launch command, privacy warning, and interpretation limits.
