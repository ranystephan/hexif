"""HEXIF training utilities.

Per-model trainer scripts live under ``experiments/train/`` and import
the shared infrastructure (dataset, loss bookkeeping, training loop,
metric tables) from here. The trainer scripts only own the parts that
differ between models — argparse defaults and model construction.
"""
