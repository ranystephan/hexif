"""Unit tests for the ``label_set`` dispatch in ``cell_table_to_targets``.

Covers the SPACEc gating (SPACEc gating v1) wiring described in
``experiments/configs/spacec_gating_v1.md``. No GPU, no scanpy, no bundle:
purely synthetic DataFrames that exercise the column-family selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hexif.cell_phenotype import (
    FOCUSED_MARKERS,
    VALID_LABEL_SETS,
    cell_table_to_targets,
)


def _synthetic_table(
    *,
    n_cells: int = 8,
    with_consensus: bool = True,
    with_disagree: bool = True,
    with_gmm_orig: bool = True,
    with_spacec: bool = False,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build a deterministic synthetic cell table.

    Returns the DataFrame plus a dict mapping each column-family key
    (``"pos"`` for ``chXX_pos``, ``"consensus"`` for ``chXX_pos_consensus``,
    ``"gmm_orig"`` for ``chXX_pos_gmm_orig``, ``"spacec"`` for
    ``chXX_pos_spacec``) to the stacked ``(N, n_markers)`` bool array we
    wrote into the DataFrame, so callers can assert exact equality.
    """
    rng = np.random.default_rng(seed)
    n_markers = len(FOCUSED_MARKERS)
    # Three independent random matrices so the families genuinely disagree.
    pos = rng.integers(0, 2, size=(n_cells, n_markers)).astype(bool)
    consensus = (~pos).copy()  # forced to disagree with pos column-by-column
    gmm_orig = pos.copy()  # same as pos by convention (the GMM original)
    spacec = rng.integers(0, 2, size=(n_cells, n_markers)).astype(bool)
    disagree = rng.integers(0, 2, size=(n_cells, n_markers)).astype(bool)

    data: dict[str, np.ndarray] = {}
    for j, ch in enumerate(FOCUSED_MARKERS):
        data[f"ch{ch:02d}_pos"] = pos[:, j]
        if with_consensus:
            data[f"ch{ch:02d}_pos_consensus"] = consensus[:, j]
        if with_disagree:
            data[f"ch{ch:02d}_pos_disagree"] = disagree[:, j]
        if with_gmm_orig:
            data[f"ch{ch:02d}_pos_gmm_orig"] = gmm_orig[:, j]
        if with_spacec:
            data[f"ch{ch:02d}_pos_spacec"] = spacec[:, j]

    df = pd.DataFrame(data)
    expected = {"pos": pos, "consensus": consensus, "gmm_orig": gmm_orig}
    if with_spacec:
        expected["spacec"] = spacec
    return df, expected


def test_label_set_consensus_reads_consensus_columns() -> None:
    df, expected = _synthetic_table()
    marker_y, _, _, _ = cell_table_to_targets(df, label_set="consensus")
    np.testing.assert_array_equal(marker_y, expected["consensus"])
    # And the consensus column is the one that disagrees with chXX_pos, so
    # we can also assert the dispatch did NOT silently fall through to gmm.
    assert not np.array_equal(marker_y, expected["pos"])


def test_label_set_gmm_reads_gmm_columns() -> None:
    df, expected = _synthetic_table()
    marker_y, _, _, _ = cell_table_to_targets(df, label_set="gmm")
    # The gmm branch prefers chXX_pos_gmm_orig when present.
    np.testing.assert_array_equal(marker_y, expected["gmm_orig"])
    # Cross-check that this is genuinely the gmm-orig column, not the
    # consensus column the function would have read by default.
    assert not np.array_equal(marker_y, expected["consensus"])


def test_label_set_spacec_reads_spacec_columns() -> None:
    df, expected = _synthetic_table(with_spacec=True)
    marker_y, disagree, _, _ = cell_table_to_targets(df, label_set="spacec")
    np.testing.assert_array_equal(marker_y, expected["spacec"])
    # marker_disagree_mask must be all-False under spacec, regardless of
    # whether chXX_pos_disagree exists in the frame (it does in this fixture).
    assert disagree.dtype == bool
    assert disagree.shape == marker_y.shape
    assert not disagree.any()


def test_label_set_spacec_missing_columns_raises() -> None:
    df, _ = _synthetic_table(with_spacec=False)
    with pytest.raises(ValueError) as excinfo:
        cell_table_to_targets(df, label_set="spacec")
    msg = str(excinfo.value)
    # The error message must name the first missing column so the operator
    # can spot which channel to investigate without a stack-trace dive.
    first_missing = f"ch{FOCUSED_MARKERS[0]:02d}_pos_spacec"
    assert first_missing in msg


def test_use_consensus_true_is_deprecated_alias() -> None:
    df, expected = _synthetic_table()
    with pytest.deprecated_call():
        marker_y, _, _, _ = cell_table_to_targets(df, use_consensus=True)
    np.testing.assert_array_equal(marker_y, expected["consensus"])


def test_use_consensus_false_is_deprecated_alias_to_gmm() -> None:
    df, expected = _synthetic_table()
    with pytest.deprecated_call():
        marker_y, _, _, _ = cell_table_to_targets(df, use_consensus=False)
    np.testing.assert_array_equal(marker_y, expected["gmm_orig"])


def test_use_consensus_and_label_set_together_raises() -> None:
    df, _ = _synthetic_table()
    with pytest.raises(ValueError) as excinfo:
        cell_table_to_targets(df, label_set="spacec", use_consensus=True)
    msg = str(excinfo.value)
    assert "label_set" in msg
    assert "use_consensus" in msg


def test_invalid_label_set_raises() -> None:
    df, _ = _synthetic_table()
    with pytest.raises(ValueError) as excinfo:
        cell_table_to_targets(df, label_set="bogus")  # type: ignore[arg-type]
    msg = str(excinfo.value)
    # The error must enumerate the valid values so the operator knows what
    # to pick; sourcing this from VALID_LABEL_SETS guarantees one truth.
    for legal in VALID_LABEL_SETS:
        assert legal in msg
