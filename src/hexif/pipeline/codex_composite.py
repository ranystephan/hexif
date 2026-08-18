"""Build the CODEX RGB composite image for the Workbench base layer.

The composite uses a fixed three-channel mapping:

* **Blue**  = channel 0  (DAPI; nuclei)
* **Red**   = channel 7  (CD45; immune)
* **Green** = channel 52 (panCK; tumor)

Channel indices are positions in the 53-channel CODEX panel (see
:data:`hexif.cell_phenotype.FOCUSED_MARKERS` — DAPI / CD45
/ panCK live at the first, second-to-last, and last positions of the
focused-12 marker list). This mapping separates nuclei, immune signal,
and tumor epithelium: blue marks every nucleus, red flags
immune infiltrate, green outlines tumor epithelium.

Why the inverse-scale step matters
----------------------------------

The Workbench bundle's :class:`hexif.scaling.QuantileScaler` maps raw CODEX
intensities to ``log1p`` quantile-
scaled values for the model. If the on-disk array is the scaled form,
the visible composite needs the inverse so percentile clipping happens
in the same intensity space the user expects when reading the slide
("CD45-bright lymphocyte = high raw signal").

Registered training pairs may be stored as **raw uint16**
(channel-first, 53 channels). For that case the inverse-scaler is a
no-op — :func:`_to_raw_intensities` detects scaled vs raw and routes
accordingly. The function still accepts log-quantile-scaled float32
inputs because future bundles may store pre-scaled arrays (the spec
text quotes the scaled-shape contract verbatim).

CODEX channels are highly sparse — most pixels
are background. Computing the 1st/99th percentile over **nonzero**
pixels prevents the noise floor from dominating the rescale and gives
us a per-channel display dynamic range that survives at all zoom
levels (the percentile cache lands in
``manifest.composite_percentiles`` so runtime never has to recompute
it).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hexif.scaling import QuantileScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed composite channel mapping.
# ---------------------------------------------------------------------------

# Why a frozen tuple of (rgb-slot, channel-index, marker-name) tuples:
# pinning the mapping here means callers (build path, route layer,
# tests) cannot drift apart. The composite is *defined* by this table.
_COMPOSITE_CHANNELS: tuple[tuple[str, int, str], ...] = (
    ("red", 7, "CD45"),
    ("green", 52, "panCK"),
    ("blue", 0, "DAPI"),
)

# The minimum number of CODEX channels we need to extract from the
# loaded array. Anything below 53 means the npy is not the canonical
# CODEX panel and the load is a hard error.
_MIN_CODEX_CHANNELS: int = 53

# Default percentile window for the per-channel rescale. The values
# match the spec's table and the existing
# :func:`hexif.pipeline.bundle._render_codex_truth_png` convention
# (1.0–99.0; the truth thumb uses 99.5 because it visualizes one
# channel at a time and tolerates more clipping on the bright tail).
DEFAULT_PERCENTILE_WINDOW: tuple[float, float] = (1.0, 99.0)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_codex_composite(
    codex_npy_path: str | Path,
    scaler: QuantileScaler,
    percentile_window: tuple[float, float] = DEFAULT_PERCENTILE_WINDOW,
) -> tuple[np.ndarray, dict]:
    """Render the (H, W, 3) uint8 CODEX composite for one core.

    Parameters
    ----------
    codex_npy_path
        Path to ``<basename>_CODEX.npy``. The canonical layout is
        ``(C, H, W) uint16`` with ``C >= 53`` (raw scanner counts); a
        ``(H, W, C) float32`` log-quantile-scaled array (post-training
        format) is also accepted. ``ValueError`` for anything else.
    scaler
        The bundle's :class:`hexif.scaling.QuantileScaler` (loaded from
        ``manifest.json:sources.scaler`` or wherever the bundle pinned
        it). Used to inverse-transform a scaled input back to raw
        intensity space; ignored when the input is already raw.
    percentile_window
        ``(p_low, p_high)`` to clip each of the 3 channels to.
        Computed over **nonzero** pixels only; including background
        10 on sparsity.

    Returns
    -------
    rgb
        ``(H, W, 3) uint8`` with the channel order (R, G, B) =
        (CD45, panCK, DAPI). All-zero channels yield an all-zero output
        plane (no spurious noise from a divide-by-zero rescale).
    info
        Dict to land in ``manifest.composite_percentiles[basename]``.
        Carries the percentile values per RGB slot, the channel name
        mapping, and the percentile window used so future readers can
        reproduce the rescale.
    """
    p_low, p_high = percentile_window
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(
            f"percentile_window={percentile_window} invalid; need 0 <= p_low < p_high <= 100"
        )

    codex_npy_path = Path(codex_npy_path)
    arr = np.load(codex_npy_path, mmap_mode="r")
    raw_channels = _to_raw_intensities(arr, scaler)
    height, width = raw_channels.shape[1], raw_channels.shape[2]

    # Allocate the output and build the info dict in one pass; the
    # dict is the audit trail for the rescale and lands verbatim in
    # the manifest, so we write each percentile pair the moment we
    # know it (one fewer place a future refactor can drift).
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    channels_info: list[dict] = []
    rgb_slot_to_idx = {"red": 0, "green": 1, "blue": 2}
    for rgb_name, channel_idx, marker_name in _COMPOSITE_CHANNELS:
        channel = np.asarray(raw_channels[channel_idx], dtype=np.float32)
        rescaled, p_low_value, p_high_value = _rescale_channel(channel, p_low, p_high)
        rgb[..., rgb_slot_to_idx[rgb_name]] = rescaled
        channels_info.append(
            {
                "idx": int(channel_idx),
                "name": marker_name,
                "rgb": rgb_name,
                "p_low": float(p_low_value),
                "p_high": float(p_high_value),
            }
        )
        logger.debug(
            "codex composite ch=%d (%s → %s): p%.1f=%.2f p%.1f=%.2f",
            channel_idx,
            marker_name,
            rgb_name,
            p_low,
            p_low_value,
            p_high,
            p_high_value,
        )

    info: dict = {
        "channels": channels_info,
        "percentile_window": [float(p_low), float(p_high)],
        "image_size": [int(width), int(height)],
    }
    return rgb, info


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _to_raw_intensities(arr: np.ndarray, scaler: QuantileScaler) -> np.ndarray:
    """Return a channel-first ``(C, H, W)`` raw-intensity array.

    Three input shapes are recognised:

    1. ``(C, H, W) uint16`` — canonical training-pairs layout: raw
       scanner counts, no scaling needed. Returned as-is (still uint16;
       caller will float-cast per channel).
    2. ``(H, W, C) float32`` — log-quantile-scaled layout per
       Inverse-transformed via the scaler
       (``raw = expm1(scaled) * (qhi - qlo) + qlo``) and transposed
       back to ``(C, H, W)`` for uniform downstream access.
    3. ``(C, H, W) float32`` — same scaled semantics, channel-first
       layout. Inverse-transformed in place.

    Anything else (wrong rank, wrong channel count, unsupported dtype)
    raises :class:`ValueError`. Failing loud here means the build
    pipeline catches a malformed bundle long before a user opens the
    workbench.
    """
    if arr.ndim != 3:
        raise ValueError(f"CODEX array must be 3-D; got shape {arr.shape}")

    # Disambiguate channel layout. The canonical 53-channel CODEX panel
    # is the minimum we require. The two valid layouts are channel-first
    # (C, H, W) with C >= MIN, or channel-last (H, W, C) with C >= MIN.
    # When both ends are >= MIN (e.g., a tiny 53×64×64 synthetic-test
    # array) we prefer channel-first — the canonical training-pairs
    # layout — so the function stays predictable in test fixtures.
    h_first, _h_mid, h_last = arr.shape
    if h_first >= _MIN_CODEX_CHANNELS:
        channel_first = True
        n_channels = h_first
    elif h_last >= _MIN_CODEX_CHANNELS:
        channel_first = False
        n_channels = h_last
    else:
        raise ValueError(
            f"CODEX array shape {arr.shape} does not match a canonical "
            f"(C>=53, H, W) or (H, W, C>=53) layout; cannot pick channels for the composite"
        )
    if n_channels > scaler.C:
        logger.debug(
            "CODEX array has %d channels; scaler covers %d. Using channels in [0, %d).",
            n_channels,
            scaler.C,
            scaler.C,
        )

    if arr.dtype == np.uint16:
        # Canonical training-pairs layout — already raw scanner counts.
        # The composite percentiles run over raw intensities; no scaler
        # inverse is required. Cast to float32 lazily inside the per-
        # channel loop to keep memory flat on a 2950² × 53-channel core
        # (~900 MB if we float'd everything up front).
        return arr if channel_first else np.transpose(arr, (2, 0, 1))

    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"CODEX array dtype {arr.dtype} is neither uint16 nor floating; "
            "cannot interpret as raw or log-quantile-scaled"
        )

    # Floating point => assume log-quantile-scaled. The scaler's
    # forward step is z = clip((raw - qlo) / (qhi - qlo), 0, inf);
    # log1p(z). Inverse: raw = expm1(scaled) * (qhi - qlo) + qlo.
    # We do it channel-by-channel so we never materialize a 53-channel
    # float64 buffer.
    cf = arr if channel_first else np.transpose(arr, (2, 0, 1))
    raw_channels = np.empty_like(cf, dtype=np.float32)
    n_inverse = min(n_channels, scaler.C)
    for c in range(n_inverse):
        scaled = np.asarray(cf[c], dtype=np.float32)
        denom = float(scaler.qhi[c] - scaler.qlo[c])
        raw_channels[c] = np.expm1(scaled) * denom + float(scaler.qlo[c])
    if n_channels > n_inverse:
        # Extra channels beyond the scaler's coverage are kept verbatim
        # (the only consumers are channels 0, 7, 52 — all inside the
        # canonical 20-/53-channel scaler — so this branch is defensive,
        # not a hot path).
        raw_channels[n_inverse:] = np.asarray(cf[n_inverse:], dtype=np.float32)
    return raw_channels


def _rescale_channel(
    channel: np.ndarray,
    p_low: float,
    p_high: float,
) -> tuple[np.ndarray, float, float]:
    """Linearly rescale one channel to uint8 via nonzero percentiles.

    Returns ``(uint8_plane, p_low_value, p_high_value)``. The percentile
    values are computed over **nonzero pixels only** because CODEX is
    sparse: including the zero background
    collapses both percentiles toward zero and yields a saturated /
    uninformative composite.

    An all-zero channel returns an all-zero plane with `p_low_value =
    p_high_value = 0` so the manifest still carries a faithful record.
    """
    nonzero = channel[channel > 0]
    if nonzero.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8), 0.0, 0.0
    # ``np.percentile`` is the convention used elsewhere in the bundle
    # pipeline (grep: ``np.percentile`` in src/hexif/pipeline/bundle.py
    # and src/hexif/data.py) — stay consistent rather than reaching
    # for np.quantile.
    p_low_value = float(np.percentile(nonzero, p_low))
    p_high_value = float(np.percentile(nonzero, p_high))
    span = p_high_value - p_low_value
    if span <= 0:
        # Edge case: every nonzero pixel is the same value. The naive
        # rescale would divide by zero. Map every nonzero pixel to
        # 255 so the user still sees *something*; the manifest record
        # makes the degenerate case auditable.
        out = np.where(channel > 0, np.uint8(255), np.uint8(0)).astype(np.uint8)
        return out, p_low_value, p_high_value
    clipped = np.clip(channel, p_low_value, p_high_value)
    scaled = (clipped - p_low_value) / span  # in [0, 1]
    # Round-to-nearest beats truncation here: a stride-2 downsample of
    # the resulting uint8 (as the DZI pyramid does at every level)
    # otherwise biases the tile means low by ~0.5/255.
    return (scaled * 255.0 + 0.5).astype(np.uint8), p_low_value, p_high_value
