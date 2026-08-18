"""DeepZoom pyramid helpers for the Workbench HE viewer.

OpenSeadragon ships a native DeepZoom client. Serving the H&E as a tile
pyramid (DZI + per-level JPEG tiles) means the browser only fetches
the tiles intersecting the current viewport, so a 2950² core scrolls
at 60 fps over a slow connection with no canvas-side resampling.

Build path
----------

The fast build is :func:`pyvips.Image.dzsave` — a C-level libvips routine
that emits the entire pyramid in one pass. When ``pyvips`` is available
(it is in the ``web`` extra), we use it; otherwise we fall back to a
pure-PIL pyramid (slower but dependency-free at runtime for legacy
bundles).

The pyramid is built **lazily** on first ``/api/he/{core}.dzi`` request
for bundles missing a pre-built pyramid. Subsequent
requests read straight off disk. Idempotent: if ``he_dz.dzi`` already
exists, :func:`ensure_dzi_pyramid` is a no-op.

Security
--------

Path validation lives here because raw ``{level}/{x}_{y}.{ext}`` triples
come straight off the URL. We refuse:

* Non-integer level / x / y.
* ``ext`` outside the allowed JPEG/PNG set.
* Any tile path that escapes ``thumbs/<basename>/he_dz_files/``.

Refusing at this layer means the FastAPI route can be a one-liner and
the test gate exercises both the happy path and the traversal attempt
against the same helper.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.responses import FileResponse, Response

logger = logging.getLogger(__name__)


# Tile geometry chosen to match the OpenSeadragon default. 256-px tiles
# at JPEG Q=85 give ~10 KB/tile on H&E — light enough to ship the whole
# pyramid for a 2950² core in under a second on a LAN.
_TILE_SIZE: int = 256
_OVERLAP: int = 1
_JPEG_QUALITY: int = 85

# Maximum 24-h cache; matches the spec budget. The tile bytes are
# content-addressed by (level, x, y) so cache invalidation only matters
# when we rebuild the pyramid — which we never do once it's on disk.
_TILE_CACHE_CONTROL: str = "public, max-age=86400"

# Allowed tile extensions. PNG is included for the PIL fallback path
# (which emits PNG by default to stay outside the libjpeg dependency).
_ALLOWED_EXTS: frozenset[str] = frozenset({"jpg", "jpeg", "png"})

# The pyramid's directory layout mirrors libvips' convention so a
# pyvips-built pyramid and a PIL-built pyramid are bit-compatible on
# the disk layout that the route reads.
_DZI_BASENAME: str = "he_dz"
_DZI_XML: str = "he_dz.dzi"
_DZI_FILES_DIR: str = "he_dz_files"

# CODEX composite: parallel pyramid for the measured CODEX composite
# (DAPI / panCK / CD45 → blue / green / red). Lives next to the H&E
# pyramid in the same ``thumbs/<basename>/`` directory.
_CODEX_DZI_BASENAME: str = "codex_composite_dz"
_CODEX_DZI_XML: str = "codex_composite_dz.dzi"
_CODEX_DZI_FILES_DIR: str = "codex_composite_dz_files"
_CODEX_COMPOSITE_PNG: str = "codex_composite.png"


def _per_marker_codex_basename(channel_idx: int) -> str:
    """DZI basename for the per-marker CODEX truth pyramid.

    Mirrors the bundle's ``ch{NN}_codex.png`` naming so callers can move
    between the source PNG and the pyramid by string substitution.
    """
    return f"codex_ch{channel_idx:02d}_dz"


def _per_marker_codex_png(channel_idx: int) -> str:
    return f"ch{channel_idx:02d}_codex.png"


class DZIError(RuntimeError):
    """Raised when the DZI pyramid cannot be built or read.

    Distinct from ``FileNotFoundError`` so the route layer can return
    a 500 (build failure) vs 404 (legitimate out-of-range tile) without
    string-matching on the message.
    """


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _thumbs_core_dir(bundle_dir: Path, basename: str) -> Path:
    """Resolve the ``thumbs/<basename>`` dir, validating the basename.

    Why: the basename is caller-supplied (URL path). Refusing slashes
    and ``..`` up front means the rest of this module can pass the
    basename to :class:`pathlib.Path` without worrying about traversal.
    """
    if not basename or any(c in basename for c in ("/", "\\", "..")):
        raise ValueError(f"invalid core basename: {basename!r}")
    return bundle_dir / "thumbs" / basename


def _load_he_array(bundle_dir: Path, basename: str) -> np.ndarray:
    """Read the (downscaled) ``thumbs/<basename>/he.png`` as an RGB uint8 array.

    This is the thumbnail fallback source for the DZI pyramid. Polygons in
    cell-level GeoJSON use full-resolution coordinates, so a pyramid built
    from a downscaled thumbnail cannot provide an aligned overlay.
    :func:`ensure_dzi_pyramid` therefore prefers the full-resolution
    ``.npy`` resolved via :func:`_load_he_full_array`; this helper is
    only invoked when the manifest doesn't carry a ``sources.pairs_dir``
    or the matching ``<basename>_HE.npy`` isn't on disk.
    """
    he_path = _thumbs_core_dir(bundle_dir, basename) / "he.png"
    if not he_path.exists():
        raise FileNotFoundError(f"HE thumbnail missing: {he_path}")
    img = Image.open(he_path)
    if img.mode != "RGB":
        # Strip alpha so libvips
        # doesn't emit 4-channel JPEGs (some browsers reject them).
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resolve_pairs_dir(bundle_dir: Path) -> Path | None:
    """Read ``manifest.json:sources.pairs_dir`` from the bundle.

    Returns ``None`` when the manifest is missing, malformed, or the
    ``sources.pairs_dir`` key is absent. Callers use that ``None`` to
    fall back to the legacy thumb-based pyramid path.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    sources = manifest.get("sources") or {}
    pairs_dir = sources.get("pairs_dir")
    if not pairs_dir:
        return None
    return Path(pairs_dir)


def _load_he_full_array(bundle_dir: Path, basename: str) -> np.ndarray | None:
    """Read the full-resolution ``<basename>_HE.npy`` from ``sources.pairs_dir``.

    Returns the array (always uint8 ``(H, W, 3)``) on success, ``None``
    when the pairs dir / npy file isn't resolvable so the caller can
    fall back. ``.npy`` HE arrays in the canonical layout are uint8 in
    [0, 255] (shape ``(H, W, 3)``); for safety we also accept float
    arrays in [0, 1] and convert to uint8.
    """
    pairs_dir = _resolve_pairs_dir(bundle_dir)
    if pairs_dir is None:
        return None
    he_npy = pairs_dir / f"{basename}_HE.npy"
    if not he_npy.exists():
        return None
    arr = np.load(he_npy, mmap_mode="r")
    # mmap returns a numpy memmap; coerce to a plain ndarray so libvips
    # / PIL get a contiguous buffer.
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        logger.warning(
            "DZI source %s has unexpected shape %s; falling back to thumb",
            he_npy,
            arr.shape,
        )
        return None
    if arr.dtype == np.uint8:
        return arr
    # Float arrays in [0, 1] are the alternate canonical representation
    # Anything else is unusual; warn and bail.
    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.min()) < -0.01 or float(arr.max()) > 1.01:
            logger.warning(
                "DZI source %s float HE out of [0, 1] (min=%s max=%s); falling back",
                he_npy,
                arr.min(),
                arr.max(),
            )
            return None
        return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    logger.warning(
        "DZI source %s has unsupported dtype %s; falling back to thumb",
        he_npy,
        arr.dtype,
    )
    return None


def _build_with_pyvips(he: np.ndarray, out_base: Path) -> None:
    """Build the DZI pyramid via libvips. Fast path."""
    # Why: pyvips is in the `web` extra (Workbench contract §4); localize the
    # import so a missing libvips system lib only breaks the DZI path.
    import pyvips

    img = pyvips.Image.new_from_array(he)
    img.dzsave(
        str(out_base),
        tile_size=_TILE_SIZE,
        overlap=_OVERLAP,
        suffix=f".jpg[Q={_JPEG_QUALITY}]",
    )


def _build_with_pil(he: np.ndarray, out_base: Path) -> None:
    """Build the DZI pyramid via PIL. Slow but pyvips-free fallback.

    This re-implements the DeepZoom level structure manually:

    * Level N (top) is the full-resolution image.
    * Each subsequent level halves both dimensions.
    * Level 0 is 1×1 (or as small as the halving sequence reaches).

    The XML manifest format matches the libvips ``dzsave`` output so
    OpenSeadragon cannot tell which builder was used.
    """
    h, w = he.shape[:2]

    out_base.parent.mkdir(parents=True, exist_ok=True)
    files_dir = out_base.parent / f"{out_base.name}_files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Write the DZI XML manifest. libvips uses jpg; we mirror that for
    # the route layer that infers the extension from the manifest.
    dzi_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        '  Format="jpg"\n'
        f'  Overlap="{_OVERLAP}"\n'
        f'  TileSize="{_TILE_SIZE}"\n'
        "  >\n"
        "  <Size\n"
        f'    Height="{h}"\n'
        f'    Width="{w}"\n'
        "  />\n"
        "</Image>\n"
    )
    out_base.with_suffix(".dzi").write_text(dzi_xml)

    current = Image.fromarray(he)
    # DeepZoom level numbering: level (n_levels-1) is the full image, 0 is 1x1.
    levels: list[Image.Image] = [current]
    while levels[-1].size != (1, 1):
        cw, ch = levels[-1].size
        nw = max(1, cw // 2)
        nh = max(1, ch // 2)
        levels.append(levels[-1].resize((nw, nh), Image.BILINEAR))

    for level_idx, lvl_img in enumerate(reversed(levels)):
        # Reversed so level 0 is the smallest. DeepZoom expects this.
        level = level_idx
        lvl_dir = files_dir / str(level)
        lvl_dir.mkdir(parents=True, exist_ok=True)
        lw, lh = lvl_img.size
        n_cols = max(1, math.ceil(lw / _TILE_SIZE))
        n_rows = max(1, math.ceil(lh / _TILE_SIZE))
        for col in range(n_cols):
            for row in range(n_rows):
                x0 = col * _TILE_SIZE - (_OVERLAP if col > 0 else 0)
                y0 = row * _TILE_SIZE - (_OVERLAP if row > 0 else 0)
                x1 = min(lw, (col + 1) * _TILE_SIZE + _OVERLAP)
                y1 = min(lh, (row + 1) * _TILE_SIZE + _OVERLAP)
                tile = lvl_img.crop((x0, y0, x1, y1))
                # Output as JPEG to match the libvips path so the route
                # layer cannot tell the difference.
                tile.convert("RGB").save(
                    lvl_dir / f"{col}_{row}.jpg",
                    format="JPEG",
                    quality=_JPEG_QUALITY,
                )


def ensure_dzi_pyramid(bundle_dir: Path, basename: str) -> Path:
    """Build ``thumbs/<basename>/he_dz.dzi`` if it doesn't exist; return the path.

    Idempotent: if the pyramid is already on disk, returns the existing
    ``.dzi`` path without touching it. This is the single entrypoint the
    routes call so pre-built and on-demand pyramids behave identically.

    Source-resolution priority:

    1. **Full-resolution** ``<pairs_dir>/<basename>_HE.npy`` resolved
       via the bundle's ``manifest.json:sources.pairs_dir`` field.
       Cell polygons are emitted in this image space (e.g. 2950×2950
       for ccRCC TMA cores), so building the pyramid here keeps the
       polygon overlay geometrically aligned with the H&E.
    2. **Legacy fallback**: the ``thumbs/<basename>/he.png`` thumbnail
       baked by ``_render_he_png`` (max_side=1024 → 840×839 for those
       same cores). Pyramid will display the H&E at thumb resolution,
       so the polygon layer will appear at the wrong scale — this is
       only the path for very old / hand-assembled bundles that don't
       carry ``sources.pairs_dir`` and is logged at WARN.

    Prefers ``pyvips`` (C-speed) and falls back to PIL when libvips is
    not installed. Failure to build via *both* paths raises
    :class:`DZIError` so the route can 500 with a structured body.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    dzi_path = core_dir / _DZI_XML
    if dzi_path.exists():
        return dzi_path

    core_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: prefer the full-resolution HE from the pairs dir.
    he = _load_he_full_array(bundle_dir, basename)
    if he is None:
        logger.warning(
            "DZI source for %s falling back to thumb resolution (pairs_dir or "
            "%s_HE.npy unavailable); polygon overlay will mis-align unless the "
            "frontend scales coords by the H&E thumb factor",
            basename,
            basename,
        )
        he = _load_he_array(bundle_dir, basename)

    out_base = core_dir / _DZI_BASENAME

    try:
        # Localized import so a missing libvips on macOS triggers the
        # PIL fallback below rather than failing module load.
        import pyvips  # noqa: F401

        _build_with_pyvips(he, out_base)
        logger.info(
            "built DZI pyramid via pyvips for %s (source=%dx%d)",
            basename,
            he.shape[1],
            he.shape[0],
        )
    except ImportError:
        # Why: libvips Python bindings are an optional system dep on
        # macOS — fall back to the pure-Python PIL path. Any other
        # exception from the pyvips call is a real bug and re-raises.
        logger.warning(
            "pyvips unavailable; building DZI pyramid via PIL fallback for %s (source=%dx%d)",
            basename,
            he.shape[1],
            he.shape[0],
        )
        _build_with_pil(he, out_base)

    if not dzi_path.exists():
        raise DZIError(f"DZI pyramid build did not produce {dzi_path}")
    return dzi_path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_dzi_xml(dzi_path: Path) -> str:
    """Return the XML manifest content. Pure helper for the route layer.

    Kept separate from :func:`serve_dzi_tile` so callers that want the
    manifest as a string (e.g., tests, OpenAPI examples) don't need a
    ``FileResponse``.
    """
    if not dzi_path.exists():
        raise FileNotFoundError(f"DZI manifest missing: {dzi_path}")
    return dzi_path.read_text()


def _parse_tile_spec(level: str, tile: str) -> tuple[int, int, int, str]:
    """Parse ``level`` and ``tile`` URL segments into validated integers.

    ``tile`` is the full ``{x}_{y}.{ext}`` URL segment. We split it here
    rather than asking FastAPI to multiplex three path params so the
    OpenSeadragon URL convention round-trips byte-for-byte.

    Returns ``(level_int, x, y, ext)``. Raises ``ValueError`` on any
    malformed input — the route layer turns that into a 400.
    """
    try:
        level_int = int(level)
    except (TypeError, ValueError) as e:
        raise ValueError(f"level must be an integer; got {level!r}") from e
    if level_int < 0 or level_int > 64:
        # 2**64 px squared is well past any realistic image; this is a
        # cheap sanity bound that traps overflow attempts.
        raise ValueError(f"level out of range: {level_int}")

    # ``tile`` looks like ``12_7.jpg``. Split on the last dot for the
    # extension, then on the underscore for the coords. ``..`` and
    # slashes are caught here before we ever touch the filesystem.
    if "/" in tile or "\\" in tile or ".." in tile:
        raise ValueError(f"invalid tile spec: {tile!r}")
    if "." not in tile:
        raise ValueError(f"tile missing extension: {tile!r}")
    coords, ext = tile.rsplit(".", 1)
    ext_lower = ext.lower()
    if ext_lower not in _ALLOWED_EXTS:
        raise ValueError(f"unsupported tile ext: {ext!r}")
    if "_" not in coords:
        raise ValueError(f"tile spec missing underscore: {tile!r}")
    x_str, y_str = coords.split("_", 1)
    try:
        x = int(x_str)
        y = int(y_str)
    except ValueError as e:
        raise ValueError(f"tile coords must be integers; got {coords!r}") from e
    if x < 0 or y < 0:
        raise ValueError(f"tile coords must be non-negative; got ({x}, {y})")
    return level_int, x, y, ext_lower


def serve_dzi_tile(
    bundle_dir: Path,
    basename: str,
    level: str,
    tile: str,
) -> Response:
    """Resolve and serve a DeepZoom tile.

    Returns a :class:`FileResponse` for a known tile, a 404 (as a plain
    :class:`Response`) for an out-of-range tile, and raises
    :class:`ValueError` for malformed input. The route layer translates
    ``ValueError`` → 400 and ``FileNotFoundError`` → 404; non-existent
    tiles inside an existing pyramid (e.g., level=99) come back as 404
    rather than 400 so OpenSeadragon's edge-of-image probing doesn't
    spam the logs with 400s.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    level_int, x, y, ext = _parse_tile_spec(level, tile)
    tiles_root = core_dir / _DZI_FILES_DIR
    if not tiles_root.exists():
        raise FileNotFoundError(f"DZI pyramid not built: {tiles_root}")

    # Resolve the tile path *after* validating each segment. resolve()
    # then guarantees the final path is inside tiles_root — a defense in
    # depth against any traversal sequence we missed in parsing.
    tile_path = (tiles_root / str(level_int) / f"{x}_{y}.{ext}").resolve()
    tiles_root_resolved = tiles_root.resolve()
    try:
        tile_path.relative_to(tiles_root_resolved)
    except ValueError as e:
        # Why: path escaped the pyramid root despite the parser. Treat
        # as a hostile request, not a missing tile.
        raise ValueError(f"tile path escapes pyramid root: {tile_path}") from e
    if not tile_path.exists():
        # Out-of-range tile (e.g., OpenSeadragon probing one tile past
        # the right edge). Distinct from a malformed path — 404, not 400.
        raise FileNotFoundError(f"tile not found: level={level_int} x={x} y={y}")
    media_type = "image/jpeg" if ext in {"jpg", "jpeg"} else "image/png"
    return FileResponse(
        tile_path,
        media_type=media_type,
        headers={"Cache-Control": _TILE_CACHE_CONTROL},
    )


# ---------------------------------------------------------------------------
# CODEX composite DZI pyramid (CODEX composite)
# ---------------------------------------------------------------------------
#
# Why a parallel set of helpers (rather than parameterizing the H&E
# ones): the source files live in the same ``thumbs/<basename>/``
# directory and follow the same DZI layout, but the H&E lazy-build
# fallback (full-res ``.npy`` → tile pyramid) does **not** apply to
# CODEX. CODEX rendering is patient-derived intensity processing that
# belongs in :mod:`hexif.pipeline.codex_composite` at bundle-build
# time, not at the route layer. The route only builds a pyramid from a
# pre-rendered ``codex_composite.png``; if that PNG is absent the
# request is a 404, not a lazy render.


def _has_codex_composite_png(bundle_dir: Path, basename: str) -> bool:
    """Return True iff the pre-rendered composite PNG is on disk.

    The PNG is the single source of truth for whether the CODEX pyramid
    can be served. Manifest entries can drift after a partial rebuild,
    so checking the file on disk is the trustworthy answer to "does
    this core have a paired-CODEX composite?".
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    return (core_dir / _CODEX_COMPOSITE_PNG).exists()


def ensure_codex_dzi_pyramid(bundle_dir: Path, basename: str) -> Path:
    """Build ``thumbs/<basename>/codex_composite_dz.dzi`` if absent.

    Idempotent (matches :func:`ensure_dzi_pyramid` semantics). Source is
    the pre-rendered composite PNG written by
    :func:`hexif.pipeline.bundle.build_bundle` /
    :func:`hexif.pipeline.bundle.rebuild_codex_composite`. If that PNG
    is missing, raises :class:`FileNotFoundError` so the route returns
    a structured 404 — CODEX-less cores legitimately have no composite
    (e.g., the v0.5 upload-mode case).

    Builds via ``pyvips`` when available, falls back to PIL otherwise
    (same dual-path logic as the H&E pyramid). Failure to build via
    *both* paths raises :class:`DZIError`.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    dzi_path = core_dir / _CODEX_DZI_XML
    if dzi_path.exists():
        return dzi_path

    composite_png = core_dir / _CODEX_COMPOSITE_PNG
    if not composite_png.exists():
        raise FileNotFoundError(
            f"CODEX composite PNG missing for {basename!r}: {composite_png}. "
            "Run `hexif rebuild-codex-composite --bundle <dir>` to build it."
        )

    img = Image.open(composite_png)
    if img.mode != "RGB":
        img = img.convert("RGB")
    composite_array = np.asarray(img, dtype=np.uint8)
    out_base = core_dir / _CODEX_DZI_BASENAME

    try:
        # Same localized import pattern as the H&E path: a missing
        # libvips system lib should fall through to PIL rather than
        # break module load.
        import pyvips  # noqa: F401

        _build_with_pyvips(composite_array, out_base)
        logger.info(
            "built CODEX DZI pyramid via pyvips for %s (source=%dx%d)",
            basename,
            composite_array.shape[1],
            composite_array.shape[0],
        )
    except ImportError:
        logger.warning(
            "pyvips unavailable; building CODEX DZI pyramid via PIL fallback for %s (%dx%d)",
            basename,
            composite_array.shape[1],
            composite_array.shape[0],
        )
        _build_with_pil(composite_array, out_base)

    if not dzi_path.exists():
        raise DZIError(f"CODEX DZI pyramid build did not produce {dzi_path}")
    return dzi_path


def serve_codex_dzi_tile(
    bundle_dir: Path,
    basename: str,
    level: str,
    tile: str,
) -> Response:
    """Serve a CODEX DZI tile (mirror of :func:`serve_dzi_tile` for H&E).

    The validation contract is identical to the H&E tile route
    (path traversal refused, out-of-range tiles are 404, bad extensions
    are 400). Different tiles-root than the H&E path — see
    ``_CODEX_DZI_FILES_DIR``.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    level_int, x, y, ext = _parse_tile_spec(level, tile)
    tiles_root = core_dir / _CODEX_DZI_FILES_DIR
    if not tiles_root.exists():
        raise FileNotFoundError(f"CODEX DZI pyramid not built: {tiles_root}")

    tile_path = (tiles_root / str(level_int) / f"{x}_{y}.{ext}").resolve()
    tiles_root_resolved = tiles_root.resolve()
    try:
        tile_path.relative_to(tiles_root_resolved)
    except ValueError as e:
        raise ValueError(f"tile path escapes pyramid root: {tile_path}") from e
    if not tile_path.exists():
        raise FileNotFoundError(f"tile not found: level={level_int} x={x} y={y}")
    media_type = "image/jpeg" if ext in {"jpg", "jpeg"} else "image/png"
    return FileResponse(
        tile_path,
        media_type=media_type,
        headers={"Cache-Control": _TILE_CACHE_CONTROL},
    )


# ---------------------------------------------------------------------------
# Per-marker CODEX DZI pyramid (visualizes raw CODEX staining for one of
# the 12 focused markers, e.g. Ki67). Source PNG is ``ch{NN}_codex.png``
# rendered at bundle-build time by
# :func:`hexif.pipeline.bundle._render_codex_truth_png` — a magma-mapped
# p1/p99.5 normalization of the raw uint16 channel. The pyramid is
# built lazily on first request (mirrors the composite path so a
# missing pyramid is not a startup gate).
# ---------------------------------------------------------------------------


def list_per_marker_codex_pngs(bundle_dir: Path, basename: str) -> list[int]:
    """Return the FOCUSED_MARKERS channel indices that have a truth PNG.

    The frontend reads this list to decide which entries appear in the
    "show CODEX truth for marker" selector. Returns an empty list when
    none of the per-marker PNGs are on disk (legacy bundle that only
    rendered the composite).
    """
    from hexif.cell_phenotype import FOCUSED_MARKERS  # local import: optional dep order

    core_dir = _thumbs_core_dir(bundle_dir, basename)
    available: list[int] = []
    for ch in FOCUSED_MARKERS:
        if (core_dir / _per_marker_codex_png(ch)).exists():
            available.append(ch)
    return available


def ensure_per_marker_codex_dzi_pyramid(bundle_dir: Path, basename: str, channel_idx: int) -> Path:
    """Build the DZI pyramid for a single CODEX channel's truth PNG.

    Source is ``thumbs/<basename>/ch{NN}_codex.png``. Missing source PNG
    is a :class:`FileNotFoundError` (the bundle hasn't rendered per-marker
    truth thumbs — the route returns 404). Idempotent: a pre-built
    pyramid is returned without re-rendering.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    basename_dzi = _per_marker_codex_basename(channel_idx)
    dzi_path = core_dir / f"{basename_dzi}.dzi"
    if dzi_path.exists():
        return dzi_path

    src_png = core_dir / _per_marker_codex_png(channel_idx)
    if not src_png.exists():
        raise FileNotFoundError(
            f"CODEX truth PNG missing for {basename!r} channel {channel_idx}: "
            f"{src_png}. Run `hexif rebuild-codex-marker-thumbs --bundle <dir>`."
        )

    img = Image.open(src_png)
    # The truth render is grayscale magma-mapped; .convert("RGB") gives
    # libvips / PIL a 3-channel buffer regardless of source mode.
    if img.mode != "RGB":
        img = img.convert("RGB")
    src_array = np.asarray(img, dtype=np.uint8)
    out_base = core_dir / basename_dzi

    try:
        import pyvips  # noqa: F401

        _build_with_pyvips(src_array, out_base)
        logger.info(
            "built per-marker CODEX DZI pyramid via pyvips for %s ch%02d (%dx%d)",
            basename,
            channel_idx,
            src_array.shape[1],
            src_array.shape[0],
        )
    except ImportError:
        logger.warning(
            "pyvips unavailable; building per-marker CODEX DZI pyramid via PIL "
            "fallback for %s ch%02d (%dx%d)",
            basename,
            channel_idx,
            src_array.shape[1],
            src_array.shape[0],
        )
        _build_with_pil(src_array, out_base)

    if not dzi_path.exists():
        raise DZIError(f"per-marker CODEX DZI pyramid build did not produce {dzi_path}")
    return dzi_path


def serve_per_marker_codex_dzi_tile(
    bundle_dir: Path,
    basename: str,
    channel_idx: int,
    level: str,
    tile: str,
) -> Response:
    """Serve one tile of a per-marker CODEX truth pyramid.

    Validation matches the composite tile route: path traversal refused,
    out-of-range tiles are 404, bad extensions are 400.
    """
    core_dir = _thumbs_core_dir(bundle_dir, basename)
    level_int, x, y, ext = _parse_tile_spec(level, tile)
    tiles_root = core_dir / f"{_per_marker_codex_basename(channel_idx)}_files"
    if not tiles_root.exists():
        raise FileNotFoundError(f"per-marker CODEX DZI pyramid not built: {tiles_root}")

    tile_path = (tiles_root / str(level_int) / f"{x}_{y}.{ext}").resolve()
    tiles_root_resolved = tiles_root.resolve()
    try:
        tile_path.relative_to(tiles_root_resolved)
    except ValueError as e:
        raise ValueError(f"tile path escapes pyramid root: {tile_path}") from e
    if not tile_path.exists():
        raise FileNotFoundError(f"tile not found: level={level_int} x={x} y={y}")
    media_type = "image/jpeg" if ext in {"jpg", "jpeg"} else "image/png"
    return FileResponse(
        tile_path,
        media_type=media_type,
        headers={"Cache-Control": _TILE_CACHE_CONTROL},
    )
