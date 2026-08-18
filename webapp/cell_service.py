"""CellPhenotypeService — read-only service over a Workbench bundle.

Loads the parquet bundle produced by ``hexif build-cell-tables`` (see
:mod:`hexif.pipeline.bundle`) and exposes the slicing operations the
FastAPI workbench routes need.

Design contract
---------------
* No GPU.  No model loading.  Pure parquet + PNG serving.
* Lazy where useful (e.g. AnnData export is only built on demand).
* Thread-safe: pandas DataFrames are read-only, slicing is per-request.
* Cache immutability: every DataFrame published into ``_cells_cache``,
  ``_polygons_cache``, ``_spatial_cache``, or ``_per_core_cache`` is
  treated as immutable after publication. Readers under the matching
  lock receive a stable snapshot; mutations (including the precomputed
  ring columns) happen on a freshly-built local before insertion.
* Filename / path validation: any caller-supplied basename is checked
  against the manifest before opening any file on disk.

This service is limited to read-only browsing of precomputed bundles.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from hexif.cell_phenotype import (
    FOCUSED_MARKERS,
    MARKER_NAMES,
    PHENOTYPE_NAMES,
)
from hexif.pipeline.thresholds import load_thresholds

from .schemas import (
    CellDetailResponse,
    CellFeature,
    CellFeatures,
    CellGeometry,
    CellsFeatureProperties,
    CellsQuery,
    CellsResponse,
    CellsResponseMetadata,
    CodexChannelInfo,
    CodexInfoResponse,
    CodexMarkerInfo,
    CohortFilter,
    CohortStatsResponse,
    MarkerEntry,
    MarkerPositivityRow,
    ModelCardResponse,
    ModelEntry,
    ModelListResponse,
    PerCoreStatsResponse,
    PhenotypeDistRow,
    PhenotypeEntry,
    TruthComparisonRow,
    ValidationRow,
    ValidationStatsResponse,
)

logger = logging.getLogger(__name__)

# Marker name → channel int lookup. Built once at import; used by the
# /api/cells handler to route a user-supplied marker name to the right
# parquet column. The integer column carries the underscore-prefixed
# zero-padded form (``ch{NN:02d}``) — see _marker_columns().
_MARKER_NAME_TO_CHANNEL: dict[str, int] = dict(zip(MARKER_NAMES, FOCUSED_MARKERS, strict=True))

# LRU bound on the per-core joined cells+polygons cache. ~10 cores ×
# ~3 KB/cell × ~3 KB/poly ≈ 200 MB worst-case, which is fine for a
# single-user local app.
_PER_CORE_CACHE_LIMIT: int = 10

# LRU bound on the statistics stats cache. Each entry is a few KB of
# precomputed rows (per-core or per-cohort-filter), so a ~32-entry cap
# fits comfortably in memory while covering the realistic working set
# (one entry per (core, model) the user has visited + a few cohort
# filter combinations).
_STATS_CACHE_LIMIT: int = 32


# ---------------------------------------------------------------------------
# Filter spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreFilter:
    tma: str | None = None
    tissue: str | None = None
    split: str | None = None
    min_cells: int | None = None
    max_rigid_d_px: float | None = None
    confidence: str | None = None  # not yet used; kept for forward compat

    def apply(self, cores: pd.DataFrame) -> pd.DataFrame:
        df = cores
        if self.tma:
            df = df[df["tma"].astype(str) == self.tma]
        if self.tissue:
            df = df[df["tissue"].astype(str) == self.tissue]
        if self.split:
            df = df[df["split"].astype(str) == self.split]
        if self.min_cells is not None:
            df = df[df["n_cells"].astype(int) >= int(self.min_cells)]
        if self.max_rigid_d_px is not None:
            df = df[df["rigid_D_px"].astype(float) <= float(self.max_rigid_d_px)]
        return df


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CellPhenotypeService:
    """Read-only service over a Workbench bundle directory.

    The bundle must contain at minimum::

        manifest.json
        core_summary.parquet
        cells.parquet

    Optional but recommended::

        spatial_summary.parquet
        spatial_edges.parquet
        thumbs/<basename>/*.png
        reports/<basename>.pdf
    """

    def __init__(self, bundle_dir: str | Path, *, model_id: str | None = None) -> None:
        bundle_dir = Path(bundle_dir).resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"bundle_dir does not exist: {bundle_dir}")
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in {bundle_dir}")
        self.bundle_dir = bundle_dir
        self.manifest: dict = json.loads(manifest_path.read_text())
        self.cores = pd.read_parquet(bundle_dir / "core_summary.parquet")
        self._cells_path = bundle_dir / "cells.parquet"
        self._spatial_path = bundle_dir / "spatial_summary.parquet"
        self._edges_path = bundle_dir / "spatial_edges.parquet"
        self._polygons_path = bundle_dir / "cell_polygons.parquet"
        # Model metadata and metrics must come from the validated bundle.
        # Missing metadata is an error: the UI must never invent values.
        self._models: list[dict] = self._read_models()
        if not self._models:
            raise ValueError(f"manifest at {manifest_path} has no models entry")
        valid_ids = [m["id"] for m in self._models]
        if model_id is not None and model_id not in valid_ids:
            raise ValueError(
                f"requested default model_id={model_id!r} is not in manifest.models "
                f"(valid: {valid_ids})"
            )
        self._default_model_id: str = model_id if model_id is not None else self._models[0]["id"]
        threshold_ref = self.manifest.get("threshold_source")
        if not isinstance(threshold_ref, str) or not threshold_ref:
            raise ValueError("manifest.threshold_source must name a bundle artifact")
        threshold_path = (bundle_dir / threshold_ref).resolve()
        if bundle_dir not in threshold_path.parents:
            raise ValueError("manifest.threshold_source must remain inside the bundle")
        self._thresholds = load_thresholds(threshold_path)
        # Lazy caches
        self._cells_cache: pd.DataFrame | None = None
        self._spatial_cache: pd.DataFrame | None = None
        self._polygons_cache: pd.DataFrame | None = None
        # LRU of per-core joined (cells + polygons) DataFrames keyed by
        # basename. Kept tight (≤_PER_CORE_CACHE_LIMIT) so the workbench
        # never balloons past a few-hundred-MB working set.
        self._per_core_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cells_lock = threading.Lock()
        self._spatial_lock = threading.Lock()
        self._polygons_lock = threading.Lock()
        self._per_core_lock = threading.Lock()
        # model selection: dedicated lock for the model-list / model-card
        # accessors so they remain race-free even though the underlying
        # ``self._models`` list is otherwise immutable. We match the
        # ``_per_core_lock`` idiom rather than reuse it to keep concerns
        # separate.
        self._models_lock = threading.Lock()
        # statistics stats cache: keyed on opaque tuples (per-core vs
        # cohort entries share the same dict — the key shape disambig).
        # Values are the fully-built Pydantic response models so a cache
        # hit short-circuits all of the pandas work plus the schema
        # validation.
        self._stats_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._stats_lock = threading.Lock()
        logger.info(
            "CellPhenotypeService: bundle=%s  n_cores=%d  n_cells=%s  models=%s  default=%s",
            bundle_dir,
            len(self.cores),
            self.manifest.get("n_cells", "?"),
            valid_ids,
            self._default_model_id,
        )

    def _read_models(self) -> list[dict]:
        """Return validated model entries from ``manifest.models``."""
        raw = self.manifest.get("models")
        if isinstance(raw, list) and raw:
            return [dict(m) for m in raw]
        raise ValueError("manifest.models must contain at least one validated model entry")

    # ----- lazy loaders -----
    def _cells(self) -> pd.DataFrame:
        with self._cells_lock:
            if self._cells_cache is None:
                self._cells_cache = pd.read_parquet(self._cells_path)
                logger.info(
                    "loaded cells.parquet: %d rows × %d cols",
                    len(self._cells_cache),
                    len(self._cells_cache.columns),
                )
        return self._cells_cache

    def _spatial(self) -> pd.DataFrame:
        if not self._spatial_path.exists():
            return pd.DataFrame()
        with self._spatial_lock:
            if self._spatial_cache is None:
                self._spatial_cache = pd.read_parquet(self._spatial_path)
        return self._spatial_cache

    def _polygons(self) -> pd.DataFrame:
        """Load ``cell_polygons.parquet`` lazily.

        Returns an empty DataFrame when the bundle predates bundle builder.
        Callers that depend on polygons surface a 404 in that branch.
        """
        if not self._polygons_path.exists():
            return pd.DataFrame()
        with self._polygons_lock:
            if self._polygons_cache is None:
                self._polygons_cache = pd.read_parquet(self._polygons_path)
                logger.info(
                    "loaded cell_polygons.parquet: %d rows × %d cols",
                    len(self._polygons_cache),
                    len(self._polygons_cache.columns),
                )
        return self._polygons_cache

    def _per_core_join(self, basename: str) -> pd.DataFrame:
        """Return the join of cells.parquet × cell_polygons.parquet for one core.

        Cached LRU-style — the workbench tends to revisit the same
        core multiple times in a session (color toggles, mode flips,
        phenotype filters) and each visit otherwise re-pays a
        ~40-ms pandas merge.

        We also pre-compute the per-cell polygon ring lists once and
        attach them as ``_nucleus_ring`` / ``_expanded_ring`` columns
        on the cached DataFrame. The ``tolist`` round-trip is the
        single hottest operation in :meth:`cells_geojson_dicts`; doing
        it once per core (vs once per request) drops p95 by ~70 ms.
        """
        with self._per_core_lock:
            if basename in self._per_core_cache:
                # Move to MRU end so it survives eviction longer.
                self._per_core_cache.move_to_end(basename)
                return self._per_core_cache[basename]

        cells = self._cells()
        polys = self._polygons()
        if polys.empty:
            raise FileNotFoundError(
                f"cell_polygons.parquet missing in bundle {self.bundle_dir}; rebuild it via "
                "`hexif rebuild-polygons --bundle <dir>` (see workbench Workbench contract §3.1)"
            )
        cell_sub = cells[cells["basename"] == basename]
        poly_sub = polys[polys["basename"] == basename]
        if cell_sub.empty or poly_sub.empty:
            raise KeyError(f"core has no rows in cells or polygons: {basename!r}")
        # Inner join on (basename, cell_id). The bundle builder pipeline
        # guarantees one polygon row per cell, but we drop duplicates
        # defensively — a duplicate would silently inflate counts.
        joined = cell_sub.merge(
            poly_sub,
            on=["basename", "cell_id"],
            how="inner",
            suffixes=("", "_poly"),
            validate="one_to_one",
        )
        # Pre-compute and attach the polygon ring lists. Doing this
        # once per core amortizes the cost across every subsequent
        # color / phenotype-filter request for the same core.
        joined["_nucleus_ring"] = [
            _polygon_to_geojson_ring(p) for p in joined["nucleus_xy"].to_numpy()
        ]
        joined["_expanded_ring"] = [
            _polygon_to_geojson_ring(p) for p in joined["expanded_xy"].to_numpy()
        ]
        # Cached DataFrames are treated as immutable after publication —
        # readers under ``_per_core_lock`` get a stable snapshot, and the
        # ring columns above are built on the local ``joined`` before
        # insertion so no in-place mutation is ever visible to a reader.
        with self._per_core_lock:
            self._per_core_cache[basename] = joined
            while len(self._per_core_cache) > _PER_CORE_CACHE_LIMIT:
                self._per_core_cache.popitem(last=False)
        return joined

    # ----- public API -----
    @property
    def model_id(self) -> str:
        # Backwards-compat alias for earlier release callers (report, h5ad export)
        # that pre-dated the multi-model API. Always returns the
        # bundle's default model id.
        return self._default_model_id

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    @property
    def valid_model_ids(self) -> list[str]:
        return [m["id"] for m in self._models]

    def resolve_model_id(self, model_id: str | None) -> str:
        """Return ``model_id`` if valid, else the default; raise on bad id.

        Centralizing this in one method lets the route handlers stay
        thin: they just pass the query-param value through and get back
        a guaranteed-valid string. Unknown ids raise ``KeyError`` so the
        route can translate to a structured 422 response carrying the
        list of valid ids (per the API contract).
        """
        if model_id is None or model_id == "":
            return self._default_model_id
        valid = self.valid_model_ids
        if model_id not in valid:
            raise KeyError(f"unknown model id {model_id!r}; valid ids are {valid}")
        return model_id

    def models(self) -> ModelListResponse:
        """Return the model listing payload for ``GET /api/models``."""
        with self._models_lock:
            entries = [
                ModelEntry(
                    id=str(m["id"]),
                    name=str(m["name"]),
                    backbone=str(m["backbone"]),
                    training_split=str(m["training_split"]),
                    macro_marker_ap=float(m["macro_marker_ap"]),
                    macro_phenotype_ap=float(m["macro_phenotype_ap"]),
                    is_default=str(m["id"]) == self._default_model_id,
                )
                for m in self._models
            ]
        return ModelListResponse(models=entries, default_id=self._default_model_id)

    def model_card(self, model_id: str) -> ModelCardResponse:
        """Return the full per-model card payload, or raise KeyError."""
        with self._models_lock:
            for m in self._models:
                if str(m["id"]) == model_id:
                    return ModelCardResponse(
                        id=str(m["id"]),
                        name=str(m["name"]),
                        backbone=str(m["backbone"]),
                        training_split=str(m["training_split"]),
                        macro_marker_ap=float(m["macro_marker_ap"]),
                        macro_phenotype_ap=float(m["macro_phenotype_ap"]),
                        per_marker_ap={str(k): float(v) for k, v in m["per_marker_ap"].items()},
                        per_phenotype_ap={
                            str(k): float(v) for k, v in m["per_phenotype_ap"].items()
                        },
                        notes=str(m.get("notes", "")),
                    )
        raise KeyError(f"model not found: {model_id!r}")

    def info(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "manifest": self.manifest,
            "model_id": self.model_id,
            "n_cores": len(self.cores),
            "tmas": sorted(self.cores["tma"].dropna().unique().tolist())
            if "tma" in self.cores.columns
            else [],
            "tissues": sorted(self.cores["tissue"].dropna().unique().tolist())
            if "tissue" in self.cores.columns
            else [],
            "splits": sorted(self.cores["split"].dropna().unique().tolist())
            if "split" in self.cores.columns
            else [],
            "marker_confidence": dict(self._thresholds.marker_confidence),
            "phenotype_confidence": dict(self._thresholds.phenotype_confidence),
            "thresholds": self._thresholds.to_dict(),
        }

    def list_cores(self, flt: CoreFilter | None = None) -> list[dict[str, Any]]:
        df = self.cores
        if flt is not None:
            df = flt.apply(df)
        keep_cols = ["basename", "tma", "tissue", "split", "n_cells", "rigid_D_px"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        return df[keep_cols].to_dict(orient="records")

    def assert_core(self, basename: str) -> None:
        if basename not in set(self.cores["basename"]):
            raise KeyError(f"core not in bundle: {basename!r}")

    def cell_table_slice(
        self,
        basename: str,
        *,
        limit: int | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self.assert_core(basename)
        cells = self._cells()
        sub = cells[cells["basename"] == basename]
        if columns:
            cols = [c for c in columns if c in sub.columns]
            sub = sub[cols]
        if limit is not None:
            sub = sub.head(int(limit))
        return sub

    def phenotype_summary(self, basename: str) -> dict[str, Any]:
        """Per-core composition + confidence flags."""
        self.assert_core(basename)
        core_row = self.cores[self.cores["basename"] == basename].iloc[0]
        markers = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False):
            pred_col = f"frac_ch{ch:02d}_pos_{self._default_model_id}"
            truth_col = f"frac_ch{ch:02d}_pos_truth"
            markers.append(
                {
                    "channel": int(ch),
                    "name": name,
                    "frac_pred": _safe_float(core_row.get(pred_col)),
                    "frac_truth": _safe_float(core_row.get(truth_col)),
                    "threshold": self._thresholds.marker_threshold(int(ch)),
                    "confidence": self._thresholds.marker_confidence[name],
                }
            )
        phenotypes = []
        for name in PHENOTYPE_NAMES:
            pred_col = f"frac_phenotype_{name}_call_{self._default_model_id}"
            truth_col = f"frac_phenotype_{name}_truth"
            phenotypes.append(
                {
                    "name": name,
                    "frac_pred": _safe_float(core_row.get(pred_col)),
                    "frac_truth": _safe_float(core_row.get(truth_col)),
                    "threshold": self._thresholds.phenotype_threshold(name),
                    "confidence": self._thresholds.phenotype_confidence[name],
                }
            )
        return {
            "basename": basename,
            "tma": str(core_row.get("tma", "")),
            "tissue": str(core_row.get("tissue", "")),
            "split": str(core_row.get("split", "")),
            "n_cells": int(core_row.get("n_cells", 0)),
            "rigid_d_px": _safe_float(core_row.get("rigid_D_px")),
            "model_id": self._default_model_id,
            "markers": markers,
            "phenotypes": phenotypes,
        }

    # ----- CODEX composite: CODEX composite info -----
    def codex_info(self, basename: str) -> CodexInfoResponse | None:
        """Return the CODEX composite info for one core, or ``None``.

        ``None`` signals "this core has no paired CODEX composite" — the
        route layer translates that to a 404 with a structured detail.
        A non-None response always carries three :class:`CodexChannelInfo`
        rows because the current release composite is the fixed DAPI/CD45/panCK
        triple; ``has_composite`` is therefore True whenever a record is
        returned, but kept on the wire for forward-compat with future
        composites that may legitimately have a zero-channel "no
        composite" record (e.g., the upload-mode core in v0.5+).

        The check is **both** the on-disk PNG AND the manifest entry —
        either one being absent is treated as "no composite", because a
        rebuild that wrote the PNG but failed to update the manifest
        leaves the percentile cache stale.
        """
        self.assert_core(basename)
        png_path = self.bundle_dir / "thumbs" / basename / "codex_composite.png"
        manifest_entry = (self.manifest.get("composite_percentiles") or {}).get(basename)
        if not png_path.exists() or manifest_entry is None:
            return None

        # Coerce the manifest's channels list into the Pydantic shape.
        # Defensively skip an entry whose keys are wrong — a malformed
        # manifest should not poison the response, but the test gate
        # picks it up because the channels list would come up short.
        channels: list[CodexChannelInfo] = []
        for ch in manifest_entry.get("channels", []):
            try:
                channels.append(
                    CodexChannelInfo(
                        idx=int(ch["idx"]),
                        name=str(ch["name"]),
                        rgb=str(ch["rgb"]),
                        p_low=float(ch["p_low"]),
                        p_high=float(ch["p_high"]),
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "composite_percentiles[%s] channel record malformed (%s): %s",
                    basename,
                    e,
                    ch,
                )
                continue

        image_size_raw = manifest_entry.get("image_size") or [2950, 2950]
        # Reuse :meth:`_core_image_size` so the CODEX info reports the
        # *same* pixel dimensions the cells GeoJSON metadata reports —
        # otherwise a frontend that trusts CODEX info to size the
        # viewport would mis-align polygons by the composite's edge.
        canonical_size = self._core_image_size(basename)
        if (
            len(image_size_raw) == 2
            and (int(image_size_raw[0]), int(image_size_raw[1])) != canonical_size
        ):
            logger.warning(
                "composite_percentiles[%s] image_size %s != bundle image_size %s; "
                "trusting the bundle's value",
                basename,
                image_size_raw,
                canonical_size,
            )

        # Per-marker truth PNGs are written by render_core_thumbs at
        # bundle-build time. Probe the directory directly so a stale
        # manifest entry doesn't lie about availability.
        core_dir = self.bundle_dir / "thumbs" / basename
        available_markers: list[CodexMarkerInfo] = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
            if (core_dir / f"ch{ch:02d}_codex.png").exists():
                available_markers.append(
                    CodexMarkerInfo(
                        channel_idx=ch,
                        name=name,
                        dzi_url=f"/api/codex/{basename}/marker/{name}.dzi",
                    )
                )

        return CodexInfoResponse(
            core=basename,
            has_composite=True,
            channels=channels,
            image_size=canonical_size,
            available_markers=available_markers,
        )

    # ----- earlier release: bulk cell polygons -----
    def cells_geojson(self, basename: str, query: CellsQuery) -> CellsResponse:
        """Return the GeoJSON FeatureCollection for one core.

        Joins ``cells.parquet`` × ``cell_polygons.parquet`` per core
        (cached LRU-style), resolves the per-cell color and confidence,
        and constructs a :class:`CellsResponse`. The model is built via
        ``model_construct`` on the hot path — Pydantic validation is
        opt-in via :meth:`CellsResponse.model_validate` for tests.

        Coloring rules (see the API contract):

        * ``phenotype``     → integer phenotype index (or -1 = no call).
        * ``marker_pred``   → ``ch{NN}_pred_<model_id>`` prob ∈ [0, 1].
        * ``marker_truth``  → ``ch{NN}_pos`` coerced to {0.0, 1.0};
                              raises if the truth column is absent.
        * ``marker_error``  → ``pred − truth`` in [-1, 1]; same caveat.

        ``query.model`` selects the model id; if ``None`` we fall back
        to the bundle's default. The route handler should pre-validate
        the id via :meth:`resolve_model_id` so a 422 surfaces before
        we open any parquet.
        """
        feature_dicts, image_size, color_meta_marker = self._cells_geojson_payload(basename, query)
        # Build the response via model_construct so Pydantic validation
        # only runs at the route boundary (response_model) or in tests.
        # The dicts already match the schema field-for-field.
        features = [
            CellFeature.model_construct(
                type="Feature",
                id=fd["id"],
                geometry=CellGeometry.model_construct(
                    type="Polygon", coordinates=fd["geometry"]["coordinates"]
                ),
                properties=CellsFeatureProperties.model_construct(**fd["properties"]),
            )
            for fd in feature_dicts
        ]
        metadata = CellsResponseMetadata(
            core=basename,
            n_features=len(features),
            image_size=image_size,
            color_by=query.color_by,
            marker=color_meta_marker,
            truth_label_set=(
                query.truth_label_set
                if query.color_by in ("marker_truth", "marker_error")
                else None
            ),
        )
        return CellsResponse.model_construct(
            type="FeatureCollection", metadata=metadata, features=features
        )

    def cells_geojson_dicts(
        self, basename: str, query: CellsQuery
    ) -> tuple[list[dict[str, Any]], tuple[int, int], str | None]:
        """Return raw feature dicts + metadata for the streaming response.

        Returns ``(features, image_size, marker_echo)``. The streaming
        route serializes the dicts directly via ``json.dumps`` to avoid
        the ~150 ms Pydantic round-trip on a 2k-cell core.
        """
        return self._cells_geojson_payload(basename, query)

    def _cells_geojson_payload(
        self, basename: str, query: CellsQuery
    ) -> tuple[list[dict[str, Any]], tuple[int, int], str | None]:
        """Resolve the per-cell rows and return the raw payload triple.

        Single source of truth for the column-resolution logic. Both
        :meth:`cells_geojson` and :meth:`cells_geojson_dicts` dispatch
        through here so any bug fix lands once.
        """
        self.assert_core(basename)
        # Validate / default the model id once. Routes that go through
        # the schema layer already restrict to known ids, but exposing
        # the helper here keeps unit-test callers (which bypass the
        # route) honest.
        model_id = self.resolve_model_id(query.model)
        joined = self._per_core_join(basename)
        joined = self._apply_phenotype_filter(joined, query.phenotypes, model_id=model_id)

        ring_col = "_nucleus_ring" if query.mode == "nucleus" else "_expanded_ring"
        if ring_col not in joined.columns:
            raise KeyError(f"polygon ring column missing: {ring_col!r}")

        color_values, color_meta_marker = self._resolve_color_values(
            joined,
            color_by=query.color_by,
            marker=query.marker,
            model_id=model_id,
            truth_label_set=query.truth_label_set,
        )
        phenotype_calls = self._resolve_phenotype_call_names(joined, model_id=model_id)
        confidence_strata = self._resolve_confidence_strata(
            phenotype_calls=phenotype_calls,
            color_by=query.color_by,
            marker=query.marker,
        )
        # Per-cell predicted probability for the active marker, sent as
        # ``properties.pred_prob`` alongside ``color_value``. The frontend
        # uses this to drive the probability filter (Min probability
        # slider) regardless of which color_by view is active — so the
        # filter persists when the user toggles Pred → Truth → Error.
        # In phenotype mode there's no active marker, so this is None.
        pred_probs = (
            self._resolve_pred_probs(joined, marker=query.marker, model_id=model_id)
            if query.color_by != "phenotype"
            else None
        )
        feature_dicts = self._build_feature_dicts(
            joined=joined,
            ring_col=ring_col,
            color_values=color_values,
            phenotype_calls=phenotype_calls,
            confidence_strata=confidence_strata,
            pred_probs=pred_probs,
        )
        image_size = self._core_image_size(basename)
        return feature_dicts, image_size, color_meta_marker

    def iter_cells_geojson_features(
        self, basename: str, query: CellsQuery
    ) -> Iterator[CellFeature]:
        """Yield :class:`CellFeature` objects one at a time.

        Used by callers that want validated Pydantic models per feature
        (e.g., tests). The streaming response uses
        :meth:`cells_geojson_dicts` directly for speed.
        """
        response = self.cells_geojson(basename, query)
        yield from response.features

    def cell_detail(
        self, basename: str, cell_id: int, *, model_id: str | None = None
    ) -> CellDetailResponse:
        """Per-cell detail payload — backs ``GET /api/cell/{core}/{cell_id}``.

        Joins cells × polygons × thresholds. Raises :class:`KeyError`
        when the (core, cell_id) pair is absent so the route can
        translate that into a structured 404.

        ``model_id`` picks which model's columns to read; ``None``
        falls back to the bundle's default. Unknown ids raise
        :class:`KeyError` via :meth:`resolve_model_id`.
        """
        self.assert_core(basename)
        resolved_model = self.resolve_model_id(model_id)
        cells = self._cells()
        row = cells[(cells["basename"] == basename) & (cells["cell_id"] == int(cell_id))]
        if row.empty:
            raise KeyError(f"cell not found: core={basename!r} cell_id={cell_id}")
        cell_row = row.iloc[0]

        polys = self._polygons()
        poly_match = (
            polys[(polys["basename"] == basename) & (polys["cell_id"] == int(cell_id))]
            if not polys.empty
            else polys
        )
        if poly_match.empty:
            # The cell may have been skipped during polygon extraction
            # (tiny / degenerate). Fall back to cell-table morphology;
            # the spec allows it but flags the side panel.
            area_nucleus = _safe_float(cell_row.get("area_nucleus_px")) or 0.0
            area_expanded = _safe_float(cell_row.get("area_expanded_px")) or 0.0
        else:
            poly_row = poly_match.iloc[0]
            area_nucleus = float(poly_row["area_nucleus_px"])
            area_expanded = float(poly_row["area_expanded_px"])
        eccentricity = _safe_float(cell_row.get("eccentricity")) or 0.0

        markers = self._marker_entries(cell_row, model_id=resolved_model)
        phenotypes = self._phenotype_entries(cell_row, model_id=resolved_model)

        return CellDetailResponse(
            core=basename,
            cell_id=int(cell_id),
            centroid={
                "x": float(cell_row["centroid_x"]),
                "y": float(cell_row["centroid_y"]),
            },
            features=CellFeatures(
                area_nucleus_px=float(area_nucleus),
                area_expanded_px=float(area_expanded),
                eccentricity=float(eccentricity),
            ),
            markers=markers,
            phenotypes=phenotypes,
        )

    # ----- statistics: stats tab -----
    def per_core_stats(self, basename: str, model_id: str) -> PerCoreStatsResponse:
        """Per-core phenotype distribution + marker positivity + (optional) truth comparison.

        All numbers are computed from ``cells.parquet`` — no new
        computations at serve time other than thresholding the
        ``chXX_pred_<model>`` columns and counting. Threshold values
        come from the bundled v1.1 calibration; non-v1.1 model ids
        currently share the v1.1 thresholds because the bundle pipeline
        emits a single ``thresholds.py`` calibration. That's an
        explicit choice — using per-model thresholds would mean a model
        could "win" the positivity table by setting a generous cut,
        which would mislead the side-by-side comparison.

        Note: the 9 phenotype labels are **hierarchical and overlap**
        (CD8 T-cell ⊂ CD45 immune, etc.), so the per-phenotype counts
        in the response will sum to MORE than n_cells. The derived
        ``no_call`` 10th row is the disjoint complement of "any
        positive call" — only ``no_call + any_positive == n_cells``
        holds. See :class:`PhenotypeDistRow` for the full contract.

        Cached under ``("per_core", basename, model_id)`` so repeated
        loads (the frontend reloads on tab focus) skip the pandas work.
        """
        self.assert_core(basename)
        resolved_model = self.resolve_model_id(model_id)
        key: tuple = ("per_core", basename, resolved_model)
        with self._stats_lock:
            cached = self._stats_cache.get(key)
            if cached is not None:
                self._stats_cache.move_to_end(key)
                return cached

        cells = self._cells()
        sub = cells[cells["basename"] == basename]
        n_cells = len(sub)
        # Phenotype distribution — count cells whose call column is 1.
        # The 9 phenotype labels are hierarchical and OVERLAP (e.g., a
        # CD8 T-cell row also counts toward CD45 immune), so the 9
        # row-counts sum to >= the cells-with-any-positive-call total.
        # We append a derived "no_call" bucket (cells with all call columns
        # = 0) which is the *complement* of "any positive call"; the
        # partition guarantee is therefore:
        #   no_call.count + (cells with >= 1 positive call) == n_cells
        # NOT: sum(all 10 rows) == n_cells. See PhenotypeDistRow doc.
        phenotypes: list[PhenotypeDistRow] = []
        any_pos = pd.Series(False, index=sub.index)
        for name in PHENOTYPE_NAMES:
            col = f"phenotype_{name}_call_{resolved_model}"
            if col in sub.columns:
                pos_mask = sub[col].astype(int) > 0
                count = int(pos_mask.sum())
                any_pos = any_pos | pos_mask
            else:
                count = 0
            fraction = (count / n_cells) if n_cells > 0 else 0.0
            phenotypes.append(PhenotypeDistRow(name=name, count=count, fraction=fraction))
        no_call_count = int((~any_pos).sum()) if n_cells > 0 else 0
        no_call_frac = (no_call_count / n_cells) if n_cells > 0 else 0.0
        phenotypes.append(
            PhenotypeDistRow(name="no_call", count=no_call_count, fraction=no_call_frac)
        )

        # Marker positivity — count cells whose pred prob exceeds the
        # calibrated threshold. We pin to the v1.1 thresholds; the
        # rationale + caveat is documented in the docstring above.
        markers: list[MarkerPositivityRow] = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
            pred_col = f"ch{ch:02d}_pred_{resolved_model}"
            threshold = self._thresholds.marker_threshold(int(ch))
            if pred_col in sub.columns and n_cells > 0:
                pred = sub[pred_col].astype(float).to_numpy()
                n_pos = int((pred > threshold).sum())
            else:
                n_pos = 0
            fraction = (n_pos / n_cells) if n_cells > 0 else 0.0
            markers.append(
                MarkerPositivityRow(
                    name=name,
                    n_positive=n_pos,
                    fraction_positive=fraction,
                    confidence_stratum=_coerce_stratum(self._thresholds.marker_confidence[name]),
                )
            )

        # Truth comparison — gated on the presence of ANY chXX_pos
        # column for this core. The core may have a column at the
        # parquet level but all-NaN values when the CODEX registration
        # was rejected; we treat that as "no truth" too.
        has_truth, comparison = self._compute_truth_comparison(
            sub, model_id=resolved_model, n_cells=n_cells
        )

        response = PerCoreStatsResponse(
            core=basename,
            model=resolved_model,
            n_cells=n_cells,
            phenotypes=phenotypes,
            markers=markers,
            has_truth=has_truth,
            comparison=comparison,
        )
        with self._stats_lock:
            self._stats_cache[key] = response
            while len(self._stats_cache) > _STATS_CACHE_LIMIT:
                self._stats_cache.popitem(last=False)
        return response

    def _compute_truth_comparison(
        self, sub: pd.DataFrame, *, model_id: str, n_cells: int
    ) -> tuple[bool, list[TruthComparisonRow] | None]:
        """Compute the 12-marker truth-comparison rows for one core.

        Returns ``(has_truth, rows_or_none)``. ``has_truth`` is True iff
        at least one focused-12 channel has a non-NaN ``chXX_pos``
        column on this core — that's the spec's contract ("when truth
        exists"). The AP column uses sklearn's
        ``average_precision_score`` and may return ``None`` for a
        degenerate truth column (all-zero / all-one) where AP is
        undefined; the frontend renders that as an em-dash.
        """
        if n_cells == 0:
            return False, None
        # Probe truth columns once — if none have non-NaN rows the
        # comparison block is omitted entirely.
        truth_present = False
        for ch in FOCUSED_MARKERS:
            col = f"ch{ch:02d}_pos"
            if col in sub.columns and sub[col].notna().any():
                truth_present = True
                break
        if not truth_present:
            return False, None

        try:
            from sklearn.metrics import average_precision_score
        except ImportError as e:
            logger.warning(
                "sklearn unavailable in this env (%s); per-marker AP in the "
                "truth-comparison table will be null. Install hexif[web] to "
                "enable it.",
                e,
            )
            average_precision_score = None  # type: ignore[assignment]

        rows: list[TruthComparisonRow] = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
            pred_col = f"ch{ch:02d}_pred_{model_id}"
            truth_col = f"ch{ch:02d}_pos"
            threshold = self._thresholds.marker_threshold(int(ch))
            if pred_col in sub.columns:
                pred = sub[pred_col].astype(float).to_numpy()
                predicted_fraction = float((pred > threshold).mean()) if len(pred) > 0 else 0.0
            else:
                pred = np.full(len(sub), float("nan"), dtype=np.float64)
                predicted_fraction = 0.0
            if truth_col in sub.columns:
                truth_raw = sub[truth_col].to_numpy()
                truth = np.where(np.isnan(truth_raw.astype(float)), 0.0, truth_raw.astype(float))
                truth_fraction = float(truth.mean()) if len(truth) > 0 else 0.0
            else:
                truth = np.zeros(len(sub), dtype=np.float64)
                truth_fraction = 0.0
            abs_err = float(abs(predicted_fraction - truth_fraction))
            ap: float | None = None
            if average_precision_score is not None and truth_col in sub.columns:
                # AP is undefined when the positive class is empty / full
                # — sklearn would emit a warning and return a nonsense
                # value. Detect that branch and emit None instead.
                pos_count = int(truth.sum())
                if 0 < pos_count < len(truth) and pred_col in sub.columns:
                    try:
                        ap = float(average_precision_score(truth.astype(int), pred))
                    except ValueError as e:
                        logger.warning(
                            "average_precision_score failed for %s on truth ch%02d: %s",
                            model_id,
                            ch,
                            e,
                        )
                        ap = None
            rows.append(
                TruthComparisonRow(
                    name=name,
                    predicted_fraction=predicted_fraction,
                    truth_fraction=truth_fraction,
                    absolute_error=abs_err,
                    ap=ap,
                )
            )
        return True, rows

    def cohort_stats(
        self, filt: CohortFilter, *, histogram_phenotype: str = "tumor_ca9_or_panck"
    ) -> CohortStatsResponse:
        """Aggregate per-core phenotype + marker stats under a filter.

        Filtering happens at the core-summary level (tissue / tma /
        split) and then propagates to cells.parquet via the ``basename``
        column. The TMA filter accepts either the bare suffix
        (``"TMA1"``) or the bundle-internal prefix form
        (``"ccRCC_TMA1"``); the suffix form matches via ``endswith``
        so the frontend can pass the short string.

        The histogram is computed per-core: for each core, compute the
        fraction of cells where ``phenotype_<name>_call_<model>``
        is positive; then bin those fractions into the fixed 10-bin
        ``[0, 1]`` grid. Cached under
        ``("cohort", tissue, tma, split, model, histogram_phenotype)``.
        """
        resolved_model = self.resolve_model_id(filt.model)
        if histogram_phenotype not in PHENOTYPE_NAMES:
            raise KeyError(
                f"unknown histogram phenotype {histogram_phenotype!r}; "
                f"valid: {list(PHENOTYPE_NAMES)}"
            )
        key: tuple = (
            "cohort",
            filt.tissue,
            filt.tma,
            filt.split,
            resolved_model,
            histogram_phenotype,
        )
        with self._stats_lock:
            cached = self._stats_cache.get(key)
            if cached is not None:
                self._stats_cache.move_to_end(key)
                return cached

        # Resolve the filtered core set first — that's the universe
        # for both the per-cell aggregates and the per-core histogram.
        cores_df = self._filter_cores(filt)
        valid_tissues = sorted(self.cores["tissue"].dropna().unique().tolist())
        valid_splits = sorted(self.cores["split"].dropna().unique().tolist())
        if filt.tissue is not None and filt.tissue not in valid_tissues:
            raise KeyError(f"unknown tissue {filt.tissue!r}; valid: {valid_tissues}")
        if filt.split is not None and filt.split not in valid_splits:
            raise KeyError(f"unknown split {filt.split!r}; valid: {valid_splits}")
        basenames = cores_df["basename"].astype(str).tolist()
        n_cores = len(basenames)
        if filt.tma is not None and n_cores == 0:
            # We accept "TMA1" or "ccRCC_TMA1" — but if neither form
            # matched we still report 0 cores rather than 422 so the
            # frontend can show "no cores in this slice" naturally. The
            # tissue/split filters above are 422 because the user
            # entered a typo; tma filter combinations with tissue can
            # legitimately produce 0 cores.
            logger.info(
                "cohort_stats: filter %r returned 0 cores (tissue=%r tma=%r split=%r)",
                filt,
                filt.tissue,
                filt.tma,
                filt.split,
            )

        cells = self._cells()
        sub = cells[cells["basename"].isin(basenames)] if n_cores > 0 else cells.iloc[0:0]
        n_cells_total = len(sub)

        # Phenotype summary — same shape as per_core_stats but pooled.
        phenotype_summary: list[PhenotypeDistRow] = []
        any_pos = pd.Series(False, index=sub.index)
        for name in PHENOTYPE_NAMES:
            col = f"phenotype_{name}_call_{resolved_model}"
            if col in sub.columns and n_cells_total > 0:
                pos_mask = sub[col].astype(int) > 0
                count = int(pos_mask.sum())
                any_pos = any_pos | pos_mask
            else:
                count = 0
            fraction = (count / n_cells_total) if n_cells_total > 0 else 0.0
            phenotype_summary.append(PhenotypeDistRow(name=name, count=count, fraction=fraction))
        no_call_count = int((~any_pos).sum()) if n_cells_total > 0 else 0
        no_call_frac = (no_call_count / n_cells_total) if n_cells_total > 0 else 0.0
        phenotype_summary.append(
            PhenotypeDistRow(name="no_call", count=no_call_count, fraction=no_call_frac)
        )

        # Marker summary — pooled positivity counts.
        marker_summary: list[MarkerPositivityRow] = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
            pred_col = f"ch{ch:02d}_pred_{resolved_model}"
            threshold = self._thresholds.marker_threshold(int(ch))
            if pred_col in sub.columns and n_cells_total > 0:
                pred = sub[pred_col].astype(float).to_numpy()
                n_pos = int((pred > threshold).sum())
            else:
                n_pos = 0
            fraction = (n_pos / n_cells_total) if n_cells_total > 0 else 0.0
            marker_summary.append(
                MarkerPositivityRow(
                    name=name,
                    n_positive=n_pos,
                    fraction_positive=fraction,
                    confidence_stratum=_coerce_stratum(self._thresholds.marker_confidence[name]),
                )
            )

        # Per-core histogram for the chosen phenotype.
        bins = [i / 10.0 for i in range(11)]
        histogram_counts = [0] * 10
        hist_col = f"phenotype_{histogram_phenotype}_call_{resolved_model}"
        if hist_col in sub.columns:
            for bn in basenames:
                bn_mask = sub["basename"] == bn
                bn_cells = sub[bn_mask]
                if len(bn_cells) == 0:
                    continue
                frac = float(bn_cells[hist_col].astype(int).mean())
                # Place into the [0, 1] grid; clamp the rightmost edge
                # so a perfect-1 fraction lands in bin 9, not out-of-range.
                idx = min(int(frac * 10), 9)
                histogram_counts[idx] += 1

        echo_filter = CohortFilter(
            tissue=filt.tissue,
            tma=filt.tma,
            split=filt.split,
            model=resolved_model,
        )
        response = CohortStatsResponse(
            filter=echo_filter,
            n_cores=n_cores,
            n_cells_total=n_cells_total,
            phenotype_summary=phenotype_summary,
            marker_summary=marker_summary,
            histogram_phenotype=histogram_phenotype,
            histogram_bins=bins,
            histogram_counts=histogram_counts,
        )
        with self._stats_lock:
            self._stats_cache[key] = response
            while len(self._stats_cache) > _STATS_CACHE_LIMIT:
                self._stats_cache.popitem(last=False)
        return response

    def _filter_cores(self, filt: CohortFilter) -> pd.DataFrame:
        """Apply the cohort filter to ``core_summary``.

        Why this is a private helper and not :meth:`CoreFilter.apply`:
        :class:`CohortFilter` is the wire shape (with the bundle-default
        model resolution), while :class:`CoreFilter` is the broader
        Atlas-mode filter that also carries ``min_cells`` /
        ``max_rigid_d_px``. Keeping them separate stops a CohortFilter
        from accidentally inheriting Atlas-mode defaults.

        TMA matching is intentionally suffix-tolerant: the bundle
        stores ``"ccRCC_TMA1"`` etc, but the frontend cohort filter
        sends just ``"TMA1"``. We match via ``endswith`` so both
        forms work — see the schema docstring.
        """
        df = self.cores
        if filt.tissue is not None:
            df = df[df["tissue"].astype(str) == str(filt.tissue)]
        if filt.tma is not None:
            tma_str = str(filt.tma)
            tma_col = df["tma"].astype(str)
            mask = (tma_col == tma_str) | tma_col.str.endswith(f"_{tma_str}")
            df = df[mask]
        if filt.split is not None:
            df = df[df["split"].astype(str) == str(filt.split)]
        return df

    def validation_stats(self, model_id: str) -> ValidationStatsResponse:
        """Per-marker / per-phenotype AP table for one model.

        Reads from ``manifest.models[<id>].per_marker_ap`` /
        ``per_phenotype_ap`` — these are the **source of truth** for the
        AP claims (populated by ``hexif rebuild-cell-tables``). The
        ``n_cores_with_truth`` count counts cores where at least one
        chXX_pos column has a non-NaN value (for markers) or any
        phenotype_<name>_label has a non-NaN value (for phenotypes).

        ``per_core_ap`` is the per-core AP list from the bundle's
        ``per_core_ap.csv`` when
        present; otherwise ``None``. The frontend skips variability
        statistics when the source bundle does not contain them.
        """
        resolved_model = self.resolve_model_id(model_id)
        key: tuple = ("validation", resolved_model)
        with self._stats_lock:
            cached = self._stats_cache.get(key)
            if cached is not None:
                self._stats_cache.move_to_end(key)
                return cached

        with self._models_lock:
            entry = next((m for m in self._models if str(m["id"]) == resolved_model), None)
        if entry is None:
            raise KeyError(f"model not found: {resolved_model!r}")
        per_marker_ap = entry["per_marker_ap"]
        per_phenotype_ap = entry["per_phenotype_ap"]
        missing_markers = set(MARKER_NAMES) - set(per_marker_ap)
        missing_phenotypes = set(PHENOTYPE_NAMES) - set(per_phenotype_ap)
        if missing_markers or missing_phenotypes:
            raise ValueError(
                "model validation metadata is incomplete: "
                f"markers={sorted(missing_markers)}, phenotypes={sorted(missing_phenotypes)}"
            )

        # n_cores_with_truth — count cores in core_summary that carry
        # any non-NaN truth signal for the marker / phenotype. The
        # bundle currently doesn't break this down per channel, so we
        # use a single global count for all rows; tightening this
        # would require an additional per-marker pass over cells.parquet
        # which is exactly the kind of "serve-time computation" the
        # spec forbids.
        n_cores_truth = self._n_cores_with_any_truth()

        per_core_csv: dict[str, list[float]] = self._maybe_load_per_core_ap_csv(resolved_model)

        markers: list[ValidationRow] = []
        for name in MARKER_NAMES:
            macro = float(per_marker_ap[name])
            pc = per_core_csv.get(name)
            markers.append(
                ValidationRow(
                    name=name,
                    macro_ap=macro,
                    n_cores_with_truth=n_cores_truth,
                    confidence_stratum=_coerce_stratum(self._thresholds.marker_confidence[name]),
                    per_core_ap=pc,
                )
            )
        phenotypes: list[ValidationRow] = []
        for name in PHENOTYPE_NAMES:
            macro = float(per_phenotype_ap[name])
            pc = per_core_csv.get(name)
            phenotypes.append(
                ValidationRow(
                    name=name,
                    macro_ap=macro,
                    n_cores_with_truth=n_cores_truth,
                    confidence_stratum=_coerce_stratum(self._thresholds.phenotype_confidence[name]),
                    per_core_ap=pc,
                )
            )
        response = ValidationStatsResponse(
            model=resolved_model,
            macro_marker_ap=float(entry["macro_marker_ap"]),
            macro_phenotype_ap=float(entry["macro_phenotype_ap"]),
            markers=markers,
            phenotypes=phenotypes,
        )
        with self._stats_lock:
            self._stats_cache[key] = response
            while len(self._stats_cache) > _STATS_CACHE_LIMIT:
                self._stats_cache.popitem(last=False)
        return response

    def _n_cores_with_any_truth(self) -> int:
        """Count cores whose core_summary row has ANY non-NaN truth signal.

        The core_summary table carries ``frac_chXX_pos_truth`` columns
        per focused marker; a core with at least one finite value
        contributes to ``n_cores_with_truth``. The value is computed from
        the current bundle rather than assumed from cohort history.
        """
        truth_cols = [f"frac_ch{ch:02d}_pos_truth" for ch in FOCUSED_MARKERS]
        present_cols = [c for c in truth_cols if c in self.cores.columns]
        if not present_cols:
            return 0
        any_truth = self.cores[present_cols].notna().any(axis=1)
        return int(any_truth.sum())

    def _maybe_load_per_core_ap_csv(self, model_id: str) -> dict[str, list[float]]:
        """Try to load a per-core AP CSV from the benchmarks tree.

        Returns ``{name: [ap_per_core, ...]}`` when a CSV at one of the
        expected paths exists; an empty dict otherwise. The schema
        contract says ``per_core_ap`` is ``None`` when this returns no
        rows — the caller wires the ``None`` straight into the response.

        We probe multiple candidate paths so a future bundle that ships
        the CSV inside the bundle works without code
        changes.
        """
        candidates = [
            self.bundle_dir / "benchmarks" / "codex_cell_level" / model_id / "per_core_ap.csv",
            self.bundle_dir / "per_core_ap.csv",
        ]
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                except (pd.errors.ParserError, ValueError) as e:
                    logger.warning("per-core AP CSV malformed at %s: %s", path, e)
                    return {}
                # Expected columns: 'name' (marker or phenotype) and
                # one column per basename. We pivot to {name: [ap...]}
                # but only when the CSV is structured that way; older
                # layouts fall through to an empty dict.
                if "name" not in df.columns:
                    logger.warning("per-core AP CSV at %s lacks 'name' column", path)
                    return {}
                out: dict[str, list[float]] = {}
                value_cols = [c for c in df.columns if c != "name"]
                for _, row in df.iterrows():
                    vals = [float(row[c]) for c in value_cols if pd.notna(row[c])]
                    if vals:
                        out[str(row["name"])] = vals
                return out
        return {}

    # ----- internal helpers for cells_geojson -----
    def _apply_phenotype_filter(
        self, joined: pd.DataFrame, phenotypes_csv: str | None, *, model_id: str
    ) -> pd.DataFrame:
        """Return rows whose dominant phenotype call is in the CSV filter."""
        if not phenotypes_csv:
            return joined
        wanted = {p.strip() for p in phenotypes_csv.split(",") if p.strip()}
        if not wanted:
            return joined
        unknown = wanted - set(PHENOTYPE_NAMES)
        if unknown:
            raise KeyError(f"unknown phenotype filter token(s): {sorted(unknown)}")
        # A cell passes the filter if *any* of its requested phenotype
        # call columns is positive. That matches the spec's "set
        # intersection" intent — the frontend chips are OR'd.
        mask = pd.Series(False, index=joined.index)
        for name in wanted:
            col = f"phenotype_{name}_call_{model_id}"
            if col in joined.columns:
                mask = mask | (joined[col].astype(int) > 0)
        return joined[mask]

    # Mapping from the ``truth_label_set`` query parameter to the bundle's
    # column suffix for per-cell positivity. Kept on the service class so
    # tests can introspect it without parsing strings.
    _TRUTH_LABEL_SET_TO_SUFFIX: ClassVar[dict[str, str]] = {
        "gmm": "",  # bare chXX_pos — v1.0 default (current workbench behavior)
        "consensus": "_consensus",  # chXX_pos_consensus — v1.1 training labels
        "spacec": "_spacec",  # chXX_pos_spacec — v2 cluster-derived labels
    }

    def _truth_column_for(self, channel: int, truth_label_set: str) -> str:
        """Resolve the cells.parquet column for a marker truth view.

        Args:
            channel: ORION/CODEX channel index (e.g. 27 for CD68).
            truth_label_set: one of ``gmm`` / ``consensus`` / ``spacec``.

        Returns:
            The column name, e.g. ``ch27_pos`` (gmm), ``ch27_pos_consensus``,
            or ``ch27_pos_spacec``.

        Raises:
            ValueError: if ``truth_label_set`` is unknown.
        """
        if truth_label_set not in self._TRUTH_LABEL_SET_TO_SUFFIX:
            raise ValueError(
                f"unknown truth_label_set: {truth_label_set!r}; expected one of "
                f"{sorted(self._TRUTH_LABEL_SET_TO_SUFFIX)}"
            )
        suffix = self._TRUTH_LABEL_SET_TO_SUFFIX[truth_label_set]
        return f"ch{channel:02d}_pos{suffix}"

    def _resolve_color_values(
        self,
        joined: pd.DataFrame,
        *,
        color_by: str,
        marker: str | None,
        model_id: str,
        truth_label_set: str = "gmm",
    ) -> tuple[np.ndarray, str | None]:
        """Build the per-row ``color_value`` array and resolve the marker meta.

        Args:
            joined: per-core cells × polygons frame.
            color_by: ``phenotype`` / ``marker_pred`` / ``marker_truth`` /
                ``marker_error``.
            marker: focused-panel marker name (required when ``color_by`` !=
                ``phenotype``).
            model_id: which model's predictions to read for pred/error views.
            truth_label_set: which positivity column family to read as the
                marker truth — ``gmm`` (``chXX_pos`` — workbench default for
                backwards compat), ``consensus`` (``chXX_pos_consensus`` —
                v1.1 training labels), or ``spacec`` (``chXX_pos_spacec`` —
                v2 cluster-derived labels). Ignored for color_by in
                {``phenotype``, ``marker_pred``} which don't read the truth
                column.

        Returns ``(color_values, marker_echo)``. ``marker_echo`` is the
        marker name we echo back in metadata — None for ``phenotype``.
        """
        if color_by == "phenotype":
            # Use the highest-scoring phenotype call as the categorical
            # value. -1 sentinel = no positive call (i.e., "no_call").
            return self._phenotype_color_values(joined, model_id=model_id), None

        if marker is None:
            # The Pydantic layer enforces this, but defensive here for
            # callers bypassing the schema (tests, internal use).
            raise ValueError(f"marker required for color_by={color_by!r}")
        if marker not in _MARKER_NAME_TO_CHANNEL:
            raise KeyError(f"unknown marker: {marker!r}")
        channel = _MARKER_NAME_TO_CHANNEL[marker]
        pred_col = f"ch{channel:02d}_pred_{model_id}"
        truth_col = self._truth_column_for(channel, truth_label_set)

        if color_by == "marker_pred":
            if pred_col not in joined.columns:
                # A non-default model (e.g., MIPHEI's 9-marker panel)
                # may not cover this channel. Returning a NaN-filled
                # array preserves the row order so the GeoJSON feature
                # count never drifts between model selections; the
                # client renders NaN as a "no data" swatch.
                return np.full(len(joined), float("nan"), dtype=np.float64), marker
            return joined[pred_col].to_numpy(dtype=np.float64), marker
        if color_by == "marker_truth":
            if truth_col not in joined.columns:
                raise KeyError(f"truth column missing for marker {marker!r}: {truth_col!r}")
            return joined[truth_col].astype(float).to_numpy(dtype=np.float64), marker
        if color_by == "marker_error":
            if truth_col not in joined.columns:
                raise KeyError(f"truth column missing for marker {marker!r}: {truth_col!r}")
            if pred_col not in joined.columns:
                # As with marker_pred — NaN preserves row order rather
                # than dropping the feature for a model that doesn't
                # cover this marker.
                return np.full(len(joined), float("nan"), dtype=np.float64), marker
            pred = joined[pred_col].to_numpy(dtype=np.float64)
            truth = joined[truth_col].astype(float).to_numpy(dtype=np.float64)
            return pred - truth, marker
        raise ValueError(f"unhandled color_by: {color_by!r}")

    def _resolve_pred_probs(
        self,
        joined: pd.DataFrame,
        *,
        marker: str | None,
        model_id: str,
    ) -> np.ndarray | None:
        """Return the predicted probability per cell for the active marker.

        Independent of color_by — same value in pred / truth / error
        views so the frontend's probability filter survives a view
        toggle. ``None`` when there is no marker (phenotype mode) or
        when the model doesn't carry a column for this channel.
        """
        if marker is None or marker not in _MARKER_NAME_TO_CHANNEL:
            return None
        channel = _MARKER_NAME_TO_CHANNEL[marker]
        pred_col = f"ch{channel:02d}_pred_{model_id}"
        if pred_col not in joined.columns:
            return None
        return joined[pred_col].to_numpy(dtype=np.float64)

    def _phenotype_color_values(self, joined: pd.DataFrame, *, model_id: str) -> np.ndarray:
        """Return an int-coded phenotype index per row (-1 = no call).

        Ties are broken by the order of :data:`PHENOTYPE_NAMES` (first
        positive call wins). The choice is documented here rather than
        configured because it is paired with the frontend's palette key
        order — changing one without the other corrupts the rendering.
        """
        n = len(joined)
        result = np.full(n, -1, dtype=np.int64)
        for idx, name in enumerate(PHENOTYPE_NAMES):
            col = f"phenotype_{name}_call_{model_id}"
            if col not in joined.columns:
                continue
            calls = joined[col].astype(int).to_numpy()
            # First-positive-wins: only update cells that don't yet
            # have an assigned call.
            unassigned = result < 0
            assign = unassigned & (calls > 0)
            result = np.where(assign, idx, result)
        return result.astype(np.float64)

    def _resolve_phenotype_call_names(
        self, joined: pd.DataFrame, *, model_id: str
    ) -> list[str | None]:
        """Resolve each row's dominant phenotype call to a name (or None)."""
        codes = self._phenotype_color_values(joined, model_id=model_id).astype(int)
        return [PHENOTYPE_NAMES[c] if 0 <= c < len(PHENOTYPE_NAMES) else None for c in codes]

    def _resolve_confidence_strata(
        self,
        *,
        phenotype_calls: list[str | None],
        color_by: str,
        marker: str | None,
    ) -> list[str]:
        """Return the per-feature confidence stratum.

        For ``color_by=phenotype`` we use the phenotype confidence; for
        the marker modes we use the marker confidence. Both tables come
        from :mod:`hexif.pipeline.thresholds`.
        """
        # "weak" is the conservative default for unknown / no-call rows
        # — the side panel renders that as a yellow badge so the user
        # is never misled into trusting a no-call cell.
        if color_by == "phenotype":
            return [
                self._thresholds.phenotype_confidence[name] if name is not None else "unknown"
                for name in phenotype_calls
            ]
        # Marker modes: stratum is a function of the marker, not the cell.
        if marker is None:
            return ["weak"] * len(phenotype_calls)
        stratum = self._thresholds.marker_confidence[marker]
        return [stratum] * len(phenotype_calls)

    def _build_feature_dicts(
        self,
        *,
        joined: pd.DataFrame,
        ring_col: str,
        color_values: np.ndarray,
        phenotype_calls: list[str | None],
        confidence_strata: list[str],
        pred_probs: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Construct the raw GeoJSON-shaped feature dicts.

        Returns plain Python dicts (not Pydantic models) so the
        streaming response can ``json.dumps`` each feature with zero
        Pydantic overhead. The dict layout matches
        :class:`CellFeature` field-for-field; the round-trip
        ``CellsResponse.model_validate({...})`` is what tests use to
        guarantee the shape.
        """
        cell_ids = joined["cell_id"].to_numpy(dtype=np.int64).tolist()
        # Pre-resolved polygon rings live as a column on the cached
        # join; pulling them via ``.to_numpy()`` returns an object
        # ndarray of list[list[float]] without further coercion.
        rings = joined[ring_col].tolist()
        color_list = color_values.astype(np.float64, copy=False).tolist()
        # Pred-probability list — same length as the others, populated
        # only in marker modes. The frontend's probability filter reads
        # this; when None / NaN the filter treats the cell as ineligible
        # and hides it once the threshold is > 0.
        pred_prob_list: list[float | None]
        if pred_probs is not None:
            arr = pred_probs.astype(np.float64, copy=False)
            pred_prob_list = [None if not np.isfinite(v) else float(v) for v in arr]
        else:
            pred_prob_list = [None] * len(joined)
        strata_list = [_coerce_stratum(s) for s in confidence_strata]
        out: list[dict[str, Any]] = []
        for i in range(len(joined)):
            coords = rings[i]
            if coords is None:
                # Cell missing a polygon ring is dropped silently here
                # but is *logged*: the bundle builder pipeline guarantees
                # every cell either has a polygon or appears in
                # polygons_skipped.csv, so this branch is unreachable
                # on a well-formed bundle.
                logger.debug("skipping cell with empty polygon: cell_id=%d", cell_ids[i])
                continue
            cid = cell_ids[i]
            out.append(
                {
                    "type": "Feature",
                    "id": cid,
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "cell_id": cid,
                        "color_value": color_list[i],
                        "pred_prob": pred_prob_list[i],
                        "phenotype_call": phenotype_calls[i],
                        "confidence_stratum": strata_list[i],
                    },
                }
            )
        return out

    def _core_image_size(self, basename: str) -> tuple[int, int]:
        """Resolve the recorded ``(width, height)`` of a core image."""
        sizes = self.manifest.get("image_sizes", {})
        if isinstance(sizes, dict) and basename in sizes:
            size = sizes[basename]
            if isinstance(size, list | tuple) and len(size) == 2:
                width, height = int(size[0]), int(size[1])
                if width > 0 and height > 0:
                    return (width, height)
        raise ValueError(f"manifest.image_sizes has no valid entry for {basename!r}")

    # ----- internal helpers for cell_detail -----
    def _marker_entries(self, cell_row: pd.Series, *, model_id: str) -> list[MarkerEntry]:
        """Build the 12-marker side-panel rows for one cell.

        ``pred_prob`` is ``None`` when the requested model doesn't carry
        a column for this channel (e.g., MIPHEI doesn't predict the
        three PDL1-context markers). The schema explicitly admits a
        nullable ``pred_prob`` so the client can render an empty bar
        rather than misleadingly showing 0.
        """
        out: list[MarkerEntry] = []
        for ch, name in zip(FOCUSED_MARKERS, MARKER_NAMES, strict=True):
            pred_col = f"ch{ch:02d}_pred_{model_id}"
            truth_col = f"ch{ch:02d}_pos"
            pred = _safe_float(cell_row.get(pred_col)) if pred_col in cell_row.index else None
            truth = cell_row.get(truth_col) if truth_col in cell_row.index else None
            truth_pos: bool | None
            if truth is None or (isinstance(truth, float) and np.isnan(truth)):
                truth_pos = None
            else:
                truth_pos = bool(truth)
            out.append(
                MarkerEntry(
                    channel=int(ch),
                    name=name,
                    pred_prob=(None if pred is None else float(pred)),
                    truth_pos=truth_pos,
                    # Use the same coercion as the GeoJSON path so a raw
                    # "unknown" from MARKER_CONFIDENCE downgrades to
                    # "weak" instead of bypassing the
                    # Literal["strong","moderate","weak"] enum.
                    confidence_stratum=_coerce_stratum(self._thresholds.marker_confidence[name]),
                    model_id=model_id,
                )
            )
        return out

    def _phenotype_entries(self, cell_row: pd.Series, *, model_id: str) -> list[PhenotypeEntry]:
        """Build the 9-phenotype side-panel rows for one cell."""
        out: list[PhenotypeEntry] = []
        for name in PHENOTYPE_NAMES:
            # The v1.1 bundle stores per-cell phenotype scores under
            # ``phenotype_<name>_score`` (no model suffix) — older
            # exports used a ``_score_v1_1`` suffix. Try both so the
            # service stays forward-compatible with future bundles.
            score_col_versioned = f"phenotype_{name}_score_{model_id}"
            score_col_bare = f"phenotype_{name}_score"
            call_col = f"phenotype_{name}_call_{model_id}"
            label_col = f"phenotype_{name}_label"

            if score_col_versioned in cell_row.index:
                score = _safe_float(cell_row.get(score_col_versioned))
            elif score_col_bare in cell_row.index and model_id == "v1_1":
                score = _safe_float(cell_row.get(score_col_bare))
            else:
                score = None
            call = cell_row.get(call_col, 0) if call_col in cell_row.index else 0
            label = cell_row.get(label_col) if label_col in cell_row.index else None
            label_truth: bool | None
            if label is None or (isinstance(label, float) and np.isnan(label)):
                label_truth = None
            else:
                label_truth = bool(label)

            out.append(
                PhenotypeEntry(
                    name=name,
                    score=(None if score is None else float(score)),
                    call=bool(int(call)),
                    label_truth=label_truth,
                    # Match marker-entry coercion to keep cell-detail strata
                    # consistent with the bulk GeoJSON path.
                    confidence_stratum=_coerce_stratum(self._thresholds.phenotype_confidence[name]),
                    model_id=model_id,
                )
            )
        return out

    def spatial_summary_slice(
        self,
        basename: str,
        *,
        exclude_self_pairs: bool = True,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        self.assert_core(basename)
        ss = self._spatial()
        if ss.empty:
            return []
        sub = ss[ss["basename"] == basename].copy()
        if exclude_self_pairs:
            sub = sub[sub["phenotype_a"] != sub["phenotype_b"]]
        sub = sub.dropna(subset=["z"])
        sub = sub.reindex(sub["z"].abs().sort_values(ascending=False).index)
        if top_n is not None:
            sub = sub.head(int(top_n))
        keep = [
            "phenotype_a",
            "phenotype_b",
            "observed",
            "expected",
            "z",
            "p_two_sided",
            "obs_over_exp",
            "hotspot_overlap_iou",
            "n_a",
            "n_b",
            "n_total",
            "n_edges",
        ]
        keep = [c for c in keep if c in sub.columns]
        return sub[keep].to_dict(orient="records")

    # ----- file paths (used by image routes) -----
    def thumb_path(self, basename: str, fname: str) -> Path:
        self.assert_core(basename)
        # SECURITY: prevent path traversal — only allow simple basenames
        if any(ch in fname for ch in ("/", "\\", "..")):
            raise ValueError(f"invalid thumb filename: {fname!r}")
        p = self.bundle_dir / "thumbs" / basename / fname
        if not p.exists():
            raise FileNotFoundError(f"thumb not found: {p}")
        return p

    def report_path(self, basename: str) -> Path:
        self.assert_core(basename)
        p = self.bundle_dir / "reports" / f"{basename}.pdf"
        if not p.exists():
            raise FileNotFoundError(f"report not built: {p}")
        return p

    # ----- export -----
    def export_zip(self, basename: str) -> bytes:
        """Return a zip blob: cells.csv (this core only), spatial.csv, anndata.h5ad."""
        self.assert_core(basename)
        cells = self.cell_table_slice(basename)
        spatial = self._spatial()
        spatial_sub = (
            spatial[spatial["basename"] == basename] if not spatial.empty else pd.DataFrame()
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            cells_csv = cells.to_csv(index=False).encode()
            z.writestr(f"{basename}/cells.csv", cells_csv)
            if not spatial_sub.empty:
                z.writestr(
                    f"{basename}/spatial_summary.csv", spatial_sub.to_csv(index=False).encode()
                )
            # AnnData export (lazy)
            try:
                h5ad_bytes = _cells_to_h5ad_bytes(cells, basename, self._default_model_id)
                z.writestr(f"{basename}/cells.h5ad", h5ad_bytes)
            except Exception as e:
                logger.warning("anndata export failed for %s: %s", basename, e)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_STRATA: frozenset[str] = frozenset({"strong", "moderate", "weak"})


def _coerce_stratum(name: str) -> str:
    """Coerce a stratum string to one of the spec's three values.

    The :class:`CellsFeatureProperties` Pydantic model enforces the
    Literal["strong","moderate","weak"] enum. The thresholds module
    occasionally returns "unknown" for markers we haven't curated; we
    map that to "weak" here so the response always validates and the
    user is shown the lowest-confidence badge by default.
    """
    return name if name in _VALID_STRATA else "weak"


def _polygon_to_geojson_ring(poly: Any) -> list[list[float]] | None:
    """Convert a parquet polygon cell to a closed GeoJSON ring.

    The bundle builder pipeline stores each polygon as a list-of-pairs
    (parquet's ``list<list<float64>>``). Pandas rehydrates that column
    as an object-dtype ``np.ndarray`` of 2-element float64 ndarrays —
    we hot-path that representation directly: a per-vertex
    ``ndarray.tolist`` is ~5× faster than ``np.stack``/``np.asarray``
    because the inner arrays are uniformly tiny (≤ 32 × 2) and the
    intermediate buffer allocation is the dominant cost.

    Returns ``None`` for a degenerate / empty polygon.
    """
    if poly is None:
        return None
    if hasattr(poly, "__len__") and len(poly) == 0:
        return None
    if isinstance(poly, np.ndarray) and poly.dtype == object:
        # Fast path: each element is a 2-element float64 ndarray.
        try:
            ring: list[list[float]] = [p.tolist() for p in poly]
        except AttributeError:
            return _fallback_ring(poly)
    elif isinstance(poly, np.ndarray):
        # Native 2D numpy array — single ``tolist`` is best.
        ring = poly.astype(np.float64, copy=False).tolist()
    else:
        # Python list-of-lists / list-of-tuples — fall back to the
        # safe per-vertex coercion.
        return _fallback_ring(poly)
    if len(ring) < 3:
        return None
    # Close the ring if the pipeline somehow emitted it open. The
    # invariant is upheld by bundle builder but we re-assert here so the
    # GeoJSON output is always RFC 7946 compliant.
    if ring[0] != ring[-1]:
        ring.append([ring[0][0], ring[0][1]])
    return ring


def _fallback_ring(poly: Any) -> list[list[float]] | None:
    """Slow per-vertex ring conversion. Reached only on malformed input."""
    out: list[list[float]] = []
    for p in poly:
        try:
            out.append([float(p[0]), float(p[1])])
        except (TypeError, ValueError):
            return None
    if len(out) < 3:
        return None
    if out[0] != out[-1]:
        out.append([out[0][0], out[0][1]])
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _cells_to_h5ad_bytes(cells: pd.DataFrame, basename: str, model_id: str) -> bytes:
    """Build a single-core AnnData and serialize to h5ad bytes.

    Schema mirrors ``scripts/benchmark/to_anndata.py``:
        .X      = true marker binary matrix (chXX_pos)
        .layers["pred_<model_id>"] = predicted prob matrix
        .obs    = basename, cell_id, centroid_x/y, phenotype labels + calls
        .obsm["spatial"] = (centroid_x, centroid_y)
        .var    = channel, name, prevalence
    """
    import anndata as ad

    n = len(cells)
    truth = np.zeros((n, len(FOCUSED_MARKERS)), dtype=np.float32)
    pred = np.full((n, len(FOCUSED_MARKERS)), np.nan, dtype=np.float32)
    var_rows = []
    for j, (ch, name) in enumerate(zip(FOCUSED_MARKERS, MARKER_NAMES, strict=False)):
        t_col = f"ch{ch:02d}_pos"
        p_col = f"ch{ch:02d}_pred_{model_id}"
        if t_col in cells.columns:
            truth[:, j] = cells[t_col].to_numpy(dtype=np.float32)
        if p_col in cells.columns:
            pred[:, j] = cells[p_col].to_numpy(dtype=np.float32)
        prev = (
            float(np.nanmean(cells.get(t_col, np.nan))) if t_col in cells.columns else float("nan")
        )
        var_rows.append({"channel": int(ch), "name": name, "prevalence": prev})
    var = pd.DataFrame(var_rows, index=MARKER_NAMES)
    obs_cols: dict[str, Any] = {
        "basename": cells["basename"].astype("category"),
        "cell_id": cells["cell_id"].astype(np.int64),
        "centroid_x": cells["centroid_x"].astype(np.int32),
        "centroid_y": cells["centroid_y"].astype(np.int32),
    }
    for name in PHENOTYPE_NAMES:
        label_col = f"phenotype_{name}_label"
        call_col = f"phenotype_{name}_call_{model_id}"
        score_col = f"phenotype_{name}_score_{model_id}"
        bare_score = f"phenotype_{name}_score"
        if label_col in cells.columns:
            obs_cols[label_col] = cells[label_col].astype(np.int8)
        if call_col in cells.columns:
            obs_cols[call_col] = cells[call_col].astype(np.int8)
        if score_col in cells.columns:
            obs_cols[score_col] = cells[score_col].astype(np.float32)
        elif bare_score in cells.columns and model_id == "v1_1":
            obs_cols[f"phenotype_{name}_score_{model_id}"] = cells[bare_score].astype(np.float32)
    obs = pd.DataFrame(obs_cols)
    obs.index = [f"cell_{i}" for i in range(n)]
    obsm = {
        "spatial": np.stack(
            [
                cells["centroid_x"].to_numpy(dtype=np.int32),
                cells["centroid_y"].to_numpy(dtype=np.int32),
            ],
            axis=1,
        ),
    }
    adata = ad.AnnData(
        X=truth,
        obs=obs,
        var=var,
        layers={f"pred_{model_id}": pred},
        obsm=obsm,
    )
    adata.uns["model_id"] = model_id
    adata.uns["basename"] = basename
    adata.uns["marker_channels"] = list(map(int, FOCUSED_MARKERS))
    adata.uns["marker_names"] = list(MARKER_NAMES)
    adata.uns["phenotype_names"] = list(PHENOTYPE_NAMES)
    # anndata's write_h5ad wants a filesystem path; round-trip via tempfile.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=True) as tf:
        adata.write_h5ad(tf.name, compression="gzip")
        return Path(tf.name).read_bytes()
