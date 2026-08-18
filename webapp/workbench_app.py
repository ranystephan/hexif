"""HEXIF Spatial Cytometry Workbench — FastAPI app.

Read-only browser over a Workbench bundle built by
``hexif build-cell-tables`` + ``hexif spatial-summary`` +
``hexif report``.  Three modes inside one page (Atlas Explorer,
Virtual Cytometry Report, Spatial Architecture).

Run locally with an authorized bundle::

    hexif serve-workbench --bundle /path/to/validated_bundle --port 7860

or directly::

    python -m webapp.workbench_app --bundle /path/to/validated_bundle \\n        --host 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from hexif.cell_phenotype import FOCUSED_MARKERS, MARKER_NAMES, PHENOTYPE_NAMES

from . import dzi as dzi_mod
from . import help as help_mod
from .cell_service import CellPhenotypeService, CoreFilter
from .schemas import (
    CellDetailResponse,
    CellsQuery,
    CohortFilter,
    CohortStatsResponse,
    HelpPageInfo,
    HelpPageList,
    ModelCardResponse,
    ModelListResponse,
    PerCoreStatsResponse,
    StrataResponse,
    StratumBucket,
    ValidationStatsResponse,
)

logger = logging.getLogger("hexif.webapp.workbench")

STATIC_DIR = Path(__file__).parent / "static"
WORKBENCH_DIR = STATIC_DIR / "workbench"
PALETTE_DIR = WORKBENCH_DIR


def create_workbench_app(service: CellPhenotypeService) -> FastAPI:
    app = FastAPI(
        title="HEXIF Spatial Cytometry Workbench",
        description=(
            "Read-only browser for cell-level marker probabilities, "
            "phenotype calls, and spatial summaries from an explicitly "
            "supplied validated bundle. Research use only."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    # GZip every response over ~1 KB. The /api/cells/{core} payload is
    # the hot path: a 2k-cell core compresses ~5× over JSON which is
    # the difference between hitting the 250 ms p95 budget and missing
    # it. Spec §3.2.1 mandates this middleware.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ----------------------------- frontend -----------------------------
    # Single canonical workbench. The older release template and the legacy
    # version-suffixed aliases (`/v2`, `/v2/`, and the literal
    # `/workbench_` plus version stub) were removed in Workbench current release
    # the API contract routing; everything collapses onto `/` now.
    _INDEX = WORKBENCH_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not _INDEX.exists():
            raise HTTPException(
                status_code=500,
                detail=f"workbench index.html missing at {_INDEX}",
            )
        return HTMLResponse(_INDEX.read_text())

    @app.get("/favicon.svg")
    def favicon():
        f = STATIC_DIR / "favicon.svg"
        if f.exists():
            return FileResponse(f)
        raise HTTPException(404)

    if STATIC_DIR.exists():
        # commit 9 — Cache-Control: no-cache forces every browser to send
        # a conditional GET (If-None-Match) instead of serving the file
        # from its disk cache. FastAPI's StaticFiles still returns a 304
        # when the ETag matches, so this is cheap — the browser saves
        # the body and only re-fetches when the asset actually changed.
        # Why we need this: ES module imports (main.js → state.js →
        # canvas_overlay.js → …) are pulled by URL, and aggressive
        # browser caching at any one of those URLs leaves the user with
        # a mismatched bundle where main.js calls a method that the
        # cached canvas_overlay.js does not yet define. Safari is
        # particularly sticky here even on a "hard reload."
        class _NoCacheStatic(StaticFiles):
            def file_response(self, *args, **kwargs):
                response = super().file_response(*args, **kwargs)
                # The browser may still cache the body so subsequent
                # conditional GETs are cheap; ``no-cache`` says "you
                # must revalidate every time" (= send If-None-Match),
                # which is exactly what we want for hot-iterating JS.
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
                return response

        app.mount("/static", _NoCacheStatic(directory=STATIC_DIR), name="static")

    # ----------------------------- meta -----------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "hexif-workbench"}

    @app.get("/api/info")
    def info() -> dict:
        return service.info()

    # ----------------------------- models (model selection) -----------------------------
    @app.get(
        "/api/models",
        response_model=ModelListResponse,
        responses={200: {"description": "List of available models in the bundle"}},
    )
    def list_models() -> ModelListResponse:
        # The model list comes straight from manifest.models — the
        # service reads the manifest once at boot and caches; this
        # route is therefore O(1) regardless of bundle size.
        return service.models()

    @app.get(
        "/api/model/{model_id}",
        response_model=ModelCardResponse,
        responses={
            200: {"description": "Full per-model card payload"},
            404: {"description": "Model id not in manifest.models"},
        },
    )
    def model_card(model_id: str) -> ModelCardResponse:
        try:
            return service.model_card(model_id)
        except KeyError as e:
            # Structured 404 carrying both the bad id and the valid set
            # so the client can render a useful error message without a
            # second round-trip to /api/models.
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "id": model_id,
                    "valid_ids": service.valid_model_ids,
                    "message": str(e),
                },
            ) from e

    # ----------------------------- cores -----------------------------
    @app.get("/api/cores")
    def list_cores(
        tma: str | None = Query(None),
        tissue: str | None = Query(None),
        split: str | None = Query(None),
        min_cells: int | None = Query(None, ge=0),
        max_rigid_d_px: float | None = Query(None, ge=0),
    ) -> dict:
        flt = CoreFilter(
            tma=tma,
            tissue=tissue,
            split=split,
            min_cells=min_cells,
            max_rigid_d_px=max_rigid_d_px,
        )
        return {"cores": service.list_cores(flt)}

    # NOTE: /api/cell-table/{basename} was removed in Workbench contract §3.1
    # (routing). The richer GeoJSON `/api/cells/{core}` route is
    # the canonical bulk-data path; the legacy tabular endpoint had no
    # remaining consumers after older release was retired.

    @app.get("/api/phenotypes/{basename}")
    def phenotypes(basename: str) -> dict:
        try:
            return service.phenotype_summary(basename)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/spatial/{basename}")
    def spatial(
        basename: str,
        top_n: int | None = Query(None, ge=1, le=200),
        include_self: bool = Query(False),
    ) -> dict:
        try:
            rows = service.spatial_summary_slice(
                basename,
                exclude_self_pairs=not include_self,
                top_n=top_n,
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return {"basename": basename, "n": len(rows), "rows": rows}

    # ----------------------------- thumbs -----------------------------
    @app.get("/api/thumb/{basename}/{fname}")
    def thumb(basename: str, fname: str):
        try:
            p = service.thumb_path(basename, fname)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return FileResponse(p, media_type="image/png")

    # ----- earlier release HE DeepZoom routes (declared before the legacy
    # /api/he/{basename} so FastAPI matches them first; the legacy
    # path's {basename} is greedy and would otherwise swallow the
    # ".dzi" / "_files/..." suffixes).
    @app.get("/api/he/{basename}.dzi")
    def he_dzi_manifest(basename: str) -> Response:
        # Spec §3.2.3 — lazy-build the pyramid on first access for
        # bundles built before earlier release (no DZI). Subsequent requests are
        # cached on disk and never hit the build path.
        try:
            dzi_path = dzi_mod.ensure_dzi_pyramid(service.bundle_dir, basename)
            xml = dzi_mod.read_dzi_xml(dzi_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except dzi_mod.DZIError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(
            content=xml,
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/he/{basename}_files/{level}/{tile}")
    def he_dzi_tile(basename: str, level: str, tile: str) -> Response:
        try:
            return dzi_mod.serve_dzi_tile(service.bundle_dir, basename, level, tile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # NOTE: the bare `/api/he/{basename}` PNG endpoint was removed in
    # Workbench contract §3.1 (routing). The DZI tile route
    # `/api/he/{basename}.dzi` + `_files/{level}/{tile}` is the
    # canonical H&E access path now (full-res 2950² tiles).

    # ----- CODEX composite: CODEX composite DZI -----
    # Mirrors the HE tile routes, but the lazy-build resolves the
    # source from the pre-rendered ``thumbs/<basename>/codex_composite.png``
    # (written at bundle-build time). If the PNG is absent the route
    # is a 404 — composite is *not* lazily rendered from the raw npy
    # at request time (that's bundle pipeline work).
    @app.get("/api/codex/{basename}.dzi")
    def codex_dzi_manifest(basename: str) -> Response:
        try:
            dzi_path = dzi_mod.ensure_codex_dzi_pyramid(service.bundle_dir, basename)
            xml = dzi_mod.read_dzi_xml(dzi_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except dzi_mod.DZIError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(
            content=xml,
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/codex/{basename}_files/{level}/{tile}")
    def codex_dzi_tile(basename: str, level: str, tile: str) -> Response:
        try:
            return dzi_mod.serve_codex_dzi_tile(service.bundle_dir, basename, level, tile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # ----- Per-marker CODEX truth DZI -----
    # Visualizes one of the 12 focused CODEX channels (e.g. Ki67) as a
    # standalone base layer. Used by the workbench when the user picks
    # a marker and toggles base=CODEX — instead of showing the fixed
    # DAPI/CD45/panCK composite, the viewer fetches this per-marker
    # pyramid so the visible staining matches the selected marker. The
    # marker path segment is the human-readable name (``Ki67``); we
    # resolve it to a channel index server-side rather than expose
    # ch<NN> in the URL to keep the API contract stable across panel
    # changes.
    _MARKER_TO_CHANNEL: dict[str, int] = dict(zip(MARKER_NAMES, FOCUSED_MARKERS, strict=True))

    def _resolve_marker_channel(marker: str) -> int:
        ch = _MARKER_TO_CHANNEL.get(marker)
        if ch is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown marker {marker!r} (expected one of {list(MARKER_NAMES)})",
            )
        return ch

    @app.get("/api/codex/{basename}/marker/{marker}.dzi")
    def codex_marker_dzi_manifest(basename: str, marker: str) -> Response:
        ch = _resolve_marker_channel(marker)
        try:
            dzi_path = dzi_mod.ensure_per_marker_codex_dzi_pyramid(service.bundle_dir, basename, ch)
            xml = dzi_mod.read_dzi_xml(dzi_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except dzi_mod.DZIError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(
            content=xml,
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/codex/{basename}/marker/{marker}_files/{level}/{tile}")
    def codex_marker_dzi_tile(basename: str, marker: str, level: str, tile: str) -> Response:
        ch = _resolve_marker_channel(marker)
        try:
            return dzi_mod.serve_per_marker_codex_dzi_tile(
                service.bundle_dir, basename, ch, level, tile
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get(
        "/api/codex/{basename}/info",
        responses={
            200: {"description": "CodexInfoResponse"},
            404: {"description": "Core not in bundle or no paired CODEX"},
        },
    )
    def codex_info(basename: str):
        try:
            info = service.codex_info(basename)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"no paired CODEX composite for {basename!r}",
            )
        return info

    # ----- confidence display: confidence stratum lookup -----
    # /api/strata is a static echo of MARKER_CONFIDENCE +
    # PHENOTYPE_CONFIDENCE from src/hexif/pipeline/thresholds.py — the
    # canonical source of which markers / phenotypes the v1.1 model
    # delivers reliably. Sourced from the route (not the service) since
    # the bundle plays no part: thresholds.py ships in the repo and the
    # buckets are model-agnostic in current release.
    @app.get(
        "/api/strata",
        response_model=StrataResponse,
        responses={
            200: {"description": "Confidence-stratum lookup for legend chips"},
        },
    )
    def strata() -> StrataResponse:
        return _strata_response(
            service._thresholds.marker_confidence,
            service._thresholds.phenotype_confidence,
        )

    # ----- help system: in-site help (markdown-rendered) -----
    # The markdown sources ship in the repo (``webapp/help/*.md``) so
    # there's no XSS surface — see the API contract risk row 5. Path-traversal
    # protection lives in :func:`webapp.help.render_page` (regex on the
    # ``page_id`` + explicit ``is_file()`` check) and the route maps the
    # exception shape to HTTP semantics: ``ValueError`` → 422 (bad
    # ``page_id`` syntax, including ``..`` traversal attempts);
    # ``FileNotFoundError`` → 404 (regex-valid id, no such page on disk).
    @app.get(
        "/api/help",
        response_model=HelpPageList,
        responses={
            200: {"description": "Ordered list of available help pages"},
        },
    )
    def help_pages() -> HelpPageList:
        return HelpPageList(
            pages=[HelpPageInfo(id=p["id"], title=p["title"]) for p in help_mod.list_pages()]
        )

    @app.get(
        "/api/help/{page_id}",
        response_class=HTMLResponse,
        responses={
            200: {
                "content": {"text/html": {}},
                "description": "Rendered markdown body (no <html> wrapper)",
            },
            404: {"description": "Page id is valid syntax but missing on disk"},
            422: {"description": "Page id failed the [a-z0-9_-]+ regex"},
        },
    )
    def help_page(page_id: str) -> HTMLResponse:
        # Spec §3.5 contract: ``page_id`` must match ``^[a-z0-9_-]+$``.
        # The regex runs against the *raw* id (no lowercase first), so
        # a mixed-case input like ``NotAPage`` is rejected with 422
        # rather than silently lowercased and 404-ed. Lowercasing before
        # the regex would mean uppercase ids never reach the regex gate
        # — which contradicts the spec's "rejects uppercase" promise.
        try:
            html = help_mod.render_page(page_id)
        except ValueError as e:
            # Bad page_id syntax (including path-traversal attempts like
            # "../etc_passwd") — 422 with structured detail so the client
            # can distinguish "you typo'd" from "doesn't exist".
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_page_id", "id": page_id, "message": str(e)},
            ) from e
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": "page_not_found", "id": page_id, "message": str(e)},
            ) from e
        return HTMLResponse(content=html, media_type="text/html; charset=utf-8")

    # ----------------------------- statistics: stats tab -----------------------------
    # Three read-only stats routes over the parquet bundle + manifest.
    # No new computations at serve time other than per-core thresholding
    # of the pred columns + counting. Per-core entries are cached in
    # the service's _stats_cache LRU so the second tab-open is free.
    @app.get(
        "/api/stats/per-core/{basename}",
        response_model=PerCoreStatsResponse,
        responses={
            200: {"description": "Per-core phenotype + marker + truth-comparison tables"},
            404: {"description": "Core not in bundle"},
            422: {"description": "Unknown model id"},
        },
    )
    def stats_per_core(basename: str, model: str | None = Query(None)) -> PerCoreStatsResponse:
        # Same 422-before-touching-disk pattern as /api/cells: reject an
        # unknown model id with the structured echoes-back-valid-ids
        # payload before we open any parquet.
        if model is not None and model not in service.valid_model_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_model_id",
                    "id": model,
                    "valid_ids": service.valid_model_ids,
                },
            )
        try:
            return service.per_core_stats(basename, model or service.default_model_id)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": "core_not_found", "core": basename, "message": str(e)},
            ) from e

    @app.get(
        "/api/stats/cohort",
        response_model=CohortStatsResponse,
        responses={
            200: {"description": "Cohort-aggregated phenotype + marker + histogram"},
            422: {"description": "Unknown filter value (tissue / split / phenotype / model)"},
        },
    )
    def stats_cohort(
        tissue: str | None = Query(None),
        tma: str | None = Query(None),
        split: str | None = Query(None),
        model: str | None = Query(None),
        histogram_phenotype: str = Query("tumor_ca9_or_panck"),
    ) -> CohortStatsResponse:
        # Reject unknown model before touching parquet, same shape as
        # /api/cells; reject unknown histogram_phenotype here so the
        # service layer's KeyError doesn't need to disambiguate. Tissue
        # and split are validated in the service layer because the
        # valid set is bundle-dependent.
        if model is not None and model not in service.valid_model_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_model_id",
                    "id": model,
                    "valid_ids": service.valid_model_ids,
                },
            )
        if histogram_phenotype not in PHENOTYPE_NAMES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_phenotype",
                    "id": histogram_phenotype,
                    "valid_phenotypes": list(PHENOTYPE_NAMES),
                },
            )
        try:
            filt = CohortFilter(tissue=tissue, tma=tma, split=split, model=model)
            return service.cohort_stats(filt, histogram_phenotype=histogram_phenotype)
        except KeyError as e:
            # Unknown tissue / split / phenotype-name fall through here
            # — the service layer raises KeyError with a structured
            # message that we echo back as 422.
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_filter", "message": str(e)},
            ) from e

    @app.get(
        "/api/stats/validation",
        response_model=ValidationStatsResponse,
        responses={
            200: {"description": "Per-marker / per-phenotype AP table for one model"},
            422: {"description": "Unknown model id"},
        },
    )
    def stats_validation(model: str | None = Query(None)) -> ValidationStatsResponse:
        if model is not None and model not in service.valid_model_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_model_id",
                    "id": model,
                    "valid_ids": service.valid_model_ids,
                },
            )
        try:
            return service.validation_stats(model or service.default_model_id)
        except KeyError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # ----------------------------- report / export -----------------------------
    @app.get("/api/report/{basename}")
    def report(basename: str):
        try:
            p = service.report_path(basename)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(
                404,
                f"PDF not built for {basename!r} — run `hexif report --bundle <bundle> {basename}`",
            ) from e
        return FileResponse(
            p,
            media_type="application/pdf",
            filename=f"hexif_{basename}.pdf",
        )

    @app.get("/api/export/{basename}")
    def export(basename: str):
        try:
            data = service.export_zip(basename)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": (f'attachment; filename="hexif_{basename}.zip"'),
            },
        )

    # ----------------------------- cell-level routes -----------------------------
    @app.get(
        "/api/cells/{basename}",
        response_model=None,
        responses={
            200: {
                "content": {"application/geo+json": {}},
                "description": "GeoJSON FeatureCollection",
            },
            404: {"description": "Core or polygons not in bundle"},
            422: {"description": "Invalid query parameters"},
        },
    )
    def cells_geojson(
        basename: str,
        mode: Literal["nucleus", "expanded"] = Query("expanded"),
        color_by: Literal["phenotype", "marker_pred", "marker_truth", "marker_error"] = Query(
            "phenotype"
        ),
        marker: str | None = Query(None),
        phenotypes: str | None = Query(None),
        model: str | None = Query(None),
        truth_label_set: Literal["gmm", "consensus", "spacec"] = Query(
            "gmm",
            description=(
                "Which positivity column family backs marker_truth / marker_error: "
                "'gmm' (chXX_pos, v1.0 default), 'consensus' (chXX_pos_consensus, "
                "v1.1 training labels), or 'spacec' (chXX_pos_spacec, v2 cluster-"
                "derived labels). Ignored for phenotype / marker_pred views."
            ),
        ),
    ) -> Response:
        # Validate via the Pydantic schema so the 422 path is centralized.
        try:
            query = CellsQuery(
                mode=mode,
                color_by=color_by,
                marker=marker,
                phenotypes=phenotypes,
                model=model,
                truth_label_set=truth_label_set,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        # model selection: reject unknown model ids with a structured 422
        # before we touch any parquet. Echoing the valid id list saves
        # the client a /api/models round-trip on the error path.
        if model is not None and model not in service.valid_model_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_model_id",
                    "id": model,
                    "valid_ids": service.valid_model_ids,
                },
            )
        try:
            features, image_size, marker_echo = service.cells_geojson_dicts(basename, query)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        metadata = {
            "core": basename,
            "n_features": len(features),
            "image_size": list(image_size),
            "color_by": query.color_by,
            "marker": marker_echo,
            "model": service.resolve_model_id(model),
            # Echo truth_label_set ONLY for the views that actually consume
            # it. None on phenotype / marker_pred keeps the client's URL
            # parser from treating an irrelevant axis as live state.
            "truth_label_set": (
                query.truth_label_set
                if query.color_by in ("marker_truth", "marker_error")
                else None
            ),
        }
        return _geojson_streaming_response(metadata, features)

    @app.get(
        "/api/cell/{basename}/{cell_id}",
        response_model=CellDetailResponse,
        responses={
            404: {"description": "Cell or core not in bundle"},
        },
    )
    def cell_detail(
        basename: str,
        cell_id: int,
        model: str | None = Query(None),
    ) -> CellDetailResponse:
        if model is not None and model not in service.valid_model_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_model_id",
                    "id": model,
                    "valid_ids": service.valid_model_ids,
                },
            )
        try:
            return service.cell_detail(basename, cell_id, model_id=model)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/api/palette/{name}")
    def palette(name: str) -> Response:
        # Reject path-traversal in the name before touching disk.
        if not name or any(c in name for c in ("/", "\\", "..")):
            raise HTTPException(status_code=400, detail=f"invalid palette name: {name!r}")
        # The palettes.json file ships all palettes in one document; we
        # serve a specific entry by name. 404 if not present.
        palettes_path = PALETTE_DIR / "palettes.json"
        if not palettes_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"palettes.json missing at {palettes_path}",
            )
        try:
            palettes = json.loads(palettes_path.read_text())
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"palettes.json is not valid JSON: {e}",
            ) from e
        if not isinstance(palettes, dict) or name not in palettes:
            raise HTTPException(status_code=404, detail=f"palette not found: {name!r}")
        return Response(
            content=json.dumps(palettes[name]),
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app


def _strata_response(
    marker_confidence: dict[str, str], phenotype_confidence: dict[str, str]
) -> StrataResponse:
    """Build confidence buckets from the validated bundle artifact."""
    buckets: dict[str, dict[str, list[str]]] = {
        "strong": {"markers": [], "phenotypes": []},
        "moderate": {"markers": [], "phenotypes": []},
        "weak": {"markers": [], "phenotypes": []},
    }
    for name, stratum in marker_confidence.items():
        if stratum in buckets:
            buckets[stratum]["markers"].append(name)
        else:  # pragma: no cover - defensive; thresholds.py invariant
            raise ValueError(f"unknown stratum {stratum!r} for {name}")
    for name, stratum in phenotype_confidence.items():
        if stratum in buckets:
            buckets[stratum]["phenotypes"].append(name)
        else:  # pragma: no cover - defensive; thresholds.py invariant
            raise ValueError(f"unknown stratum {stratum!r} for {name}")
    return StrataResponse(
        strong=StratumBucket(**buckets["strong"]),
        moderate=StratumBucket(**buckets["moderate"]),
        weak=StratumBucket(**buckets["weak"]),
    )


def _geojson_streaming_response(
    metadata: dict[str, object], features: list[dict[str, object]]
) -> StreamingResponse:
    """Stream a GeoJSON FeatureCollection as ``application/geo+json``.

    Hand-rolls the JSON serialization so the response generator's
    memory bound is O(batch) rather than O(N features). Features are
    flushed in 256-row batches: small enough that the generator never
    holds more than ~50 KB in flight, large enough that ASGI's
    per-chunk overhead (each chunk is its own ``Body`` event) does
    not dominate the wall-clock for 2k-cell cores.

    The pydantic ``model_dump_json`` round-trip would defeat the
    streaming intent — we'd materialize the whole string anyway, and
    pay the per-model construction cost twice on the hot path.
    """
    metadata_json = json.dumps(metadata, separators=(",", ":"))
    # 256-feature batches keep per-chunk size around ~30 KB after JSON
    # encode, which is well-matched to ASGI's per-send overhead while
    # still bounding peak memory to a small multiple of the batch
    # payload. Tuned against the 3622-cell ccRCC_TMA1__A-10 benchmark.
    batch_size = 256

    def _iter() -> Iterator[bytes]:
        yield (
            b'{"type":"FeatureCollection","metadata":'
            + metadata_json.encode("utf-8")
            + b',"features":['
        )
        buf: list[str] = []
        for i, feat in enumerate(features):
            if i > 0:
                buf.append(",")
            buf.append(json.dumps(feat, separators=(",", ":")))
            if len(buf) >= batch_size:
                yield "".join(buf).encode("utf-8")
                buf = []
        if buf:
            yield "".join(buf).encode("utf-8")
        yield b"]}"

    return StreamingResponse(_iter(), media_type="application/geo+json")


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI entry
    p = argparse.ArgumentParser(description="HEXIF Spatial Cytometry Workbench")
    p.add_argument(
        "--bundle",
        required=True,
        help="Workbench bundle dir (built by `hexif build-cell-tables`)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--model-id", default="v1_1")
    return p.parse_args()


def main() -> None:  # pragma: no cover - CLI entry (covered by integration smoke)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    service = CellPhenotypeService(args.bundle, model_id=args.model_id)
    app = create_workbench_app(service)
    import uvicorn

    print(f"\n  HEXIF Workbench at http://{args.host}:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
