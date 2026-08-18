"""Pydantic response schemas for the Workbench API.

Every cell-level route in :mod:`webapp.workbench_app` pins its payload
shape with one of the models in this module. Pydantic v2 buys us three
things at once:

1. The OpenAPI spec is generated automatically — the frontend can
   regenerate typed fetchers without ever reading the Python source.
2. The response is validated *outbound*, so a regression in
   :mod:`webapp.cell_service` that silently changes a column name fails
   loud in CI instead of corrupting the UI.
3. Tests round-trip the JSON through these models, which means a 200
   that violates the contract still trips the test gate even if the
   route returned 200.

Centroids remain plain ``dict[str, float]`` values because they are small
GeoJSON-compatible leaves, while each top-level response is a Pydantic model.

This module is the canonical API contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hexif.cell_phenotype import MARKER_NAMES

# Frozen tuple of focused-12 marker names; used by the validator on
# :class:`CellsQuery` to reject typos before they reach the service
# layer. Kept as a module-level constant so the validator does not
# rebuild the set on every request.
_VALID_MARKER_NAMES: frozenset[str] = frozenset(MARKER_NAMES)


# ---------------------------------------------------------------------------
# /api/cells/{core}
# ---------------------------------------------------------------------------


class CellsQuery(BaseModel):
    """Query-parameter model for ``GET /api/cells/{core}``.

    The query carries the rendering intent (which polygon variant to
    emit, what scalar to attach as ``color_value``) — the GeoJSON
    response is otherwise unparameterised. Validation lives here, not in
    the service layer, so a malformed request is a 422 *before* any
    parquet is touched.

    The optional ``model`` axis selects a bundle model. When unset, the
    service layer falls back to ``manifest.models[0].id`` (typically
    ``v1_1``). Unknown ids are rejected by the route handler with a 422
    so the schema layer remains model-agnostic; only the bundle knows
    which model ids are valid for any given snapshot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["nucleus", "expanded"] = "expanded"
    color_by: Literal["phenotype", "marker_pred", "marker_truth", "marker_error"] = "phenotype"
    marker: str | None = None
    phenotypes: str | None = Field(
        default=None,
        description="Comma-separated phenotype names; intersect-filter on the call column.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model id (e.g., 'v1_1', 'v1', 'miphei_vit', 'miphei_b2'). "
            "Defaults to the bundle's default model when unset."
        ),
    )
    truth_label_set: Literal["gmm", "consensus", "spacec"] = Field(
        default="gmm",
        description=(
            "Which positivity column family backs the ``marker_truth`` and "
            "``marker_error`` views: 'gmm' (chXX_pos — v1.0 cohort threshold, "
            "current default for backwards compat), 'consensus' (chXX_pos_consensus "
            "— v1.1 2-of-3 cohort vote), or 'spacec' (chXX_pos_spacec — v2 "
            "cluster-derived labels; see docs/reproducibility.md). "
            "Ignored for color_by='phenotype' or 'marker_pred' (those don't read "
            "the truth column)."
        ),
    )

    @field_validator("marker")
    @classmethod
    def _marker_must_be_known(cls, v: str | None) -> str | None:
        # Why: rejecting unknown markers at the schema layer keeps the
        # 422 path tight; the service code can assume v is either None
        # or one of MARKER_NAMES. We accept None here because the
        # marker-required-when rule is a cross-field check handled
        # below.
        if v is None:
            return v
        if v not in _VALID_MARKER_NAMES:
            raise ValueError(f"unknown marker {v!r}; expected one of {sorted(_VALID_MARKER_NAMES)}")
        return v

    @model_validator(mode="after")
    def _marker_required_iff_not_phenotype(self) -> CellsQuery:
        # Marker is required for the three
        # non-phenotype color modes and meaningless for phenotype mode.
        # Enforcing it here means downstream code never sees an
        # inconsistent (color_by, marker) tuple.
        needs_marker = self.color_by != "phenotype"
        if needs_marker and self.marker is None:
            raise ValueError(f"marker is required when color_by={self.color_by!r}")
        return self


class CellsResponseMetadata(BaseModel):
    """Top-level metadata for a ``CellsResponse``.

    Echoes back the resolved query plus the dimensions the frontend
    needs to seed the OpenSeadragon viewport. ``n_features`` is the
    post-filter count, not the bundle's total cell count for the core.

    ``model`` carries the resolved model id back
    to the client so its in-memory state stays in sync with what the
    server actually rendered.
    """

    model_config = ConfigDict(extra="forbid")

    core: str
    n_features: int
    image_size: tuple[int, int]
    color_by: str
    marker: str | None = None
    model: str | None = None
    truth_label_set: str | None = None


class CellsFeatureProperties(BaseModel):
    """Per-feature properties block of a GeoJSON ``Feature``.

    ``color_value`` is the scalar the frontend maps through the active
    palette. Kept as ``float`` for all four color modes — the
    phenotype-call case packs the integer phenotype index into a float
    so the wire format is uniform.

    ``pred_prob`` is the predicted probability for the active marker,
    independent of color_by. It lets the frontend's "Min probability"
    slider filter cells consistently across Pred / Truth / Error views
    (the same set of cells stays visible when you toggle views). None
    in phenotype mode or when the model doesn't predict this marker.
    """

    model_config = ConfigDict(extra="forbid")

    cell_id: int
    color_value: float
    pred_prob: float | None = None
    phenotype_call: str | None = None
    confidence_stratum: Literal["strong", "moderate", "weak"]


class CellGeometry(BaseModel):
    """GeoJSON Polygon geometry — single outer ring, no holes.

    The polygon pipeline elides holes deliberately; the
    geometry field is therefore guaranteed-shape ``Polygon``, never
    ``MultiPolygon``. The coordinate ring is closed (last == first) by
    the pipeline; we re-assert that invariant at response time.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]


class CellFeature(BaseModel):
    """One GeoJSON Feature in the FeatureCollection.

    The ``id`` field at the Feature level mirrors ``properties.cell_id``
    so OpenSeadragon's hit-test plugins (which key by ``feature.id``)
    work without dipping into properties. Both fields are retained as part
    of the API contract.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["Feature"] = "Feature"
    id: int
    geometry: CellGeometry
    properties: CellsFeatureProperties


class CellsResponse(BaseModel):
    """GeoJSON FeatureCollection augmented with a metadata block.

    Augmenting with ``metadata`` keeps the response a valid GeoJSON
    payload — RFC 7946 §6 explicitly permits foreign members — while
    letting the frontend skip a second round-trip to learn the image
    size and the resolved color mode.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"] = "FeatureCollection"
    metadata: CellsResponseMetadata
    features: list[CellFeature]


# ---------------------------------------------------------------------------
# /api/cell/{core}/{cell_id}
# ---------------------------------------------------------------------------


class MarkerEntry(BaseModel):
    """One row of the 12-marker side panel.

    ``truth_pos`` is optional: cores without paired CODEX (or cores
    where the upstream call pipeline did not emit ``chXX_pos``) report
    the prediction without a truth comparison.

    ``pred_prob`` is optional too: a non-default model (e.g., MIPHEI's
    9-marker panel) may not predict every channel. The client renders a
    "—" cell in that case rather than rendering a 0 bar.

    ``confidence_stratum`` matches the same three-value enum used on
    the GeoJSON path — the cell-detail emitter must route raw labels
    (e.g., ``"unknown"`` from a not-yet-curated marker) through
    :func:`webapp.cell_service._coerce_stratum` before constructing
    this model so the cell-detail and bulk-polygons surfaces stay
    consistent.

    ``model_id`` echoes back the model that produced ``pred_prob`` so
    the client can confirm what it is looking at.
    """

    model_config = ConfigDict(extra="forbid")

    channel: int
    name: str
    pred_prob: float | None
    truth_pos: bool | None = None
    confidence_stratum: Literal["strong", "moderate", "weak"]
    model_id: str | None = None


class PhenotypeEntry(BaseModel):
    """One row of the 9-phenotype side panel.

    ``score`` is the sigmoid output, ``call`` is the thresholded
    decision, ``label_truth`` is the consensus CODEX-derived label
    where available.

    See :class:`MarkerEntry` for the rationale behind the tightened
    ``confidence_stratum`` enum and the ``model_id`` echo field.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float | None
    call: bool
    label_truth: bool | None = None
    confidence_stratum: Literal["strong", "moderate", "weak"]
    model_id: str | None = None


class CellFeatures(BaseModel):
    """Morphology features shown in the side panel.

    Eccentricity is computed on the nucleus mask (the canonical shape);
    expansion is an isotropic dilation so its eccentricity is degenerate.
    """

    model_config = ConfigDict(extra="forbid")

    area_nucleus_px: float
    area_expanded_px: float
    eccentricity: float


class CellDetailResponse(BaseModel):
    """Per-cell detail payload for ``GET /api/cell/{core}/{cell_id}``.

    ``centroid`` is intentionally a plain ``dict[str, float]`` rather
    than a named model — see the module docstring for the rationale.
    """

    model_config = ConfigDict(extra="forbid")

    core: str
    cell_id: int
    centroid: dict[str, float]
    features: CellFeatures
    markers: list[MarkerEntry]
    phenotypes: list[PhenotypeEntry]


# ---------------------------------------------------------------------------
# /api/models  +  /api/model/{id}  (model selection — multi-model API)
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """Summary row for one model in the bundle's model zoo.

    The list response carries one of these per model; ``is_default`` is
    true for exactly one entry (``manifest.models[0]``). The full per-
    marker / per-phenotype AP tables live in :class:`ModelCardResponse`
    so the listing stays light.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    backbone: str
    training_split: str
    macro_marker_ap: float
    macro_phenotype_ap: float
    is_default: bool


class ModelListResponse(BaseModel):
    """Payload of ``GET /api/models``.

    ``default_id`` is the bundle's default model id (the first entry in
    ``manifest.models``). Echoing it out separately spares the client
    from scanning the list for ``is_default``.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[ModelEntry]
    default_id: str


class ModelCardResponse(BaseModel):
    """Full per-model card payload for ``GET /api/model/{id}``.

    Carries the macro metrics (echoed for symmetry with the listing
    endpoint) plus the per-marker and per-phenotype AP tables that the
    "details" popover renders on the frontend.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    backbone: str
    training_split: str
    macro_marker_ap: float
    macro_phenotype_ap: float
    per_marker_ap: dict[str, float]
    per_phenotype_ap: dict[str, float]
    notes: str = ""


# ---------------------------------------------------------------------------
# /api/palette/{name}
# ---------------------------------------------------------------------------


class PaletteEntry(BaseModel):
    """One row of a categorical palette (e.g., phenotype name → hex)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    hex: str


class PaletteStop(BaseModel):
    """One stop of a sequential / diverging palette (t ∈ [0, 1] → hex)."""

    model_config = ConfigDict(extra="forbid")

    t: float
    hex: str


class PaletteResponse(BaseModel):
    """The palettes.json schema.

    Two palette shapes are supported: categorical (``entries``) and
    sequential / diverging (``stops``). Either one may be present per
    palette; the frontend dispatches on ``type``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["categorical", "sequential", "diverging"]
    entries: list[PaletteEntry] | None = None
    stops: list[PaletteStop] | None = None


# ---------------------------------------------------------------------------
# /api/codex/{core}/info   (CODEX composite)
# ---------------------------------------------------------------------------


class CodexChannelInfo(BaseModel):
    """One CODEX channel of the composite (DAPI / CD45 / panCK).

    ``idx`` indexes into the 53-channel CODEX panel; ``rgb`` is the
    fixed composite mapping. ``p_low`` /
    ``p_high`` are the **nonzero** percentile values used to rescale
    the channel to uint8 — pinned at bundle build, served verbatim
    here so the legend popover and any export script see the same
    intensity calibration.
    """

    model_config = ConfigDict(extra="forbid")

    idx: int
    name: str
    rgb: Literal["red", "green", "blue"]
    p_low: float
    p_high: float


class CodexMarkerInfo(BaseModel):
    """One CODEX channel that has a per-marker truth thumbnail on disk.

    Surfaced from ``GET /api/codex/{core}/info`` so the frontend knows
    which markers can be shown as a CODEX base layer (versus the fixed
    3-channel composite). ``channel_idx`` is the index into the 53-channel
    CODEX panel; ``name`` is the focused-12 marker name (e.g. ``"Ki67"``);
    ``dzi_url`` is the relative URL of the per-marker DZI pyramid that
    OpenSeadragon should fetch.
    """

    model_config = ConfigDict(extra="forbid")

    channel_idx: int
    name: str
    dzi_url: str


class CodexInfoResponse(BaseModel):
    """Response shape for ``GET /api/codex/{core}/info``.

    ``has_composite`` is the gating flag the frontend reads — it is
    True iff the composite PNG is on disk **and** the manifest carries
    the matching percentile cache. ``channels`` carries three entries
    in that case, an empty list otherwise. ``image_size`` is the
    composite's pixel dimensions, which equal the H&E dimensions and define
    the image space used by cell polygons.

    ``available_markers`` lists the focused-12 markers that have a
    per-marker CODEX truth PNG rendered for this core. Empty list when
    the bundle did not render them; a non-empty list
    means the frontend can swap the CODEX base layer to a single-channel
    visualization for that marker.
    """

    model_config = ConfigDict(extra="forbid")

    core: str
    has_composite: bool
    channels: list[CodexChannelInfo]
    image_size: tuple[int, int]
    available_markers: list[CodexMarkerInfo] = []


# ---------------------------------------------------------------------------
# /api/strata   (confidence display — confidence stratum lookup)
# ---------------------------------------------------------------------------


class StratumBucket(BaseModel):
    """One confidence stratum (strong / moderate / weak).

    Carries the canonical marker + phenotype names that fall into the
    bucket per :mod:`hexif.pipeline.thresholds`. The frontend reads this
    payload once at boot and renders one chip per bucket; the chip's
    hover-text lists the members so the user can match the side-panel
    badge against the underlying calibration without leaving the page.
    """

    model_config = ConfigDict(extra="forbid")

    markers: list[str]
    phenotypes: list[str]


class StrataResponse(BaseModel):
    """Response shape for ``GET /api/strata``.

    Mirrors :data:`hexif.pipeline.thresholds.MARKER_CONFIDENCE` and
    :data:`PHENOTYPE_CONFIDENCE` — the validated bundle's three v1.1
    buckets (strong / moderate / weak). The route is a thin echo: every
    marker / phenotype in the focused panel lands in exactly one bucket
    and the totals match :data:`MARKER_NAMES` / :data:`PHENOTYPE_NAMES`.
    """

    model_config = ConfigDict(extra="forbid")

    strong: StratumBucket
    moderate: StratumBucket
    weak: StratumBucket


# ---------------------------------------------------------------------------
# /api/help   (help system — in-site documentation)
# ---------------------------------------------------------------------------


class HelpPageInfo(BaseModel):
    """One entry in the help-panel tab strip.

    ``id`` is the regex-bounded slug used in the route path
    (``GET /api/help/{page_id}``). ``title`` comes from the first ``#``
    (or fallback first ``##``) heading of the markdown source so docs
    edits propagate without code changes — see
    :func:`webapp.help._read_title`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str


class HelpPageList(BaseModel):
    """Response shape for ``GET /api/help``.

    Carries the ordered list of pages so the frontend renders the tab
    strip in a deterministic sequence (overview → controls → faq) rather
    than the filesystem-glob order. The list is empty only when the
    repo ships with no markdown sources at all — which would itself be
    a packaging regression.
    """

    model_config = ConfigDict(extra="forbid")

    pages: list[HelpPageInfo]


# ---------------------------------------------------------------------------
# /api/stats/*   (statistics — stats tab)
# ---------------------------------------------------------------------------


class PhenotypeDistRow(BaseModel):
    """One row of the per-core phenotype distribution table.

    ``count`` is the number of cells whose ``phenotype_<name>_call_<model>``
    column is positive; ``fraction = count / n_cells`` is precomputed so
    the frontend table renderer stays a pure dumb-write.

    The 9 phenotype labels are **hierarchical, not mutually exclusive**
    (e.g., a CD8 T-cell row also counts toward CD45 immune, and an
    M2-like macrophage also counts toward macrophage). The 10th row is
    the derived ``no_call`` bucket (cells with every call=0); it is
    the complement of "any positive call". The only partition guarantee
    is::

        no_call.count + (cells with >= 1 positive call) == n_cells

    Summing all 10 row counts will OVERCOUNT by the overlap mass. For
    example on ``ccRCC_TMA1__A-10`` the 10 rows sum to ~3867 against
    ``n_cells = 3622`` (+245 from biologically-correct overlap).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    count: int
    fraction: float


class MarkerPositivityRow(BaseModel):
    """One row of the per-core marker-positivity table.

    Computed at request time from ``chXX_pred_<model>`` thresholded
    against the v1.1 calibrated cut (see
    :data:`hexif.pipeline.thresholds.load_v1_1_thresholds`). The
    ``confidence_stratum`` mirrors the model-agnostic v1.1 stratum from
    :data:`hexif.pipeline.thresholds.MARKER_CONFIDENCE` — confidence
    buckets are a property of the marker biology, not the predictor.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    n_positive: int
    fraction_positive: float
    confidence_stratum: Literal["strong", "moderate", "weak"]


class TruthComparisonRow(BaseModel):
    """One row of the paired-CODEX comparison table for a single core.

    Only emitted when the core has paired CODEX truth (``chXX_pos``
    columns). ``ap`` is the within-core average-precision — Pearson and
    Spearman aren't useful at a single-core grain. ``ap`` may be ``None``
    when the truth column is degenerate (all-zero or all-one); the
    frontend renders ``None`` as an em dash rather than zero.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    predicted_fraction: float
    truth_fraction: float
    absolute_error: float
    ap: float | None = None


class PerCoreStatsResponse(BaseModel):
    """Response shape for ``GET /api/stats/per-core/{core}``.

    ``phenotypes`` always has 10 entries (9 phenotypes + ``no_call``);
    the 9 phenotype labels are hierarchical and OVERLAP — summing the
    row counts will overcount n_cells by the overlap mass. See
    :class:`PhenotypeDistRow` for the partition guarantee.

    ``markers`` always has 12 entries. ``comparison`` is ``None`` when
    the core has no paired CODEX; otherwise 12 entries (one per
    focused-12 marker). All numeric values are float-typed so the
    frontend can render uniformly; integer ``count`` / ``n_positive``
    fields stay int-typed because the table cell is "N cells", not "N
    fraction" — Pydantic surfaces the type difference in the OpenAPI
    spec which the frontend uses for table-row rendering.
    """

    model_config = ConfigDict(extra="forbid")

    core: str
    model: str
    n_cells: int
    phenotypes: list[PhenotypeDistRow]
    markers: list[MarkerPositivityRow]
    has_truth: bool
    comparison: list[TruthComparisonRow] | None = None


class CohortFilter(BaseModel):
    """Filter spec for ``GET /api/stats/cohort``.

    ``tma`` accepts either the bare suffix (``"TMA1"``) or the full
    bundle-internal string (``"ccRCC_TMA1"``); the service matches the
    suffix so the frontend can pass the human-facing value. ``model``
    falls back to the bundle default when ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    tissue: str | None = None
    tma: str | None = None
    split: str | None = None
    model: str | None = None


class CohortStatsResponse(BaseModel):
    """Response shape for ``GET /api/stats/cohort``.

    The histogram describes the distribution of one chosen phenotype's
    per-core fraction across the filtered cohort. ``histogram_bins`` is
    a fixed 10-bin grid on ``[0, 1]`` (11 edges). The frontend reads
    the counts directly — no rebinning needed. Aggregate marker /
    phenotype counts are pooled across all cells in the filtered
    cohort, so ``sum(phenotype.count) >= n_cells_total`` is *not*
    guaranteed (a cell can be positive for multiple phenotypes) — but
    we still emit ``no_call`` so the table tells the full story.
    """

    model_config = ConfigDict(extra="forbid")

    filter: CohortFilter
    n_cores: int
    n_cells_total: int
    phenotype_summary: list[PhenotypeDistRow]
    marker_summary: list[MarkerPositivityRow]
    histogram_phenotype: str
    histogram_bins: list[float]
    histogram_counts: list[int]


class ValidationRow(BaseModel):
    """One row of the per-marker / per-phenotype validation table.

    ``macro_ap`` is read verbatim from
    ``manifest.models[<id>].per_marker_ap`` (or ``per_phenotype_ap``);
    this is the source of truth for the displayed AP. ``per_core_ap`` is
    ``None`` until a per-core CSV ships with the bundle; the frontend
    skips the variability stats in that case rather than computing
    fabricated spread numbers.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    macro_ap: float
    n_cores_with_truth: int
    confidence_stratum: Literal["strong", "moderate", "weak"]
    per_core_ap: list[float] | None = None


class ValidationStatsResponse(BaseModel):
    """Response shape for ``GET /api/stats/validation``.

    The 12 marker rows + 9 phenotype rows are emitted in the canonical
    panel order (:data:`hexif.cell_phenotype.MARKER_NAMES` /
    ``PHENOTYPE_NAMES``) — *not* sorted by AP — so the frontend can
    diff the table against the spec without re-sorting. Every numeric
    AP value is read from ``manifest.models[<id>]``.

    ``macro_marker_ap`` / ``macro_phenotype_ap`` are the bundle-recorded
    macros (a verbatim echo of ``manifest.models[<id>]``) so the
    frontend can render the highlighted "headline" AP at the top of
    the table without re-deriving it from the per-row values — those
    differ slightly because the bundle's macro is taken over the
    cores-with-truth subset, while a naive mean of the per-row values
    would re-weight unmapped channels as zero.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    macro_marker_ap: float
    macro_phenotype_ap: float
    markers: list[ValidationRow]
    phenotypes: list[ValidationRow]
