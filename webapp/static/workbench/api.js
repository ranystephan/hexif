// Typed fetchers for each /api endpoint backed by API.
//
// Why centralize here: every fetch has the same error shape
// (FastAPI's {"detail": "..."}, with 404/422/500 cases), and every
// caller needs the same Accept-header + AbortController plumbing.
// Returning typed shapes (a plain JS object with the field names that
// match webapp/schemas.py) keeps callers from re-deriving the field
// names from the OpenAPI doc.

/**
 * Resolve relative '/api/...' against the page origin, honoring
 * file:// and dev-server hosts. Tests inject a baseUrl override.
 * @param {string} path
 * @param {string|null} baseUrl
 */
function resolve(path, baseUrl) {
  if (baseUrl) return baseUrl.replace(/\/$/, '') + path;
  return path;
}

async function jsonOrThrow(resp, label) {
  if (!resp.ok) {
    let detail = `${label}: HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body && body.detail) detail += ` — ${body.detail}`;
    } catch (err) {
      // Why: an upstream 5xx without a JSON body is common (uvicorn's
      // default 500 page is HTML); we tolerate the parse error and keep
      // the original HTTP status in the rejection.
      void err;
    }
    throw new Error(detail);
  }
  return resp.json();
}

/**
 * GET /api/cores — returns the full list of cores in the bundle.
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<{cores: Array<Object>}>}
 */
export async function fetchCores(opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(resolve('/api/cores', baseUrl), {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/cores');
}

/**
 * GET /api/cells/{core} — GeoJSON FeatureCollection of polygons.
 * Returns the parsed JSON object (parse cost is dominated by the
 * gzipped wire transfer anyway; streaming-parse would buy <5 ms).
 *
 * Note (API contract): the browser frontend deliberately does not forward
 * a `phenotypes` query param. Phenotype toggling is a client-side
 * filter over `properties.phenotype_call` — no server round trip. The
 * server endpoint still accepts `?phenotypes=` for non-browser API
 * clients (API feature), but the JS frontend doesn't use that
 * path.
 *
 * @param {string} core
 * @param {{mode?: string, color_by?: string, marker?: string|null,
 *          baseUrl?: string,
 *          signal?: AbortSignal}} q
 * @returns {Promise<Object>}
 */
export async function fetchCellsGeoJson(core, q) {
  const params = new URLSearchParams();
  if (q.mode) params.set('mode', q.mode);
  if (q.color_by) params.set('color_by', q.color_by);
  if (q.marker) params.set('marker', q.marker);
  // model selection: forward the active model id so the server picks the
  // right ch??_pred_<id> columns. The server defaults to manifest's
  // default model when this is absent, so we only attach it when the
  // client wants a non-default — but we always send it once
  // state.model is bootstrapped because the server's default and the
  // client's default agree by construction.
  if (q.model) params.set('model', q.model);
  const qs = params.toString();
  const url = resolve(
    `/api/cells/${encodeURIComponent(core)}${qs ? '?' + qs : ''}`,
    q.baseUrl,
  );
  const resp = await fetch(url, {
    headers: { Accept: 'application/geo+json, application/json' },
    signal: q.signal,
  });
  return jsonOrThrow(resp, `GET /api/cells/${core}`);
}

/**
 * GET /api/cell/{core}/{cell_id} — per-cell detail payload.
 *
 * @param {string} core
 * @param {number} cellId
 * @param {{model?: string|null, baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchCellDetail(core, cellId, opts) {
  const { model = null, baseUrl = null, signal } = opts || {};
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  const qs = params.toString();
  const url = resolve(
    `/api/cell/${encodeURIComponent(core)}/${cellId}${qs ? '?' + qs : ''}`,
    baseUrl,
  );
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, `GET /api/cell/${core}/${cellId}`);
}

/**
 * GET /api/models — list of available models (model selection).
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<{models: Array<Object>, default_id: string}>}
 */
export async function fetchModels(opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(resolve('/api/models', baseUrl), {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/models');
}

/**
 * GET /api/model/{id} — full model card (model selection).
 * @param {string} id
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchModelCard(id, opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(
    resolve(`/api/model/${encodeURIComponent(id)}`, baseUrl),
    { headers: { Accept: 'application/json' }, signal },
  );
  return jsonOrThrow(resp, `GET /api/model/${id}`);
}

/**
 * GET /api/palette/{name} — palette entries by name.
 * @param {string} name
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchPalette(name, opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(
    resolve(`/api/palette/${encodeURIComponent(name)}`, baseUrl),
    { headers: { Accept: 'application/json' }, signal },
  );
  return jsonOrThrow(resp, `GET /api/palette/${name}`);
}

/**
 * GET /api/phenotypes/{core} — used for the comparison-mode probe.
 * @param {string} core
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchCorePhenotypes(core, opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(
    resolve(`/api/phenotypes/${encodeURIComponent(core)}`, baseUrl),
    { headers: { Accept: 'application/json' }, signal },
  );
  return jsonOrThrow(resp, `GET /api/phenotypes/${core}`);
}

/**
 * Build the URL that OpenSeadragon's tileSource fetches. Letting OSD
 * pull the DZI directly (rather than parsing it here) keeps the OSD
 * tile loader's request scheduler intact.
 *
 * @param {string} core
 * @param {string|null} [baseUrl]
 * @returns {string}
 */
export function heDziUrl(core, baseUrl) {
  return resolve(`/api/he/${encodeURIComponent(core)}.dzi`, baseUrl || null);
}

/**
 * CODEX composite — DZI URL for the CODEX composite tile pyramid.
 * Returns the same shape as :func:`heDziUrl` so OpenSeadragon's
 * tileSource swap is a one-liner: ``viewer.open(codexDziUrl(core))``.
 *
 * @param {string} core
 * @param {string|null} [baseUrl]
 * @returns {string}
 */
export function codexDziUrl(core, baseUrl) {
  return resolve(`/api/codex/${encodeURIComponent(core)}.dzi`, baseUrl || null);
}

/**
 * DZI URL for one focused CODEX channel's raw truth (e.g. Ki67). Backed
 * by the per-marker truth PNG rendered at bundle build time
 * (``thumbs/<core>/ch{NN}_codex.png``). The frontend uses this when
 * the CODEX base layer should track the selected marker — instead of
 * the fixed DAPI/CD45/panCK composite, the viewer paints raw Ki67
 * staining underneath the cell overlay.
 *
 * @param {string} core
 * @param {string} marker  Focused-12 marker name (e.g. ``"Ki67"``).
 * @param {string|null} [baseUrl]
 * @returns {string}
 */
export function codexMarkerDziUrl(core, marker, baseUrl) {
  return resolve(
    `/api/codex/${encodeURIComponent(core)}/marker/${encodeURIComponent(marker)}.dzi`,
    baseUrl || null,
  );
}

/**
 * GET /api/codex/{core}/info — gates the CODEX toggle in the controls.
 * Returns ``{has_composite, channels, image_size, core}`` per
 * :class:`webapp.schemas.CodexInfoResponse`. A 404 means the core has
 * no paired CODEX — we surface that as ``{has_composite: false}`` so
 * callers don't have to switch on HTTP errors.
 *
 * @param {string} core
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchCodexInfo(core, opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(
    resolve(`/api/codex/${encodeURIComponent(core)}/info`, baseUrl),
    { headers: { Accept: 'application/json' }, signal },
  );
  if (resp.status === 404) {
    // Treat the missing-composite case as a structured response so the
    // controls layer's `disabled` decision is a single property read.
    return { core, has_composite: false, channels: [], image_size: [0, 0] };
  }
  return jsonOrThrow(resp, `GET /api/codex/${core}/info`);
}

/**
 * GET /api/strata — confidence-stratum lookup (confidence display).
 *
 * Returns ``{strong, moderate, weak}`` where each bucket carries the
 * canonical marker + phenotype names that fall into the stratum. The
 * legend renders one chip per bucket and uses the lists for the chip's
 * ``title=`` hover-text. The payload is small (< 1 kB) and the route
 * is static, so we cache the promise across calls for the page.
 *
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<{strong: {markers: string[], phenotypes: string[]},
 *                    moderate: {markers: string[], phenotypes: string[]},
 *                    weak: {markers: string[], phenotypes: string[]}}>}
 */
export async function fetchStrata(opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(resolve('/api/strata', baseUrl), {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/strata');
}

/**
 * URL of the per-core PDF report. Linked from the side panel footer.
 * @param {string} core
 * @returns {string}
 */
export function reportUrl(core) {
  return `/api/report/${encodeURIComponent(core)}`;
}

/**
 * statistics — GET /api/stats/per-core/{core}.
 * Phenotype distribution + marker positivity + (optional) truth
 * comparison for one core. The stats tab calls this on tab focus and
 * caches the result client-side per (core, model).
 *
 * @param {string} core
 * @param {{model?: string|null, baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchStatsPerCore(core, opts) {
  const { model = null, baseUrl = null, signal } = opts || {};
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  const qs = params.toString();
  const url = resolve(
    `/api/stats/per-core/${encodeURIComponent(core)}${qs ? '?' + qs : ''}`,
    baseUrl,
  );
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, `GET /api/stats/per-core/${core}`);
}

/**
 * statistics — GET /api/stats/cohort.
 * Filtered cohort aggregate + per-core histogram for one phenotype.
 *
 * @param {{tissue?: string|null, tma?: string|null, split?: string|null,
 *          model?: string|null, histogram_phenotype?: string|null,
 *          baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchStatsCohort(opts) {
  const {
    tissue = null,
    tma = null,
    split = null,
    model = null,
    histogram_phenotype = null,
    baseUrl = null,
    signal,
  } = opts || {};
  const params = new URLSearchParams();
  if (tissue) params.set('tissue', tissue);
  if (tma) params.set('tma', tma);
  if (split) params.set('split', split);
  if (model) params.set('model', model);
  if (histogram_phenotype) params.set('histogram_phenotype', histogram_phenotype);
  const qs = params.toString();
  const url = resolve(
    `/api/stats/cohort${qs ? '?' + qs : ''}`,
    baseUrl,
  );
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/stats/cohort');
}

/**
 * statistics — GET /api/stats/validation.
 * Per-marker / per-phenotype AP table for one model — sourced from
 * ``manifest.models[<id>]``.
 *
 * @param {string|null} model
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<Object>}
 */
export async function fetchStatsValidation(model, opts) {
  const { baseUrl = null, signal } = opts || {};
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  const qs = params.toString();
  const url = resolve(
    `/api/stats/validation${qs ? '?' + qs : ''}`,
    baseUrl,
  );
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/stats/validation');
}

/**
 * help system — GET /api/help — list of available help pages.
 *
 * Returns ``{pages: [{id, title}, ...]}`` ordered by the canonical
 * tab-strip sequence (overview → controls → faq). Failures surface
 * with the same ``{detail}`` shape as every other route; the help
 * panel collapses to a one-line error rather than blocking boot.
 *
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<{pages: Array<{id: string, title: string}>}>}
 */
export async function fetchHelpPages(opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(resolve('/api/help', baseUrl), {
    headers: { Accept: 'application/json' },
    signal,
  });
  return jsonOrThrow(resp, 'GET /api/help');
}

/**
 * help system — GET /api/help/{page_id} — rendered HTML body.
 *
 * Returns the raw HTML body (no ``<html>`` wrapper). The caller injects
 * it into a scrollable container; markdown-it's renderer escapes the
 * markdown content so this is safe to insert via ``innerHTML``.
 *
 * @param {string} pageId
 * @param {{baseUrl?: string, signal?: AbortSignal}} [opts]
 * @returns {Promise<string>}
 */
export async function fetchHelpPageHtml(pageId, opts) {
  const { baseUrl = null, signal } = opts || {};
  const resp = await fetch(
    resolve(`/api/help/${encodeURIComponent(pageId)}`, baseUrl),
    { headers: { Accept: 'text/html' }, signal },
  );
  if (!resp.ok) {
    let detail = `GET /api/help/${pageId}: HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body && body.detail) {
        detail += ` — ${JSON.stringify(body.detail)}`;
      }
    } catch (err) {
      // Why: an upstream 5xx without a JSON body is rare but possible
      // on the help route; we tolerate the parse failure and keep the
      // original HTTP status in the rejection message.
      void err;
    }
    throw new Error(detail);
  }
  return resp.text();
}
