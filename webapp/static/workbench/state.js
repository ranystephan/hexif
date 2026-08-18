// URL <-> state encoding/decoding for the Workbench frontend.
//
// Why a separate module: the URL is the source of truth for the
// renderable view (per spec section 3.3). Every control change funnels
// through parseUrlToState / stateToUrl / applyState, so a bug in the
// state machine is bug-once-fix-once instead of scattered across the
// overlay, panels, and bootstrap. Keeping the contract in one file also
// makes the round-trip property easy to test from Node (no DOM needed
// for the parser).

const VALID_MODES = new Set(['nucleus', 'expanded']);
const VALID_COLOR_BY = new Set([
  'phenotype',
  'marker_pred',
  'marker_truth',
  'marker_error',
]);
// Truth label set (added 2026-05-15) — which column family the Truth /
// Error views read from. Mirrors webapp/schemas.py CellsQuery.
// truth_label_set. Backwards-compat default is 'gmm' (the column the
// workbench has always read); 'consensus' is v1.1's training labels;
// 'spacec' is v2's cluster-derived labels (fixes label audit bidirectional
// failures — see docs/reproducibility.md).
const VALID_TRUTH_LABEL_SETS = new Set(['gmm', 'consensus', 'spacec']);
export const DEFAULT_TRUTH_LABEL_SET = 'gmm';
// CODEX composite — base-layer values: H&E, CODEX, or side-by-side
// split. URL key `base=he|codex|split` round-trips through state.js
// but is *not* part of fetchKey: switching only changes how the
// viewer(s) paint, never the per-core polygon GeoJSON.
const VALID_BASES = new Set(['he', 'codex', 'split']);
// Focused-12 markers — kept in sync with hexif.cell_phenotype.MARKER_NAMES.
// The frontend validates against this set so a typo'd ?marker= falls
// back to the default rather than firing a doomed /api/cells request.
export const VALID_MARKERS = [
  'DAPI', 'CD45', 'CD8', 'Ki67', 'CA9', 'Vimentin',
  'CD68', 'FAP', 'CD163', 'PDL1', 'aSMA', 'panCK',
];
const VALID_MARKER_SET = new Set(VALID_MARKERS);

// Phenotypes plus a derived 'no_call' chip. Order mirrors the
// palette entries in palettes.json so the chip layout reads
// left-to-right in the same order as the legend.
export const PHENOTYPE_NAMES = [
  'immune_cd45',
  't_cell_cd45_cd8',
  'tumor_ca9_or_panck',
  'macrophage_cd68_or_cd163',
  'm2_like_cd68_cd163',
  'caf_fap_or_asma',
  'proliferating_ki67',
  'pdl1_positive',
  'pdl1_tumor_like',
];
export const NO_CALL_KEY = 'no_call';

// Default marker when color_by != 'phenotype' and the URL omits one.
// Ki67 is the panel's signature marker (proliferation index), spans
// strong-confidence and shows clear positive cells against H&E.
export const DEFAULT_MARKER = 'Ki67';

// model selection: default model id used by the URL parser when
// the bundle's /api/models hasn't been fetched yet. The runtime
// default model is bootstrapped from /api/models at boot — this
// constant is the fallback so deep-link parsing works synchronously
// (Playwright tests, server-side state tests, the first paint before
// the model list resolves). Matches the spec's "manifest.models[0]
// is v1_1 by default."
export const DEFAULT_MODEL_ID = 'v1_1';

/**
 * @typedef {Object} State
 * @property {string|null} core       Core basename. null means "no core selected yet".
 * @property {'nucleus'|'expanded'} mode
 * @property {'phenotype'|'marker_pred'|'marker_truth'|'marker_error'} color_by
 * @property {string|null} marker     Required iff color_by != 'phenotype'.
 * @property {Set<string>|null} phenotypes  Visible phenotype names (incl. 'no_call'). null = all on.
 * @property {number|null} selected   Selected cell_id, or null.
 * @property {boolean} compare        Side-panel comparison toggle (pred vs truth).
 * @property {string} model           Active model id; runtime-bootstrapped
 *                                    from /api/models, defaults to v1_1.
 * @property {'he'|'codex'} base      CODEX composite base-layer toggle.
 */

/**
 * Parse a URL (string or URL) into a fully-defaulted State object.
 * Unknown / invalid values fall back to their defaults; we never
 * throw, because the URL is user-supplied input that should degrade
 * gracefully (the controls then resynchronize the URL via stateToUrl).
 *
 * @param {string|URL} url
 * @returns {State}
 */
export function parseUrlToState(url) {
  const u = typeof url === 'string' ? new URL(url, 'http://localhost') : url;
  const q = u.searchParams;

  const core = q.get('core') || null;

  let mode = q.get('mode');
  if (!VALID_MODES.has(mode)) mode = 'expanded';

  let colorBy = q.get('color_by');
  if (!VALID_COLOR_BY.has(colorBy)) colorBy = 'phenotype';

  let marker = q.get('marker');
  if (marker !== null && !VALID_MARKER_SET.has(marker)) marker = null;
  // commit 3 — selection refactor: `marker` is now preserved across
  // color_by modes. Rationale: the same marker drives the cell overlay
  // color (in Marker focus), the side-panel marker bars, and the CODEX
  // base layer when base=codex. Forcing it to null in phenotype mode
  // meant the user had to repick the marker every time they toggled
  // Phenotype ↔ Marker. The UI hides the marker dropdown in Phenotype
  // focus so the option isn't visually noisy.
  if (colorBy !== 'phenotype' && marker === null) {
    // Why: shareable URLs that pin color_by=marker_pred but forget the
    // marker should resolve to a deterministic view rather than blank.
    // The console.warn surfaces in DevTools but does not block render.
    console.warn(
      `state: color_by=${colorBy} requires marker; defaulting to ${DEFAULT_MARKER}`,
    );
    marker = DEFAULT_MARKER;
  }

  let phenotypes = null;
  const phenoCsv = q.get('phenotypes');
  if (phenoCsv !== null) {
    const allowed = new Set([...PHENOTYPE_NAMES, NO_CALL_KEY]);
    const seen = new Set();
    for (const token of phenoCsv.split(',')) {
      const t = token.trim();
      if (allowed.has(t)) seen.add(t);
    }
    if (seen.size > 0) phenotypes = seen;
  }

  let selected = null;
  const selRaw = q.get('selected');
  if (selRaw !== null) {
    const n = Number.parseInt(selRaw, 10);
    if (Number.isFinite(n) && n >= 0) selected = n;
  }

  const compare = q.get('compare') === '1';

  // Model: a non-empty string is accepted as-is; the server validates
  // against manifest.models on the next /api/cells fetch and returns
  // 422 with the valid list if it isn't present. We don't pre-validate
  // here because state.js shouldn't carry a hard-coded model list —
  // the bundle's manifest is the source of truth.
  const modelRaw = q.get('model');
  const model = modelRaw && modelRaw.trim() ? modelRaw.trim() : DEFAULT_MODEL_ID;

  let base = q.get('base');
  if (!VALID_BASES.has(base)) base = 'he';

  // truth_label_set — only meaningful for marker_truth / marker_error
  // views; we still parse it in every mode so the user's pinned setting
  // survives a toggle to phenotype/pred and back. Invalid values fall
  // back to the default rather than erroring.
  let truthLabelSet = q.get('truth_label_set');
  if (!VALID_TRUTH_LABEL_SETS.has(truthLabelSet)) {
    truthLabelSet = DEFAULT_TRUTH_LABEL_SET;
  }

  // commit 7 — probability threshold for the overlay filter. Only
  // meaningful in marker_pred view; we still parse it in every mode so
  // a URL that pinned ?min_prob=0.7&color_by=phenotype is back-compat
  // when the user later toggles to a marker view.
  let minProb = 0;
  const minProbRaw = q.get('min_prob');
  if (minProbRaw !== null) {
    const parsed = Number.parseFloat(minProbRaw);
    if (Number.isFinite(parsed)) {
      minProb = Math.max(0, Math.min(1, parsed));
    }
  }

  return {
    core,
    mode,
    color_by: colorBy,
    marker,
    phenotypes,
    selected,
    compare,
    model,
    base,
    min_prob: minProb,
    truth_label_set: truthLabelSet,
  };
}

/**
 * Serialize a State object to a URL search-string ('?core=...&...').
 * Keys at their default value are omitted so shareable URLs stay
 * compact ("?core=X" rather than "?core=X&mode=expanded&color_by=...").
 *
 * @param {State} state
 * @returns {string} The leading '?' is included when non-empty.
 */
export function stateToUrl(state) {
  const parts = [];
  if (state.core) parts.push(`core=${encodeURIComponent(state.core)}`);
  if (state.mode && state.mode !== 'expanded') {
    parts.push(`mode=${encodeURIComponent(state.mode)}`);
  }
  if (state.color_by && state.color_by !== 'phenotype') {
    parts.push(`color_by=${encodeURIComponent(state.color_by)}`);
  }
  // Marker is now meaningful in every mode (drives the CODEX base layer
  // + side-panel marker bars even in phenotype focus), so we round-trip
  // it whenever set — not only when color_by != phenotype.
  if (state.marker) {
    parts.push(`marker=${encodeURIComponent(state.marker)}`);
  }
  if (state.phenotypes && state.phenotypes.size > 0) {
    const all = new Set([...PHENOTYPE_NAMES, NO_CALL_KEY]);
    // If every phenotype is on, omit the param so the URL stays clean.
    if (state.phenotypes.size !== all.size) {
      const ordered = [...PHENOTYPE_NAMES, NO_CALL_KEY].filter((k) =>
        state.phenotypes.has(k),
      );
      parts.push(`phenotypes=${encodeURIComponent(ordered.join(','))}`);
    }
  }
  if (state.selected !== null && state.selected !== undefined) {
    parts.push(`selected=${encodeURIComponent(String(state.selected))}`);
  }
  if (state.compare) parts.push('compare=1');
  // Model: omit from URL when it equals the default so shareable
  // links stay compact for the common case (?model=v1_1 is noise
  // when the manifest's default is also v1_1).
  if (state.model && state.model !== DEFAULT_MODEL_ID) {
    parts.push(`model=${encodeURIComponent(state.model)}`);
  }
  // Only emit `base=` when non-default so shared URLs stay compact.
  if (state.base && state.base !== 'he') {
    parts.push(`base=${encodeURIComponent(state.base)}`);
  }
  // commit 7 — probability threshold round-trip. Default 0 is omitted
  // so shareable URLs stay compact when the filter is disabled.
  if (typeof state.min_prob === 'number' && state.min_prob > 0) {
    // Cap to 2 decimal places — anything finer than a single tick is
    // noise in a 0-1 range slider.
    parts.push(`min_prob=${state.min_prob.toFixed(2)}`);
  }
  // truth_label_set — emit only when non-default. The view only consumes
  // it for marker_truth / marker_error; the serializer doesn't gate on
  // color_by so a pinned setting survives an idle toggle, matching the
  // parser's "preserve across modes" behavior.
  if (
    state.truth_label_set &&
    state.truth_label_set !== DEFAULT_TRUTH_LABEL_SET
  ) {
    parts.push(`truth_label_set=${encodeURIComponent(state.truth_label_set)}`);
  }
  return parts.length === 0 ? '' : `?${parts.join('&')}`;
}

/**
 * Push or replace history with the serialized state. The default is
 * replaceState so the user's Back button doesn't fill up with every
 * tiny tweak (a phenotype-chip toggle is not a navigation event).
 *
 * @param {State} state
 * @param {{replace?: boolean}} [opts]
 */
export function applyState(state, opts) {
  const replace = !opts || opts.replace !== false;
  const qs = stateToUrl(state);
  const url = `${window.location.pathname}${qs}`;
  if (replace) window.history.replaceState(null, '', url);
  else window.history.pushState(null, '', url);
}

/**
 * Deep equality on two State objects. Used by main.js to skip
 * redundant re-fetches when a control change reproduces the current
 * state (e.g., clicking the active mode button).
 *
 * @param {State} a
 * @param {State} b
 * @returns {boolean}
 */
export function stateEquals(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.core !== b.core) return false;
  if (a.mode !== b.mode) return false;
  if (a.color_by !== b.color_by) return false;
  if (a.marker !== b.marker) return false;
  if (a.selected !== b.selected) return false;
  if (a.compare !== b.compare) return false;
  if (a.model !== b.model) return false;
  if ((a.base || 'he') !== (b.base || 'he')) return false;
  if ((a.min_prob || 0) !== (b.min_prob || 0)) return false;
  if (
    (a.truth_label_set || DEFAULT_TRUTH_LABEL_SET) !==
    (b.truth_label_set || DEFAULT_TRUTH_LABEL_SET)
  ) {
    return false;
  }
  const ap = a.phenotypes, bp = b.phenotypes;
  if ((ap === null) !== (bp === null)) return false;
  if (ap !== null && bp !== null) {
    if (ap.size !== bp.size) return false;
    for (const k of ap) if (!bp.has(k)) return false;
  }
  return true;
}

/**
 * Return the subset of axes that controls the /api/cells/{core} fetch.
 * Two states with equal "fetch-key" return identical bytes from the
 * server, so the cache key is exactly this tuple.
 *
 * Per spec section 3.3, the phenotype toggle is a **frontend filter**
 * over GeoJSON properties — it must NOT trigger a re-fetch. Likewise
 * the selected cell + compare toggle are pure UI state. Only `core`,
 * `mode`, `color_by`, and `marker` change the bytes returned by the
 * server, so those four define the cache key.
 *
 * @param {State} s
 * @returns {string}
 */
export function fetchKey(s) {
  // model selection: model id is included in the cache key because each
  // model is a separate column family in cells.parquet — different
  // models return different color_value bytes for the same core /
  // mode / color_by / marker tuple.
  // truth_label_set changes the bytes for marker_truth / marker_error
  // views (different columns → different color_value), but is irrelevant
  // for phenotype / marker_pred. We include it in the key whenever the
  // *view* is one of those two so toggling the label set re-fetches; in
  // other views the key collapses to the same value regardless of
  // setting so changes there don't bust the cache.
  const tlsKey =
    s.color_by === 'marker_truth' || s.color_by === 'marker_error'
      ? s.truth_label_set || DEFAULT_TRUTH_LABEL_SET
      : '';
  return `${s.core}|${s.mode}|${s.color_by}|${s.marker || ''}|${s.model || ''}|${tlsKey}`;
}

// ---------------------------------------------------------------------
// Inline self-test (spec section 3.3) — gated by a window flag so it
// only fires under Playwright control. Result is stashed on
// `window.__HEXIF_STATE_SELFTEST_RESULT__` as either `{ok: true, n}` or
// `{ok: false, failures: [...]}`. We use this instead of a unit test
// harness because the module is loaded as a real ES module from the
// same path the browser uses, so a bug in module bundling that breaks
// state.js in the browser also fails this test.
//
// The Node-driven URL↔state tests in tests/test_workbench_state_js.py
// cover the same contract from a server-side perspective; this is the
// browser-side mirror.
// ---------------------------------------------------------------------
if (typeof window !== 'undefined' && window.__HEXIF_STATE_SELFTEST__) {
  const failures = [];
  const check = (label, cond) => {
    if (!cond) failures.push(label);
  };

  // 1. Empty URL → canonical defaults.
  const s0 = parseUrlToState('http://x/');
  check('defaults.core null', s0.core === null);
  check('defaults.mode expanded', s0.mode === 'expanded');
  check('defaults.color_by phenotype', s0.color_by === 'phenotype');
  check('defaults.marker null', s0.marker === null);
  check('defaults.phenotypes null', s0.phenotypes === null);
  check('defaults.selected null', s0.selected === null);
  check('defaults.compare false', s0.compare === false);

  // 2. marker_pred round trip preserves marker.
  const s1 = parseUrlToState('http://x/?core=A&color_by=marker_pred&marker=CD8');
  check('marker_pred.core', s1.core === 'A');
  check('marker_pred.color_by', s1.color_by === 'marker_pred');
  check('marker_pred.marker', s1.marker === 'CD8');
  const u1 = stateToUrl(s1);
  check('marker_pred URL has core', u1.includes('core=A'));
  check('marker_pred URL has color_by', u1.includes('color_by=marker_pred'));
  check('marker_pred URL has marker', u1.includes('marker=CD8'));
  check('marker_pred URL omits default mode', !u1.includes('mode=expanded'));

  // 3. Unknown marker falls back to the default.
  const s2 = parseUrlToState('http://x/?color_by=marker_pred&marker=BOGUS');
  check('unknown marker → default', s2.marker === DEFAULT_MARKER);

  // 4. Phenotype CSV → Set.
  const s3 = parseUrlToState('http://x/?phenotypes=immune_cd45,no_call');
  check('pheno CSV parsed', s3.phenotypes instanceof Set);
  check('pheno set has immune_cd45', s3.phenotypes && s3.phenotypes.has('immune_cd45'));
  check('pheno set has no_call', s3.phenotypes && s3.phenotypes.has('no_call'));

  // 5. "All chips on" collapses param out of URL.
  const allPheno = [...PHENOTYPE_NAMES, NO_CALL_KEY].join(',');
  const s4 = parseUrlToState(`http://x/?phenotypes=${allPheno}`);
  check('all-on collapses URL', !stateToUrl(s4).includes('phenotypes='));

  // 6. stateEquals order-insensitive on the phenotype Set.
  const sa = parseUrlToState('http://x/?phenotypes=immune_cd45,t_cell_cd45_cd8');
  const sb = parseUrlToState('http://x/?phenotypes=t_cell_cd45_cd8,immune_cd45');
  check('stateEquals set-equal', stateEquals(sa, sb));

  // 7. fetchKey is stable across selection changes.
  const fk1 = fetchKey(parseUrlToState('http://x/?core=A&selected=10'));
  const fk2 = fetchKey(parseUrlToState('http://x/?core=A&selected=999'));
  check('fetchKey ignores selected', fk1 === fk2);

  // 7b. fetchKey is stable across phenotype-toggle changes (API contract:
  // phenotypes are a frontend-only filter; toggling must not refetch).
  const fk3 = fetchKey(parseUrlToState('http://x/?core=A'));
  const fk4 = fetchKey(parseUrlToState('http://x/?core=A&phenotypes=immune_cd45'));
  check('fetchKey ignores phenotypes', fk3 === fk4);

  // 8. compare=1 survives round trip.
  const s5 = parseUrlToState('http://x/?compare=1');
  check('compare flag parsed', s5.compare === true);
  check('compare flag re-serialized', stateToUrl(s5).includes('compare=1'));

  // 9. model selection: model round-trips through parse/serialize and is
  // omitted from the URL when it equals the default.
  const s6 = parseUrlToState('http://x/?core=A&model=miphei_vit');
  check('model parsed', s6.model === 'miphei_vit');
  check('model re-serialized', stateToUrl(s6).includes('model=miphei_vit'));
  const sDefault = parseUrlToState('http://x/?core=A');
  check('model defaults to v1_1', sDefault.model === 'v1_1');
  check('default model omitted from URL', !stateToUrl(sDefault).includes('model='));
  // 10. fetchKey changes when model changes.
  const fkV = fetchKey(parseUrlToState('http://x/?core=A&model=v1'));
  const fkV1 = fetchKey(parseUrlToState('http://x/?core=A&model=v1_1'));
  check('fetchKey distinguishes models', fkV !== fkV1);

  // 10b. Commit-3: marker is preserved when color_by=phenotype (it now
  // drives the CODEX base layer + side-panel marker bars even in
  // phenotype focus).
  const sM1 = parseUrlToState('http://x/?core=A&color_by=phenotype&marker=Ki67');
  check('marker preserved in phenotype mode', sM1.marker === 'Ki67');
  check(
    'marker re-serialized in phenotype mode',
    stateToUrl(sM1).includes('marker=Ki67'),
  );

  // 11. CODEX composite: base=codex round-trips; default he stays out of the URL.
  const sB1 = parseUrlToState('http://x/?core=A&base=codex');
  check('base parsed = codex', sB1.base === 'codex');
  check('base serialized = codex', stateToUrl(sB1).includes('base=codex'));
  const sB2 = parseUrlToState('http://x/?core=A');
  check('base default = he', sB2.base === 'he');
  check('base default omitted from URL', !stateToUrl(sB2).includes('base='));
  // 11b. commit 8 — base=split is a third option (H&E + CODEX side-by-
  //      side with synced pan/zoom). Round-trips through the URL like
  //      'codex' does.
  const sB3 = parseUrlToState('http://x/?core=A&base=split');
  check('base parsed = split', sB3.base === 'split');
  check('base serialized = split', stateToUrl(sB3).includes('base=split'));

  // 12. switching only `base` must NOT change fetchKey — switching
  //     base swaps the tile source but the polygon GeoJSON is unchanged.
  const fk_he = fetchKey(parseUrlToState('http://x/?core=A'));
  const fk_codex = fetchKey(parseUrlToState('http://x/?core=A&base=codex'));
  check('fetchKey ignores base', fk_he === fk_codex);

  // 13. commit 7 — min_prob round-trips through URL and is omitted at
  //     the default value (0).
  const sP1 = parseUrlToState('http://x/?core=A&min_prob=0.75');
  check('min_prob parsed', sP1.min_prob === 0.75);
  check('min_prob serialized', stateToUrl(sP1).includes('min_prob=0.75'));
  const sP0 = parseUrlToState('http://x/?core=A');
  check('min_prob default = 0', sP0.min_prob === 0);
  check('min_prob default omitted from URL', !stateToUrl(sP0).includes('min_prob'));
  // Out-of-range clamps to [0, 1].
  const sP2 = parseUrlToState('http://x/?core=A&min_prob=2');
  check('min_prob clamps high to 1', sP2.min_prob === 1);
  const sP3 = parseUrlToState('http://x/?core=A&min_prob=-0.5');
  check('min_prob clamps low to 0', sP3.min_prob === 0);

  const result =
    failures.length === 0
      ? { ok: true, n: 13 }
      : { ok: false, failures };
  window.__HEXIF_STATE_SELFTEST_RESULT__ = result;
}
