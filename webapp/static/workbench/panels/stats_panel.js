// Stats tab panel (statistics).
//
// Three vanilla tables stacked vertically: per-core stats for the
// currently-selected core, cohort summary filterable by tissue / TMA
// / split, and a per-model validation panel. All data is read-only
// from the bundle (manifest.models for AP, cells.parquet for counts).
//
// Why vanilla <table> and not DataTables / a charting library: the
// spec (§3.6) explicitly says "keep it vanilla". The numbers don't
// move at sort time (the rows are stable per fetch) so client-side
// sorting was deliberately scoped out.
//
// URL state: the stats-tab toggle is purely transient UI state. We
// do NOT round-trip it in the URL — opening the workbench at
// `?core=X` should default to the cell-viewer tab. See the API contract

import {
  fetchStatsPerCore,
  fetchStatsCohort,
  fetchStatsValidation,
} from '../api.js';
import { PHENOTYPE_NAMES, NO_CALL_KEY } from '../state.js';

// Default histogram selection — mirrors the backend default in
// webapp/cell_service.py:cohort_stats. tumor_ca9_or_panck is the
// panel's most-populated phenotype on ccRCC/ccOC and yields a
// non-degenerate histogram across nearly every cohort filter.
const DEFAULT_HISTOGRAM_PHENOTYPE = 'tumor_ca9_or_panck';

/**
 * @typedef {Object} StatsCallbacks
 * @property {() => any} getState               Read the current state (.core, .model).
 * @property {() => {models: Array, default_id: string}|null} [getModels]
 *   Optional model list loaded at boot. Used as a validation-metric fallback.
 * @property {() => boolean} isActive           True when the stats tab is the active view.
 */

/**
 * Mount the stats tab into ``root``. The caller is responsible for
 * showing / hiding ``root`` based on the active tab — this function
 * just manages the panel's internal DOM and data lifecycle.
 *
 * Returns a handle with:
 *   refresh()             — re-fetch all three panels against the current state
 *   refreshPerCore()      — refresh just the per-core table (e.g. after core switch)
 *   refreshValidation()   — refresh just the validation table (e.g. after model switch)
 *
 * @param {HTMLElement} root
 * @param {StatsCallbacks} cb
 */
export function mountStatsPanel(root, cb) {
  root.innerHTML = '';
  root.classList.add('stats-panel');

  // --- KPI cards (top of stats tab) -------------------------------------
  // Three at-a-glance numbers: cells in the active core, validation
  // macro marker AP, validation macro phenotype AP. These mirror data
  // that lives elsewhere in the panel (per-core table; validation
  // headlines) — surfacing them here means the user sees the headline
  // numbers without scrolling.
  const kpiRow = document.createElement('section');
  kpiRow.className = 'stats-kpis';
  kpiRow.innerHTML = `
    <div class="stats-kpi" data-role="kpi-cells">
      <span class="stats-kpi-label">Cells in core</span>
      <span class="stats-kpi-value" data-role="kpi-cells-value">—</span>
      <span class="stats-kpi-sub" data-role="kpi-cells-sub">—</span>
    </div>
    <div class="stats-kpi" data-role="kpi-dominant">
      <span class="stats-kpi-label">Dominant phenotype</span>
      <span class="stats-kpi-value" data-role="kpi-dominant-value">—</span>
      <span class="stats-kpi-sub" data-role="kpi-dominant-sub">—</span>
    </div>
    <div class="stats-kpi" data-role="kpi-marker-ap">
      <span class="stats-kpi-label">Macro marker AP</span>
      <span class="stats-kpi-value" data-role="kpi-marker-ap-value">—</span>
      <span class="stats-kpi-sub">val cohort</span>
    </div>
    <div class="stats-kpi" data-role="kpi-pheno-ap">
      <span class="stats-kpi-label">Macro phenotype AP</span>
      <span class="stats-kpi-value" data-role="kpi-pheno-ap-value">—</span>
      <span class="stats-kpi-sub">val cohort</span>
    </div>
  `;
  root.appendChild(kpiRow);

  // --- Per-core section -------------------------------------------------
  const perCoreSection = document.createElement('section');
  perCoreSection.className = 'stats-section stats-per-core';
  perCoreSection.innerHTML = `
    <h3 class="stats-section-title">Per-core stats</h3>
    <div class="stats-section-sub" data-role="per-core-subtitle">—</div>
    <div class="stats-tables" data-role="per-core-tables">
      <div class="stats-placeholder">Loading…</div>
    </div>
  `;
  root.appendChild(perCoreSection);

  // --- Cohort section ---------------------------------------------------
  const cohortSection = document.createElement('section');
  cohortSection.className = 'stats-section stats-cohort';
  cohortSection.innerHTML = `
    <h3 class="stats-section-title">Cohort summary</h3>
    <div class="stats-cohort-filters">
      <label class="stats-filter">
        <span>Tissue</span>
        <select data-filter="tissue">
          <option value="">All</option>
          <option value="ccRCC">ccRCC</option>
          <option value="ccOC">ccOC</option>
        </select>
      </label>
      <label class="stats-filter">
        <span>TMA</span>
        <select data-filter="tma">
          <option value="">All</option>
          <option value="TMA1">TMA1</option>
          <option value="TMA2">TMA2</option>
          <option value="TMA3">TMA3</option>
        </select>
      </label>
      <label class="stats-filter">
        <span>Split</span>
        <select data-filter="split">
          <option value="">All</option>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
      </label>
      <button type="button" class="stats-cohort-refresh" data-role="cohort-refresh">
        Refresh
      </button>
    </div>
    <div class="stats-section-sub" data-role="cohort-subtitle">—</div>
    <div class="stats-tables" data-role="cohort-tables">
      <div class="stats-placeholder">Loading…</div>
    </div>
  `;
  root.appendChild(cohortSection);

  // --- Validation section -----------------------------------------------
  const validationSection = document.createElement('section');
  validationSection.className = 'stats-section stats-validation';
  validationSection.innerHTML = `
    <h3 class="stats-section-title">Validation</h3>
    <div class="stats-section-sub" data-role="validation-subtitle">—</div>
    <div class="stats-validation-headlines" data-role="validation-headlines"></div>
    <div class="stats-tables" data-role="validation-tables">
      <div class="stats-placeholder">Loading…</div>
    </div>
  `;
  root.appendChild(validationSection);

  // Cohort filter UI hookup.
  const cohortRefresh = cohortSection.querySelector('[data-role="cohort-refresh"]');
  cohortRefresh.addEventListener('click', () => {
    void refreshCohort();
  });
  // Auto-refresh on filter change is a UX nicety — many users miss the
  // Refresh button and assume the filter is "live". The cost is one
  // extra fetch per click and the result is cached server-side.
  for (const sel of cohortSection.querySelectorAll('select[data-filter]')) {
    sel.addEventListener('change', () => {
      void refreshCohort();
    });
  }

  // Transient per-tab UI state — which phenotype the histogram is
  // showing. Not round-tripped through the URL per the API contract (the
  // stats tab itself is transient). Initialized to the backend's
  // default so the first render matches a no-param GET.
  let selectedHistogramPhenotype = DEFAULT_HISTOGRAM_PHENOTYPE;

  // -----------------------------------------------------------------------
  // Per-core rendering
  // -----------------------------------------------------------------------
  async function refreshPerCore() {
    const state = cb.getState() || {};
    const core = state.core;
    const model = state.model;
    const sub = perCoreSection.querySelector('[data-role="per-core-subtitle"]');
    const tables = perCoreSection.querySelector('[data-role="per-core-tables"]');
    if (!core) {
      sub.textContent = 'No core selected.';
      tables.innerHTML = '';
      return;
    }
    sub.textContent = `Core: ${core} · Model: ${model || '(default)'}`;
    tables.innerHTML = '<div class="stats-placeholder">Loading…</div>';
    let body;
    try {
      body = await fetchStatsPerCore(core, { model });
    } catch (err) {
      tables.innerHTML = '';
      tables.appendChild(buildError(`Per-core stats failed: ${err.message || err}`));
      return;
    }
    tables.innerHTML = '';
    tables.appendChild(buildPerCorePhenoTable(body));
    tables.appendChild(buildPerCoreMarkerTable(body));
    if (body.has_truth && Array.isArray(body.comparison)) {
      tables.appendChild(buildPerCoreComparisonTable(body));
    } else {
      const note = document.createElement('div');
      note.className = 'stats-note';
      note.textContent = 'No paired CODEX truth for this core — comparison hidden.';
      tables.appendChild(note);
    }
    updateKpisFromPerCore(body);
  }

  /**
   * Populate the top-row KPI cards from a /api/stats/per-core payload.
   * Driven from the per-core response so the dominant-phenotype card
   * tracks whichever core / model the user has loaded.
   */
  function updateKpisFromPerCore(body) {
    const cellsValue = kpiRow.querySelector('[data-role="kpi-cells-value"]');
    const cellsSub   = kpiRow.querySelector('[data-role="kpi-cells-sub"]');
    const domValue   = kpiRow.querySelector('[data-role="kpi-dominant-value"]');
    const domSub     = kpiRow.querySelector('[data-role="kpi-dominant-sub"]');

    cellsValue.textContent = formatInt(body.n_cells);
    const state = cb.getState() || {};
    cellsSub.textContent = `core ${state.core || '—'}`;

    const ranked = [...(body.phenotypes || [])]
      .filter((p) => p && p.name !== 'no_call')
      .sort((a, b) => b.count - a.count);
    const top = ranked[0];
    if (top) {
      domValue.textContent = top.name;
      domSub.textContent = `${formatPct(top.fraction)} of cells`;
    } else {
      domValue.textContent = '—';
      domSub.textContent = '—';
    }
  }

  function buildPerCorePhenoTable(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = `<div class="stats-table-caption">Phenotype distribution (n_cells = ${body.n_cells})</div>`;
    // Sort: descending count, but keep "no_call" at the bottom because
    // it dominates the table on most cores and burying it elsewhere
    // misleads on the long-tail distribution.
    const rows = [...(body.phenotypes || [])];
    rows.sort((a, b) => {
      if (a.name === 'no_call') return 1;
      if (b.name === 'no_call') return -1;
      return b.count - a.count;
    });
    const table = document.createElement('table');
    table.className = 'stats-table';
    table.innerHTML = `
      <thead><tr><th>Phenotype</th><th class="num">Count</th><th class="num">Fraction</th></tr></thead>
    `;
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.dataset.name = r.name;
      tr.innerHTML = `
        <td>${escapeHtml(r.name)}</td>
        <td class="num">${formatInt(r.count)}</td>
        <td class="num">${formatPct(r.fraction)}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function buildPerCoreMarkerTable(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = '<div class="stats-table-caption">Marker positivity (above calibrated threshold)</div>';
    // Descending fraction so the dominant markers land at the top —
    // matches the spec's "sorted by fraction desc".
    const rows = [...(body.markers || [])];
    rows.sort((a, b) => b.fraction_positive - a.fraction_positive);
    const table = document.createElement('table');
    table.className = 'stats-table';
    table.innerHTML = `
      <thead><tr><th>Marker</th><th class="num">n_positive</th><th class="num">Fraction</th><th>Confidence</th></tr></thead>
    `;
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.dataset.name = r.name;
      tr.innerHTML = `
        <td>${escapeHtml(r.name)}</td>
        <td class="num">${formatInt(r.n_positive)}</td>
        <td class="num">${formatPct(r.fraction_positive)}</td>
        <td><span class="stats-stratum-${escapeHtml(r.confidence_stratum)}">${escapeHtml(r.confidence_stratum)}</span></td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function buildPerCoreComparisonTable(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = '<div class="stats-table-caption">Paired CODEX comparison</div>';
    const rows = body.comparison || [];
    const table = document.createElement('table');
    table.className = 'stats-table';
    table.innerHTML = `
      <thead><tr><th>Marker</th><th class="num">Pred frac</th><th class="num">Truth frac</th><th class="num">|Δ|</th><th class="num">AP</th></tr></thead>
    `;
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.dataset.name = r.name;
      tr.innerHTML = `
        <td>${escapeHtml(r.name)}</td>
        <td class="num">${formatPct(r.predicted_fraction)}</td>
        <td class="num">${formatPct(r.truth_fraction)}</td>
        <td class="num">${formatPct(r.absolute_error)}</td>
        <td class="num">${r.ap === null || r.ap === undefined ? '—' : r.ap.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  // -----------------------------------------------------------------------
  // Cohort rendering
  // -----------------------------------------------------------------------
  async function refreshCohort() {
    const state = cb.getState() || {};
    const tissue = cohortSection.querySelector('[data-filter="tissue"]').value || null;
    const tma = cohortSection.querySelector('[data-filter="tma"]').value || null;
    const split = cohortSection.querySelector('[data-filter="split"]').value || null;
    const sub = cohortSection.querySelector('[data-role="cohort-subtitle"]');
    const tables = cohortSection.querySelector('[data-role="cohort-tables"]');
    sub.textContent = `Filter: tissue=${tissue || 'all'} · tma=${tma || 'all'} · split=${split || 'all'}`;
    tables.innerHTML = '<div class="stats-placeholder">Loading…</div>';
    let body;
    try {
      body = await fetchStatsCohort({
        tissue,
        tma,
        split,
        model: state.model,
        histogram_phenotype: selectedHistogramPhenotype,
      });
    } catch (err) {
      tables.innerHTML = '';
      tables.appendChild(buildError(`Cohort stats failed: ${err.message || err}`));
      return;
    }
    sub.textContent =
      `Filter: tissue=${body.filter.tissue || 'all'} · tma=${body.filter.tma || 'all'} `
      + `· split=${body.filter.split || 'all'} · n_cores=${body.n_cores} · n_cells=${formatInt(body.n_cells_total)}`;
    tables.innerHTML = '';
    tables.appendChild(buildCohortPhenoTable(body));
    tables.appendChild(buildCohortMarkerTable(body));
    tables.appendChild(buildCohortHistogram(body));
  }

  // Re-fetch ONLY the histogram for the currently-selected phenotype
  // and swap it in place. The pheno/marker tables don't change with
  // histogram_phenotype, so they're left untouched. The fetch reuses
  // the current cohort filter values, and the backend response is
  // cached server-side per (filter, model, histogram_phenotype) key.
  async function refreshHistogramOnly() {
    const state = cb.getState() || {};
    const tissue = cohortSection.querySelector('[data-filter="tissue"]').value || null;
    const tma = cohortSection.querySelector('[data-filter="tma"]').value || null;
    const split = cohortSection.querySelector('[data-filter="split"]').value || null;
    const tables = cohortSection.querySelector('[data-role="cohort-tables"]');
    const old = tables.querySelector('[data-role="cohort-histogram"]');
    if (!old) return; // panel not yet rendered; refreshCohort will handle it.
    let body;
    try {
      body = await fetchStatsCohort({
        tissue,
        tma,
        split,
        model: state.model,
        histogram_phenotype: selectedHistogramPhenotype,
      });
    } catch (err) {
      old.replaceWith(buildError(`Histogram refresh failed: ${err.message || err}`));
      return;
    }
    old.replaceWith(buildCohortHistogram(body));
  }

  function buildCohortPhenoTable(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = '<div class="stats-table-caption">Phenotype distribution (pooled)</div>';
    const rows = [...(body.phenotype_summary || [])];
    rows.sort((a, b) => {
      if (a.name === 'no_call') return 1;
      if (b.name === 'no_call') return -1;
      return b.count - a.count;
    });
    const table = document.createElement('table');
    table.className = 'stats-table';
    table.innerHTML = `
      <thead><tr><th>Phenotype</th><th class="num">Count</th><th class="num">Fraction</th></tr></thead>
    `;
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${escapeHtml(r.name)}</td>
        <td class="num">${formatInt(r.count)}</td>
        <td class="num">${formatPct(r.fraction)}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function buildCohortMarkerTable(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = '<div class="stats-table-caption">Marker positivity (pooled)</div>';
    const rows = [...(body.marker_summary || [])];
    rows.sort((a, b) => b.fraction_positive - a.fraction_positive);
    const table = document.createElement('table');
    table.className = 'stats-table';
    table.innerHTML = `
      <thead><tr><th>Marker</th><th class="num">n_positive</th><th class="num">Fraction</th><th>Confidence</th></tr></thead>
    `;
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${escapeHtml(r.name)}</td>
        <td class="num">${formatInt(r.n_positive)}</td>
        <td class="num">${formatPct(r.fraction_positive)}</td>
        <td><span class="stats-stratum-${escapeHtml(r.confidence_stratum)}">${escapeHtml(r.confidence_stratum)}</span></td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function buildCohortHistogram(body) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap stats-histogram-wrap';
    wrap.dataset.role = 'cohort-histogram';
    // Caption with inline phenotype selector. The select lives next to
    // the caption so the user reads "histogram for <phenotype>" as one
    // unit. Options = the 9 phenotypes + no_call (the backend accepts
    // all 10; no_call is sometimes useful to surface "how empty" the
    // cohort is on a per-core basis).
    const captionRow = document.createElement('div');
    captionRow.className = 'stats-histogram-caption-row';
    const caption = document.createElement('div');
    caption.className = 'stats-table-caption';
    caption.textContent = 'Per-core fraction histogram';
    captionRow.appendChild(caption);
    const selectLabel = document.createElement('label');
    selectLabel.className = 'stats-histogram-phenotype-select';
    selectLabel.innerHTML = '<span>Phenotype</span>';
    const select = document.createElement('select');
    select.setAttribute('data-role', 'histogram-phenotype-select');
    const optionNames = [...PHENOTYPE_NAMES, NO_CALL_KEY];
    for (const name of optionNames) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === (body.histogram_phenotype || selectedHistogramPhenotype)) {
        opt.selected = true;
      }
      select.appendChild(opt);
    }
    select.addEventListener('change', (ev) => {
      const next = ev.target.value;
      if (!next || next === selectedHistogramPhenotype) return;
      selectedHistogramPhenotype = next;
      void refreshHistogramOnly();
    });
    selectLabel.appendChild(select);
    captionRow.appendChild(selectLabel);
    wrap.appendChild(captionRow);
    const counts = body.histogram_counts || [];
    const bins = body.histogram_bins || [];
    const max = counts.reduce((m, v) => Math.max(m, v), 0) || 1;
    // SVG bar chart — 10 bins on [0, 1]. We keep it inline + dimension-
    // less so the panel reflows naturally with the parent width.
    const W = 360;
    const H = 120;
    const PAD_LEFT = 28;
    const PAD_RIGHT = 8;
    const PAD_TOP = 6;
    const PAD_BOTTOM = 22;
    const innerW = W - PAD_LEFT - PAD_RIGHT;
    const innerH = H - PAD_TOP - PAD_BOTTOM;
    const barW = innerW / 10;
    const parts = [];
    parts.push(
      `<svg viewBox="0 0 ${W} ${H}" class="stats-histogram-svg" role="img" `
      + `aria-label="Per-core fraction histogram for ${escapeHtml(body.histogram_phenotype)}">`,
    );
    // Bars.
    for (let i = 0; i < 10; i++) {
      const c = counts[i] || 0;
      const h = c === 0 ? 0 : Math.max(1, (c / max) * innerH);
      const x = PAD_LEFT + i * barW;
      const y = PAD_TOP + (innerH - h);
      parts.push(
        `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(barW - 1).toFixed(1)}" `
        + `height="${h.toFixed(1)}" fill="#3b82f6" stroke="#1e40af" stroke-width="0.5">`
        + `<title>[${(bins[i] || 0).toFixed(1)}, ${(bins[i + 1] || 1).toFixed(1)}): ${c} cores</title>`
        + '</rect>',
      );
    }
    // Axis labels.
    parts.push(`<text x="${PAD_LEFT}" y="${H - 6}" font-size="9" fill="#6b7280">0</text>`);
    parts.push(
      `<text x="${PAD_LEFT + innerW - 6}" y="${H - 6}" font-size="9" `
      + 'fill="#6b7280" text-anchor="end">1.0</text>',
    );
    parts.push(`<text x="2" y="${PAD_TOP + 8}" font-size="9" fill="#6b7280">${max}</text>`);
    parts.push(`<text x="2" y="${PAD_TOP + innerH}" font-size="9" fill="#6b7280">0</text>`);
    parts.push('</svg>');
    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-histogram-svg-wrap';
    svgWrap.innerHTML = parts.join('');
    wrap.appendChild(svgWrap);
    return wrap;
  }

  // -----------------------------------------------------------------------
  // Validation rendering
  // -----------------------------------------------------------------------
  async function refreshValidation() {
    const state = cb.getState() || {};
    const model = state.model;
    const sub = validationSection.querySelector('[data-role="validation-subtitle"]');
    const headlines = validationSection.querySelector('[data-role="validation-headlines"]');
    const tables = validationSection.querySelector('[data-role="validation-tables"]');
    sub.textContent = `Model: ${model || '(default)'}`;
    headlines.innerHTML = '';
    tables.innerHTML = '<div class="stats-placeholder">Loading…</div>';
    let body;
    try {
      body = await fetchStatsValidation(model);
    } catch (err) {
      tables.innerHTML = '';
      tables.appendChild(buildError(`Validation stats failed: ${err.message || err}`));
      return;
    }
    sub.textContent = `Model: ${body.model} · cores with truth: ${body.markers[0] ? body.markers[0].n_cores_with_truth : 0}`;
    const modelEntry = modelEntryFor(body.model);
    const markerMacro = metricOrFallback(body.macro_marker_ap, modelEntry && modelEntry.macro_marker_ap);
    const phenoMacro = metricOrFallback(body.macro_phenotype_ap, modelEntry && modelEntry.macro_phenotype_ap);
    const validationAvailable = metricAvailable(markerMacro)
      || metricAvailable(phenoMacro)
      || (body.markers || []).some((r) => metricAvailable(r.macro_ap))
      || (body.phenotypes || []).some((r) => metricAvailable(r.macro_ap));
    headlines.innerHTML = `
      <span class="stats-headline">
        <span class="stats-headline-label">Macro marker AP</span>
        <span class="stats-headline-value">${formatAp(markerMacro, { unavailableZero: !validationAvailable })}</span>
      </span>
      <span class="stats-headline">
        <span class="stats-headline-label">Macro phenotype AP</span>
        <span class="stats-headline-value">${formatAp(phenoMacro, { unavailableZero: !validationAvailable })}</span>
      </span>
    `;
    if (!validationAvailable) {
      const note = document.createElement('div');
      note.className = 'stats-note';
      note.textContent = 'Validation AP is not recorded in this bundle.';
      headlines.appendChild(note);
    }
    tables.innerHTML = '';
    tables.appendChild(buildValidationTable(body.markers || [], 'Per-marker AP'));
    tables.appendChild(buildValidationTable(body.phenotypes || [], 'Per-phenotype AP'));

    // Top-row KPI mirrors of the validation headlines.
    const markerKpi = kpiRow.querySelector('[data-role="kpi-marker-ap-value"]');
    const phenoKpi  = kpiRow.querySelector('[data-role="kpi-pheno-ap-value"]');
    markerKpi.textContent = formatAp(markerMacro, { digits: 3, unavailableZero: !validationAvailable });
    phenoKpi.textContent  = formatAp(phenoMacro, { digits: 3, unavailableZero: !validationAvailable });
  }

  function buildValidationTable(rows, caption) {
    const wrap = document.createElement('div');
    wrap.className = 'stats-table-wrap';
    wrap.innerHTML = `<div class="stats-table-caption">${escapeHtml(caption)}</div>`;
    const table = document.createElement('table');
    table.className = 'stats-table';
    const hasPerCore = rows.some((r) => Array.isArray(r.per_core_ap) && r.per_core_ap.length > 0);
    if (hasPerCore) {
      table.innerHTML = `
        <thead><tr><th>Name</th><th class="num">Macro AP</th><th class="num">Mean</th><th class="num">Median</th><th class="num">Std</th><th>Confidence</th></tr></thead>
      `;
    } else {
      table.innerHTML = `
        <thead><tr><th>Name</th><th class="num">Macro AP</th><th class="num">Cores w/ truth</th><th>Confidence</th></tr></thead>
      `;
    }
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      if (hasPerCore) {
        const { mean, median, std } = summarize(r.per_core_ap || []);
        tr.innerHTML = `
          <td>${escapeHtml(r.name)}</td>
          <td class="num">${formatAp(r.macro_ap, { digits: 3 })}</td>
          <td class="num">${mean === null ? '—' : mean.toFixed(3)}</td>
          <td class="num">${median === null ? '—' : median.toFixed(3)}</td>
          <td class="num">${std === null ? '—' : std.toFixed(3)}</td>
          <td><span class="stats-stratum-${escapeHtml(r.confidence_stratum)}">${escapeHtml(r.confidence_stratum)}</span></td>
        `;
      } else {
        tr.innerHTML = `
          <td>${escapeHtml(r.name)}</td>
          <td class="num">${formatAp(r.macro_ap, { digits: 3 })}</td>
          <td class="num">${formatInt(r.n_cores_with_truth)}</td>
          <td><span class="stats-stratum-${escapeHtml(r.confidence_stratum)}">${escapeHtml(r.confidence_stratum)}</span></td>
        `;
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  function formatInt(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    return Number(n).toLocaleString();
  }
  function formatPct(f) {
    if (f === null || f === undefined || Number.isNaN(f)) return '—';
    if (f > 0 && f < 0.001) return '<0.1%';
    return (f * 100).toFixed(1) + '%';
  }
  function formatAp(v, opts) {
    const { digits = 4, unavailableZero = false } = opts || {};
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    const n = Number(v);
    if (unavailableZero && n === 0) return 'n/a';
    if (n > 0 && n < 10 ** -digits) return `<${(10 ** -digits).toFixed(digits)}`;
    return n.toFixed(digits);
  }
  function metricAvailable(v) {
    return v !== null && v !== undefined && Number.isFinite(Number(v)) && Number(v) > 0;
  }
  function metricOrFallback(primary, fallback) {
    if (metricAvailable(primary)) return Number(primary);
    if (metricAvailable(fallback)) return Number(fallback);
    return primary;
  }
  function modelEntryFor(modelId) {
    if (!cb.getModels) return null;
    const modelList = cb.getModels();
    if (!modelList || !Array.isArray(modelList.models)) return null;
    return modelList.models.find((m) => m.id === modelId) || null;
  }
  function escapeHtml(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function summarize(arr) {
    if (!arr || arr.length === 0) return { mean: null, median: null, std: null };
    const sorted = [...arr].sort((a, b) => a - b);
    const n = sorted.length;
    const mean = sorted.reduce((a, b) => a + b, 0) / n;
    const median = n % 2 === 1
      ? sorted[(n - 1) >> 1]
      : (sorted[(n / 2) - 1] + sorted[n / 2]) / 2;
    const variance = sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    return { mean, median, std };
  }
  function buildError(msg) {
    const div = document.createElement('div');
    div.className = 'stats-error';
    div.textContent = msg;
    return div;
  }

  async function refresh() {
    await Promise.all([refreshPerCore(), refreshCohort(), refreshValidation()]);
  }

  return {
    refresh,
    refreshPerCore,
    refreshValidation,
    refreshCohort,
  };
}
