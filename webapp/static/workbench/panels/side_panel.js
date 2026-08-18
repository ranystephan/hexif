// Cell-detail side panel.
//
// Three render states:
//   - `showEmpty(summary)` — no cell selected; renders an inviting empty
//     state with a "hover or click a cell" prompt and a per-core
//     summary card (n cells, dominant phenotype, top markers).
//   - `show(cellId)` — fetches /api/cell/{core}/{cell_id} and renders
//     the marker bars + phenotype table + morphology.
//   - error / loading — small inline strips while the request resolves.
//
// Layout inside `show`:
//   header           Cell #N · phenotype call · confidence badge
//   marker bars      12 horizontal viridis-scaled bars w/ truth tick
//   phenotype rows   9 rows: name | score | call | truth
//   features grid    area_nucleus_px, area_expanded_px, eccentricity
//   footer           link to per-core PDF report
//
// Comparison mode (state.compare == true) splits the marker block into
// two columns (pred | truth). The compare toggle in controls.js is
// disabled when the core has no CODEX truth.

import { fetchCellDetail, reportUrl } from '../api.js';
import { resolveColor } from '../palette.js';
import { VALID_MARKERS, PHENOTYPE_NAMES, NO_CALL_KEY } from '../state.js';

/**
 * @typedef {Object} SidePanelCallbacks
 * @property {() => any} getState
 * @property {(state: any) => void} onStateChange
 * @property {() => any} getPalettes
 * @property {() => any} [getCorePhenotypes]   Optional: latest /api/phenotypes/{core} payload for the empty state.
 */

/**
 * Mount the side panel into `root`. Returns a controller object.
 *
 * @param {HTMLElement} root
 * @param {string} core
 * @param {SidePanelCallbacks} cb
 */
export function mountSidePanel(root, core, cb) {
  root.classList.add('side-panel');
  /** @type {AbortController|null} */
  let inflight = null;

  /** @type {number|null} */
  let currentCellId = null;

  function close() {
    const cur = cb.getState();
    cb.onStateChange({ ...cur, selected: null });
    currentCellId = null;
    // The state-change handler in main.js will call back into
    // showEmpty(); we just bail out here.
  }

  function renderEmpty(summary) {
    // Empty-state pane: stays present whenever no cell is selected, so
    // the right rail never reads as broken. We provide enough scaffolding
    // (instructions, keyboard hints, a small per-core overview) that the
    // user has something useful to do before clicking a cell.
    root.classList.remove('side-panel-open');
    root.innerHTML = '';

    const headerLabel = document.createElement('div');
    headerLabel.className = 'sp-empty-header';
    headerLabel.textContent = 'Cell explorer';
    root.appendChild(headerLabel);

    const prompt = document.createElement('div');
    prompt.className = 'sp-empty-prompt';
    prompt.innerHTML = `
      <h3>Click any cell to inspect</h3>
      <p>
        Each polygon in the viewer is a segmented cell. Hover to highlight,
        click to lock and read its 12-marker prediction plus phenotype
        scoring here. Press <span class="kbd">Esc</span> to release.
      </p>
      <div class="sp-empty-hints">
        <div class="sp-empty-hint">
          <span class="kbd">1</span><span class="kbd">2</span><span class="kbd">3</span><span class="kbd">4</span>
          <span>Switch color-by mode (phenotype / pred / truth / error).</span>
        </div>
        <div class="sp-empty-hint">
          <span class="kbd">]</span><span>Hide this panel to maximise the viewer.</span>
        </div>
        <div class="sp-empty-hint">
          <span class="kbd">?</span><span>Open the full shortcut reference.</span>
        </div>
      </div>
    `;
    root.appendChild(prompt);

    // Per-core overview card. We synthesise it from corePheno (probed at
    // boot) plus the metadata row in the cores list; both are best-effort
    // — if either is missing we just skip that block.
    const summaryWrap = document.createElement('div');
    summaryWrap.className = 'sp-empty-summary';

    if (summary && summary.meta) {
      const m = summary.meta;
      const stat = document.createElement('div');
      stat.className = 'sp-summary-stat';
      stat.innerHTML = `
        <div class="sp-summary-stat-label">Core</div>
        <div class="sp-summary-stat-value">${escapeHtml(summary.core || '—')}</div>
        <div class="sp-summary-stat-sub">${[m.tissue, m.tma, m.split].filter(Boolean).map(escapeHtml).join(' · ') || '—'}</div>
      `;
      summaryWrap.appendChild(stat);

      if (typeof m.n_cells === 'number') {
        const cellsStat = document.createElement('div');
        cellsStat.className = 'sp-summary-stat';
        cellsStat.innerHTML = `
          <div class="sp-summary-stat-label">Cells segmented</div>
          <div class="sp-summary-stat-value">${formatInt(m.n_cells)}</div>
        `;
        summaryWrap.appendChild(cellsStat);
      }
    }

    // Dominant-phenotype block — read from corePheno if available.
    const cp = summary && summary.corePheno;
    if (cp && Array.isArray(cp.phenotypes) && cp.phenotypes.length > 0) {
      const ranked = [...cp.phenotypes]
        .filter((p) => p && p.name !== NO_CALL_KEY)
        .sort((a, b) => (b.frac_pred || 0) - (a.frac_pred || 0))
        .slice(0, 3);
      const totalForBar =
        ranked.reduce((acc, p) => Math.max(acc, p.frac_pred || 0), 0) || 1;
      if (ranked.length > 0) {
        const stat = document.createElement('div');
        stat.className = 'sp-summary-stat';
        const bars = ranked.map((p) => {
          const pct = ((p.frac_pred || 0) * 100).toFixed(1);
          const width = (((p.frac_pred || 0) / totalForBar) * 100).toFixed(1);
          return `
            <div class="sp-summary-bar-row">
              <span class="sp-summary-bar-name">${escapeHtml(p.name)}</span>
              <span class="sp-summary-bar-track">
                <span class="sp-summary-bar-fill" style="width:${width}%"></span>
              </span>
              <span class="sp-summary-bar-value">${pct}%</span>
            </div>
          `;
        }).join('');
        stat.innerHTML = `
          <div class="sp-summary-stat-label">Top phenotypes (predicted)</div>
          <div class="sp-summary-bars">${bars}</div>
        `;
        summaryWrap.appendChild(stat);
      }
    }

    if (summaryWrap.children.length > 0) root.appendChild(summaryWrap);
  }

  function renderLoading(cellId) {
    root.innerHTML = '';
    const h = document.createElement('div');
    h.className = 'side-panel-header';
    h.innerHTML = `
      <div class="sp-title">
        <span class="sp-title-id">Cell #${cellId}</span>
        <span class="sp-pheno-name">loading…</span>
      </div>
    `;
    root.appendChild(h);
    root.classList.add('side-panel-open');
  }

  function renderDetail(detail) {
    const compare = !!cb.getState().compare;
    const pals = cb.getPalettes();

    root.innerHTML = '';
    root.classList.add('side-panel-open');

    // --- header
    const header = document.createElement('div');
    header.className = 'side-panel-header';

    const title = document.createElement('div');
    title.className = 'sp-title';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'sp-close';
    closeBtn.setAttribute('aria-label', 'Close side panel');
    closeBtn.title = 'Close the cell detail panel and deselect the cell';
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', close);

    let dominantPheno = 'no_call';
    let dominantStratum = 'weak';
    for (const p of detail.phenotypes) {
      if (p.call) {
        dominantPheno = p.name;
        dominantStratum = p.confidence_stratum;
        break;
      }
    }
    const phenoIdx = PHENOTYPE_NAMES.indexOf(dominantPheno);
    const phenoSwatch = pals && pals.phenotypeHex
      ? pals.phenotypeHex[phenoIdx >= 0 ? phenoIdx : PHENOTYPE_NAMES.length]
      : '#888888';

    title.innerHTML =
      `<span class="sp-swatch" style="background:${phenoSwatch}" aria-hidden="true"></span>` +
      `<span class="sp-title-id">#${detail.cell_id}</span>` +
      `<span class="sp-pheno-name">${escapeHtml(dominantPheno)}</span>` +
      `<span class="sp-stratum sp-stratum-${escapeHtml(dominantStratum)}">${escapeHtml(dominantStratum)}</span>`;
    header.appendChild(title);
    header.appendChild(closeBtn);
    root.appendChild(header);

    // --- markers
    const markersTitle = document.createElement('div');
    markersTitle.className = 'sp-section-title';
    markersTitle.textContent = compare ? 'Markers · pred vs truth' : 'Markers';
    root.appendChild(markersTitle);

    if (compare) {
      const headerRow = document.createElement('div');
      headerRow.className = 'sp-markers-header';
      const hPred = document.createElement('div');
      hPred.className = 'sp-col-header';
      hPred.textContent = 'Prediction';
      const hTruth = document.createElement('div');
      hTruth.className = 'sp-col-header';
      hTruth.textContent = 'CODEX truth';
      headerRow.appendChild(hPred);
      headerRow.appendChild(hTruth);
      root.appendChild(headerRow);

      const markerBox = document.createElement('div');
      markerBox.className = 'sp-markers sp-markers-compare';
      for (const m of detail.markers) {
        markerBox.appendChild(renderMarkerCompareRow(m, pals));
      }
      root.appendChild(markerBox);
    } else {
      const markerBox = document.createElement('div');
      markerBox.className = 'sp-markers';
      for (const m of detail.markers) {
        markerBox.appendChild(renderMarkerRow(m, false, pals));
      }
      root.appendChild(markerBox);
    }

    // --- phenotype table
    const phenoTitle = document.createElement('div');
    phenoTitle.className = 'sp-section-title';
    phenoTitle.textContent = 'Phenotype scores';
    root.appendChild(phenoTitle);

    const table = document.createElement('table');
    table.className = 'sp-phenotype-table';
    const thead = document.createElement('thead');
    thead.innerHTML =
      '<tr><th>Name</th><th class="num">Score</th><th>Call</th><th>Truth</th></tr>';
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    for (const p of detail.phenotypes) {
      const tr = document.createElement('tr');
      tr.dataset.phenotype = p.name;
      const truthCell =
        p.label_truth === null || p.label_truth === undefined
          ? '<span class="sp-na">—</span>'
          : (p.label_truth ? '✓' : '');
      const scoreCell = p.score == null ? '<span class="sp-na">—</span>' : p.score.toFixed(3);
      tr.innerHTML =
        `<td>${escapeHtml(p.name)}</td>` +
        `<td class="num">${scoreCell}</td>` +
        `<td>${p.call ? '✓' : ''}</td>` +
        `<td>${truthCell}</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    root.appendChild(table);

    // --- morphology features grid
    const featsTitle = document.createElement('div');
    featsTitle.className = 'sp-section-title';
    featsTitle.textContent = 'Morphology';
    root.appendChild(featsTitle);

    const featsGrid = document.createElement('div');
    featsGrid.className = 'sp-features-grid';
    featsGrid.innerHTML = `
      <div class="sp-feature">
        <div class="sp-feature-label">Nucleus area</div>
        <div class="sp-feature-value">${formatInt(detail.features.area_nucleus_px)} px²</div>
      </div>
      <div class="sp-feature">
        <div class="sp-feature-label">Expanded area</div>
        <div class="sp-feature-value">${formatInt(detail.features.area_expanded_px)} px²</div>
      </div>
      <div class="sp-feature">
        <div class="sp-feature-label">Eccentricity</div>
        <div class="sp-feature-value">${detail.features.eccentricity.toFixed(2)}</div>
      </div>
      <div class="sp-feature">
        <div class="sp-feature-label">Compactness</div>
        <div class="sp-feature-value">${(detail.features.area_expanded_px / Math.max(1, detail.features.area_nucleus_px)).toFixed(2)}×</div>
      </div>
    `;
    root.appendChild(featsGrid);

    // --- footer
    const footer = document.createElement('div');
    footer.className = 'sp-footer';
    const link = document.createElement('a');
    link.href = reportUrl(core);
    link.target = '_blank';
    link.rel = 'noopener';
    link.className = 'sp-report-link';
    link.title =
      'Open the rendered per-core PDF report in a new tab (run '
      + '`hexif report` to build it if missing)';
    link.textContent = '📄 Report PDF';
    footer.appendChild(link);
    root.appendChild(footer);
  }

  function renderMarkerRow(m, _compare, pals) {
    const row = document.createElement('div');
    row.className = 'sp-marker-row';
    row.dataset.marker = m.name;

    const name = document.createElement('div');
    name.className = 'sp-marker-name';
    name.textContent = m.name;
    row.appendChild(name);

    if (m.pred_prob === null || m.pred_prob === undefined) {
      row.classList.add('sp-marker-row-na');
      const naCell = document.createElement('div');
      naCell.className = 'sp-marker-na';
      naCell.textContent = '—';
      naCell.title = 'n/a — this model does not predict this marker';
      row.appendChild(naCell);
      return row;
    }

    const bar = document.createElement('div');
    bar.className = 'sp-marker-bar';
    const fill = document.createElement('div');
    fill.className = 'sp-marker-fill';
    const prob = Math.max(0, Math.min(1, m.pred_prob));
    fill.style.width = `${(prob * 100).toFixed(1)}%`;
    if (pals) {
      const rgb = resolveColor(pals, 'marker_pred', prob);
      fill.style.background = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }
    bar.appendChild(fill);
    if (m.truth_pos !== null && m.truth_pos !== undefined) {
      const tick = document.createElement('div');
      tick.className = 'sp-marker-tick';
      tick.style.left = m.truth_pos ? 'calc(100% - 2px)' : '0%';
      bar.appendChild(tick);
    }
    row.appendChild(bar);

    const value = document.createElement('div');
    value.className = 'sp-marker-value';
    value.textContent = prob.toFixed(2);
    row.appendChild(value);

    return row;
  }

  function renderMarkerCompareRow(m, pals) {
    const row = document.createElement('div');
    row.className = 'sp-marker-compare-row';
    row.dataset.marker = m.name;

    // left: prediction
    const left = document.createElement('div');
    left.className = 'sp-marker-row sp-marker-pred-cell';
    const lname = document.createElement('div');
    lname.className = 'sp-marker-name';
    lname.textContent = m.name;
    left.appendChild(lname);

    if (m.pred_prob === null || m.pred_prob === undefined) {
      left.classList.add('sp-marker-row-na');
      const lna = document.createElement('div');
      lna.className = 'sp-marker-na';
      lna.textContent = '—';
      lna.title = 'n/a — this model does not predict this marker';
      left.appendChild(lna);
    } else {
      const lbar = document.createElement('div');
      lbar.className = 'sp-marker-bar';
      const lfill = document.createElement('div');
      lfill.className = 'sp-marker-fill';
      const prob = Math.max(0, Math.min(1, m.pred_prob));
      lfill.style.width = `${(prob * 100).toFixed(1)}%`;
      if (pals) {
        const rgb = resolveColor(pals, 'marker_pred', prob);
        lfill.style.background = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      }
      lbar.appendChild(lfill);
      left.appendChild(lbar);
      const lval = document.createElement('div');
      lval.className = 'sp-marker-value sp-pred';
      lval.textContent = prob.toFixed(2);
      left.appendChild(lval);
    }
    row.appendChild(left);

    // right: truth (or n/a)
    const right = document.createElement('div');
    right.className = 'sp-marker-row sp-marker-truth-cell';
    const rname = document.createElement('div');
    rname.className = 'sp-marker-name';
    rname.textContent = m.name;
    right.appendChild(rname);

    if (m.truth_pos === null || m.truth_pos === undefined) {
      const naCell = document.createElement('div');
      naCell.className = 'sp-marker-na';
      naCell.textContent = 'n/a';
      right.appendChild(naCell);
    } else {
      const rbar = document.createElement('div');
      rbar.className = 'sp-marker-bar sp-marker-bar-truth';
      const rfill = document.createElement('div');
      rfill.className = 'sp-marker-fill sp-marker-fill-truth';
      rfill.style.width = m.truth_pos ? '100%' : '0%';
      rbar.appendChild(rfill);
      right.appendChild(rbar);
      const rval = document.createElement('div');
      rval.className = 'sp-marker-value sp-truth';
      rval.textContent = m.truth_pos ? '1' : '0';
      right.appendChild(rval);
    }
    row.appendChild(right);
    return row;
  }

  async function load(cellId) {
    if (inflight) inflight.abort();
    inflight = new AbortController();
    renderLoading(cellId);
    try {
      const stateNow = cb.getState();
      const detail = await fetchCellDetail(core, cellId, {
        model: stateNow.model,
        signal: inflight.signal,
      });
      const cur = cb.getState();
      if (cur.selected !== cellId) return;
      renderDetail(detail);
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      const errEl = document.createElement('div');
      errEl.className = 'sp-error';
      errEl.textContent = `Failed to load cell #${cellId}: ${err.message || err}`;
      root.innerHTML = '';
      root.appendChild(errEl);
      console.warn(err);
    }
  }

  return {
    show(cellId) {
      currentCellId = cellId;
      load(cellId);
    },
    hide() {
      if (inflight) inflight.abort();
      currentCellId = null;
      renderEmpty(null);
    },
    showEmpty(summary) {
      if (inflight) inflight.abort();
      currentCellId = null;
      renderEmpty(summary || null);
    },
    close,
    get cellId() {
      return currentCellId;
    },
  };
}

/** Probe: does the core appear to have CODEX truth columns? */
export function hasTruthFromCorePhenotypes(corePheno) {
  if (!corePheno || !Array.isArray(corePheno.markers)) return false;
  for (const m of corePheno.markers) {
    if (m.frac_truth !== null && m.frac_truth !== undefined) return true;
  }
  return false;
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

function formatInt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Math.round(Number(n)).toLocaleString();
}

// Re-export for tests / debug overlays.
export const _MARKERS = VALID_MARKERS;
