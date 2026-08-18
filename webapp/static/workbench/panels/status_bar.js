// Bottom status strip. Reads transient runtime metrics (zoom, viewport
// cell count, selection) and paints a single-line strip beneath the
// viewer. The strip is read-only — every interaction lives elsewhere
// (sidebars, viewer chrome, shortcut keys).

/**
 * @typedef {Object} StatusBarHandle
 * @property {() => void} update      Re-paint from the supplied getters.
 * @property {() => void} destroy     Detach event listeners (idempotent).
 */

/**
 * @typedef {Object} StatusBarSources
 * @property {() => any} getState
 * @property {() => number} getZoom            OSD viewport.getZoom(true) for the image-zoom value.
 * @property {() => number} getCellCount       Total polygons currently bound to the canvas overlay.
 * @property {() => number} getVisibleCount    Polygons painted at the last frame (viewport-culled).
 * @property {() => string|null} getModelLabel Human-readable model name to surface.
 */

/**
 * Mount the status bar into `root`.
 *
 * @param {HTMLElement} root
 * @param {StatusBarSources} sources
 * @returns {StatusBarHandle}
 */
export function mountStatusBar(root, sources) {
  root.classList.add('statusbar');
  root.setAttribute('role', 'contentinfo');
  root.innerHTML = `
    <span class="statusbar-item" data-role="status-model">
      <span class="sb-label">Model</span>
      <span class="sb-value" data-role="status-model-value">—</span>
    </span>
    <span class="statusbar-sep">·</span>
    <span class="statusbar-item" data-role="status-zoom">
      <span class="sb-label">Zoom</span>
      <span class="sb-value" data-role="status-zoom-value">1.0×</span>
    </span>
    <span class="statusbar-sep">·</span>
    <span class="statusbar-item" data-role="status-cellcount">
      <span class="sb-label">Cells</span>
      <span class="sb-value" data-role="status-cellcount-value">0</span>
    </span>
    <span class="statusbar-sep">·</span>
    <span class="statusbar-item" data-role="status-selection">
      <span class="sb-label">Selection</span>
      <span class="sb-value" data-role="status-selection-value">—</span>
    </span>
    <span class="statusbar-spacer"></span>
    <span class="statusbar-item statusbar-item--hint">
      Press <span class="kbd">?</span> for shortcuts
    </span>
  `;

  const modelEl     = root.querySelector('[data-role="status-model-value"]');
  const zoomEl      = root.querySelector('[data-role="status-zoom-value"]');
  const cellsEl     = root.querySelector('[data-role="status-cellcount-value"]');
  const selectionEl = root.querySelector('[data-role="status-selection-value"]');

  function update() {
    const state = sources.getState() || {};
    const zoom = safeNumber(sources.getZoom());
    const cells = safeInt(sources.getCellCount());
    const visible = safeInt(sources.getVisibleCount());
    const modelLabel = sources.getModelLabel();

    modelEl.textContent = modelLabel || state.model || '—';
    zoomEl.textContent = zoom === null ? '—' : `${zoom.toFixed(2)}×`;
    cellsEl.textContent =
      cells === null ? '—' : `${formatInt(visible || 0)} / ${formatInt(cells)}`;
    selectionEl.textContent =
      state.selected === null || state.selected === undefined
        ? 'none'
        : `#${state.selected}`;
  }

  // Initial paint.
  update();

  return {
    update,
    destroy() {
      root.innerHTML = '';
    },
  };
}

function safeNumber(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return null;
  return Number(n);
}
function safeInt(n) {
  const v = safeNumber(n);
  return v === null ? null : Math.floor(v);
}
function formatInt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toLocaleString();
}
