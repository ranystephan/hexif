// Bootstraps the workbench: OpenSeadragon viewer, URL state machine,
// overlays, and every panel (top bar, controls, side panel, legend,
// stats, status bar, shortcuts, help).
//
// Single-writer rule: every state mutation goes through updateState(...).
// The function de-duplicates against the current state, persists to the
// URL, and triggers the side-effects (re-fetch GeoJSON / re-render side
// panel / repaint overlay).

import {
  parseUrlToState,
  applyState,
  stateEquals,
  fetchKey,
  DEFAULT_MARKER,
} from './state.js';
import {
  fetchCores,
  fetchCellsGeoJson,
  fetchCorePhenotypes,
  fetchModels,
  fetchCodexInfo,
  fetchStrata,
  fetchStatsValidation,
  heDziUrl,
  codexDziUrl,
  codexMarkerDziUrl,
} from './api.js';
import { loadPalettes } from './palette.js';
import { attachCanvasOverlay } from './overlays/canvas_overlay.js';
import { attachSvgOverlay } from './overlays/svg_overlay.js';
import { buildHitIndex, hitTest } from './overlays/hit_test.js';
import { mountControls } from './panels/controls.js';
import { mountSidePanel, hasTruthFromCorePhenotypes } from './panels/side_panel.js';
import { renderLegend } from './panels/legend.js';
import { mountHelpPanel } from './panels/help_panel.js';
import { mountStatsPanel } from './panels/stats_panel.js';
import { mountAppBar } from './panels/app_bar.js';
import { mountStatusBar } from './panels/status_bar.js';
import { mountShortcutsModal, installKeyBindings } from './panels/shortcuts.js';

// localStorage keys for sidebar layout. v2 adds user-resized widths
// while still reading v1 open/closed state as a migration path.
const LS_LAYOUT = 'hexif.workbench.layout.v2';
const LS_LAYOUT_V1 = 'hexif.workbench.layout.v1';
// commit 5 — theme preference key. Stored as a plain string 'light' /
// 'dark' so the change is one localStorage round trip. Default = light
// unless the OS reports prefers-color-scheme: dark on first paint.
const LS_THEME = 'hexif.workbench.theme';
const LAYOUT_DEFAULTS = {
  left: true,
  right: true,
  leftWidth: 340,
  rightWidth: 440,
};
const LAYOUT_LIMITS = {
  leftMin: 280,
  leftMax: 560,
  rightMin: 340,
  rightMax: 680,
  viewerMin: 360,
};

function readLayoutPrefs() {
  try {
    const raw = localStorage.getItem(LS_LAYOUT) || localStorage.getItem(LS_LAYOUT_V1);
    if (!raw) return { ...LAYOUT_DEFAULTS };
    const parsed = JSON.parse(raw);
    return {
      left: parsed.left !== false,
      right: parsed.right !== false,
      leftWidth: clampWidth('left', parsed.leftWidth || parsed.left_width || LAYOUT_DEFAULTS.leftWidth),
      rightWidth: clampWidth('right', parsed.rightWidth || parsed.right_width || LAYOUT_DEFAULTS.rightWidth),
    };
  } catch {
    return { ...LAYOUT_DEFAULTS };
  }
}

function writeLayoutPrefs(prefs) {
  try {
    localStorage.setItem(LS_LAYOUT, JSON.stringify(prefs));
  } catch {
    // SecurityError / quota — ignore. Sidebar prefs are nice-to-have.
  }
}

function readThemePref() {
  try {
    const v = localStorage.getItem(LS_THEME);
    if (v === 'dark' || v === 'light') return v;
  } catch {
    // SecurityError / private mode — fall through to OS preference.
  }
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }
  return 'light';
}

function writeThemePref(theme) {
  try {
    localStorage.setItem(LS_THEME, theme);
  } catch {
    // Same fallback as writeLayoutPrefs — preference is best-effort.
  }
}

function applyThemeToBody(theme) {
  if (typeof document === 'undefined') return;
  if (theme === 'dark') {
    document.body.setAttribute('data-theme', 'dark');
  } else {
    document.body.removeAttribute('data-theme');
  }
}

function clampWidth(side, value) {
  const n = Number(value);
  const fallback = side === 'left' ? LAYOUT_DEFAULTS.leftWidth : LAYOUT_DEFAULTS.rightWidth;
  if (!Number.isFinite(n)) return fallback;
  const min = side === 'left' ? LAYOUT_LIMITS.leftMin : LAYOUT_LIMITS.rightMin;
  const max = side === 'left' ? LAYOUT_LIMITS.leftMax : LAYOUT_LIMITS.rightMax;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function clampLayoutToViewport(layout) {
  const viewport = Math.max(760, window.innerWidth || 1280);
  const openCount = (layout.left ? 1 : 0) + (layout.right ? 1 : 0);
  const chrome = openCount * 8 + LAYOUT_LIMITS.viewerMin;
  const budget = Math.max(0, viewport - chrome);
  let leftWidth = clampWidth('left', layout.leftWidth);
  let rightWidth = clampWidth('right', layout.rightWidth);

  if (layout.left && layout.right && leftWidth + rightWidth > budget) {
    const total = leftWidth + rightWidth;
    const leftMin = Math.min(LAYOUT_LIMITS.leftMin, 220);
    const rightMin = Math.min(LAYOUT_LIMITS.rightMin, 260);
    leftWidth = Math.max(leftMin, Math.min(LAYOUT_LIMITS.leftMax, Math.round((leftWidth / total) * budget)));
    rightWidth = Math.max(rightMin, Math.min(LAYOUT_LIMITS.rightMax, Math.round(budget - leftWidth)));
    if (leftWidth + rightWidth > budget) {
      leftWidth = Math.max(leftMin, budget - rightWidth);
    }
  } else if (layout.left && leftWidth > budget) {
    leftWidth = Math.max(220, Math.min(LAYOUT_LIMITS.leftMax, budget));
  } else if (layout.right && rightWidth > budget) {
    rightWidth = Math.max(260, Math.min(LAYOUT_LIMITS.rightMax, budget));
  }

  return { ...layout, leftWidth, rightWidth };
}

// Globals exposed to tests / debug. The Playwright smoke test polls
// window.__osdReady; window.__hexifState exposes the current state so
// the test can synchronize without DOM scraping.
function exposeGlobal(key, value) {
  window.__hexif = window.__hexif || {};
  window.__hexif[key] = value;
}

async function main() {
  const root = document.getElementById('app');
  if (!root) {
    console.error('main: missing #app root');
    return;
  }
  root.innerHTML = `
    <header id="topbar" class="topbar" role="banner"></header>
    <div id="app-main" class="app-main" data-left-open="true" data-right-open="true">
      <div id="controls" class="controls" role="region" aria-label="Controls"></div>
      <div class="sidebar-resizer" data-side="left" role="separator" aria-orientation="vertical"
           aria-controls="controls" aria-label="Resize controls panel" tabindex="0"></div>
      <div id="viewer-area" class="viewer-area" data-split="false">
        <div id="viewer-panes" class="viewer-panes">
          <div id="osd-viewport" class="osd-viewport osd-pane osd-pane-primary"
               data-stain="he" tabindex="0"></div>
          <div id="osd-viewport-2" class="osd-viewport osd-pane osd-pane-secondary"
               data-stain="codex" hidden></div>
        </div>
        <div id="codex-badge" class="codex-badge" hidden></div>
        <div id="hover-tooltip" class="hover-tooltip" role="tooltip" hidden></div>
        <div id="legend" class="legend"></div>
      </div>
      <div class="sidebar-resizer" data-side="right" role="separator" aria-orientation="vertical"
           aria-controls="side-panel" aria-label="Resize cell detail panel" tabindex="0"></div>
      <div id="side-panel" class="side-panel" role="region" aria-label="Cell detail"></div>
    </div>
    <footer id="statusbar" class="statusbar" role="contentinfo"></footer>
  `;

  const appMain = document.getElementById('app-main');
  let layoutPrefs = clampLayoutToViewport(readLayoutPrefs());
  appMain.dataset.leftOpen = String(layoutPrefs.left);
  appMain.dataset.rightOpen = String(layoutPrefs.right);
  applyLayoutPrefs();
  installSidebarResizers();

  // commit 5 — initialize theme as early as possible to avoid the
  // "white flash" when the user has dark preference set.
  let theme = readThemePref();
  applyThemeToBody(theme);

  let state = parseUrlToState(window.location.href);
  let palettes;
  let cores = [];
  let corePheno = null;
  /** @type {AbortController|null} */
  let geoInflight = null;
  /** @type {Map<string, Object>} */
  const geoCache = new Map();
  /** @type {Object|null} */
  let currentGeo = null;
  let hitIndex = null;
  let viewer = null;
  let canvasOverlay = null;
  let svgOverlay = null;
  // commit 8 — secondary viewer for side-by-side split mode. Lazily
  // created the first time the user enters base=codex or base=split.
  // Once instantiated, it lives for the rest of the session; we just
  // hide its pane in base=he. Pan/zoom sync handlers attach only in
  // split (both panes visible).
  let viewer2 = null;
  let canvasOverlay2 = null;
  let svgOverlay2 = null;
  /** @type {(() => void)|null} */
  let detachSyncFn = null;
  let sidePanel = null;
  let expandedOnlyAtLowZoom = true;
  // Transient UI state — which CODEX channel the base layer shows when
  // base === 'codex'. ``null`` = fixed DAPI/CD45/panCK composite (the
  // historical current release behavior); a marker name (e.g. 'Ki67') = raw
  // single-channel CODEX staining at full resolution. Not persisted to
  // the URL in commit 2; commit 3 unifies it with the marker selector.
  let codexLayer = null;
  // commit 4 — Validation AP cache keyed by model id. Populated lazily
  // the first time the legend or controls panel asks; the payload is
  // small (~12 markers + 9 phenotypes per model) so we keep it
  // in-memory rather than IndexedDB.
  /** @type {Map<string, Object>} */
  const validationCache = new Map();
  /** @type {Set<string>} */
  const validationInflight = new Set();

  // Boot: kick off palette + core list + model list fetches in parallel.
  let modelList = null;
  let strata = null;
  try {
    [palettes, { cores }, modelList, strata] = await Promise.all([
      loadPalettes(),
      fetchCores(),
      fetchModels().catch((err) => {
        console.warn('fetchModels failed; falling back to single-model UI:', err);
        return null;
      }),
      fetchStrata().catch((err) => {
        console.warn('fetchStrata failed; legend stratum chips disabled:', err);
        return null;
      }),
    ]);
  } catch (err) {
    showFatal(`Failed to load palettes or cores: ${err.message || err}`);
    throw err;
  }
  exposeGlobal('strata', strata);
  exposeGlobal('palettes', palettes);
  exposeGlobal('cores', cores);
  // Bootstrap state.model from the manifest's default if the URL didn't pin one.
  const urlHasModel = new URLSearchParams(window.location.search).has('model');
  if (!urlHasModel && modelList && modelList.default_id && state.model !== modelList.default_id) {
    state = { ...state, model: modelList.default_id };
    applyState(state, { replace: true });
  }
  exposeGlobal('models', modelList);

  // Default to A-10 (canonical demo core) if the URL didn't pin one.
  if (!state.core) {
    const preferred = cores.find((c) => c.basename === 'ccRCC_TMA1__A-10');
    state = {
      ...state,
      core: preferred ? preferred.basename : (cores[0] && cores[0].basename) || null,
    };
    applyState(state, { replace: true });
  }
  if (!state.core) {
    showFatal('No cores in bundle.');
    return;
  }

  // Probe the core's truth availability (gates the comparison toggle).
  try {
    corePheno = await fetchCorePhenotypes(state.core);
  } catch (err) {
    console.warn('phenotypes probe failed; comparison toggle disabled:', err);
    corePheno = null;
  }

  // CODEX composite — probe paired CODEX availability for the core.
  let codexInfo = null;
  try {
    codexInfo = await fetchCodexInfo(state.core);
  } catch (err) {
    console.warn('codex info probe failed; CODEX base disabled:', err);
    codexInfo = { core: state.core, has_composite: false, channels: [], image_size: [0, 0] };
  }
  if (state.base === 'codex' && !codexInfo.has_composite) {
    state = { ...state, base: 'he' };
    applyState(state, { replace: true });
  }

  // Mount the help panel + shortcuts modal first; downstream components
  // wire deep-links and shortcut handlers against them.
  const helpPanel = mountHelpPanel(document.body);
  exposeGlobal('helpPanel', helpPanel);
  const shortcutsModal = mountShortcutsModal(document.body);
  exposeGlobal('shortcutsModal', shortcutsModal);

  // --- Top app bar ----------------------------------------------------
  const topbarRoot = document.getElementById('topbar');
  const appBar = mountAppBar(topbarRoot, {
    getState: () => state,
    getCores: () => cores,
    isLeftOpen: () => appMain.dataset.leftOpen === 'true',
    isRightOpen: () => appMain.dataset.rightOpen === 'true',
    onToggleLeft: () => toggleSidebar('left'),
    onToggleRight: () => toggleSidebar('right'),
    onOpenHelp: () => {
      if (helpPanel.isOpen()) helpPanel.close();
      else void helpPanel.openTo('overview');
    },
    onOpenShortcuts: () => shortcutsModal.open(),
    isDark: () => theme === 'dark',
    onToggleTheme: () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      applyThemeToBody(theme);
      writeThemePref(theme);
    },
  });
  exposeGlobal('appBar', appBar);

  // --- Status bar -----------------------------------------------------
  const statusRoot = document.getElementById('statusbar');
  const statusBar = mountStatusBar(statusRoot, {
    getState: () => state,
    getZoom: () => {
      if (!viewer || !viewer.viewport) return 0;
      try {
        return viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
      } catch {
        return 0;
      }
    },
    getCellCount: () => (canvasOverlay ? canvasOverlay.featureCount : 0),
    getVisibleCount: () => (canvasOverlay ? canvasOverlay.getVisibleCount() : 0),
    getModelLabel: () => {
      if (!modelList || !Array.isArray(modelList.models)) return state.model || null;
      const entry = modelList.models.find((m) => m.id === state.model);
      return (entry && entry.name) || state.model || null;
    },
  });
  exposeGlobal('statusBar', statusBar);

  // --- Side panel (right) --------------------------------------------
  const sidePanelRoot = document.getElementById('side-panel');
  sidePanel = mountSidePanel(sidePanelRoot, state.core, {
    getState: () => state,
    onStateChange: (next) => updateState(next),
    getPalettes: () => palettes,
    getCorePhenotypes: () => corePheno,
  });
  exposeGlobal('sidePanel', sidePanel);

  // --- Controls (left): tabs + cells view + stats view ----------------
  const controlsRoot = document.getElementById('controls');
  controlsRoot.dataset.tab = 'cells';
  const controlsTabs = document.createElement('div');
  controlsTabs.className = 'controls-tabs';
  controlsTabs.setAttribute('role', 'tablist');
  controlsTabs.innerHTML = `
    <button type="button" class="controls-tab" role="tab" aria-selected="true"
            data-target-tab="cells">Cells</button>
    <button type="button" class="controls-tab" role="tab" aria-selected="false"
            data-target-tab="stats">Stats</button>
  `;
  controlsRoot.appendChild(controlsTabs);

  const cellsView = document.createElement('div');
  cellsView.className = 'controls-cell-view';
  controlsRoot.appendChild(cellsView);

  const statsRoot = document.createElement('div');
  statsRoot.className = 'stats-panel-root';
  controlsRoot.appendChild(statsRoot);
  const statsPanel = mountStatsPanel(statsRoot, {
    getState: () => state,
    getModels: () => modelList,
    isActive: () => controlsRoot.dataset.tab === 'stats',
  });
  exposeGlobal('statsPanel', statsPanel);

  let statsLoadedOnce = false;
  controlsTabs.querySelectorAll('button[data-target-tab]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.targetTab;
      controlsRoot.dataset.tab = target;
      for (const b of controlsTabs.querySelectorAll('button')) {
        b.setAttribute('aria-selected', String(b.dataset.targetTab === target));
      }
      if (target === 'stats') {
        if (!statsLoadedOnce) {
          statsLoadedOnce = true;
          void statsPanel.refresh();
        } else {
          void statsPanel.refreshPerCore();
          void statsPanel.refreshValidation();
        }
      }
    });
  });

  const controls = mountControls(cellsView, cores, {
    getState: () => state,
    onStateChange: (next) => updateState(next),
    getExpandedOnlyAtLowZoom: () => expandedOnlyAtLowZoom,
    setExpandedOnlyAtLowZoom: (b) => {
      expandedOnlyAtLowZoom = !!b;
      forEachCanvasOverlay((o) => o.setExpandedOnlyAtLowZoom(b));
    },
    hasTruth: () => hasTruthFromCorePhenotypes(corePheno),
    getModels: () => modelList,
    getValidationAp: getValidationAp,
    hasCodex: () => Boolean(codexInfo && codexInfo.has_composite),
    getCodexMarkers: () => (codexInfo && codexInfo.available_markers) || [],
    getCodexLayer: () => codexLayer,
    setCodexLayer: (name) => {
      const next = name && _isAvailableCodexMarker(name) ? name : null;
      if (next === codexLayer) return;
      codexLayer = next;
      // If a CODEX pane is currently visible (single or split), re-open
      // it with the new per-marker (or composite) pyramid. The primary
      // pane always shows H&E so we never call viewer.open on it.
      if ((state.base === 'codex' || state.base === 'split') && viewer2) {
        viewer2.open(secondaryTileSource());
      }
      updateCodexBadge();
    },
    getHelpPanel: () => helpPanel,
  });

  // --- OpenSeadragon (primary = H&E) ----------------------------------
  viewer = window.OpenSeadragon({
    element: document.getElementById('osd-viewport'),
    tileSources: primaryTileSource(),
    prefixUrl: '/static/workbench/vendor/openseadragon/images/',
    showNavigationControl: true,
    showFullPageControl: false,
    showHomeControl: true,
    showZoomControl: true,
    maxZoomPixelRatio: 4,
    immediateRender: true,
    crossOriginPolicy: 'Anonymous',
    gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
  });
  exposeGlobal('viewer', viewer);

  viewer.addHandler('open', () => {
    exposeGlobal('osdReady', true);
    window.__osdReady = true;
    statusBar.update();
  });
  // Refresh the status bar zoom readout while the user pans/zooms — the
  // handler fires on every animation frame, throttled by rAF inside
  // OSD, so the cost is negligible.
  viewer.addHandler('animation', () => statusBar.update());

  canvasOverlay = attachCanvasOverlay(viewer);
  canvasOverlay.setPalettes(palettes);
  canvasOverlay.setColorBy(state.color_by);
  canvasOverlay.setExpandedOnlyAtLowZoom(expandedOnlyAtLowZoom);
  canvasOverlay.setPhenotypeFilter(state.phenotypes);
  canvasOverlay.setProbabilityThreshold(state.min_prob || 0);
  exposeGlobal('canvasOverlay', canvasOverlay);

  svgOverlay = attachSvgOverlay(viewer);
  exposeGlobal('svgOverlay', svgOverlay);

  const legendRoot = document.getElementById('legend');
  renderLegend(legendRoot, state.color_by, state.marker, palettes, strata, helpPanel, getValidationAp);

  attachMouseHandlers();

  await refreshGeoJson();
  statusBar.update();
  // Bootstrap the CODEX layer to match the URL's marker selection so
  // ``?core=X&color_by=marker_truth&marker=Ki67&base=codex`` opens with
  // Ki67 staining painted underneath, not the composite.
  autoSyncCodexLayer();
  // commit 8 — apply the pane layout for the URL's `base`. This may
  // create the secondary viewer (lazily) and attach pan/zoom sync if
  // the URL pinned base=split.
  applyBaseLayout();
  updateCodexBadge();

  // Show the side-panel default state immediately so the right rail
  // never reads as broken / empty on first paint.
  if (state.selected !== null && state.selected !== undefined) {
    sidePanel.show(state.selected);
  } else {
    sidePanel.showEmpty(buildSidePanelSummary());
  }

  // --- Global keyboard shortcuts -------------------------------------
  installKeyBindings({
    toggleLeft:  () => toggleSidebar('left'),
    toggleRight: () => toggleSidebar('right'),
    openHelp:    () => {
      if (helpPanel.isOpen()) helpPanel.close();
      else void helpPanel.openTo('overview');
    },
    openShortcuts:   () => shortcutsModal.open(),
    clearSelection:  () => {
      if (state.selected !== null && state.selected !== undefined) {
        updateState({ ...state, selected: null });
      }
    },
    setColorBy: (v) => {
      // commit 3 — marker is preserved across modes so the CODEX base
      // and side-panel marker bars stay coherent across Focus toggles.
      const next = { ...state, color_by: v };
      if (v !== 'phenotype' && !state.marker) next.marker = DEFAULT_MARKER;
      // Switching to a marker view also clears any leftover phenotype
      // filter — the chips disappear in marker focus so a silent
      // filter would be hidden state.
      if (v !== 'phenotype') next.phenotypes = null;
      updateState(next);
      controls.rerender();
    },
    setMode: () => {
      const next = { ...state, mode: state.mode === 'nucleus' ? 'expanded' : 'nucleus' };
      updateState(next);
      controls.rerender();
    },
    toggleCompare: () => {
      if (!hasTruthFromCorePhenotypes(corePheno)) return;
      updateState({ ...state, compare: !state.compare });
      controls.rerender();
    },
    toggleTheme: () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      applyThemeToBody(theme);
      writeThemePref(theme);
      appBar.rerender();
    },
  });

  // ------------------------------------------------------------------
  function toggleSidebar(which) {
    const key = which === 'left' ? 'leftOpen' : 'rightOpen';
    const cur = appMain.dataset[key] !== 'false';
    appMain.dataset[key] = String(!cur);
    layoutPrefs = {
      ...layoutPrefs,
      left:  appMain.dataset.leftOpen  !== 'false',
      right: appMain.dataset.rightOpen !== 'false',
    };
    persistLayout();
    appBar.rerender();
    scheduleViewerResize(220);
  }

  function applyLayoutPrefs() {
    const next = clampLayoutToViewport(layoutPrefs);
    layoutPrefs = next;
    appMain.style.setProperty('--w-left', `${next.leftWidth}px`);
    appMain.style.setProperty('--w-right', `${next.rightWidth}px`);
    for (const handle of appMain.querySelectorAll('.sidebar-resizer')) {
      const side = handle.dataset.side;
      const width = side === 'left' ? next.leftWidth : next.rightWidth;
      handle.setAttribute('aria-valuemin', String(side === 'left' ? LAYOUT_LIMITS.leftMin : LAYOUT_LIMITS.rightMin));
      handle.setAttribute('aria-valuemax', String(side === 'left' ? LAYOUT_LIMITS.leftMax : LAYOUT_LIMITS.rightMax));
      handle.setAttribute('aria-valuenow', String(width));
      handle.title = `${side === 'left' ? 'Controls' : 'Cell detail'} width: ${width}px. Drag or use arrow keys.`;
    }
  }

  function persistLayout() {
    applyLayoutPrefs();
    writeLayoutPrefs({
      left: layoutPrefs.left,
      right: layoutPrefs.right,
      leftWidth: layoutPrefs.leftWidth,
      rightWidth: layoutPrefs.rightWidth,
    });
  }

  function scheduleViewerResize(delay = 40) {
    setTimeout(() => {
      if (viewer && typeof viewer.forceResize === 'function') viewer.forceResize();
      if (viewer2 && typeof viewer2.forceResize === 'function') viewer2.forceResize();
      // Triggers a clear + repaint at the new container size on both panes.
      forEachCanvasOverlay((o) => o.setColorBy(state.color_by));
      statusBar.update();
    }, delay);
  }

  function installSidebarResizers() {
    const handles = appMain.querySelectorAll('.sidebar-resizer');
    for (const handle of handles) {
      const side = handle.dataset.side;
      handle.addEventListener('pointerdown', (ev) => beginResize(ev, side, handle));
      handle.addEventListener('keydown', (ev) => {
        const step = ev.shiftKey ? 40 : 16;
        let delta = 0;
        if (ev.key === 'ArrowLeft') delta = side === 'left' ? -step : step;
        else if (ev.key === 'ArrowRight') delta = side === 'left' ? step : -step;
        else if (ev.key === 'Home') {
          setSidebarWidth(side, side === 'left' ? LAYOUT_LIMITS.leftMin : LAYOUT_LIMITS.rightMin);
          ev.preventDefault();
          return;
        } else if (ev.key === 'End') {
          setSidebarWidth(side, side === 'left' ? LAYOUT_LIMITS.leftMax : LAYOUT_LIMITS.rightMax);
          ev.preventDefault();
          return;
        } else {
          return;
        }
        setSidebarWidth(side, layoutPrefs[side === 'left' ? 'leftWidth' : 'rightWidth'] + delta);
        ev.preventDefault();
      });
    }

    window.addEventListener('resize', () => {
      persistLayout();
      scheduleViewerResize(60);
    });
  }

  function beginResize(ev, side, handle) {
    if (ev.button !== 0) return;
    if ((side === 'left' && appMain.dataset.leftOpen === 'false') ||
        (side === 'right' && appMain.dataset.rightOpen === 'false')) {
      return;
    }
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    appMain.classList.add('is-resizing');
    const startX = ev.clientX;
    const startWidth = side === 'left' ? layoutPrefs.leftWidth : layoutPrefs.rightWidth;

    function onMove(moveEv) {
      const dx = moveEv.clientX - startX;
      const nextWidth = side === 'left' ? startWidth + dx : startWidth - dx;
      setSidebarWidth(side, nextWidth, false);
    }
    function onUp(upEv) {
      handle.releasePointerCapture(upEv.pointerId);
      appMain.classList.remove('is-resizing');
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
      persistLayout();
      scheduleViewerResize(40);
    }

    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  }

  function setSidebarWidth(side, width, shouldPersist = true) {
    if (side === 'left') layoutPrefs = { ...layoutPrefs, leftWidth: clampWidth('left', width) };
    else layoutPrefs = { ...layoutPrefs, rightWidth: clampWidth('right', width) };
    applyLayoutPrefs();
    if (shouldPersist) persistLayout();
    scheduleViewerResize(20);
  }

  // ------------------------------------------------------------------
  function buildSidePanelSummary() {
    // Cheap synthesis from already-loaded data so the empty state has
    // something useful to show. We pull phenotype distribution + core
    // metadata from corePheno (the /api/phenotypes/{core} payload).
    const meta = (cores || []).find((c) => c.basename === state.core) || null;
    return {
      core: state.core,
      meta,
      corePheno,
    };
  }

  async function refreshGeoJson() {
    const key = fetchKey(state);
    if (geoCache.has(key)) {
      const geo = geoCache.get(key);
      currentGeo = geo;
      hitIndex = buildHitIndex(geo);
      forEachCanvasOverlay((o) => o.setGeoJson(geo));
      exposeGlobal('geoJson', geo);
      exposeGlobal('cellCount', (geo.features || []).length);
      statusBar.update();
      return;
    }
    if (geoInflight) geoInflight.abort();
    geoInflight = new AbortController();
    try {
      const geo = await fetchCellsGeoJson(state.core, {
        mode: state.mode,
        color_by: state.color_by,
        marker: state.marker,
        model: state.model,
        signal: geoInflight.signal,
      });
      geoCache.set(key, geo);
      currentGeo = geo;
      hitIndex = buildHitIndex(geo);
      forEachCanvasOverlay((o) => o.setGeoJson(geo));
      exposeGlobal('geoJson', geo);
      exposeGlobal('cellCount', (geo.features || []).length);
      statusBar.update();
      // The summary card on the right pane includes a cell-count line —
      // refresh it whenever new geo lands.
      if (state.selected === null || state.selected === undefined) {
        sidePanel.showEmpty(buildSidePanelSummary());
      }
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      console.warn('refreshGeoJson failed:', err);
    }
  }

  function attachMouseHandlers() {
    const container = viewer.container;
    let pendingMove = null;
    let lastMoveAt = 0;
    const HOVER_THROTTLE_MS = 33; // ~30 Hz per spec section 3.3

    container.addEventListener('mousemove', (ev) => {
      pendingMove = ev;
      const now = performance.now();
      if (now - lastMoveAt < HOVER_THROTTLE_MS) return;
      lastMoveAt = now;
      requestAnimationFrame(() => {
        if (!pendingMove) return;
        const evX = pendingMove.clientX, evY = pendingMove.clientY;
        pendingMove = null;
        handleHover(evX, evY);
      });
    });
    container.addEventListener('mouseleave', () => {
      // commit 6 — leaving the canvas clears the hover ring + tooltip
      // so the user isn't staring at stale info when their cursor is
      // somewhere else.
      forEachSvgOverlay((o) => o.setHover(null, null));
      hideHoverTooltip();
    });
    const DRAG_THRESHOLD_PX = 4;
    let downX = null, downY = null;
    container.addEventListener('mousedown', (ev) => {
      downX = ev.clientX;
      downY = ev.clientY;
    });
    container.addEventListener('mouseup', (ev) => {
      if (downX === null || downY === null) return;
      const dx = ev.clientX - downX;
      const dy = ev.clientY - downY;
      downX = null;
      downY = null;
      if (dx * dx + dy * dy > DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) return;
      handleClick(ev.clientX, ev.clientY);
    });
  }

  function viewerToImage(clientX, clientY) {
    if (!viewer.world || viewer.world.getItemCount() === 0) return null;
    const rect = viewer.container.getBoundingClientRect();
    const local = new window.OpenSeadragon.Point(clientX - rect.left, clientY - rect.top);
    const vpt = viewer.viewport.viewerElementToViewportCoordinates(local);
    const img = viewer.world.getItemAt(0).viewportToImageCoordinates(vpt);
    return img;
  }

  function handleHover(clientX, clientY) {
    if (!hitIndex) return;
    const img = viewerToImage(clientX, clientY);
    if (!img) return;
    const cid = hitTest(hitIndex, img.x, img.y);
    if (cid === null) {
      forEachSvgOverlay((o) => o.setHover(null, null));
      hideHoverTooltip();
      return;
    }
    const feature = featureForCellId(cid);
    const ring = feature ? feature.geometry.coordinates[0] : null;
    forEachSvgOverlay((o) => o.setHover(cid, ring));
    showHoverTooltip(clientX, clientY, feature);
  }

  // commit 6 — hover tooltip with the per-cell readout the user asked
  // for. Pulls from feature.properties (already populated by the
  // /api/cells endpoint: cell_id, color_value, phenotype_call,
  // confidence_stratum). The tooltip is positioned relative to the
  // viewer area (clientX/Y minus the viewer-area's bounding rect).
  function showHoverTooltip(clientX, clientY, feature) {
    const el = document.getElementById('hover-tooltip');
    if (!el || !feature) {
      if (el) el.hidden = true;
      return;
    }
    const props = feature.properties || {};
    const lines = [];
    lines.push(`<strong>Cell #${props.cell_id ?? feature.id ?? '?'}</strong>`);
    const pheno = props.phenotype_call;
    if (pheno) {
      const stratum = props.confidence_stratum
        ? ` <span class="hover-stratum hover-stratum-${escapeAttr(props.confidence_stratum)}">${escapeAttr(props.confidence_stratum)}</span>`
        : '';
      lines.push(`<span class="hover-label">Phenotype</span> ${escapeAttr(pheno)}${stratum}`);
    } else {
      lines.push('<span class="hover-label">Phenotype</span> <em>no call</em>');
    }
    // For marker views, color_value carries the per-cell metric
    // matching the current color_by:
    //   marker_pred  → predicted probability ∈ [0, 1]
    //   marker_truth → 0 or 1
    //   marker_error → pred − truth ∈ [-1, 1]
    if (state.color_by !== 'phenotype'
        && typeof props.color_value === 'number'
        && Number.isFinite(props.color_value)) {
      const label = state.color_by === 'marker_pred' ? 'Pred prob'
        : state.color_by === 'marker_truth' ? 'CODEX truth'
        : 'Pred − truth';
      const value = state.color_by === 'marker_truth'
        ? (props.color_value > 0.5 ? '1 (positive)' : '0 (negative)')
        : props.color_value.toFixed(2);
      lines.push(`<span class="hover-label">${escapeAttr(label)} · ${escapeAttr(state.marker || '')}</span> ${value}`);
    }
    el.innerHTML = lines.join('<br>');
    el.hidden = false;
    // Position with a small offset so the cursor doesn't sit on top of
    // the readout text. Flip to the cursor's left when too close to the
    // viewer's right edge so the tooltip doesn't get clipped.
    const viewerArea = document.getElementById('viewer-area');
    const rect = viewerArea.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const elW = el.offsetWidth || 200;
    const elH = el.offsetHeight || 60;
    const placeRight = x + 18 + elW <= rect.width;
    el.style.left = `${placeRight ? x + 14 : x - 14 - elW}px`;
    el.style.top = `${Math.max(8, Math.min(rect.height - elH - 8, y + 14))}px`;
  }

  function hideHoverTooltip() {
    const el = document.getElementById('hover-tooltip');
    if (el) el.hidden = true;
  }

  function escapeAttr(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function handleClick(clientX, clientY) {
    if (!hitIndex) return;
    const img = viewerToImage(clientX, clientY);
    if (!img) return;
    const cid = hitTest(hitIndex, img.x, img.y);
    if (cid === null) {
      // commit 6 — clicking on the canvas but not on a cell clears any
      // locked selection (and hides the hover tooltip). Easier deselect
      // than the keyboard Escape that the user mentioned.
      if (state.selected !== null && state.selected !== undefined) {
        updateState({ ...state, selected: null });
      }
      hideHoverTooltip();
      return;
    }
    updateState({ ...state, selected: cid });
  }

  function phenoKey(set) {
    if (set === null || set === undefined) return '*';
    return [...set].sort().join(',');
  }

  function featureForCellId(cellId) {
    if (!currentGeo) return null;
    const features = currentGeo.features || [];
    for (let i = 0; i < features.length; i++) {
      if (features[i].id === cellId) return features[i];
    }
    return null;
  }

  function ringForCellId(cellId) {
    if (!currentGeo) return null;
    const features = currentGeo.features || [];
    for (let i = 0; i < features.length; i++) {
      if (features[i].id === cellId) {
        return features[i].geometry.coordinates[0];
      }
    }
    return null;
  }

  // commit 8 — fixed per-pane tile sources. The primary viewer always
  // paints H&E (so toggling base back from CODEX → H&E doesn't fetch a
  // pyramid); the secondary always paints CODEX (composite or per-marker).
  function primaryTileSource() {
    return heDziUrl(state.core);
  }
  function secondaryTileSource() {
    if (codexLayer && _isAvailableCodexMarker(codexLayer)) {
      return codexMarkerDziUrl(state.core, codexLayer);
    }
    return codexDziUrl(state.core);
  }

  // Compatibility shim so older code paths (and the badge logic) can
  // still talk about a "tile source for base" abstractly. Returns the
  // source that should fill the active single pane, or the primary
  // when base=split (split paints both panes — callers that want the
  // CODEX side call secondaryTileSource() directly).
  function tileSourceForBase(base) {
    if (base === 'codex') return secondaryTileSource();
    return primaryTileSource();
  }

  function forEachCanvasOverlay(fn) {
    if (canvasOverlay) fn(canvasOverlay);
    if (canvasOverlay2) fn(canvasOverlay2);
  }
  function forEachSvgOverlay(fn) {
    if (svgOverlay) fn(svgOverlay);
    if (svgOverlay2) fn(svgOverlay2);
  }

  function ensureSecondaryViewer() {
    if (viewer2) return viewer2;
    const el = document.getElementById('osd-viewport-2');
    if (!el) return null;
    el.hidden = false;
    viewer2 = window.OpenSeadragon({
      element: el,
      tileSources: secondaryTileSource(),
      prefixUrl: '/static/workbench/vendor/openseadragon/images/',
      // The secondary pane is locked to the primary's viewport via the
      // sync handlers; hiding OSD's own nav buttons keeps the UI clean
      // (one set of controls drives both panes).
      showNavigationControl: false,
      showHomeControl: false,
      showZoomControl: false,
      showFullPageControl: false,
      maxZoomPixelRatio: 4,
      immediateRender: true,
      crossOriginPolicy: 'Anonymous',
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
    });
    canvasOverlay2 = attachCanvasOverlay(viewer2);
    canvasOverlay2.setPalettes(palettes);
    canvasOverlay2.setColorBy(state.color_by);
    canvasOverlay2.setExpandedOnlyAtLowZoom(expandedOnlyAtLowZoom);
    canvasOverlay2.setPhenotypeFilter(state.phenotypes);
    canvasOverlay2.setProbabilityThreshold(state.min_prob || 0);
    if (currentGeo) canvasOverlay2.setGeoJson(currentGeo);
    svgOverlay2 = attachSvgOverlay(viewer2);
    if (state.selected !== null && state.selected !== undefined) {
      const ring = ringForCellId(state.selected);
      svgOverlay2.setSelected(state.selected, ring);
    }
    return viewer2;
  }

  function attachSyncHandlers() {
    if (detachSyncFn) return;
    if (!viewer || !viewer2) return;
    let syncing = false;
    const syncTo = (dst, src) => () => {
      if (syncing) return;
      syncing = true;
      try {
        dst.viewport.panTo(src.viewport.getCenter(), true);
        dst.viewport.zoomTo(src.viewport.getZoom(), null, true);
      } finally {
        syncing = false;
      }
    };
    const h1 = syncTo(viewer2, viewer);
    const h2 = syncTo(viewer, viewer2);
    viewer.addHandler('animation', h1);
    viewer2.addHandler('animation', h2);
    // Initial copy so the secondary pane opens centered on the primary's
    // current viewport, not the CODEX image's default fit.
    syncing = true;
    try {
      viewer2.viewport.panTo(viewer.viewport.getCenter(), true);
      viewer2.viewport.zoomTo(viewer.viewport.getZoom(), null, true);
    } finally {
      syncing = false;
    }
    detachSyncFn = () => {
      viewer.removeHandler('animation', h1);
      viewer2.removeHandler('animation', h2);
    };
  }

  function detachSyncHandlers() {
    if (detachSyncFn) { detachSyncFn(); detachSyncFn = null; }
  }

  function applyBaseLayout() {
    const base = state.base || 'he';
    const viewerArea = document.getElementById('viewer-area');
    const primaryEl = document.getElementById('osd-viewport');
    const secondaryEl = document.getElementById('osd-viewport-2');
    if (!viewerArea || !primaryEl || !secondaryEl) return;

    viewerArea.dataset.base = base;
    viewerArea.dataset.split = String(base === 'split');

    if (base === 'codex' || base === 'split') {
      ensureSecondaryViewer();
    }

    if (base === 'he') {
      primaryEl.hidden = false;
      secondaryEl.hidden = true;
      detachSyncHandlers();
    } else if (base === 'codex') {
      primaryEl.hidden = true;
      secondaryEl.hidden = false;
      detachSyncHandlers();
    } else {
      primaryEl.hidden = false;
      secondaryEl.hidden = false;
      attachSyncHandlers();
    }

    // OSD needs to be told to re-measure when its container changes.
    setTimeout(() => {
      if (viewer && typeof viewer.forceResize === 'function') viewer.forceResize();
      if (viewer2 && typeof viewer2.forceResize === 'function') viewer2.forceResize();
      // Force overlay repaint at the new dimensions.
      forEachCanvasOverlay((o) => o.setColorBy(state.color_by));
      statusBar.update();
    }, 60);
  }

  function _isAvailableCodexMarker(name) {
    if (!codexInfo || !Array.isArray(codexInfo.available_markers)) return false;
    return codexInfo.available_markers.some((m) => m.name === name);
  }

  // commit 4 — async fetch + cache for /api/stats/validation. The
  // controls + legend call ``getValidationAp(kind, name)`` synchronously
  // and get a number-or-null; a parallel fetch warms the cache and
  // triggers a re-render once data lands so the AP value snaps in.
  async function ensureValidation(modelId) {
    if (validationCache.has(modelId)) return validationCache.get(modelId);
    if (validationInflight.has(modelId)) return null;
    validationInflight.add(modelId);
    try {
      const body = await fetchStatsValidation(modelId);
      validationCache.set(modelId, body);
      // Re-render the panels that care about the new data. Light-touch:
      // controls.rerender() reflows the AP badges in-place; legend just
      // re-renders against the same state.
      if (controls && typeof controls.rerender === 'function') controls.rerender();
      renderLegend(legendRoot, state.color_by, state.marker, palettes, strata, helpPanel, getValidationAp);
      return body;
    } catch (err) {
      console.warn(`fetchStatsValidation(${modelId}) failed:`, err);
      // Cache a sentinel so we don't pummel the server with retries.
      validationCache.set(modelId, { markers: [], phenotypes: [] });
      return null;
    } finally {
      validationInflight.delete(modelId);
    }
  }

  function getValidationAp(kind, name) {
    if (!name) return null;
    const body = validationCache.get(state.model);
    if (!body) {
      // Kick off the fetch — the callback will rerender once it lands.
      void ensureValidation(state.model);
      return null;
    }
    const rows = kind === 'marker' ? body.markers : body.phenotypes;
    if (!Array.isArray(rows)) return null;
    const row = rows.find((r) => r.name === name);
    return row ? row.macro_ap : null;
  }

  function updateCodexBadge() {
    const el = document.getElementById('codex-badge');
    if (!el) return;
    if (state.base !== 'codex') {
      el.hidden = true;
      el.textContent = '';
      return;
    }
    if (codexLayer && _isAvailableCodexMarker(codexLayer)) {
      el.hidden = false;
      el.textContent = `CODEX truth: ${codexLayer}`;
    } else {
      el.hidden = false;
      el.textContent = 'CODEX composite (DAPI · CD45 · panCK)';
    }
  }

  // commit 3 — when the user picks a marker in Marker focus, the CODEX
  // base layer auto-tracks it so the staining the viewer paints matches
  // the marker name shown on screen. Phenotype focus reverts to the
  // composite (no single marker to track). The user can still pick a
  // different CODEX channel manually via the dropdown; this only
  // overrides when the *driving* state (color_by, marker) changes —
  // the dropdown's setCodexLayer callback doesn't go through here.
  function autoSyncCodexLayer() {
    let target = null;
    if (state.color_by !== 'phenotype'
        && state.marker
        && _isAvailableCodexMarker(state.marker)) {
      target = state.marker;
    }
    if (target === codexLayer) return false;
    codexLayer = target;
    // If a CODEX pane is currently in view (single or split), re-open
    // it on viewer2. The primary always paints H&E so we never call
    // viewer.open on it from here.
    if ((state.base === 'codex' || state.base === 'split') && viewer2
        && viewer2.world && viewer2.world.getItemCount() > 0) {
      viewer2.open(secondaryTileSource());
    }
    updateCodexBadge();
    return true;
  }

  async function updateState(next) {
    if (stateEquals(state, next)) return;
    const prev = state;
    state = next;
    applyState(state, { replace: true });
    exposeGlobal('state', state);

    if (fetchKey(prev) !== fetchKey(state)) {
      forEachCanvasOverlay((o) => o.setColorBy(state.color_by));
      await refreshGeoJson();
    }

    if (phenoKey(prev.phenotypes) !== phenoKey(state.phenotypes)) {
      forEachCanvasOverlay((o) => o.setPhenotypeFilter(state.phenotypes));
    }

    if ((prev.min_prob || 0) !== (state.min_prob || 0)) {
      forEachCanvasOverlay((o) => o.setProbabilityThreshold(state.min_prob || 0));
    }

    if (prev.color_by !== state.color_by || prev.marker !== state.marker) {
      renderLegend(legendRoot, state.color_by, state.marker, palettes, strata, helpPanel, getValidationAp);
      autoSyncCodexLayer();
    }

    if (prev.selected !== state.selected) {
      if (state.selected !== null && state.selected !== undefined) {
        sidePanel.show(state.selected);
        const ring = ringForCellId(state.selected);
        forEachSvgOverlay((o) => o.setSelected(state.selected, ring));
      } else {
        sidePanel.showEmpty(buildSidePanelSummary());
        forEachSvgOverlay((o) => o.setSelected(null, null));
      }
    } else if (prev.compare !== state.compare && state.selected !== null && state.selected !== undefined) {
      sidePanel.show(state.selected);
    }

    if (prev.mode !== state.mode && state.selected !== null && state.selected !== undefined) {
      const ring = ringForCellId(state.selected);
      forEachSvgOverlay((o) => o.setSelected(state.selected, ring));
    }

    if (prev.model !== state.model) {
      if (controls && typeof controls.rerender === 'function') controls.rerender();
      if (state.selected !== null && state.selected !== undefined) {
        sidePanel.show(state.selected);
      }
      if (statsLoadedOnce) {
        void statsPanel.refreshPerCore();
        void statsPanel.refreshValidation();
      }
    }

    if ((prev.base || 'he') !== (state.base || 'he')) {
      // commit 8 — apply the pane layout for the new base. Always
      // through this helper so single-↔-split transitions handle the
      // viewer-resize + sync-handler attach/detach in one place.
      applyBaseLayout();
      updateCodexBadge();
    }

    appBar.rerender();
    statusBar.update();
  }

  window.addEventListener('popstate', () => {
    const next = parseUrlToState(window.location.href);
    updateState({ ...next, core: state.core });
  });

  exposeGlobal('state', state);
  exposeGlobal('codexInfo', codexInfo);
  exposeGlobal('refreshGeoJson', refreshGeoJson);
  exposeGlobal('updateState', updateState);
  exposeGlobal('DEFAULT_MARKER', DEFAULT_MARKER);
}

function showFatal(msg) {
  const root = document.getElementById('app');
  if (!root) {
    console.error(msg);
    return;
  }
  root.innerHTML = `<div class="fatal">${msg}</div>`;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    main().catch((err) => console.error('boot failed:', err));
  });
} else {
  main().catch((err) => console.error('boot failed:', err));
}
