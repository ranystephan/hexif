// Top app bar - brand, core breadcrumb, sidebar toggles, help, shortcuts.
//
// The bar is purely presentational; it reads state through getState/getCore
// callbacks and dispatches user intents back through onToggleLeft /
// onToggleRight / onOpenHelp / onOpenShortcuts. Persistent state (which
// sidebars are open) is owned by main.js. This module just paints.

/**
 * @typedef {Object} AppBarCallbacks
 * @property {() => any} getState
 * @property {() => Array<{basename: string, tma?: string, tissue?: string, split?: string, n_cells?: number}>} getCores
 * @property {() => boolean} isLeftOpen
 * @property {() => boolean} isRightOpen
 * @property {() => void} onToggleLeft
 * @property {() => void} onToggleRight
 * @property {() => void} onOpenHelp
 * @property {() => void} onOpenShortcuts
 */

/**
 * Mount the top app bar into `root`.
 *
 * @param {HTMLElement} root
 * @param {AppBarCallbacks} cb
 * @returns {{ rerender: () => void }}
 */
export function mountAppBar(root, cb) {
  root.innerHTML = '';
  root.classList.add('topbar');
  root.setAttribute('role', 'banner');

  const brand = document.createElement('div');
  brand.className = 'topbar-brand';
  brand.innerHTML = `
    <img class="topbar-logo" src="/static/workbench/assets/hexif_logo.png"
         alt="" aria-hidden="true" />
    <span class="topbar-titles">
      <span class="topbar-title">HEXIF</span>
      <span class="topbar-subtitle">Spatial Cytometry Workbench</span>
    </span>
  `;
  root.appendChild(brand);

  const crumb = document.createElement('nav');
  crumb.className = 'topbar-breadcrumb';
  crumb.setAttribute('aria-label', 'Active core');
  root.appendChild(crumb);

  const actions = document.createElement('div');
  actions.className = 'topbar-actions';
  actions.innerHTML = `
    <button type="button" class="topbar-action" data-action="toggle-left" aria-pressed="true"
            aria-label="Hide or show the controls panel"
            title="Hide or show controls ([). Drag the divider to resize.">
      <span class="topbar-action-icon" aria-hidden="true">⊟</span>
      <span>Controls</span>
      <span class="topbar-action-kbd">[</span>
    </button>
    <button type="button" class="topbar-action" data-action="toggle-right" aria-pressed="true"
            aria-label="Hide or show the cell detail panel"
            title="Hide or show cell detail (]). Drag the divider to resize.">
      <span class="topbar-action-icon" aria-hidden="true">⊟</span>
      <span>Cell</span>
      <span class="topbar-action-kbd">]</span>
    </button>
    <button type="button" class="topbar-action topbar-theme-toggle" data-action="toggle-theme"
            aria-pressed="false"
            aria-label="Toggle dark mode"
            title="Toggle dark mode ( D )">
      <span class="topbar-theme-icon" aria-hidden="true">☾</span>
      <span class="topbar-action-kbd">D</span>
    </button>
    <button type="button" class="topbar-action" data-action="shortcuts"
            title="Keyboard shortcuts ( ? )">
      <span>Shortcuts</span>
      <span class="topbar-action-kbd">?</span>
    </button>
    <button type="button" class="topbar-action" id="help-toggle" data-action="help"
            title="Open the help panel ( H )"
            aria-label="Open the workbench help panel">
      <span>Help</span>
      <span class="topbar-action-kbd">H</span>
    </button>
    <a class="topbar-action topbar-link" href="https://ajglab.org/" target="_blank"
       rel="noopener noreferrer" aria-label="Open the Gentles Lab website"
       title="Open the Gentles Lab website">
      <span>Lab</span>
    </a>
    <a class="topbar-action topbar-link" href="https://github.com/ranystephan/hexif" target="_blank"
       rel="noopener noreferrer" aria-label="Open the HEXIF GitHub repository"
       title="Open the HEXIF GitHub repository">
      <span>GitHub</span>
    </a>
  `;
  root.appendChild(actions);

  // Wire the actions.
  actions.querySelector('[data-action="toggle-left"]').addEventListener('click', () => {
    cb.onToggleLeft();
    rerender();
  });
  actions.querySelector('[data-action="toggle-right"]').addEventListener('click', () => {
    cb.onToggleRight();
    rerender();
  });
  actions.querySelector('[data-action="shortcuts"]').addEventListener('click', () => {
    cb.onOpenShortcuts();
  });
  actions.querySelector('[data-action="help"]').addEventListener('click', () => {
    cb.onOpenHelp();
  });
  actions.querySelector('[data-action="toggle-theme"]').addEventListener('click', () => {
    if (cb.onToggleTheme) cb.onToggleTheme();
    rerender();
  });

  function rerender() {
    // Breadcrumb: TMA · tissue · core · split tag · cell-count chip.
    crumb.innerHTML = '';
    const state = cb.getState();
    const cores = cb.getCores() || [];
    const meta = cores.find((c) => c.basename === state.core) || {};

    if (meta.tma) {
      const span = document.createElement('span');
      span.textContent = meta.tma;
      crumb.appendChild(span);
      crumb.appendChild(makeSep());
    }
    if (meta.tissue) {
      const span = document.createElement('span');
      span.textContent = meta.tissue;
      crumb.appendChild(span);
      crumb.appendChild(makeSep());
    }
    const coreChip = document.createElement('span');
    coreChip.className = 'bc-core';
    coreChip.textContent = state.core || '—';
    crumb.appendChild(coreChip);

    if (meta.split) {
      const tag = document.createElement('span');
      tag.className = `bc-tag bc-tag-${meta.split}`;
      tag.textContent = meta.split;
      crumb.appendChild(tag);
    }
    if (typeof meta.n_cells === 'number') {
      crumb.appendChild(makeSep());
      const cells = document.createElement('span');
      cells.textContent = `${formatInt(meta.n_cells)} cells`;
      crumb.appendChild(cells);
    }

    // Toggle button pressed-state mirrors the sidebar open/close.
    const leftBtn = actions.querySelector('[data-action="toggle-left"]');
    const rightBtn = actions.querySelector('[data-action="toggle-right"]');
    leftBtn.setAttribute('aria-pressed', String(cb.isLeftOpen()));
    rightBtn.setAttribute('aria-pressed', String(cb.isRightOpen()));
    // Theme toggle — icon flips between moon (in light mode → click to
    // go dark) and sun (in dark mode → click to go light).
    const themeBtn = actions.querySelector('[data-action="toggle-theme"]');
    if (themeBtn && cb.isDark) {
      const dark = cb.isDark();
      themeBtn.setAttribute('aria-pressed', String(dark));
      const icon = themeBtn.querySelector('.topbar-theme-icon');
      if (icon) icon.textContent = dark ? '☀' : '☾';
      themeBtn.title = dark
        ? 'Switch to light mode ( D )'
        : 'Switch to dark mode ( D )';
    }
  }

  rerender();
  return { rerender };
}

function makeSep() {
  const sep = document.createElement('span');
  sep.className = 'bc-sep';
  sep.textContent = '/';
  return sep;
}

function formatInt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toLocaleString();
}
