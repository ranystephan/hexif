// Keyboard shortcut handling + the "press ? to see them" reference modal.
//
// Two concerns in one module:
//   1. `mountShortcutsModal(parent)` builds a hidden <dialog> that
//      lists every shortcut, opened by clicking the topbar "Shortcuts"
//      button or pressing `?` (Shift + /).
//   2. `installKeyBindings(handlers)` attaches a single document-level
//      keydown listener that dispatches to the supplied handlers.
//
// The bindings honour standard expectations: keys are ignored when the
// user is typing in an <input>, <textarea>, or <select>; the OSD viewer
// (a focusable div) is fine to receive them because it doesn't accept
// text. Modifier-key chords (Ctrl/Cmd/Alt) are passed through to the
// browser. The workbench owns plain keys only.

/**
 * @typedef {Object} ShortcutHandlers
 * @property {() => void} [toggleLeft]        `[` collapses / expands left sidebar.
 * @property {() => void} [toggleRight]       `]` collapses / expands right sidebar.
 * @property {() => void} [openHelp]          `H` opens the help panel.
 * @property {() => void} [openShortcuts]     `?` opens this shortcuts modal.
 * @property {() => void} [clearSelection]    `Esc` deselects the active cell.
 * @property {(v: string) => void} [setColorBy] 1 / 2 / 3 / 4 sets phenotype / pred / truth / error.
 * @property {(v: string) => void} [setMode]   M flips nucleus ↔ expanded mode.
 * @property {() => void} [toggleCompare]     C toggles pred-vs-truth comparison.
 */

/**
 * Mount a hidden shortcuts modal under `parent`. Returns a handle to
 * open / close it.
 *
 * @param {HTMLElement} parent
 * @returns {{ open: () => void, close: () => void, isOpen: () => boolean }}
 */
export function mountShortcutsModal(parent) {
  const existing = parent.querySelector('dialog.shortcuts-dialog');
  if (existing && existing.__hexifShortcutsHandle) {
    return existing.__hexifShortcutsHandle;
  }

  const dialog = existing || document.createElement('dialog');
  dialog.className = 'shortcuts-dialog';
  dialog.setAttribute('aria-label', 'Keyboard shortcuts');
  dialog.innerHTML = `
    <div class="shortcuts-header">
      <h2>Keyboard shortcuts</h2>
      <button type="button" class="shortcuts-close" aria-label="Close shortcuts">×</button>
    </div>
    <div class="shortcuts-body"></div>
  `;
  if (!existing) parent.appendChild(dialog);

  const body = dialog.querySelector('.shortcuts-body');
  body.innerHTML = renderShortcutsHtml();

  dialog.querySelector('.shortcuts-close').addEventListener('click', () => dialog.close());
  // Click-outside-to-dismiss for the native <dialog>.
  dialog.addEventListener('click', (ev) => {
    if (ev.target === dialog) dialog.close();
  });

  const handle = {
    open() {
      if (!dialog.open) dialog.showModal();
    },
    close() {
      if (dialog.open) dialog.close();
    },
    isOpen() {
      return Boolean(dialog.open);
    },
  };
  dialog.__hexifShortcutsHandle = handle;
  return handle;
}

function renderShortcutsHtml() {
  const groups = [
    {
      title: 'Layout',
      rows: [
        { keys: ['['], desc: 'Hide / show controls panel' },
        { keys: [']'], desc: 'Hide / show cell-detail panel' },
        { keys: ['Drag'], desc: 'Resize either sidebar divider' },
        { keys: ['←', '→'], desc: 'Resize a focused sidebar divider' },
        { keys: ['H'], desc: 'Open the help panel' },
        { keys: ['?'], desc: 'Open this shortcuts modal' },
      ],
    },
    {
      title: 'Cell coloring',
      rows: [
        { keys: ['1'], desc: 'Color by phenotype' },
        { keys: ['2'], desc: 'Color by marker prediction' },
        { keys: ['3'], desc: 'Color by CODEX truth' },
        { keys: ['4'], desc: 'Color by prediction − truth' },
      ],
    },
    {
      title: 'Cell view',
      rows: [
        { keys: ['M'], desc: 'Flip nucleus / expanded mode' },
        { keys: ['C'], desc: 'Toggle pred-vs-truth comparison' },
        { keys: ['Esc'], desc: 'Clear current cell selection' },
      ],
    },
    {
      title: 'Appearance',
      rows: [
        { keys: ['D'], desc: 'Toggle dark mode' },
      ],
    },
    {
      title: 'Viewer (OpenSeadragon)',
      rows: [
        { keys: ['Drag'], desc: 'Pan the slide' },
        { keys: ['Wheel'], desc: 'Zoom around the cursor' },
        { keys: ['Click'], desc: 'Select a cell' },
        { keys: ['Dbl-click'], desc: 'Zoom-to-fit / reset view' },
      ],
    },
  ];
  return groups
    .map(
      (g) => `
        <div class="shortcuts-group">
          <div class="shortcuts-group-title">${escapeHtml(g.title)}</div>
          <div class="shortcuts-list">
            ${g.rows
              .map(
                (r) => `
                  <div class="shortcuts-row">
                    <span class="shortcuts-desc">${escapeHtml(r.desc)}</span>
                    <span class="shortcuts-keys">${r.keys
                      .map((k) => `<span class="kbd">${escapeHtml(k)}</span>`)
                      .join('')}</span>
                  </div>
                `,
              )
              .join('')}
          </div>
        </div>
      `,
    )
    .join('');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Wire global keydown shortcuts to the supplied handlers.
 *
 * @param {ShortcutHandlers} handlers
 * @returns {() => void} A teardown function that removes the listener.
 */
export function installKeyBindings(handlers) {
  function onKey(ev) {
    // Skip when a popover / modal / open input wants the key. We also
    // bail when any modifier is held — chords belong to the browser
    // (Cmd+L for address bar, Ctrl+R for reload, etc.).
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const target = ev.target;
    if (target && target.matches && target.matches('input, textarea, select, [contenteditable=""], [contenteditable="true"]')) {
      return;
    }

    switch (ev.key) {
      case '[':
        if (handlers.toggleLeft) {
          handlers.toggleLeft();
          ev.preventDefault();
        }
        break;
      case ']':
        if (handlers.toggleRight) {
          handlers.toggleRight();
          ev.preventDefault();
        }
        break;
      case 'h':
      case 'H':
        if (handlers.openHelp) {
          handlers.openHelp();
          ev.preventDefault();
        }
        break;
      case '?':
        if (handlers.openShortcuts) {
          handlers.openShortcuts();
          ev.preventDefault();
        }
        break;
      case 'Escape':
        if (handlers.clearSelection) {
          // Don't preventDefault — Esc also closes dialogs/popovers; let
          // any open popover claim the key first via its own listener.
          handlers.clearSelection();
        }
        break;
      case '1':
        if (handlers.setColorBy) { handlers.setColorBy('phenotype');     ev.preventDefault(); }
        break;
      case '2':
        if (handlers.setColorBy) { handlers.setColorBy('marker_pred');   ev.preventDefault(); }
        break;
      case '3':
        if (handlers.setColorBy) { handlers.setColorBy('marker_truth');  ev.preventDefault(); }
        break;
      case '4':
        if (handlers.setColorBy) { handlers.setColorBy('marker_error');  ev.preventDefault(); }
        break;
      case 'm':
      case 'M':
        if (handlers.setMode) {
          handlers.setMode();
          ev.preventDefault();
        }
        break;
      case 'c':
      case 'C':
        if (handlers.toggleCompare) {
          handlers.toggleCompare();
          ev.preventDefault();
        }
        break;
      case 'd':
      case 'D':
        if (handlers.toggleTheme) {
          handlers.toggleTheme();
          ev.preventDefault();
        }
        break;
      default:
        break;
    }
  }

  document.addEventListener('keydown', onKey);
  return () => document.removeEventListener('keydown', onKey);
}
