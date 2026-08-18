// Model card + dropdown (model selection, Workbench contract §3.2).
//
// Renders at the top of the controls panel. Two pieces of UI:
//   1. A 4-line model card summarizing the active model (name +
//      backbone, training split, macro AP, "▶ details" affordance).
//   2. A <select> dropdown that lets the user switch between every
//      model the bundle ships.
//
// The "▶ details" button opens a <dialog> modal carrying the full
// per-marker / per-phenotype AP table fetched from /api/model/{id}.
// We use a native <dialog> (Chromium / Firefox ≥ 98 / Safari ≥ 15.4
// — current at the Workbench contract date) so we don't need a modal-helper
// library; the dismiss-on-click-outside + Esc behaviour comes for
// free.
//
// The component is pure: it never reads window.* or document outside
// the supplied root. State / callbacks travel through ``cb`` so the
// single-writer rule in main.js is preserved.

import { fetchModelCard } from '../api.js';

/**
 * @typedef {Object} ModelEntry
 * @property {string} id
 * @property {string} name
 * @property {string} backbone
 * @property {string} training_split
 * @property {number} macro_marker_ap
 * @property {number} macro_phenotype_ap
 * @property {boolean} is_default
 */

/**
 * @typedef {Object} ModelCardCallbacks
 * @property {() => any} getState
 * @property {(state: any) => void} onStateChange
 * @property {() => {models: ModelEntry[], default_id: string}|null} getModels
 */

/**
 * Mount the model card + dropdown into a container element. The
 * caller is responsible for clearing the container before remount.
 *
 * Returns an object with a ``rerender`` method that the parent
 * controls panel can call when state.model changes so the card lines
 * + dropdown selection re-sync without rebuilding the DOM (saves the
 * <select>'s focus state across the swap).
 *
 * @param {HTMLElement} root
 * @param {ModelCardCallbacks} cb
 * @returns {{rerender: () => void, root: HTMLElement}}
 */
export function mountModelCard(root, cb) {
  const wrap = document.createElement('div');
  wrap.className = 'model-card';
  wrap.setAttribute('aria-label', 'Active model');

  const lineName = document.createElement('div');
  lineName.className = 'model-card-name';
  const lineSplit = document.createElement('div');
  lineSplit.className = 'model-card-split';
  const lineAp = document.createElement('div');
  lineAp.className = 'model-card-ap';

  const detailsButton = document.createElement('button');
  detailsButton.type = 'button';
  detailsButton.className = 'model-card-details';
  detailsButton.textContent = '▶ details';
  detailsButton.setAttribute(
    'title',
    'Open the per-marker / per-phenotype AP table for this model',
  );
  detailsButton.addEventListener('click', async () => {
    const cur = cb.getState();
    try {
      const card = await fetchModelCard(cur.model);
      openDetailsDialog(card);
    } catch (err) {
      console.warn(`fetchModelCard(${cur.model}) failed:`, err);
    }
  });

  wrap.appendChild(lineName);
  wrap.appendChild(lineSplit);
  wrap.appendChild(lineAp);
  wrap.appendChild(detailsButton);

  // Dropdown -----------------------------------------------------
  const selectWrap = document.createElement('div');
  selectWrap.className = 'model-card-select';
  const selectLabel = document.createElement('label');
  selectLabel.textContent = 'Model: ';
  selectLabel.className = 'model-card-select-label';
  const select = document.createElement('select');
  select.className = 'model-select';
  select.setAttribute('aria-label', 'Active model selector');
  select.title =
    'Switch the prediction source. The model card above shows each model'
    + "'s calibrated macro AP on the val cores.";
  selectLabel.appendChild(select);
  selectWrap.appendChild(selectLabel);

  const modelList = cb.getModels();
  if (modelList && Array.isArray(modelList.models)) {
    for (const entry of modelList.models) {
      const opt = document.createElement('option');
      opt.value = entry.id;
      opt.textContent = entry.name || entry.id;
      select.appendChild(opt);
    }
  } else {
    select.disabled = true;
    select.title = 'Model metadata unavailable';
  }
  select.addEventListener('change', () => {
    const cur = cb.getState();
    cb.onStateChange({ ...cur, model: select.value });
  });

  root.appendChild(wrap);
  root.appendChild(selectWrap);

  function entryById(id) {
    const ml = cb.getModels();
    if (!ml || !Array.isArray(ml.models)) return null;
    for (const e of ml.models) if (e.id === id) return e;
    return null;
  }

  function rerender() {
    const cur = cb.getState();
    const entry = entryById(cur.model);
    if (entry) {
      lineName.textContent =
        `${entry.name}${entry.backbone ? ' · ' + entry.backbone : ''}`;
      lineSplit.textContent = entry.training_split || '';
      // Pill-style AP readouts so the two macro numbers read as distinct
      // chips rather than a comma-separated run-on. The inner text is
      // still readable (tests compare innerText across model switches).
      lineAp.innerHTML =
        `<span class="model-card-ap-pill">` +
          `<span class="model-card-ap-pill-label">marker</span>` +
          `<span class="model-card-ap-pill-value">${escapeHtml(fmtAp(entry.macro_marker_ap))}</span>` +
        `</span>` +
        `<span class="model-card-ap-pill">` +
          `<span class="model-card-ap-pill-label">phenotype</span>` +
          `<span class="model-card-ap-pill-value">${escapeHtml(fmtAp(entry.macro_phenotype_ap))}</span>` +
        `</span>`;
      detailsButton.disabled = false;
    } else {
      // Either /api/models failed at boot, or state.model isn't in
      // the bundle. We still show the card outline so the user
      // doesn't see a jump when the data arrives later.
      lineName.textContent = cur.model || '(no model)';
      lineSplit.textContent = '';
      lineAp.textContent = '';
      detailsButton.disabled = !cur.model;
    }
    if (select.value !== cur.model && cur.model) select.value = cur.model;
  }

  rerender();
  return { rerender, root };
}

function fmtAp(v) {
  // 4-decimal display matches the the API contract numbers and the
  // values come from the active bundle manifest.
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(4);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function openDetailsDialog(card) {
  // Build the dialog lazily — keeping the DOM out until the user asks
  // for it avoids carrying ~5 KB of unused markup on the initial
  // paint of every workbench session.
  let dialog = document.querySelector('dialog.model-card-dialog');
  if (!dialog) {
    dialog = document.createElement('dialog');
    dialog.className = 'model-card-dialog';
    // Click-outside dismiss: the native <dialog> doesn't do this by
    // default, but the <form method="dialog"> form-submit pattern
    // closes via Esc. We add the click-on-backdrop handler below.
    dialog.addEventListener('click', (ev) => {
      if (ev.target === dialog) dialog.close();
    });
    document.body.appendChild(dialog);
  }
  dialog.innerHTML = '';

  const closeButton = document.createElement('button');
  closeButton.type = 'button';
  closeButton.className = 'model-card-dialog-close';
  closeButton.textContent = '×';
  closeButton.setAttribute('aria-label', 'Close model details');
  closeButton.addEventListener('click', () => dialog.close());
  dialog.appendChild(closeButton);

  const h = document.createElement('h2');
  h.textContent = card.name || card.id;
  dialog.appendChild(h);
  if (card.backbone) {
    const sub = document.createElement('p');
    sub.className = 'model-card-dialog-sub';
    sub.textContent = card.backbone;
    dialog.appendChild(sub);
  }
  if (card.training_split) {
    const sub = document.createElement('p');
    sub.className = 'model-card-dialog-sub';
    sub.textContent = card.training_split;
    dialog.appendChild(sub);
  }

  const macroBlock = document.createElement('div');
  macroBlock.className = 'model-card-dialog-macro';
  macroBlock.textContent =
    `macro marker AP ${fmtAp(card.macro_marker_ap)} · ` +
    `macro phenotype AP ${fmtAp(card.macro_phenotype_ap)}`;
  dialog.appendChild(macroBlock);

  const tables = document.createElement('div');
  tables.className = 'model-card-dialog-tables';
  tables.appendChild(buildApTable('Per-marker AP', card.per_marker_ap || {}));
  tables.appendChild(
    buildApTable('Per-phenotype AP', card.per_phenotype_ap || {}),
  );
  dialog.appendChild(tables);

  if (card.notes) {
    const notes = document.createElement('p');
    notes.className = 'model-card-dialog-notes';
    notes.textContent = card.notes;
    dialog.appendChild(notes);
  }

  dialog.showModal();
}

function buildApTable(title, apMap) {
  const section = document.createElement('section');
  const h = document.createElement('h3');
  h.textContent = title;
  section.appendChild(h);
  const table = document.createElement('table');
  table.className = 'model-card-ap-table';
  const tbody = document.createElement('tbody');
  // Sort by AP descending so the strong markers/phenotypes lead.
  const entries = Object.entries(apMap).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 2;
    td.textContent = '(no AP available)';
    tr.appendChild(td);
    tbody.appendChild(tr);
  } else {
    for (const [name, ap] of entries) {
      const tr = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = name;
      const apCell = document.createElement('td');
      apCell.textContent = fmtAp(ap);
      apCell.className = 'model-card-ap-cell';
      tr.appendChild(nameCell);
      tr.appendChild(apCell);
      tbody.appendChild(tr);
    }
  }
  table.appendChild(tbody);
  section.appendChild(table);
  return section;
}
