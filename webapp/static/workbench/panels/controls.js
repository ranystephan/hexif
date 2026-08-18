// Left-hand controls drawer.
//
// Renders: model card + dropdown (model selection), core selector,
// color-by buttons, marker selector, mode buttons, phenotype chips,
// low-zoom-expanded-only toggle, comparison toggle. Every change
// funnels through the onChange callback so the state machine in
// main.js stays single-writer.

import {
  PHENOTYPE_NAMES,
  NO_CALL_KEY,
  VALID_MARKERS,
  DEFAULT_MARKER,
} from '../state.js';
import { mountModelCard } from './model_card.js';

// confidence display — per-button hover text for the color-by segments.
// Kept in sync with the legend captions (legend.js COLOR_BY_CAPTIONS):
// the title= on the button is the same one-liner the legend shows, so
// hovering a button previews what the legend will say after click.
// Commit 3 — the "Phenotype" entry moved up to the Focus radio above
// this row, so this section is now strictly the *view* sub-selector
// (Pred / Truth / Error) within Marker focus.
const COLOR_BY_TITLES = {
  phenotype:
    'Each cell is colored by its phenotype call. Click a chip to filter.',
  marker_pred:
    'Each cell is colored by the predicted probability that the marker is positive (0 = blue, 1 = yellow).',
  marker_truth:
    'Each cell is colored by the measured CODEX truth (0 / 1). Toggle base to CODEX to see the truth signal directly.',
  marker_error:
    'Each cell is colored by `prediction − truth`; red = overprediction, blue = underprediction.',
};

// commit 3 — top-of-panel "Focus" radio. The state.color_by field is
// the canonical URL contract (commit 1's spec mandate); this helper
// derives the user-facing focus mode from it so toggling Marker /
// Phenotype is a one-line state edit.
function _focusOf(colorBy) {
  return colorBy === 'phenotype' ? 'phenotype' : 'marker';
}

// HTML for the "?" popover that explains all four color-by modes
// side-by-side. Plain string concatenation keeps the renderer dead
// simple — no templating engine, no escaping logic. The popover is
// dismissed by clicking outside or pressing Escape (wired below).
const COLOR_BY_POPOVER_HTML = `
  <div class="cb-popover-header">
    <span class="cb-popover-title">Color-by modes</span>
    <button type="button" class="cb-popover-close" aria-label="Close">×</button>
  </div>
  <table class="cb-popover-table">
    <tbody>
      <tr>
        <th>Phenotype</th>
        <td>Each cell colored by its phenotype call.</td>
      </tr>
      <tr>
        <th>Marker pred</th>
        <td>Cell colored by predicted marker probability (viridis 0&nbsp;→&nbsp;1).</td>
      </tr>
      <tr>
        <th>Marker truth</th>
        <td>Cell colored by measured CODEX truth (binary, 0&nbsp;/&nbsp;1).</td>
      </tr>
      <tr>
        <th>Marker error</th>
        <td>Cell colored by prediction&nbsp;−&nbsp;truth (blue underpredict / red overpredict).</td>
      </tr>
    </tbody>
  </table>
`;

/**
 * @typedef {Object} ControlsCallbacks
 * @property {(state: any) => void} onStateChange  Apply a partial state update.
 * @property {() => any} getState                  Return the current state.
 * @property {() => boolean} getExpandedOnlyAtLowZoom
 * @property {(b: boolean) => void} setExpandedOnlyAtLowZoom
 * @property {() => boolean} hasTruth              True when comparison mode is unlocked.
 * @property {() => ({models: Array, default_id: string}|null)} [getModels]
 *   model selection: bundle's /api/models response (or null when the
 *   fetch failed). Used by the top-of-panel model card.
 * @property {() => boolean} [hasCodex]            CODEX composite: True iff the active core has a paired CODEX composite.
 * @property {() => Array<{channel_idx: number, name: string, dzi_url: string}>} [getCodexMarkers]
 *   List of focused-12 markers that have per-marker CODEX truth PNGs on
 *   disk for the active core. Empty when the bundle hasn't rendered them.
 * @property {() => (string|null)} [getCodexLayer]
 *   Current CODEX base-layer selection — ``null`` = composite, a marker
 *   name = single-channel raw staining. Transient (not URL-persisted).
 * @property {(name: string|null) => void} [setCodexLayer]
 *   Switch the CODEX base layer to a marker name (or composite when null).
 * @property {((kind: 'marker'|'phenotype', name: string) => (number|null))} [getValidationAp]
 *   commit 4 — async-friendly accessor for the per-marker / per-phenotype
 *   macro AP. Returns null when validation data isn't loaded yet; the
 *   caller kicks off the fetch and re-renders us when it lands.
 * @property {() => ({openTo: (id: string, anchor?: string|null) => Promise<void>}|null)} [getHelpPanel]
 *   help system: handle to the slide-out help panel. When wired,
 *   ambiguous controls (model card, base-layer toggle) gain a "?"
 *   affordance that deep-links into the relevant anchor in controls.md.
 */

/**
 * Build a small "?" button that opens the help panel deep-linked to a
 * specific anchor inside the controls.md page.
 *
 * @param {string} anchor   The slugified heading id (no leading '#').
 * @param {string} label    aria-label / title text.
 * @param {(() => ({openTo: (id: string, anchor?: string|null) => Promise<void>}|null))|undefined} getHelpPanel
 * @returns {HTMLButtonElement}
 */
function buildHelpDeepLink(anchor, label, getHelpPanel) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'help-deep-link';
  btn.textContent = '?';
  btn.setAttribute('aria-label', label);
  btn.title = label;
  btn.dataset.helpAnchor = anchor;
  btn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    const panel = getHelpPanel ? getHelpPanel() : null;
    if (panel && typeof panel.openTo === 'function') {
      void panel.openTo('controls', `#${anchor}`);
    }
  });
  return btn;
}

/**
 * Mount the controls panel into `root` (a #controls element).
 *
 * @param {HTMLElement} root
 * @param {Array<{basename: string}>} cores
 * @param {ControlsCallbacks} cb
 */
export function mountControls(root, cores, cb) {
  root.innerHTML = '';

  // --- Model card + dropdown (model selection) --------------------------
  // The card lands above the core selector per the API contract: "Model card
  // at the top of the controls panel, above the core selector".
  // Skip when the caller didn't wire getModels — keeps the controls
  // panel backwards-compat with non-G test harnesses.
  let modelCard = null;
  if (typeof cb.getModels === 'function') {
    const modelSection = section('Model');
    modelSection.classList.add('controls-section-model');
    // help system — discoverability "?" icon next to the section
    // label that deep-links to controls.md#top-left-model-card-dropdown.
    if (typeof cb.getHelpPanel === 'function') {
      const labelEl = modelSection.querySelector('.controls-label');
      if (labelEl) {
        labelEl.appendChild(
          buildHelpDeepLink(
            'top-left-model-card-dropdown',
            'Learn about the model card and dropdown',
            cb.getHelpPanel,
          ),
        );
      }
    }
    modelCard = mountModelCard(modelSection, cb);
    root.appendChild(modelSection);
  }

  // --- Core selector ------------------------------------------------
  const coreSection = section('Core');
  const coreSelect = document.createElement('select');
  coreSelect.className = 'core-select';
  coreSelect.title = 'Select which TMA core to view';
  for (const c of cores) {
    const opt = document.createElement('option');
    opt.value = c.basename;
    opt.textContent = c.basename;
    coreSelect.appendChild(opt);
  }
  coreSelect.value = cb.getState().core || (cores[0] && cores[0].basename) || '';
  coreSelect.addEventListener('change', () => {
    // A core switch is a full reload — OSD doesn't re-init cleanly
    // across radically different tile sources (spec section 3.3).
    const target = `${window.location.pathname}?core=${encodeURIComponent(coreSelect.value)}`;
    window.location.assign(target);
  });
  coreSection.appendChild(coreSelect);
  root.appendChild(coreSection);

  // --- Base layer toggle (CODEX composite) -------------------------
  // Mounted between the core selector and the color-by section so the
  // base-layer choice (which determines what the viewer paints) is
  // visually grouped with the core selector. Disabled when the active
  // core lacks a paired CODEX composite — the disabled tooltip cites
  // the same wording the spec mandates.
  const baseSection = section('Base layer');
  // help system — discoverability "?" next to the section label
  // that deep-links to controls.md#base-layer-he-codex.
  if (typeof cb.getHelpPanel === 'function') {
    const labelEl = baseSection.querySelector('.controls-label');
    if (labelEl) {
      labelEl.appendChild(
        buildHelpDeepLink(
          'base-layer-he-codex',
          'Learn about the H&E ↔ CODEX base-layer toggle',
          cb.getHelpPanel,
        ),
      );
    }
  }
  const baseGroup = buttonGroup(
    [
      ['he', 'H&E'],
      ['split', 'Split'],
      ['codex', 'CODEX'],
    ],
    cb.getState().base || 'he',
    (v) => {
      const cur = cb.getState();
      cb.onStateChange({ ...cur, base: v });
      rerender();
    },
    {
      he: 'H&E histology base layer (the input image to the model)',
      split:
        'Side-by-side: H&E on the left, CODEX on the right, pan + zoom '
        + 'synced so cells line up across the two stains.',
      codex:
        'Measured CODEX composite — DAPI (blue), panCK (green), '
        + 'CD45 (red)',
    },
  );
  baseSection.appendChild(baseGroup);
  root.appendChild(baseSection);

  // --- CODEX channel selector (visible only when base === 'codex') ---
  // The fixed 3-channel composite (DAPI/CD45/panCK) does not show the
  // marker the user picked above. When the bundle has per-marker truth
  // PNGs, swap the base layer to that channel so what's painted matches
  // the marker name on screen. The selector is intentionally hidden in
  // H&E mode so the controls panel doesn't grow extra rows the user
  // can't act on.
  const codexLayerSection = section('CODEX channel');
  codexLayerSection.classList.add('controls-section-codex-layer');
  const codexLayerSelect = document.createElement('select');
  codexLayerSelect.className = 'codex-layer-select';
  codexLayerSelect.title =
    'Pick which CODEX channel to paint as the base layer. "Composite" = '
    + 'the fixed DAPI/CD45/panCK overlay; a marker name = the raw '
    + 'single-channel truth (magma-mapped).';
  function refreshCodexLayerOptions() {
    const markers = (cb.getCodexMarkers && cb.getCodexMarkers()) || [];
    const current = (cb.getCodexLayer && cb.getCodexLayer()) || '';
    codexLayerSelect.innerHTML = '';
    const compositeOpt = document.createElement('option');
    compositeOpt.value = '';
    compositeOpt.textContent = 'Composite (DAPI/CD45/panCK)';
    codexLayerSelect.appendChild(compositeOpt);
    for (const m of markers) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} (raw truth)`;
      codexLayerSelect.appendChild(opt);
    }
    codexLayerSelect.value = current;
    // Disable when there are no per-marker PNGs — only composite is on
    // offer, so the select is a no-op (still visible so the user knows
    // why "Truth: Ki67 staining" isn't available).
    codexLayerSelect.disabled = markers.length === 0;
  }
  codexLayerSelect.addEventListener('change', () => {
    if (cb.setCodexLayer) cb.setCodexLayer(codexLayerSelect.value || null);
  });
  codexLayerSection.appendChild(codexLayerSelect);
  root.appendChild(codexLayerSection);

  // --- Focus mode (Marker XOR Phenotype) ----------------------------
  // commit 3 — the user picks one thing to look at: a single marker
  // (with a view sub-selector Pred / Truth / Error) OR phenotype
  // call coloring. The two used to coexist (marker dropdown + phenotype
  // chips simultaneously active), which made the viewer's coloring
  // ambiguous. This radio is the one-stop entry point.
  const focusSection = section('Focus');
  focusSection.classList.add('controls-section-focus');
  const focusGroup = buttonGroup(
    [
      ['marker', 'Marker'],
      ['phenotype', 'Phenotype'],
    ],
    _focusOf(cb.getState().color_by),
    (v) => {
      const cur = cb.getState();
      let next;
      if (v === 'phenotype') {
        // Switching to Phenotype focus: drop to phenotype coloring.
        // We preserve state.marker (the marker still drives the CODEX
        // base layer + side-panel marker bars) but the cell-overlay
        // color flips to phenotype calls.
        next = { ...cur, color_by: 'phenotype' };
      } else {
        // Switching to Marker focus: ensure a marker is set, pick a
        // default view if the current color_by is phenotype, and drop
        // any leftover phenotype filter. Keeping the filter would
        // silently hide cells while the chips are off-screen — that's
        // the "hidden state" UX bug commit 3 is fixing.
        const view = cur.color_by !== 'phenotype' ? cur.color_by : 'marker_pred';
        next = {
          ...cur,
          color_by: view,
          marker: cur.marker || DEFAULT_MARKER,
          phenotypes: null,
        };
      }
      cb.onStateChange(next);
      rerender();
    },
    {
      marker:
        'Color each cell by one marker (Pred / Truth / Error). The same '
        + 'marker drives the side-panel bars and the CODEX base layer.',
      phenotype:
        'Color each cell by its dominant phenotype call. The phenotype '
        + 'chips below let you filter to specific cell types.',
    },
  );
  focusSection.appendChild(focusGroup);
  root.appendChild(focusSection);

  // --- View (Pred / Truth / Error) — visible only in Marker focus ----
  // The color-by section also carries the "?" popover (API contract)
  // explaining all four modes side-by-side. The popover is a sibling
  // of the buttonGroup so its `position: absolute` anchors against the
  // controls-section block.
  const colorBySection = section('View');
  colorBySection.classList.add('controls-section-color-by');
  const colorByRow = document.createElement('div');
  colorByRow.className = 'color-by-row';
  const colorByGroup = buttonGroup(
    [
      ['marker_pred', 'Pred'],
      ['marker_truth', 'Truth'],
      ['marker_error', 'Error'],
    ],
    // Default the segmented button to 'marker_pred' when the user is
    // currently in Phenotype focus — the View row is hidden in that
    // case, so the default only matters on the next Marker-focus visit.
    cb.getState().color_by === 'phenotype' ? 'marker_pred' : cb.getState().color_by,
    (v) => {
      const cur = cb.getState();
      const next = { ...cur, color_by: v };
      if (!cur.marker) next.marker = DEFAULT_MARKER;
      cb.onStateChange(next);
      rerender();
    },
    /* titlesByValue */ COLOR_BY_TITLES,
  );
  colorByRow.appendChild(colorByGroup);

  // The "?" trigger button + popover panel. We use a click-toggle (not
  // hover) so keyboard users can reach it too; the panel also closes
  // on outside-click and Escape.
  const helpBtn = document.createElement('button');
  helpBtn.type = 'button';
  helpBtn.className = 'cb-help-btn';
  helpBtn.textContent = '?';
  helpBtn.setAttribute('aria-label', 'Explain color-by modes');
  helpBtn.setAttribute('aria-expanded', 'false');
  helpBtn.title = 'What do the color-by modes mean?';

  const popover = document.createElement('div');
  popover.className = 'cb-popover';
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-label', 'Color-by mode explanations');
  popover.hidden = true;
  popover.innerHTML = COLOR_BY_POPOVER_HTML;

  function closePopover() {
    popover.hidden = true;
    helpBtn.setAttribute('aria-expanded', 'false');
    document.removeEventListener('mousedown', onDocMouseDown, true);
    document.removeEventListener('keydown', onKeyDown, true);
  }
  function openPopover() {
    popover.hidden = false;
    helpBtn.setAttribute('aria-expanded', 'true');
    document.addEventListener('mousedown', onDocMouseDown, true);
    document.addEventListener('keydown', onKeyDown, true);
  }
  // Defensive: if mountControls re-renders while the popover is open,
  // the popover/helpBtn nodes are detached from the document but our
  // capture-phase listeners on `document` survive — and they'd keep
  // firing against dead nodes. Bail out (and unregister) the moment
  // we notice the popover is no longer in the DOM.
  function listenersOrphaned() {
    if (!popover.isConnected) {
      document.removeEventListener('mousedown', onDocMouseDown, true);
      document.removeEventListener('keydown', onKeyDown, true);
      return true;
    }
    return false;
  }
  function onDocMouseDown(ev) {
    if (listenersOrphaned()) return;
    if (popover.contains(ev.target) || helpBtn.contains(ev.target)) return;
    closePopover();
  }
  function onKeyDown(ev) {
    if (listenersOrphaned()) return;
    if (ev.key === 'Escape') closePopover();
  }
  helpBtn.addEventListener('click', () => {
    if (popover.hidden) openPopover();
    else closePopover();
  });
  // Static HTML below guarantees a `.cb-popover-close` button, but a
  // null-check keeps us safe if the template ever changes — the
  // popover still functions via outside-click / Escape dismissal.
  const closeBtn = popover.querySelector('.cb-popover-close');
  if (closeBtn) closeBtn.addEventListener('click', closePopover);

  // help system — keep the quick-reference popover (Option B in
  // the help system brief) AND add a "Learn more" deep-link to the
  // help panel's controls.md#color-by-mode anchor. The popover stays
  // useful for at-a-glance refreshes; the help link is for deeper
  // understanding.
  if (typeof cb.getHelpPanel === 'function') {
    const learnMore = document.createElement('button');
    learnMore.type = 'button';
    learnMore.className = 'cb-popover-learn-more';
    learnMore.textContent = 'Learn more →';
    learnMore.title = 'Open the help panel to "Color-by mode"';
    learnMore.addEventListener('click', (ev) => {
      ev.stopPropagation();
      closePopover();
      const panel = cb.getHelpPanel ? cb.getHelpPanel() : null;
      if (panel && typeof panel.openTo === 'function') {
        void panel.openTo('controls', '#color-by-mode');
      }
    });
    popover.appendChild(learnMore);
  }

  colorByRow.appendChild(helpBtn);
  colorBySection.appendChild(colorByRow);
  colorBySection.appendChild(popover);
  root.appendChild(colorBySection);

  // --- Marker selector ----------------------------------------------
  const markerSection = section('Marker');
  const markerSelect = document.createElement('select');
  markerSelect.className = 'marker-select';
  markerSelect.title =
    'Pick which marker the Pred / Truth / Error color-by modes show';
  for (const m of VALID_MARKERS) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    markerSelect.appendChild(opt);
  }
  markerSelect.value = cb.getState().marker || DEFAULT_MARKER;
  markerSelect.addEventListener('change', () => {
    const cur = cb.getState();
    cb.onStateChange({ ...cur, marker: markerSelect.value });
  });
  markerSection.appendChild(markerSelect);
  // commit 7 — probability filter slider. Hides cell polygons whose
  // pred probability for the active marker is below the threshold;
  // only meaningful when color_by === 'marker_pred' (the only view
  // that maps color_value to a probability). The slider stays in DOM
  // in other views but is hidden so toggling Truth / Error doesn't
  // make the filter disappear mid-session.
  const probFilterRow = document.createElement('div');
  probFilterRow.className = 'controls-prob-filter';
  const probLabel = document.createElement('label');
  probLabel.className = 'controls-prob-label';
  probLabel.innerHTML = `
    <span class="controls-prob-text">Min probability</span>
    <span class="controls-prob-value" data-role="prob-value">0.00</span>
  `;
  const probInput = document.createElement('input');
  probInput.type = 'range';
  probInput.min = '0';
  probInput.max = '1';
  probInput.step = '0.05';
  probInput.value = String(cb.getState().min_prob || 0);
  probInput.className = 'controls-prob-slider';
  probInput.title =
    'Hide cells whose predicted probability for the selected marker is '
    + 'below this threshold. Set to 0 to show every cell again.';
  probInput.addEventListener('input', () => {
    const cur = cb.getState();
    const v = parseFloat(probInput.value);
    const next = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
    const valueEl = probFilterRow.querySelector('[data-role="prob-value"]');
    if (valueEl) valueEl.textContent = next.toFixed(2);
    cb.onStateChange({ ...cur, min_prob: next });
  });
  probFilterRow.appendChild(probLabel);
  probFilterRow.appendChild(probInput);
  markerSection.appendChild(probFilterRow);
  // commit 4 — AP badge directly under the marker dropdown so the user
  // sees per-marker model performance next to the selection (without
  // hopping to the Stats tab). Text is filled in by rerender(); we
  // create the element once here so the layout doesn't reflow.
  const markerApBadge = document.createElement('div');
  markerApBadge.className = 'controls-ap-badge';
  markerApBadge.title =
    'Macro average precision on the validation cohort for the selected '
    + 'marker. Higher = better separability of positive vs negative.';
  markerSection.appendChild(markerApBadge);
  root.appendChild(markerSection);

  // --- Mode (nucleus/expanded) --------------------------------------
  const modeSection = section('Mode');
  const modeGroup = buttonGroup(
    [
      ['expanded', 'Expanded'],
      ['nucleus', 'Nucleus'],
    ],
    cb.getState().mode,
    (v) => {
      const cur = cb.getState();
      cb.onStateChange({ ...cur, mode: v });
      rerender();
    },
    {
      expanded:
        'Nucleus polygon + 8 px cytoplasmic ring — captures whole-cell '
        + 'CODEX signal',
      nucleus: 'Filled nucleus polygon only — DAPI-anchored shape',
    },
  );
  modeSection.appendChild(modeGroup);
  root.appendChild(modeSection);

  // --- Phenotype chips ----------------------------------------------
  const phenoSection = section('Phenotypes');
  const phenoChips = document.createElement('div');
  phenoChips.className = 'phenotype-chips';
  const allKeys = [...PHENOTYPE_NAMES, NO_CALL_KEY];

  function chipState(key) {
    const cur = cb.getState();
    return cur.phenotypes === null || cur.phenotypes.has(key);
  }

  for (const key of allKeys) {
    const chip = document.createElement('button');
    chip.className = 'pheno-chip';
    chip.dataset.key = key;
    chip.type = 'button';
    // commit 4 — show name + AP on the chip when validation data is
    // loaded. The AP suffix is updated in rerender() so it snaps in
    // when the async fetch resolves.
    const nameEl = document.createElement('span');
    nameEl.className = 'pheno-chip-name';
    nameEl.textContent = key;
    chip.appendChild(nameEl);
    const apEl = document.createElement('span');
    apEl.className = 'pheno-chip-ap';
    chip.appendChild(apEl);
    chip.title =
      key === NO_CALL_KEY
        ? 'Cells whose dominant phenotype score did not clear the calibrated '
          + 'threshold. Toggle to hide / show.'
        : `Show/hide ${key} cells in the viewer (filter is client-side; no fetch)`;
    chip.setAttribute('aria-pressed', String(chipState(key)));
    chip.addEventListener('click', () => {
      const cur = cb.getState();
      // Normalize: if phenotypes==null, start from "all on" so the
      // first chip click reads as "turn this one off".
      let active = cur.phenotypes ? new Set(cur.phenotypes) : new Set(allKeys);
      if (active.has(key)) active.delete(key);
      else active.add(key);
      // If every chip is on after the toggle, collapse back to null
      // (the canonical "all on" representation).
      if (active.size === allKeys.length) active = null;
      cb.onStateChange({ ...cur, phenotypes: active });
      rerender();
    });
    phenoChips.appendChild(chip);
  }
  phenoSection.appendChild(phenoChips);
  root.appendChild(phenoSection);

  // --- Low-zoom toggle ----------------------------------------------
  const zoomSection = section('At zoom < 0.5x');
  const zoomLabel = document.createElement('label');
  zoomLabel.className = 'toggle-row';
  zoomLabel.title =
    'Draw only filled outlines (no anti-aliased fills) when zoomed out '
    + 'to keep the overlay responsive on dense cores';
  const zoomToggle = document.createElement('input');
  zoomToggle.type = 'checkbox';
  zoomToggle.checked = cb.getExpandedOnlyAtLowZoom();
  zoomToggle.title = zoomLabel.title;
  zoomToggle.addEventListener('change', () => {
    cb.setExpandedOnlyAtLowZoom(zoomToggle.checked);
  });
  zoomLabel.appendChild(zoomToggle);
  zoomLabel.append(' Outline only');
  zoomSection.appendChild(zoomLabel);
  root.appendChild(zoomSection);

  // --- Comparison toggle --------------------------------------------
  const compareSection = section('Comparison');
  const compareLabel = document.createElement('label');
  compareLabel.className = 'toggle-row';
  const compareToggle = document.createElement('input');
  compareToggle.type = 'checkbox';
  compareToggle.checked = !!cb.getState().compare;
  compareToggle.disabled = !cb.hasTruth();
  compareLabel.title = cb.hasTruth()
    ? 'Compare prediction vs CODEX truth side-by-side in the cell detail panel'
    : 'No CODEX truth in this core';
  compareToggle.title = compareLabel.title;
  compareToggle.addEventListener('change', () => {
    const cur = cb.getState();
    cb.onStateChange({ ...cur, compare: compareToggle.checked });
  });
  compareLabel.appendChild(compareToggle);
  compareLabel.append(' Pred vs Truth');
  compareSection.appendChild(compareLabel);
  root.appendChild(compareSection);

  // ------------------------------------------------------------------
  function rerender() {
    // Light re-render: update the pieces whose visibility / selection
    // depends on the current state without rebuilding the DOM.
    const cur = cb.getState();
    const focus = _focusOf(cur.color_by);
    coreSelect.value = cur.core || '';
    // Focus radio
    for (const b of focusGroup.querySelectorAll('button')) {
      b.setAttribute('aria-pressed', String(b.dataset.value === focus));
    }
    // View buttons (Pred / Truth / Error) — pressed state tracks
    // color_by; the section is hidden in Phenotype focus.
    for (const b of colorByGroup.querySelectorAll('button')) {
      b.setAttribute('aria-pressed', String(b.dataset.value === cur.color_by));
    }
    colorBySection.style.display = focus === 'marker' ? '' : 'none';
    // Marker dropdown — visible only in Marker focus per the
    // exclusive-selection rule.
    markerSection.style.display = focus === 'marker' ? '' : 'none';
    if (cur.marker) markerSelect.value = cur.marker;
    // Probability slider: visible + active across every Marker focus
    // view. The backend ships pred_prob alongside color_value so the
    // same filter applies whether the cells are colored by Pred,
    // Truth, or Error — switching view keeps the same set of cells
    // visible.
    probFilterRow.style.display = focus === 'marker' ? '' : 'none';
    probInput.disabled = false;
    probInput.title =
      'Hide cells whose predicted probability for the selected marker is '
      + 'below this threshold. Applies in Pred, Truth, and Error views. '
      + 'Set to 0 to show every cell again.';
    probFilterRow.classList.remove('controls-prob-filter-disabled');
    const target = (cur.min_prob || 0).toFixed(2);
    if (probInput.value !== String(cur.min_prob || 0)) {
      probInput.value = String(cur.min_prob || 0);
    }
    const probValueEl = probFilterRow.querySelector('[data-role="prob-value"]');
    if (probValueEl) probValueEl.textContent = target;
    // commit 4 — refresh the marker AP badge. Reads the current marker
    // and asks the validation cache; the cache fires its own re-render
    // when the fetch lands so the value snaps in asynchronously.
    if (markerApBadge) {
      const ap = cb.getValidationAp ? cb.getValidationAp('marker', cur.marker) : null;
      if (cur.marker && typeof ap === 'number' && Number.isFinite(ap)) {
        markerApBadge.textContent = `${cur.marker} — AP ${ap.toFixed(2)} (val)`;
        markerApBadge.hidden = false;
      } else if (cur.marker) {
        markerApBadge.textContent = `${cur.marker} — AP loading…`;
        markerApBadge.hidden = false;
      } else {
        markerApBadge.hidden = true;
      }
    }
    // Mode group
    for (const b of modeGroup.querySelectorAll('button')) {
      b.setAttribute('aria-pressed', String(b.dataset.value === cur.mode));
    }
    // Phenotype chips — visible only in Phenotype focus. In Marker
    // focus they would silently filter the overlay (the old confusing
    // behavior); hide the whole row so the user sees one thing at a
    // time.
    phenoSection.style.display = focus === 'phenotype' ? '' : 'none';
    for (const c of phenoChips.querySelectorAll('button')) {
      const on = chipState(c.dataset.key);
      c.setAttribute('aria-pressed', String(on));
      // commit 4 — append AP next to the chip name. ``no_call`` and any
      // phenotype the AP table omits get no suffix; everything else
      // shows ``AP 0.62`` once the validation fetch lands.
      const apEl = c.querySelector('.pheno-chip-ap');
      if (apEl) {
        const key = c.dataset.key;
        if (key === NO_CALL_KEY) {
          apEl.textContent = '';
        } else {
          const ap = cb.getValidationAp
            ? cb.getValidationAp('phenotype', key) : null;
          apEl.textContent = (typeof ap === 'number' && Number.isFinite(ap))
            ? ` · ${ap.toFixed(2)}`
            : '';
        }
      }
    }
    compareToggle.checked = !!cur.compare;
    compareToggle.disabled = !cb.hasTruth();
    // model selection: keep the model card lines in sync with state.model.
    if (modelCard) modelCard.rerender();
    // CODEX composite — base-layer group: the CODEX segment is
    // disabled when the active core has no paired composite. We
    // attach the spec-mandated tooltip on the disabled segment so the
    // user sees *why* it can't be clicked.
    const codexAvailable = cb.hasCodex ? cb.hasCodex() : false;
    for (const b of baseGroup.querySelectorAll('button')) {
      const value = b.dataset.value;
      b.setAttribute('aria-pressed', String(value === (cur.base || 'he')));
      if (value === 'codex') {
        b.disabled = !codexAvailable;
        b.title = codexAvailable
          ? 'Measured CODEX composite (DAPI / panCK / CD45)'
          : 'No paired CODEX for this core';
      } else if (value === 'split') {
        // Split mode also needs the CODEX pane; gate identically.
        b.disabled = !codexAvailable;
        b.title = codexAvailable
          ? 'Side-by-side H&E + CODEX with synced pan/zoom'
          : 'No paired CODEX for this core';
      } else {
        b.title = 'H&E base layer';
      }
    }
    // CODEX channel selector: visible whenever the CODEX pane is on
    // (codex single or split). Hidden in H&E-only so the panel is compact.
    const showCodexLayerSelect = (cur.base || 'he') !== 'he';
    codexLayerSection.style.display = showCodexLayerSelect ? '' : 'none';
    refreshCodexLayerOptions();
  }

  rerender();
  return { rerender };
}

function section(label) {
  const sec = document.createElement('div');
  sec.className = 'controls-section';
  const lab = document.createElement('div');
  lab.className = 'controls-label';
  lab.textContent = label;
  sec.appendChild(lab);
  return sec;
}

/**
 * Build a horizontal segmented button group.
 *
 * @param {Array<[string, string]>} options  ``[value, label]`` pairs.
 * @param {string} current   The currently-pressed value.
 * @param {(v: string) => void} onPick
 * @param {Record<string, string>} [titlesByValue]  Optional per-button
 *   ``title=`` text. Buttons without an entry fall back to the option
 *   label, which is fine for self-explanatory ones (e.g., "H&E").
 * @returns {HTMLDivElement}
 */
function buttonGroup(options, current, onPick, titlesByValue) {
  const grp = document.createElement('div');
  grp.className = 'btn-group';
  const titles = titlesByValue || {};
  for (const [value, label] of options) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn';
    b.dataset.value = value;
    b.textContent = label;
    b.setAttribute('aria-pressed', String(value === current));
    if (titles[value]) b.title = titles[value];
    b.addEventListener('click', () => onPick(value));
    grp.appendChild(b);
  }
  return grp;
}
