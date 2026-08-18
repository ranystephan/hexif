// Legend renderer — visual key for the currently active color_by.
//
// Phenotype mode: one swatch per phenotype name with a chip-style row.
// Marker_pred / marker_truth: a horizontal viridis gradient with 0..1
// tick labels.
// Marker_error: a diverging gradient with -1, 0, +1 labels.
//
// confidence display — three legend sections:
//   1. Active palette (categorical chips OR gradient bar) — title strip.
//   2. Caption: a one-liner explaining the current color_by mode.
//   3. Confidence-stratum chips (strong / moderate / weak), driven by
//      the /api/strata payload, with `title=` hover-text listing the
//      members of each bucket.

import { PHENOTYPE_NAMES, NO_CALL_KEY } from '../state.js';
import { resolveColor } from '../palette.js';

/**
 * @typedef {Object} StratumBucket
 * @property {string[]} markers
 * @property {string[]} phenotypes
 */

/**
 * @typedef {Object} StrataLookup
 * @property {StratumBucket} strong
 * @property {StratumBucket} moderate
 * @property {StratumBucket} weak
 */

// Per-mode caption text (API contract). Kept as a module constant so the
// frontend test (and any future i18n pass) has a single source of truth.
const COLOR_BY_CAPTIONS = {
  phenotype:
    'Each cell is colored by its phenotype call. Click a chip to filter.',
  marker_pred:
    'Each cell is colored by the predicted probability that the marker is positive (0 = blue, 1 = yellow).',
  marker_truth:
    'Each cell is colored by the measured CODEX truth (0 / 1). Toggle base to CODEX to see the truth signal directly.',
  marker_error:
    'Each cell is colored by `prediction − truth`; red = overprediction, blue = underprediction.',
};

/**
 * Build the hover-text for a stratum chip given the bucket payload.
 * Exported only so the unit test can validate the format without
 * dragging the full renderer through jsdom.
 *
 * @param {StratumBucket} bucket
 * @returns {string}
 */
export function stratumHoverText(bucket) {
  const markers = (bucket && bucket.markers) ? bucket.markers.join(', ') : '';
  const phenotypes =
    (bucket && bucket.phenotypes) ? bucket.phenotypes.join(', ') : '';
  return `Markers: ${markers || '—'} · Phenotypes: ${phenotypes || '—'}`;
}

/**
 * Render the legend into a container. Idempotent: safe to call on
 * every state change. The `strata` argument is the response payload of
 * GET /api/strata; when null the stratum chips section is omitted (the
 * page is still functional — the chips are a hint, not a control).
 *
 * @param {HTMLElement} root
 * @param {string} colorBy
 * @param {string|null} marker
 * @param {Object} palettes  ResolvedPalettes from palette.js loadPalettes()
 * @param {StrataLookup|null} [strata]
 * @param {{openTo: (id: string, anchor?: string|null) => Promise<void>}|null} [helpPanel]
 *   help system: when wired, a "?" deep-link is appended next to
 *   the stratum chips that opens the help panel to the
 *   "Calibrated confidence strata" anchor in overview.md.
 * @param {((kind: 'marker'|'phenotype', name: string) => (number|null))|undefined} [getValidationAp]
 *   commit 4: validation AP getter — when provided, the legend appends
 *   the per-marker / per-phenotype macro AP to the title strip so the
 *   user reads "Pred prob · Ki67 — AP 0.66" in one glance.
 */
export function renderLegend(root, colorBy, marker, palettes, strata, helpPanel, getValidationAp) {
  root.innerHTML = '';
  root.classList.add('legend');
  if (!palettes) return;

  // --- Active palette section (title + chips/gradient) --------------
  const paletteSection = document.createElement('div');
  paletteSection.className = 'legend-section legend-section-palette';

  const title = document.createElement('div');
  title.className = 'legend-title';
  if (colorBy === 'phenotype') {
    title.textContent = 'Phenotype';
  } else if (colorBy === 'marker_pred') {
    title.textContent = `Pred prob · ${marker || ''}`;
  } else if (colorBy === 'marker_truth') {
    title.textContent = `CODEX truth · ${marker || ''}`;
  } else if (colorBy === 'marker_error') {
    title.textContent = `Pred − truth · ${marker || ''}`;
  }
  title.dataset.colorBy = colorBy;
  if (marker) title.dataset.marker = marker;
  // commit 4 — inline AP next to the marker title so the user knows
  // how reliable the rendering is *for that marker on val* in one
  // glance. Phenotype mode shows no single-target AP here (each chip
  // has its own AP — surfaced in the controls panel beside the chip).
  if (typeof getValidationAp === 'function' && marker && colorBy !== 'phenotype') {
    const ap = getValidationAp('marker', marker);
    if (typeof ap === 'number' && Number.isFinite(ap)) {
      const apBadge = document.createElement('span');
      apBadge.className = 'legend-ap';
      apBadge.textContent = ` — AP ${ap.toFixed(2)}`;
      apBadge.title = `Macro average precision on the validation cohort for ${marker}`;
      title.appendChild(apBadge);
    }
  }
  paletteSection.appendChild(title);

  if (colorBy === 'phenotype') {
    const list = document.createElement('div');
    list.className = 'legend-categorical';
    const keys = [...PHENOTYPE_NAMES, NO_CALL_KEY];
    for (let i = 0; i < keys.length; i++) {
      const row = document.createElement('div');
      row.className = 'legend-row';
      const sw = document.createElement('span');
      sw.className = 'legend-swatch';
      sw.style.background = palettes.phenotypeHex[i];
      const lab = document.createElement('span');
      lab.className = 'legend-label';
      lab.textContent = keys[i];
      row.appendChild(sw);
      row.appendChild(lab);
      // commit 4 — per-phenotype AP next to each swatch so the user
      // sees calibration confidence per cell type. ``no_call`` and any
      // phenotype the AP table omits gets no badge.
      if (typeof getValidationAp === 'function' && keys[i] !== NO_CALL_KEY) {
        const ap = getValidationAp('phenotype', keys[i]);
        if (typeof ap === 'number' && Number.isFinite(ap)) {
          const apBadge = document.createElement('span');
          apBadge.className = 'legend-ap legend-ap-row';
          apBadge.textContent = `AP ${ap.toFixed(2)}`;
          apBadge.title =
            `Macro average precision on the validation cohort for ${keys[i]}`;
          row.appendChild(apBadge);
        }
      }
      list.appendChild(row);
    }
    paletteSection.appendChild(list);
  } else {
    // Sequential / diverging: render the same sampled colors used by
    // canvas_overlay.js via resolveColor(...). Keeping this path shared
    // prevents the legend from drifting from the actual polygon colors.
    const bar = document.createElement('div');
    bar.className = 'legend-gradient';
    bar.dataset.colorBy = colorBy;
    bar.setAttribute('aria-label', gradientAriaLabel(colorBy, marker));
    bar.style.background = gradientCss(colorBy, palettes);
    paletteSection.appendChild(bar);

    const ticks = document.createElement('div');
    ticks.className = 'legend-ticks';
    const labels =
      colorBy === 'marker_error'
        ? ['−1', '0', '+1']
        : ['0', '0.5', '1'];
    for (const t of labels) {
      const s = document.createElement('span');
      s.textContent = t;
      ticks.appendChild(s);
    }
    paletteSection.appendChild(ticks);
  }
  root.appendChild(paletteSection);

  // --- Per-mode caption ---------------------------------------------
  // The caption explains *what the color means* for the active mode,
  // not just what the marker is — the title strip already covers the
  // marker name. Renders below the palette so the user's eye lands on
  // the colors first and reads the explanation in the same glance.
  const caption = document.createElement('div');
  caption.className = 'legend-caption';
  caption.dataset.colorBy = colorBy;
  caption.textContent = COLOR_BY_CAPTIONS[colorBy] || '';
  root.appendChild(caption);

  // --- Confidence-stratum chips -------------------------------------
  // Three static chips: strong / moderate / weak. The chip is purely
  // documentary — clicking does nothing. Hover-text cites the markers
  // and phenotypes that calibrated into the bucket (per
  // src/hexif/pipeline/thresholds.py MARKER_CONFIDENCE +
  // PHENOTYPE_CONFIDENCE).
  if (strata) {
    const strataSection = document.createElement('div');
    strataSection.className = 'legend-strata';

    const strataLabel = document.createElement('span');
    strataLabel.className = 'legend-strata-label';
    strataLabel.textContent = 'Confidence:';
    strataSection.appendChild(strataLabel);

    for (const name of ['strong', 'moderate', 'weak']) {
      const bucket = strata[name];
      const chip = document.createElement('span');
      chip.className = `legend-stratum-chip legend-stratum-${name}`;
      chip.dataset.stratum = name;
      chip.textContent = name;
      chip.setAttribute('title', stratumHoverText(bucket || { markers: [], phenotypes: [] }));
      strataSection.appendChild(chip);
    }

    // help system — discoverability "?" deep-link next to the
    // stratum chips. Opens the help panel directly to the
    // "Calibrated confidence strata" section of overview.md.
    if (helpPanel && typeof helpPanel.openTo === 'function') {
      const helpBtn = document.createElement('button');
      helpBtn.type = 'button';
      helpBtn.className = 'help-deep-link';
      helpBtn.textContent = '?';
      helpBtn.title = 'Learn about the confidence strata';
      helpBtn.setAttribute('aria-label', 'Learn about the confidence strata');
      helpBtn.dataset.helpAnchor = 'calibrated-confidence-strata';
      helpBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        void helpPanel.openTo('overview', '#calibrated-confidence-strata');
      });
      strataSection.appendChild(helpBtn);
    }
    root.appendChild(strataSection);
  }
}

/**
 * Build the exact CSS gradient used by the legend. For marker_pred and
 * marker_truth, values are sampled on [0, 1]. For marker_error, values
 * are sampled on [-1, 1] and placed on the [0, 100]% CSS axis.
 *
 * @param {string} colorBy
 * @param {Object} palettes
 * @returns {string}
 */
export function gradientCss(colorBy, palettes) {
  const values = colorBy === 'marker_error'
    ? [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    : [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1];
  const stops = values.map((value) => {
    const rgb = resolveColor(palettes, colorBy, value);
    const pos = colorBy === 'marker_error' ? ((value + 1) * 50) : (value * 100);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${pos.toFixed(1)}%`;
  });
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

function gradientAriaLabel(colorBy, marker) {
  const prefix = marker ? `${marker} ` : '';
  if (colorBy === 'marker_pred') return `${prefix}predicted probability color scale`;
  if (colorBy === 'marker_truth') return `${prefix}CODEX truth color scale`;
  if (colorBy === 'marker_error') return `${prefix}prediction minus truth color scale`;
  return 'color scale';
}
