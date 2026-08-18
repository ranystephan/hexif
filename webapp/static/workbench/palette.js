// Palette resolver — fetches the three /api/palette/* endpoints once
// at boot and exposes a uniform (color_by, color_value) -> [r,g,b,a]
// lookup.  The colormaps are tiny (≤ 10 entries) so we keep them in
// memory and interpolate on the fly rather than precomputing a 256-LUT
// (which would be wasted on 1-2 fixed zoom levels per session).

import { fetchPalette } from './api.js';
import { PHENOTYPE_NAMES, NO_CALL_KEY } from './state.js';

/**
 * @typedef {Object} ResolvedPalettes
 * @property {string[]}   phenotypeHex   index-aligned with PHENOTYPE_NAMES (+ no_call at end)
 * @property {Array<[number, number[]]>} viridis   [(t, [r,g,b])...] stops, t ascending
 * @property {Array<[number, number[]]>} diverging [(t, [r,g,b])...] stops, t ascending
 */

/** Parse '#rrggbb' into [r,g,b] ints. */
function hexToRgb(hex) {
  const s = hex.replace('#', '');
  return [
    Number.parseInt(s.slice(0, 2), 16),
    Number.parseInt(s.slice(2, 4), 16),
    Number.parseInt(s.slice(4, 6), 16),
  ];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Sample a sequential / diverging palette at t in [0, 1]. */
function sampleStops(stops, t) {
  if (t <= stops[0][0]) return stops[0][1];
  if (t >= stops[stops.length - 1][0]) return stops[stops.length - 1][1];
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (t <= t1) {
      const [t0, c0] = stops[i - 1];
      const u = (t - t0) / (t1 - t0);
      return [
        Math.round(lerp(c0[0], c1[0], u)),
        Math.round(lerp(c0[1], c1[1], u)),
        Math.round(lerp(c0[2], c1[2], u)),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

/**
 * Fetch and cache the three palettes. Concurrent — they're independent.
 * @param {{baseUrl?: string}} [opts]
 * @returns {Promise<ResolvedPalettes>}
 */
export async function loadPalettes(opts) {
  const baseUrl = (opts && opts.baseUrl) || null;
  const [pheno, vir, div] = await Promise.all([
    fetchPalette('phenotype', { baseUrl }),
    fetchPalette('marker_prob_viridis', { baseUrl }),
    fetchPalette('marker_error_diverging', { baseUrl }),
  ]);
  // Index the categorical palette by phenotype key so a missing entry
  // surfaces as a console.warn rather than a silent off-palette pixel.
  const byKey = new Map();
  for (const e of pheno.entries || []) byKey.set(e.key, e.hex);
  const phenotypeHex = [];
  for (const name of PHENOTYPE_NAMES) {
    const hx = byKey.get(name);
    if (!hx) {
      console.warn(`palette: phenotype missing from palettes.json: ${name}`);
      phenotypeHex.push('#888888');
    } else {
      phenotypeHex.push(hx);
    }
  }
  phenotypeHex.push(byKey.get(NO_CALL_KEY) || '#cccccc');

  const toStops = (palette) =>
    (palette.stops || [])
      .map((s) => /** @type {[number, number[]]} */ ([s.t, hexToRgb(s.hex)]))
      .sort((a, b) => a[0] - b[0]);

  return {
    phenotypeHex,
    viridis: toStops(vir),
    diverging: toStops(div),
  };
}

/**
 * Resolve (color_by, value) -> [r, g, b].
 * value is whatever the GeoJSON feature carries in `color_value`:
 *   - phenotype:      float-coded phenotype index, or -1 for no_call
 *   - marker_pred:    sigmoid prob in [0, 1]
 *   - marker_truth:   {0, 1}
 *   - marker_error:   prob diff in [-1, 1]
 *
 * @param {ResolvedPalettes} pals
 * @param {string} colorBy
 * @param {number} value
 * @returns {number[]}  [r, g, b]
 */
export function resolveColor(pals, colorBy, value) {
  if (colorBy === 'phenotype') {
    let idx = Math.round(value);
    // -1 sentinel = no call: tail entry in phenotypeHex.
    if (!Number.isFinite(idx) || idx < 0 || idx >= PHENOTYPE_NAMES.length) {
      idx = PHENOTYPE_NAMES.length; // no_call slot
    }
    return hexToRgb(pals.phenotypeHex[idx]);
  }
  if (colorBy === 'marker_pred' || colorBy === 'marker_truth') {
    let t = Number.isFinite(value) ? value : 0.0;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return sampleStops(pals.viridis, t);
  }
  if (colorBy === 'marker_error') {
    // Map [-1, 1] -> [0, 1] for the diverging palette.
    let t = Number.isFinite(value) ? (value + 1) * 0.5 : 0.5;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return sampleStops(pals.diverging, t);
  }
  return [136, 136, 136]; // fallback grey
}

/** Convert hex to a CSS rgba() with the requested alpha. */
export function hexToCssRgba(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

export { hexToRgb };
