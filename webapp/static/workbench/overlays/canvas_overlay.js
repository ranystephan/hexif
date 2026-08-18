// Bulk cell-polygon renderer on top of OpenSeadragon.
//
// One <canvas> per overlay instance, sized to the OSD viewer's drawing
// rect. We listen to OSD's animation / update-viewport / open events
// and re-blit on the next requestAnimationFrame — coalescing multiple
// events into a single repaint so a fast pan doesn't queue dozens of
// redraws (spec section 3.3).
//
// Coordinate system: GeoJSON polygons live in image-space (pixel
// coordinates of the source HE). For each visible cell we convert each
// vertex via OSD's viewport.imageToViewportCoordinates +
// viewportToViewerElementCoordinates, then stroke / fill into the
// canvas. The OSD APIs return CSS pixels; we multiply by
// devicePixelRatio at the canvas level so HiDPI screens get sharp
// outlines (the canvas backing store is sized accordingly).

import { resolveColor } from '../palette.js';

const FILL_ALPHA = 0.35;
const STROKE_ALPHA = 0.9;
const STROKE_WIDTH_CSS = 1.0; // CSS px, multiplied by DPR internally

/**
 * Mount a canvas overlay onto an OpenSeadragon viewer.
 *
 * @param {Object} viewer  OSD viewer instance
 * @returns {{
 *   setGeoJson: (geo: Object|null) => void,
 *   setPalettes: (pals: Object) => void,
 *   setColorBy: (s: string) => void,
 *   setExpandedOnlyAtLowZoom: (b: boolean) => void,
 *   setPhenotypeFilter: (activeSet: Set<string>|null) => void,
 *   getCanvas: () => HTMLCanvasElement,
 *   getDrawCount: () => number,
 *   getVisibleCount: () => number,
 *   destroy: () => void,
 * }}
 */
export function attachCanvasOverlay(viewer) {
  const container = viewer.container;
  const canvas = document.createElement('canvas');
  canvas.className = 'cell-overlay';
  canvas.style.position = 'absolute';
  canvas.style.left = '0';
  canvas.style.top = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.pointerEvents = 'none';
  canvas.style.zIndex = '40';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  /** @type {Object|null} */
  let geo = null;
  /** @type {Object|null} */
  let palettes = null;
  let colorBy = 'phenotype';
  let expandedOnlyAtLowZoom = true;
  // Phenotype filter — the API contract: phenotypes toggle is a client-side
  // filter over GeoJSON properties, not a re-fetch. `null` means "all
  // phenotypes visible" (canonical "no filter" state). A non-null Set
  // restricts the draw to features whose `properties.phenotype_call`
  // is in the set; null / missing `phenotype_call` is treated as
  // `'no_call'` so the derived chip controls those cells.
  /** @type {Set<string>|null} */
  let phenotypeFilter = null;
  // commit 7 — probability threshold for the active marker. Cells whose
  // ``properties.color_value`` falls below this are hidden from the
  // overlay. Only applied in ``marker_pred`` mode (where color_value is
  // a probability in [0, 1]); other modes ignore it. 0 = show everything.
  let probabilityThreshold = 0;
  let drawScheduled = false;
  let drawCount = 0;       // number of full repaints since attach
  let visibleCount = 0;    // n features drawn at the last repaint

  function getDpr() {
    return window.devicePixelRatio || 1;
  }

  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    const dpr = getDpr();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
  }

  function scheduleDraw() {
    if (drawScheduled) return;
    drawScheduled = true;
    requestAnimationFrame(() => {
      drawScheduled = false;
      draw();
    });
  }

  function draw() {
    resizeCanvas();
    const dpr = getDpr();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!geo || !palettes) return;
    if (!viewer.world || viewer.world.getItemCount() === 0) return;

    const features = geo.features || [];
    const n = features.length;

    // Compute the visible image-coordinate viewport rect once so the
    // bbox cull can skip in O(1) per feature.
    const tiledImage = viewer.world.getItemAt(0);
    if (!tiledImage) return;
    const viewportRect = viewer.viewport.getBounds(true);
    const tl = tiledImage.viewportToImageCoordinates(viewportRect.x, viewportRect.y);
    const br = tiledImage.viewportToImageCoordinates(
      viewportRect.x + viewportRect.width,
      viewportRect.y + viewportRect.height,
    );
    const visMinX = Math.min(tl.x, br.x);
    const visMaxX = Math.max(tl.x, br.x);
    const visMinY = Math.min(tl.y, br.y);
    const visMaxY = Math.max(tl.y, br.y);

    // Image-zoom: ratio of viewer pixels to image pixels at the
    // current zoom. We treat zoom < 0.5x (i.e., 1 image-px shown as
    // < 0.5 viewer-px) as "low zoom" and skip the nucleus stroke per
    // spec section 3.3 risk #6.
    const imageZoom = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
    const lowZoom = imageZoom < 0.5;

    ctx.lineWidth = STROKE_WIDTH_CSS * dpr;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    let drawn = 0;
    for (let i = 0; i < n; i++) {
      const f = features[i];
      const ring = f.geometry && f.geometry.coordinates && f.geometry.coordinates[0];
      if (!ring || ring.length < 3) continue;

      // Phenotype filter (client-side, per the API contract). null / missing
      // phenotype_call is normalized to 'no_call' so the derived
      // no-call chip can hide/show those cells like any other chip.
      if (phenotypeFilter !== null) {
        const props = f.properties || {};
        const call = props.phenotype_call == null ? 'no_call' : props.phenotype_call;
        if (!phenotypeFilter.has(call)) continue;
      }

      // Probability threshold: hide cells the model isn't confident
      // about. The backend ships ``properties.pred_prob`` independent
      // of color_by, so the same filter applies across Pred / Truth /
      // Error views — the user sets a threshold once and the set of
      // visible cells persists when they toggle the color mode. In
      // phenotype mode pred_prob is null on every cell, so a non-zero
      // threshold there hides everything (the slider isn't visible in
      // phenotype focus, so this is unreachable from the UI; the test
      // gate exercises the boundary anyway).
      if (probabilityThreshold > 0 && colorBy !== 'phenotype') {
        const probProp = f.properties && f.properties.pred_prob;
        const prob = typeof probProp === 'number' ? probProp : NaN;
        if (!Number.isFinite(prob) || prob < probabilityThreshold) continue;
      }

      // Inline bbox cull
      let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
      for (let j = 0; j < ring.length; j++) {
        const x = ring[j][0], y = ring[j][1];
        if (x < minx) minx = x;
        if (y < miny) miny = y;
        if (x > maxx) maxx = x;
        if (y > maxy) maxy = y;
      }
      if (maxx < visMinX || minx > visMaxX) continue;
      if (maxy < visMinY || miny > visMaxY) continue;

      // Resolve color from feature properties
      const props = f.properties || {};
      const rgb = resolveColor(palettes, colorBy, Number(props.color_value));

      // Path
      ctx.beginPath();
      for (let j = 0; j < ring.length; j++) {
        const pt = tiledImage.imageToViewerElementCoordinates(
          new window.OpenSeadragon.Point(ring[j][0], ring[j][1]),
        );
        if (j === 0) ctx.moveTo(pt.x * dpr, pt.y * dpr);
        else ctx.lineTo(pt.x * dpr, pt.y * dpr);
      }
      ctx.closePath();

      // Fill is the colored signal; stroke is a thin outline of the
      // same hue at higher alpha so cell boundaries remain visible
      // when fills overlap.  At low zoom we skip the stroke entirely;
      // the bulk fill alone reads as a tinted layer.
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${FILL_ALPHA})`;
      ctx.fill();
      if (!lowZoom || !expandedOnlyAtLowZoom) {
        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${STROKE_ALPHA})`;
        ctx.stroke();
      }
      drawn++;
    }
    drawCount += 1;
    visibleCount = drawn;
    canvas.dataset.drawCount = String(drawCount);
    canvas.dataset.visibleCount = String(drawn);
  }

  // Wire up OSD events. animation fires per-frame during pan/zoom;
  // update-viewport fires when the viewport rect changes; open fires
  // on a new tileSource; resize fires on container size changes.
  const onChange = () => scheduleDraw();
  viewer.addHandler('animation', onChange);
  viewer.addHandler('update-viewport', onChange);
  viewer.addHandler('open', onChange);
  viewer.addHandler('resize', onChange);

  // Initial paint when the viewer is already open (state recovery).
  if (viewer.world && viewer.world.getItemCount() > 0) scheduleDraw();

  return {
    setGeoJson(g) {
      geo = g;
      scheduleDraw();
    },
    setPalettes(p) {
      palettes = p;
      scheduleDraw();
    },
    setColorBy(s) {
      colorBy = s;
      scheduleDraw();
    },
    setExpandedOnlyAtLowZoom(b) {
      expandedOnlyAtLowZoom = !!b;
      scheduleDraw();
    },
    /**
     * Restrict the overlay to features whose `phenotype_call` is in
     * `activeSet`. Pass `null` to clear the filter (= all phenotypes
     * visible). Per the API contract this is the only path that should react
     * to phenotype-chip toggles — no fetch is issued.
     *
     * @param {Set<string>|null} activeSet
     */
    setPhenotypeFilter(activeSet) {
      phenotypeFilter = activeSet instanceof Set ? activeSet : null;
      scheduleDraw();
    },
    /**
     * commit 7 — set the minimum prediction probability for a cell to
     * appear in the overlay. Clamps the input to [0, 1]; values <= 0
     * disable the filter (= show every cell). The filter only applies
     * in ``marker_pred`` mode — Truth / Error palettes still draw all
     * cells.
     *
     * @param {number} threshold
     */
    setProbabilityThreshold(threshold) {
      const n = Number(threshold);
      probabilityThreshold = Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;
      scheduleDraw();
    },
    getCanvas() {
      return canvas;
    },
    getDrawCount() {
      return drawCount;
    },
    getVisibleCount() {
      return visibleCount;
    },
    /**
     * Total number of polygon features currently bound to the overlay
     * (independent of viewport culling and phenotype filtering). Tests
     * use this as a "the GeoJSON loaded" signal that doesn't race the
     * first rAF-driven repaint.
     */
    get featureCount() {
      return geo && Array.isArray(geo.features) ? geo.features.length : 0;
    },
    destroy() {
      viewer.removeHandler('animation', onChange);
      viewer.removeHandler('update-viewport', onChange);
      viewer.removeHandler('open', onChange);
      viewer.removeHandler('resize', onChange);
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    },
  };
}
