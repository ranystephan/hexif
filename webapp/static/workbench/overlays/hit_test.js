// Cell hit-test — picks a cell_id from an (x, y) point in image space.
//
// The OpenSeadragon viewer converts the mouse position to image-space
// coords via viewport.viewerElementToImageCoordinates(); we then look
// up which cell, if any, contains that point.
//
// Performance contract (spec section 3.3):
//   - O(N) bbox prefilter per mousemove, where N <= 5000 typical.
//   - At most ~2 point-in-polygon checks per move (mean: ~0.05).
//   - Throttled to 30 Hz via requestAnimationFrame in the caller.
//
// We pack the bboxes as a single Float32Array so the prefilter is one
// tight loop over contiguous memory rather than N attribute lookups.

/**
 * Build a HitIndex from a parsed GeoJSON FeatureCollection.
 *
 * @param {Object} geo  FeatureCollection
 * @returns {{
 *   bboxes: Float32Array,   // [minx, miny, maxx, maxy] per feature, length = 4*N
 *   ids: Int32Array,        // cell_id per feature
 *   rings: Array<Array<[number, number]>>,  // outer rings (closed)
 *   n: number,
 * }}
 */
export function buildHitIndex(geo) {
  const features = geo.features || [];
  const n = features.length;
  const bboxes = new Float32Array(n * 4);
  const ids = new Int32Array(n);
  const rings = new Array(n);
  for (let i = 0; i < n; i++) {
    const f = features[i];
    const ring = (f.geometry && f.geometry.coordinates && f.geometry.coordinates[0]) || [];
    rings[i] = ring;
    ids[i] = f.id;
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let j = 0; j < ring.length; j++) {
      const x = ring[j][0], y = ring[j][1];
      if (x < minx) minx = x;
      if (y < miny) miny = y;
      if (x > maxx) maxx = x;
      if (y > maxy) maxy = y;
    }
    bboxes[4 * i] = minx;
    bboxes[4 * i + 1] = miny;
    bboxes[4 * i + 2] = maxx;
    bboxes[4 * i + 3] = maxy;
  }
  return { bboxes, ids, rings, n };
}

/**
 * Return the cell_id under (x, y) in image space, or null. Picks the
 * first matching polygon — overlaps (e.g., expanded8 cytoplasm rings)
 * tie-break by feature order, which mirrors the canvas paint order.
 *
 * @param {ReturnType<typeof buildHitIndex>} idx
 * @param {number} x
 * @param {number} y
 * @returns {number|null}
 */
export function hitTest(idx, x, y) {
  const { bboxes, ids, rings, n } = idx;
  for (let i = 0; i < n; i++) {
    const o = 4 * i;
    if (x < bboxes[o] || x > bboxes[o + 2]) continue;
    if (y < bboxes[o + 1] || y > bboxes[o + 3]) continue;
    if (pointInPolygon(rings[i], x, y)) return ids[i];
  }
  return null;
}

/**
 * Ray-casting point-in-polygon for a closed ring.
 * The ring is closed (last == first); the algorithm tolerates that
 * because the wrap-edge cancels out.
 *
 * @param {Array<[number, number]>} ring
 * @param {number} x
 * @param {number} y
 * @returns {boolean}
 */
export function pointInPolygon(ring, x, y) {
  let inside = false;
  const n = ring.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const intersects =
      (yi > y) !== (yj > y) &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}
