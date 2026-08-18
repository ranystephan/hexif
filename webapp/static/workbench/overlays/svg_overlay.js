// Hover + selection highlight overlay.
//
// One inline <svg> with at most two active <path> elements (hover +
// selected). The rationale (per spec section 3.3): SVG strokes
// dash-animate cleanly and respect drop-shadow filters that the
// canvas overlay can't replicate without an expensive offscreen
// composition pass — but only on a tiny element count. Bulk drawing
// stays in canvas_overlay.js.

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Mount an SVG overlay onto an OpenSeadragon viewer.
 *
 * @param {Object} viewer  OSD viewer instance
 * @returns {{
 *   setHover: (cellId: number|null, ring: Array<[number, number]>|null) => void,
 *   setSelected: (cellId: number|null, ring: Array<[number, number]>|null) => void,
 *   destroy: () => void,
 * }}
 */
export function attachSvgOverlay(viewer) {
  const container = viewer.container;
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'cell-svg-overlay');
  svg.style.position = 'absolute';
  svg.style.left = '0';
  svg.style.top = '0';
  svg.style.width = '100%';
  svg.style.height = '100%';
  svg.style.pointerEvents = 'none';
  svg.style.zIndex = '50';
  svg.setAttribute('overflow', 'visible');
  container.appendChild(svg);

  // Two <path> elements at most: hover (dashed) and selected (solid).
  const hoverPath = document.createElementNS(SVG_NS, 'path');
  hoverPath.setAttribute('class', 'cell-hover-path');
  hoverPath.setAttribute('fill', 'none');
  hoverPath.setAttribute('stroke', '#ffffff');
  hoverPath.setAttribute('stroke-width', '1.5');
  hoverPath.setAttribute('stroke-dasharray', '3 2');
  hoverPath.setAttribute('vector-effect', 'non-scaling-stroke');
  hoverPath.setAttribute('opacity', '0');
  svg.appendChild(hoverPath);

  const selectedPath = document.createElementNS(SVG_NS, 'path');
  selectedPath.setAttribute('class', 'cell-selected-path');
  selectedPath.setAttribute('fill', 'none');
  selectedPath.setAttribute('stroke', '#ffeb3b');
  selectedPath.setAttribute('stroke-width', '2.5');
  selectedPath.setAttribute('vector-effect', 'non-scaling-stroke');
  selectedPath.setAttribute('filter', 'drop-shadow(0 0 3px rgba(255,235,59,0.9))');
  selectedPath.setAttribute('opacity', '0');
  svg.appendChild(selectedPath);

  /** @type {Array<[number, number]>|null} */
  let hoverRing = null;
  /** @type {Array<[number, number]>|null} */
  let selectedRing = null;
  let hoverId = null;
  let selectedId = null;

  function ringToD(ring) {
    if (!ring || ring.length < 2 || !viewer.world || viewer.world.getItemCount() === 0) {
      return '';
    }
    const tiledImage = viewer.world.getItemAt(0);
    const parts = [];
    for (let i = 0; i < ring.length; i++) {
      const pt = tiledImage.imageToViewerElementCoordinates(
        new window.OpenSeadragon.Point(ring[i][0], ring[i][1]),
      );
      parts.push(`${i === 0 ? 'M' : 'L'}${pt.x.toFixed(2)},${pt.y.toFixed(2)}`);
    }
    parts.push('Z');
    return parts.join(' ');
  }

  function repaint() {
    if (hoverRing) {
      hoverPath.setAttribute('d', ringToD(hoverRing));
      hoverPath.setAttribute('opacity', '1');
    } else {
      hoverPath.setAttribute('opacity', '0');
    }
    if (selectedRing) {
      selectedPath.setAttribute('d', ringToD(selectedRing));
      selectedPath.setAttribute('opacity', '1');
      selectedPath.dataset.cellId = String(selectedId);
    } else {
      selectedPath.setAttribute('opacity', '0');
      delete selectedPath.dataset.cellId;
    }
  }

  let scheduled = false;
  function scheduleRepaint() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      repaint();
    });
  }

  const onChange = () => scheduleRepaint();
  viewer.addHandler('animation', onChange);
  viewer.addHandler('update-viewport', onChange);
  viewer.addHandler('open', onChange);
  viewer.addHandler('resize', onChange);

  return {
    setHover(cellId, ring) {
      hoverId = cellId;
      hoverRing = ring || null;
      scheduleRepaint();
      // Cursor feedback — pointer when hovering a cell.
      container.style.cursor = ring ? 'pointer' : 'default';
    },
    setSelected(cellId, ring) {
      selectedId = cellId;
      selectedRing = ring || null;
      scheduleRepaint();
    },
    destroy() {
      viewer.removeHandler('animation', onChange);
      viewer.removeHandler('update-viewport', onChange);
      viewer.removeHandler('open', onChange);
      viewer.removeHandler('resize', onChange);
      if (svg.parentNode) svg.parentNode.removeChild(svg);
    },
  };
}
