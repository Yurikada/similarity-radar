// 2D Gaussian kernel density estimation.

export function stdev(coords) {
  const n = coords.length;
  let mx = 0, my = 0;
  for (const p of coords) { mx += p[0] / n; my += p[1] / n; }
  let vx = 0, vy = 0;
  for (const p of coords) { vx += (p[0] - mx) ** 2 / n; vy += (p[1] - my) ** 2 / n; }
  return Math.sqrt((vx + vy) / 2);
}

// Scott's rule bandwidth for 2D.
export function scottBandwidth(coords) {
  return stdev(coords) * Math.pow(coords.length, -1 / 6);
}

/** density at each input point (leave-one-out sum of Gaussian kernels). */
export function pointDensities(coords, h) {
  const n = coords.length;
  const inv = 1 / (2 * h * h);
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = coords[i][0] - coords[j][0], dy = coords[i][1] - coords[j][1];
      s += Math.exp(-(dx * dx + dy * dy) * inv);
    }
    out[i] = s / ((n - 1) * 2 * Math.PI * h * h);
  }
  return out;
}

/** Abramson variable-bandwidth (adaptive) KDE on a grid.
 * Pilot density -> per-point bandwidth h_i = h0*(pilot_i/G)^(-1/2) (G = geo mean).
 * Dense regions get sharper kernels, sparse tails/edges get wider ones, which
 * reduces both over-smoothing of peaks and the edge under-estimation of fixed-h KDE. */
export function adaptiveGridDensity(coords, h0, W = 64, H = 48, pad = 2.5) {
  const n = coords.length;
  const pilot = pointDensities(coords, h0);
  let logsum = 0;
  for (const p of pilot) logsum += Math.log(Math.max(p, 1e-12));
  const G = Math.exp(logsum / n);
  const h = pilot.map((p) => h0 * Math.pow(Math.max(p, 1e-12) / G, -0.5));

  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of coords) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  const hmax = Math.max(...h);
  x0 -= pad * hmax; y0 -= pad * hmax; x1 += pad * hmax; y1 += pad * hmax;
  const grid = [];
  let max = 0;
  for (let gy = 0; gy < H; gy++) {
    const row = [];
    const py = y0 + (gy / (H - 1)) * (y1 - y0);
    for (let gx = 0; gx < W; gx++) {
      const px = x0 + (gx / (W - 1)) * (x1 - x0);
      let s = 0;
      for (let j = 0; j < n; j++) {
        const dx = px - coords[j][0], dy = py - coords[j][1], hj = h[j];
        s += Math.exp(-(dx * dx + dy * dy) / (2 * hj * hj)) / (hj * hj);
      }
      row.push(s);
      if (s > max) max = s;
    }
    grid.push(row);
  }
  return { grid, max, bbox: [x0, y0, x1, y1] };
}

// Marching-squares contour extraction. Returns segments in DATA coords per level.
const MS = {
  1: [[0, 3]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[0, 3], [1, 2]],
  6: [[0, 2]], 7: [[3, 2]], 8: [[3, 2]], 9: [[0, 2]], 10: [[0, 1], [3, 2]],
  11: [[1, 2]], 12: [[3, 1]], 13: [[0, 1]], 14: [[0, 3]],
};
export function contourSegments(grid, bbox, level) {
  const H = grid.length, W = grid[0].length, [x0, y0, x1, y1] = bbox;
  const mapX = (gx) => x0 + (gx / (W - 1)) * (x1 - x0);
  const mapY = (gy) => y0 + (gy / (H - 1)) * (y1 - y0);
  const segs = [];
  for (let gy = 0; gy < H - 1; gy++) {
    for (let gx = 0; gx < W - 1; gx++) {
      const tl = grid[gy][gx], tr = grid[gy][gx + 1], br = grid[gy + 1][gx + 1], bl = grid[gy + 1][gx];
      const idx = (tl > level ? 1 : 0) | (tr > level ? 2 : 0) | (br > level ? 4 : 0) | (bl > level ? 8 : 0);
      const pairs = MS[idx];
      if (!pairs) continue;
      const lerp = (a, b) => (Math.abs(b - a) < 1e-12 ? 0.5 : (level - a) / (b - a));
      const edge = (e) => {
        if (e === 0) { const f = lerp(tl, tr); return [mapX(gx + f), mapY(gy)]; }
        if (e === 1) { const f = lerp(tr, br); return [mapX(gx + 1), mapY(gy + f)]; }
        if (e === 2) { const f = lerp(br, bl); return [mapX(gx + 1 - f), mapY(gy + 1)]; }
        const f = lerp(bl, tl); return [mapX(gx), mapY(gy + 1 - f)];
      };
      for (const [a, b] of pairs) { const p = edge(a), q = edge(b); segs.push([p[0], p[1], q[0], q[1]]); }
    }
  }
  return segs;
}

/** density on a WxH grid spanning the coords' bounding box (padded). */
export function gridDensity(coords, h, W = 60, H = 60, pad = 2.5) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of coords) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  x0 -= pad * h; y0 -= pad * h; x1 += pad * h; y1 += pad * h;
  const inv = 1 / (2 * h * h);
  const grid = [];
  let max = 0;
  for (let gy = 0; gy < H; gy++) {
    const row = [];
    const py = y0 + (gy / (H - 1)) * (y1 - y0);
    for (let gx = 0; gx < W; gx++) {
      const px = x0 + (gx / (W - 1)) * (x1 - x0);
      let s = 0;
      for (const p of coords) {
        const dx = px - p[0], dy = py - p[1];
        s += Math.exp(-(dx * dx + dy * dy) * inv);
      }
      row.push(s);
      if (s > max) max = s;
    }
    grid.push(row);
  }
  return { grid, max, bbox: [x0, y0, x1, y1] };
}
