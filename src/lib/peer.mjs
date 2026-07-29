// Peer-relative readings, with the guard the composed radar needed.
//
// D2 surfaced the failure mode: Recruit scored peer-z +7.18 against peers
// Nomura / ZOZO / Terumo / Shiseido — a group with no business in common. A
// z-score against an incoherent peer group is not a statement about the stock.
// It is a statement about the denominator: if the peers happen to have similar
// momentum, their standard deviation collapses and every deviation looks extreme.
//
// Two independent things therefore have to be reported next to every peer-z:
//
//   cohesion   is this actually a peer group? measured in identity space, as the
//              percentile of the mean focal->peer distance. High percentile = the
//              nearest neighbours are not near, which happens inside the
//              undifferentiated cluster E1 found (c10, n=41).
//   sdRatio    is the denominator trustworthy? peer sd over universe sd. Small
//              values mean the z is riding on a fragile estimate.
//
// The shrunk z floors the denominator at a fixed fraction of the universe sd.
// ALPHA is fixed here rather than chosen from the output, and the plain peer-z is
// kept alongside: D-H3 and E-H3 were pre-registered on the unshrunk estimator and
// are not retroactively rewritten by adding this.

export const ALPHA = 0.5; // denominator floor, as a fraction of the universe sd
export const FLAG = { cohesionPct: 75, sdRatio: 0.5 };

/** k nearest neighbours of every row, from a precomputed distance matrix. */
export function peerGroups(D, k) {
  const n = D.length;
  return Array.from({ length: n }, (_, i) =>
    [...Array(n).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]).slice(0, k));
}

const percentile = (v) => {
  const o = [...v.keys()].sort((a, b) => v[a] - v[b]);
  const p = new Array(v.length);
  o.forEach((i, r) => (p[i] = (100 * r) / (v.length - 1)));
  return p;
};

/** Identity-space quality of each peer group, independent of any overlay. */
export function peerCohesion(D, peers) {
  const n = D.length;
  const toPeers = peers.map((g, i) => g.reduce((a, j) => a + D[i][j], 0) / g.length);
  const among = peers.map((g) => {
    let s = 0,
      c = 0;
    for (let a = 0; a < g.length; a++)
      for (let b = a + 1; b < g.length; b++) {
        s += D[g[a]][g[b]];
        c++;
      }
    return s / c;
  });
  return {
    toPeers,
    among,
    toPeersPct: percentile(toPeers), // 0 = tightest neighbourhood in the universe
    amongPct: percentile(among),
  };
}

/** Peer-relative reading of one overlay variable, with the guard attached. */
export function peerReading(x, peers, cohesion) {
  const n = x.length;
  const muAll = x.reduce((a, b) => a + b, 0) / n;
  const sdAll = Math.sqrt(x.reduce((a, b) => a + (b - muAll) ** 2, 0) / n) || 1;
  const floor = ALPHA * sdAll;

  return x.map((_, i) => {
    const g = peers[i];
    const mu = g.reduce((a, j) => a + x[j], 0) / g.length;
    const sd = Math.sqrt(g.reduce((a, j) => a + (x[j] - mu) ** 2, 0) / g.length) || 1;
    const z = (x[i] - mu) / sd;
    const zShrunk = (x[i] - mu) / Math.max(sd, floor);
    const sdRatio = sd / sdAll;
    return {
      z,
      zShrunk,
      sdRatio,
      cohesionPct: cohesion.toPeersPct[i],
      // both conditions are reasons to distrust the number, for different reasons
      unreliable: cohesion.toPeersPct[i] > FLAG.cohesionPct || sdRatio < FLAG.sdRatio,
    };
  });
}
