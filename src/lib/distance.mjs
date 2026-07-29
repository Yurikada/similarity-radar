// Distance / similarity primitives.

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity in [-1, 1]. */
export function cosineSim(a, b) {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

export const cosineDist = (a, b) => 1 - cosineSim(a, b);

export function euclideanDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Full pairwise distance matrix using metric fn. */
export function pairwiseMatrix(vectors, metric) {
  const n = vectors.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = metric(vectors[i], vectors[j]);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
}
