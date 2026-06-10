// Zero-dependency relationship-graph helpers: shaping InspectDocument[]/RelationEdge[]
// into a bounded {nodes, links} graph, plus a deterministic pure-TS force-directed layout.
// No React, no DOM, no external libraries. Everything here is a pure function and the
// layout uses a fixed deterministic seeding formula (no Math.random) so renders are stable.

import type { InspectDocument, RelationEdge } from './types';

export interface GraphNode {
  id: string;
  label: string;
  type: string | null;
  chunks: number;
  folder: string;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

/** Derive a human-friendly label (basename) from a document path. */
function basename(path: string): string {
  if (!path) return path;
  // Split on both separators so windows/posix paths both work.
  const parts = path.split(/[\\/]+/).filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : path;
}

/**
 * Shape documents + edges into a bounded graph.
 *
 * Keeps the top `limit` (default 150) documents ranked by degree (number of incident
 * edges) and then by chunk count as a tiebreaker. Links whose source or target was
 * dropped are removed. Returns the shaped {data}, the `total` document count, and the
 * `shown` count (nodes kept).
 */
export function buildGraph(
  documents: InspectDocument[],
  edges: RelationEdge[],
  opts?: { limit?: number },
): { data: GraphData; total: number; shown: number } {
  const limit = opts?.limit ?? 150;
  const total = documents.length;

  // Only count edges that connect two documents we actually have.
  const docIds = new Set(documents.map((d) => d.id));
  const degree = new Map<string, number>();
  for (const doc of documents) degree.set(doc.id, 0);
  for (const edge of edges) {
    if (docIds.has(edge.source_id) && docIds.has(edge.target_id)) {
      degree.set(edge.source_id, (degree.get(edge.source_id) ?? 0) + 1);
      degree.set(edge.target_id, (degree.get(edge.target_id) ?? 0) + 1);
    }
  }

  // Rank by degree desc, then chunks desc, then a stable id tiebreak for determinism.
  const ranked = [...documents].sort((a, b) => {
    const da = degree.get(a.id) ?? 0;
    const db = degree.get(b.id) ?? 0;
    if (db !== da) return db - da;
    if (b.chunks !== a.chunks) return b.chunks - a.chunks;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  const kept = ranked.slice(0, Math.max(0, limit));
  const keptIds = new Set(kept.map((d) => d.id));

  const nodes: GraphNode[] = kept.map((d) => ({
    id: d.id,
    label: basename(d.path),
    type: d.type,
    chunks: d.chunks,
    folder: d.folder,
  }));

  const links: GraphLink[] = [];
  for (const edge of edges) {
    if (keptIds.has(edge.source_id) && keptIds.has(edge.target_id)) {
      links.push({ source: edge.source_id, target: edge.target_id, type: edge.type });
    }
  }

  return { data: { nodes, links }, total, shown: nodes.length };
}

export interface PositionedNode extends GraphNode {
  x: number;
  y: number;
}

/**
 * Deterministic force-directed layout: repulsion between every pair of nodes, spring
 * attraction along links, and gentle centering toward the canvas middle. Initial
 * positions are seeded deterministically from each node's index (no randomness), so the
 * same input always yields the same output. Final positions are clamped to the canvas.
 */
export function layout(
  data: GraphData,
  width: number,
  height: number,
  iterations = 150,
): { nodes: PositionedNode[]; links: GraphLink[] } {
  const n = data.nodes.length;
  const cx = width / 2;
  const cy = height / 2;

  if (n === 0) {
    return { nodes: [], links: data.links };
  }

  // Deterministic seeding: lay nodes on a phyllotaxis-style spiral around the center.
  // Using the golden angle gives an even, non-overlapping initial spread with zero RNG.
  const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
  const radius = Math.min(width, height) * 0.45;
  const xs = new Array<number>(n);
  const ys = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0;
    const r = radius * Math.sqrt(t);
    const a = i * GOLDEN_ANGLE;
    xs[i] = cx + r * Math.cos(a);
    ys[i] = cy + r * Math.sin(a);
  }

  // Index lookup so links (by id) can address position arrays.
  const indexOf = new Map<string, number>();
  for (let i = 0; i < n; i++) indexOf.set(data.nodes[i].id, i);

  // Tunable constants. Scaled by canvas size so the layout fills the available space.
  const area = width * height;
  const k = Math.sqrt(area / n); // ideal edge length
  const repulsion = k * k; // repulsive strength
  const springLength = k * 0.6;
  const springStrength = 0.02;
  const centerPull = 0.01;

  const dx = new Array<number>(n);
  const dy = new Array<number>(n);

  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < n; i++) {
      dx[i] = 0;
      dy[i] = 0;
    }

    // Pairwise repulsion (O(n^2); fine for the bounded node count).
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let ddx = xs[i] - xs[j];
        let ddy = ys[i] - ys[j];
        let distSq = ddx * ddx + ddy * ddy;
        if (distSq < 0.01) {
          // Deterministic nudge for coincident nodes (no randomness).
          ddx = (i - j) * 0.01 + 0.01;
          ddy = (i + j) * 0.01 + 0.01;
          distSq = ddx * ddx + ddy * ddy;
        }
        const dist = Math.sqrt(distSq);
        const force = repulsion / distSq;
        const fx = (ddx / dist) * force;
        const fy = (ddy / dist) * force;
        dx[i] += fx;
        dy[i] += fy;
        dx[j] -= fx;
        dy[j] -= fy;
      }
    }

    // Spring attraction along links.
    for (const link of data.links) {
      const si = indexOf.get(link.source);
      const ti = indexOf.get(link.target);
      if (si === undefined || ti === undefined) continue;
      let ddx = xs[ti] - xs[si];
      let ddy = ys[ti] - ys[si];
      let dist = Math.sqrt(ddx * ddx + ddy * ddy);
      if (dist < 0.01) dist = 0.01;
      const displacement = dist - springLength;
      const force = springStrength * displacement;
      const fx = (ddx / dist) * force;
      const fy = (ddy / dist) * force;
      dx[si] += fx;
      dy[si] += fy;
      dx[ti] -= fx;
      dy[ti] -= fy;
    }

    // Gentle centering + integration. Cooling shrinks step size over iterations.
    const cooling = 1 - iter / iterations;
    const maxStep = k * 0.5 * cooling + 1;
    for (let i = 0; i < n; i++) {
      dx[i] += (cx - xs[i]) * centerPull;
      dy[i] += (cy - ys[i]) * centerPull;

      // Limit per-iteration displacement for stability.
      const dlen = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
      if (dlen > maxStep) {
        dx[i] = (dx[i] / dlen) * maxStep;
        dy[i] = (dy[i] / dlen) * maxStep;
      }
      xs[i] += dx[i];
      ys[i] += dy[i];
    }
  }

  const nodes: PositionedNode[] = data.nodes.map((node, i) => ({
    ...node,
    x: Math.min(width, Math.max(0, xs[i])),
    y: Math.min(height, Math.max(0, ys[i])),
  }));

  return { nodes, links: data.links };
}
