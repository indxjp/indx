'use client';

import { useId, useMemo, useState } from 'react';
import type { InspectDocument, RelationEdge } from '@/lib/types';
import { buildGraph, layout, type GraphLink, type PositionedNode } from '@/lib/graph';
import { Banner } from '@/components/ui';

// Fixed SVG viewport the layout is computed against; the <svg> scales to its container.
const WIDTH = 800;
const HEIGHT = 560;
const NODE_LIMIT = 150;

// Stable, color-blind-friendly palette pulled from the Tailwind tokens already in use.
// Node fills are keyed by document type; edge strokes are keyed by relation type. Both
// fall back to a neutral slate when the value is unknown / not in the palette.
const NODE_PALETTE = [
  '#4f46e5', // indigo (accent)
  '#0ea5e9', // sky
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ec4899', // pink
  '#8b5cf6', // violet
  '#14b8a6', // teal
  '#ef4444', // rose/red
] as const;
const NODE_FALLBACK = '#94a3b8'; // slate-400

const EDGE_PALETTE = [
  '#6366f1', // indigo
  '#0891b2', // cyan-700
  '#059669', // emerald-600
  '#d97706', // amber-600
  '#db2777', // pink-600
  '#7c3aed', // violet-600
] as const;
const EDGE_FALLBACK = '#cbd5e1'; // slate-300

/** Deterministic palette lookup keyed by the sorted set of distinct values. */
function makeColorMap(values: string[], palette: readonly string[], fallback: string) {
  const distinct = Array.from(new Set(values)).sort();
  const map = new Map<string, string>();
  distinct.forEach((v, i) => map.set(v, palette[i % palette.length] ?? fallback));
  return map;
}

/** Map a chunk count to a node radius (sqrt scaling so area tracks size, not radius). */
function nodeRadius(chunks: number): number {
  return 6 + Math.sqrt(Math.max(chunks, 0)) * 2.5;
}

export interface RelationGraphProps {
  documents: InspectDocument[];
  edges: RelationEdge[];
  /** True relation count from the backend (`inspect.edge_total`); `edges` may be capped. */
  edgeTotal?: number;
  onSelectNode: (id: string) => void;
  onSelectEdge?: (link: GraphLink) => void;
}

export function RelationGraph({
  documents,
  edges,
  edgeTotal,
  onSelectNode,
  onSelectEdge,
}: RelationGraphProps) {
  const titleId = useId();
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [folderFilter, setFolderFilter] = useState<string>('');
  const [hovered, setHovered] = useState<string | null>(null);

  // Distinct types / folders across ALL documents (for the filter dropdowns), so the
  // user can always widen the view even when the current filter empties the graph.
  const { allTypes, allFolders } = useMemo(() => {
    const types = new Set<string>();
    const folders = new Set<string>();
    for (const d of documents) {
      if (d.type) types.add(d.type);
      folders.add(d.folder || '.');
    }
    return {
      allTypes: Array.from(types).sort(),
      allFolders: Array.from(folders).sort(),
    };
  }, [documents]);

  // Apply the active filters, then shape + lay out the bounded graph.
  const filtered = useMemo(
    () =>
      documents.filter((d) => {
        if (typeFilter && (d.type ?? '') !== typeFilter) return false;
        if (folderFilter && (d.folder || '.') !== folderFilter) return false;
        return true;
      }),
    [documents, typeFilter, folderFilter],
  );

  const { positioned, links, total, shown, nodeColors, edgeColors, nodeById } = useMemo(() => {
    const { data, total: t, shown: s } = buildGraph(filtered, edges, { limit: NODE_LIMIT });
    const laid = layout(data, WIDTH, HEIGHT);
    const byId = new Map<string, PositionedNode>();
    for (const node of laid.nodes) byId.set(node.id, node);
    return {
      positioned: laid.nodes,
      links: laid.links,
      total: t,
      shown: s,
      nodeColors: makeColorMap(
        data.nodes.map((n) => n.type ?? ''),
        NODE_PALETTE,
        NODE_FALLBACK,
      ),
      edgeColors: makeColorMap(
        data.links.map((l) => l.type),
        EDGE_PALETTE,
        EDGE_FALLBACK,
      ),
      nodeById: byId,
    };
  }, [filtered, edges]);

  const nodeColor = (type: string | null) => nodeColors.get(type ?? '') ?? NODE_FALLBACK;
  const edgeColor = (type: string) => edgeColors.get(type) ?? EDGE_FALLBACK;

  // Distinct relation types present in the rendered graph -> legend rows.
  const relationLegend = useMemo(() => {
    const seen = new Map<string, number>();
    for (const l of links) seen.set(l.type, (seen.get(l.type) ?? 0) + 1);
    return Array.from(seen.entries()).sort((a, b) => b[1] - a[1]);
  }, [links]);

  // Distinct node types present in the rendered graph -> legend rows.
  const typeLegend = useMemo(() => {
    const seen = new Map<string, number>();
    for (const n of positioned) seen.set(n.type ?? '(untyped)', (seen.get(n.type ?? '(untyped)') ?? 0) + 1);
    return Array.from(seen.entries()).sort((a, b) => b[1] - a[1]);
  }, [positioned]);

  // ----- Empty state: no documents at all -----
  if (documents.length === 0) {
    return (
      <div className="card">
        <Banner kind="info">
          No documents to graph yet. Organize a space to discover relationships between your
          files.
        </Banner>
      </div>
    );
  }

  const hasFilter = Boolean(typeFilter || folderFilter);

  return (
    <div className="card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">Relationship graph</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            {shown === 0
              ? 'No documents match the current filter.'
              : `${shown} document${shown === 1 ? '' : 's'} · ${links.length} link${
                  links.length === 1 ? '' : 's'
                }`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="sr-only" htmlFor={`${titleId}-type`}>
            Filter by type
          </label>
          <select
            id={`${titleId}-type`}
            className="field w-auto"
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
          >
            <option value="">All types</option>
            {allTypes.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <label className="sr-only" htmlFor={`${titleId}-folder`}>
            Filter by folder
          </label>
          <select
            id={`${titleId}-folder`}
            className="field w-auto"
            value={folderFilter}
            onChange={(e) => setFolderFilter(e.target.value)}
          >
            <option value="">All folders</option>
            {allFolders.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
          {hasFilter ? (
            <button
              type="button"
              className="btn-ghost px-3 py-2 text-xs"
              onClick={() => {
                setTypeFilter('');
                setFolderFilter('');
              }}
            >
              Clear filters
            </button>
          ) : null}
        </div>
      </div>

      {/* "showing N of M" notice when the bounded view dropped nodes. */}
      {total > shown ? (
        <div className="mb-3">
          <Banner kind="info">
            Showing {shown} of {total} documents — filter by type or folder to focus on the rest.
          </Banner>
        </div>
      ) : null}

      {/* Edge cap notice: the backend caps the relation payload, so some edges aren't drawn. */}
      {edgeTotal !== undefined && edgeTotal > edges.length ? (
        <div className="mb-3">
          <Banner kind="warning">
            Showing the first {edges.length} of {edgeTotal} relationships — large spaces are
            capped for performance.
          </Banner>
        </div>
      ) : null}

      {shown === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
          Nothing to show for this filter.{' '}
          <button
            type="button"
            className="font-medium text-accent underline-offset-2 hover:underline"
            onClick={() => {
              setTypeFilter('');
              setFolderFilter('');
            }}
          >
            Clear filters
          </button>{' '}
          to see the full graph.
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
            <svg
              role="img"
              aria-labelledby={titleId}
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="h-auto w-full"
              preserveAspectRatio="xMidYMid meet"
            >
              <title id={titleId}>
                Relationship graph of {shown} documents and {links.length} links
              </title>

              {/* Edges first so nodes paint on top. */}
              <g>
                {links.map((link, i) => {
                  const a = nodeById.get(link.source);
                  const b = nodeById.get(link.target);
                  if (!a || !b) return null;
                  const active =
                    hovered !== null && (hovered === link.source || hovered === link.target);
                  const interactive = Boolean(onSelectEdge);
                  return (
                    <g key={`${link.source}-${link.target}-${link.type}-${i}`}>
                      {/* Visible stroke. */}
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke={edgeColor(link.type)}
                        strokeWidth={active ? 2.5 : 1.25}
                        strokeOpacity={hovered !== null && !active ? 0.15 : 0.6}
                        pointerEvents="none"
                      />
                      {/* Invisible wide hit-target: a 1.25px line is nearly impossible to click,
                          so an overlaid transparent 12px stroke gives the edge a real target. */}
                      {interactive ? (
                        <line
                          x1={a.x}
                          y1={a.y}
                          x2={b.x}
                          y2={b.y}
                          stroke="transparent"
                          strokeWidth={12}
                          style={{ cursor: 'pointer' }}
                          onClick={() => onSelectEdge?.(link)}
                        >
                          <title>{link.type} — click to explain</title>
                        </line>
                      ) : (
                        <title>{link.type}</title>
                      )}
                    </g>
                  );
                })}
              </g>

              {/* Nodes. */}
              <g>
                {positioned.map((node) => {
                  const r = nodeRadius(node.chunks);
                  const dim = hovered !== null && hovered !== node.id;
                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x} ${node.y})`}
                      style={{ cursor: 'pointer' }}
                      tabIndex={0}
                      role="button"
                      aria-label={`${node.label} (${node.type ?? 'untyped'}, ${node.chunks} chunk${
                        node.chunks === 1 ? '' : 's'
                      })`}
                      onClick={() => onSelectNode(node.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onSelectNode(node.id);
                        }
                      }}
                      onMouseEnter={() => setHovered(node.id)}
                      onMouseLeave={() => setHovered((h) => (h === node.id ? null : h))}
                      onFocus={() => setHovered(node.id)}
                      onBlur={() => setHovered((h) => (h === node.id ? null : h))}
                    >
                      <circle
                        r={r}
                        fill={nodeColor(node.type)}
                        fillOpacity={dim ? 0.3 : 0.9}
                        stroke="#ffffff"
                        strokeWidth={1.5}
                      />
                      <title>
                        {node.label}
                        {node.type ? ` · ${node.type}` : ''} · {node.chunks} chunk
                        {node.chunks === 1 ? '' : 's'}
                        {node.folder ? ` · ${node.folder}` : ''}
                      </title>
                      {/* Label only for larger / hovered nodes to limit clutter. */}
                      {r >= 11 || hovered === node.id ? (
                        <text
                          x={0}
                          y={r + 11}
                          textAnchor="middle"
                          className="pointer-events-none select-none fill-slate-600 dark:fill-slate-300"
                          fontSize={10}
                          fillOpacity={dim ? 0.3 : 1}
                        >
                          {node.label.length > 22
                            ? `${node.label.slice(0, 21)}…`
                            : node.label}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>

          {/* Legends. */}
          <aside className="flex w-full flex-col gap-4 lg:w-56">
            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Document types
              </h3>
              <ul className="space-y-1.5">
                {typeLegend.map(([type, n]) => (
                  <li key={type} className="flex items-center gap-2 text-sm text-slate-600">
                    <span
                      aria-hidden="true"
                      className="inline-block h-3 w-3 shrink-0 rounded-full"
                      style={{
                        backgroundColor: nodeColor(type === '(untyped)' ? null : type),
                      }}
                    />
                    <span className="flex-1 truncate" title={type}>
                      {type}
                    </span>
                    <span className="tabular-nums text-slate-400">{n}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Relations
              </h3>
              {relationLegend.length === 0 ? (
                <p className="text-sm text-slate-400">No links in view</p>
              ) : (
                <ul className="space-y-1.5">
                  {relationLegend.map(([type, n]) => (
                    <li key={type} className="flex items-center gap-2 text-sm text-slate-600">
                      <span
                        aria-hidden="true"
                        className="inline-block h-0.5 w-4 shrink-0 rounded"
                        style={{ backgroundColor: edgeColor(type) }}
                      />
                      <span className="flex-1 truncate" title={type}>
                        {type}
                      </span>
                      <span className="tabular-nums text-slate-400">{n}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <p className="px-1 text-xs leading-relaxed text-slate-400">
              Circle size reflects chunk count. Click a node to open its details, or click a line
              between two nodes to see how they relate.
            </p>
          </aside>
        </div>
      )}
    </div>
  );
}
