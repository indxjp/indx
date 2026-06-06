'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { InspectResponse } from '@/lib/types';
import {
  Banner,
  BarChart,
  BrowseModal,
  SectionHeader,
  StatCard,
} from '@/components/ui';

export function InspectTab({
  spacePath,
  onSpaceChange,
}: {
  spacePath: string;
  onSpaceChange: (path: string) => void;
}) {
  const [data, setData] = useState<InspectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browse, setBrowse] = useState(false);

  const load = useCallback(async (path: string) => {
    const space = path.trim();
    if (!space) {
      setError('A space path (.indx file or output dir) is required.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.inspect(space);
      setData(res);
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-load when a space is handed off from the Build tab (Run demo / Inspect →).
  useEffect(() => {
    if (spacePath.trim()) void load(spacePath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spacePath]);

  const stats = data?.stats;

  return (
    <div className="space-y-5">
      <div className="card">
        <SectionHeader
          title="Inspect a knowledge space"
          description="Open a .indx archive, a directory containing one, or a jsonl output dir."
        />
        <div className="flex gap-2">
          <input
            className="field"
            placeholder="/path/to/space.indx or output dir"
            value={spacePath}
            onChange={(e) => onSpaceChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void load(spacePath);
            }}
            aria-label="Space path"
          />
          <button className="btn-ghost" onClick={() => setBrowse(true)}>
            Browse
          </button>
          <button className="btn-primary" onClick={() => load(spacePath)} disabled={loading}>
            {loading ? 'Loading…' : 'Inspect'}
          </button>
        </div>
      </div>

      {error ? <Banner kind="error">{error}</Banner> : null}

      {data && stats ? (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Documents" value={stats.documents} />
            <StatCard label="Chunks" value={stats.chunks} />
            <StatCard label="Relations" value={stats.relations} />
            <StatCard label="Embeddings" value={stats.embeddings} />
          </div>

          <div className="card">
            <SectionHeader title="Manifest" />
            <div className="grid gap-2 text-sm sm:grid-cols-2">
              <Kv k="indx version" v={data.manifest.indx_version} />
              <Kv k="schema version" v={data.manifest.schema_version} />
              <Kv k="source root" v={data.manifest.source_root || '—'} mono />
              <Kv k="embedding model" v={data.manifest.embedding_model ?? '—'} mono />
              <Kv
                k="embedding dim"
                v={data.manifest.embedding_dim != null ? String(data.manifest.embedding_dim) : '—'}
              />
              <Kv
                k="bytes (source)"
                v={stats.bytes_source.toLocaleString()}
              />
            </div>
            {Object.keys(data.manifest.components).length > 0 ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {Object.entries(data.manifest.components).map(([slot, backend]) => (
                  <span key={slot} className="badge bg-slate-100 text-slate-700">
                    {slot}: <span className="ml-1 font-mono">{backend}</span>
                  </span>
                ))}
              </div>
            ) : null}
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <div className="card">
              <SectionHeader title="Document types" />
              <BarChart data={data.types} emptyLabel="No typed documents" />
            </div>
            <div className="card">
              <SectionHeader title="Relations" />
              <BarChart data={data.relations} emptyLabel="No relations" />
            </div>
          </div>

          <div className="card">
            <SectionHeader
              title="Documents"
              description={`${data.documents.length} document${
                data.documents.length === 1 ? '' : 's'
              }`}
            />
            <div className="overflow-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-400">
                  <tr>
                    <th className="py-1 pr-3">Path</th>
                    <th className="py-1 pr-3">Type</th>
                    <th className="py-1 pr-3">Folder</th>
                    <th className="py-1 pr-3">Topics</th>
                    <th className="py-1 pr-3">Tags</th>
                    <th className="py-1 pr-3 text-right">Chunks</th>
                  </tr>
                </thead>
                <tbody>
                  {data.documents.map((d) => (
                    <tr key={d.id} className="border-t border-slate-100 align-top">
                      <td className="py-1 pr-3 font-mono text-xs">{d.path}</td>
                      <td className="py-1 pr-3">{d.type ?? '—'}</td>
                      <td className="py-1 pr-3 font-mono text-xs">{d.folder || '.'}</td>
                      <td className="py-1 pr-3 text-xs text-slate-500">
                        {d.topics.length ? d.topics.join(', ') : '—'}
                      </td>
                      <td className="py-1 pr-3 text-xs text-slate-500">
                        {d.tags.length ? d.tags.join(', ') : '—'}
                      </td>
                      <td className="py-1 pr-3 text-right tabular-nums">{d.chunks}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}

      <BrowseModal
        open={browse}
        initialPath={spacePath || undefined}
        onClose={() => setBrowse(false)}
        onSelect={(p) => {
          onSpaceChange(p);
          setBrowse(false);
          void load(p);
        }}
      />
    </div>
  );
}

function Kv({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-3 border-b border-slate-100 py-1">
      <span className="text-slate-500">{k}</span>
      <span className={`text-right text-ink ${mono ? 'font-mono text-xs' : ''}`}>{v}</span>
    </div>
  );
}
