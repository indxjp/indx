'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { SearchHit } from '@/lib/types';
import { Banner, SectionHeader } from '@/components/ui';

export function QueryTab({ initialSpace }: { initialSpace: string }) {
  const [space, setSpace] = useState(initialSpace);
  const [text, setText] = useState('');
  const [k, setK] = useState('5');
  const [type, setType] = useState('');
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Keep the space field in sync when a build/demo hands off a fresh space path.
  useEffect(() => {
    if (initialSpace) setSpace(initialSpace);
  }, [initialSpace]);

  const onSearch = useCallback(async () => {
    if (!space.trim()) {
      setError('A space path is required.');
      return;
    }
    if (!text.trim()) {
      setError('Enter a query.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const kNum = parseInt(k, 10);
      const res = await api.query({
        space: space.trim(),
        text: text.trim(),
        k: Number.isNaN(kNum) || kNum < 1 ? 5 : kNum,
        type: type.trim() || null,
      });
      setHits(res.hits);
    } catch (e) {
      setHits(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [space, text, k, type]);

  return (
    <div className="space-y-5">
      <div className="card">
        <SectionHeader
          title="Query the knowledge space"
          description="Retrieve ranked chunks with their lineage and neighbors."
        />
        <div className="mb-3">
          <label className="label" htmlFor="q-space">
            Space path
          </label>
          <input
            id="q-space"
            className="field"
            placeholder="/path/to/space.indx or output dir"
            value={space}
            onChange={(e) => setSpace(e.target.value)}
          />
        </div>
        <div className="mb-3">
          <label className="label" htmlFor="q-text">
            Query
          </label>
          <input
            id="q-text"
            className="field"
            placeholder="What does the policy say about…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void onSearch();
            }}
          />
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label" htmlFor="q-k">
              k
            </label>
            <input
              id="q-k"
              type="number"
              min={1}
              className="field w-24"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </div>
          <div className="flex-1">
            <label className="label" htmlFor="q-type">
              Type filter (optional)
            </label>
            <input
              id="q-type"
              className="field"
              placeholder="e.g. policy"
              value={type}
              onChange={(e) => setType(e.target.value)}
            />
          </div>
          <button className="btn-primary" onClick={onSearch} disabled={loading}>
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>
      </div>

      {error ? <Banner kind="error">{error}</Banner> : null}

      {hits ? (
        hits.length === 0 ? (
          <Banner kind="info">No hits for this query.</Banner>
        ) : (
          <div className="space-y-3">
            {hits.map((hit, i) => (
              <HitCard key={hit.chunk.id ?? i} hit={hit} rank={i + 1} />
            ))}
          </div>
        )
      ) : null}
    </div>
  );
}

function HitCard({ hit, rank }: { hit: SearchHit; rank: number }) {
  const source = hit.chunk.source;
  const neighbors = hit.neighbors ?? [];
  return (
    <div className="card">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="badge bg-accent-soft text-accent">#{rank}</span>
          <span className="font-mono text-xs text-slate-500">{hit.chunk.id}</span>
        </div>
        <span className="badge bg-slate-100 text-slate-700">
          score <span className="ml-1 font-mono">{hit.score.toFixed(4)}</span>
        </span>
      </div>

      {source ? (
        <p className="mb-2 text-xs text-slate-500">
          <span className="font-mono">{source.path}</span>
          {source.type ? (
            <span className="ml-2 badge bg-slate-100 text-slate-600">{source.type}</span>
          ) : null}
        </p>
      ) : null}

      <p className="whitespace-pre-wrap text-sm text-ink">{hit.chunk.text}</p>

      {neighbors.length > 0 ? (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <h4 className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
            Neighbors
          </h4>
          <ul className="space-y-2">
            {neighbors.map((n) => (
              <li key={n.id} className="text-xs text-slate-500">
                <span className="font-mono text-slate-400">{n.id}</span>
                <span className="ml-2 line-clamp-2 text-slate-600">{n.text}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
