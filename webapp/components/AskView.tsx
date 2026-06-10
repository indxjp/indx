'use client';

import { useCallback, useState } from 'react';
import { api, ApiError } from '@/lib/api';
import { useStore } from '@/lib/store';
import type { SearchHit } from '@/lib/types';
import { Banner, SectionHeader, Skeleton } from '@/components/ui';
import { AgentPanel } from '@/components/AgentPanel';

type Mode = 'search' | 'agent';

interface HistoryEntry {
  text: string;
  k: number;
  type: string | null;
}

const HISTORY_CAP = 8;

/**
 * Ask phase: a question box over /api/query with fully-lineaged ranked hits,
 * plus a separated "Use as an agent" tab hosting <AgentPanel/>. Reads the
 * active space from the store; shows an empty state when none is selected.
 */
export function AskView() {
  const store = useStore();
  const space = store.currentSpace;

  const [mode, setMode] = useState<Mode>('search');
  const [text, setText] = useState('');
  const [k, setK] = useState('5');
  const [type, setType] = useState('');
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSearch = useCallback(
    async (override?: { text?: string; k?: number; type?: string | null }) => {
      if (!space) {
        setError('Open or build a space first.');
        return;
      }
      const queryText = (override?.text ?? text).trim();
      if (!queryText) {
        setError('Enter a question.');
        return;
      }
      const parsedK = override?.k ?? parseInt(k, 10);
      const kNum = Number.isNaN(parsedK) || parsedK < 1 ? 5 : parsedK;
      const typeFilter =
        override?.type !== undefined ? override.type : type.trim() || null;

      // Reflect overrides (e.g. re-running a history entry) into the controls.
      if (override?.text !== undefined) setText(override.text);
      if (override?.k !== undefined) setK(String(override.k));
      if (override?.type !== undefined) setType(override.type ?? '');

      setLoading(true);
      setError(null);
      try {
        const res = await api.query({
          space,
          text: queryText,
          k: kNum,
          type: typeFilter,
        });
        setHits(res.hits);
        setLastQuery(queryText);
        setHistory((prev) => {
          const next = prev.filter((h) => h.text !== queryText);
          return [{ text: queryText, k: kNum, type: typeFilter }, ...next].slice(
            0,
            HISTORY_CAP,
          );
        });
      } catch (e) {
        setHits(null);
        const msg =
          e instanceof ApiError
            ? e.message
            : e instanceof Error
              ? e.message
              : String(e);
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [space, text, k, type],
  );

  if (!space) {
    return (
      <div className="space-y-5">
        <SectionHeader
          title="Ask your knowledge space"
          description="Pose questions and get ranked, fully-cited passages."
        />
        <div className="card flex flex-col items-center gap-3 py-12 text-center">
          <p className="text-sm text-slate-500">
            No space is open yet. Build or open a space to start asking.
          </p>
          <button className="btn-primary" onClick={() => store.setPhase('library')}>
            Go to Library
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <SectionHeader
          title="Ask your knowledge space"
          description="Pose questions for ranked, fully-cited passages — or wire it up as an agent."
        />
        <p className="-mt-2 mb-4 text-xs text-slate-400">
          Active space: <span className="font-mono">{space}</span>
        </p>

        {/* Mode tabs */}
        <div
          role="tablist"
          aria-label="Ask mode"
          className="mb-4 inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-sm"
        >
          <button
            role="tab"
            aria-selected={mode === 'search'}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
              mode === 'search'
                ? 'bg-accent text-white shadow-sm'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
            onClick={() => setMode('search')}
          >
            Ask a question
          </button>
          <button
            role="tab"
            aria-selected={mode === 'agent'}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
              mode === 'agent'
                ? 'bg-accent text-white shadow-sm'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
            onClick={() => setMode('agent')}
          >
            Use as an agent
          </button>
        </div>
      </div>

      {mode === 'agent' ? (
        <AgentPanel space={space} />
      ) : (
        <>
          <div className="card">
            <div className="mb-3">
              <label className="label" htmlFor="ask-text">
                Your question
              </label>
              <input
                id="ask-text"
                className="field"
                placeholder="What does the policy say about…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void runSearch();
                }}
              />
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label className="label" htmlFor="ask-k">
                  Results (k)
                </label>
                <input
                  id="ask-k"
                  type="number"
                  min={1}
                  className="field w-24"
                  value={k}
                  onChange={(e) => setK(e.target.value)}
                />
              </div>
              <div className="flex-1">
                <label className="label" htmlFor="ask-type">
                  Type filter (optional)
                </label>
                <input
                  id="ask-type"
                  className="field"
                  placeholder="e.g. policy"
                  value={type}
                  onChange={(e) => setType(e.target.value)}
                />
              </div>
              <button
                className="btn-primary"
                onClick={() => void runSearch()}
                disabled={loading}
              >
                {loading ? 'Searching…' : 'Ask'}
              </button>
            </div>

            {history.length > 0 ? (
              <div className="mt-4 border-t border-slate-100 pt-3">
                <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
                  Recent questions
                </h4>
                <div className="flex flex-wrap gap-2">
                  {history.map((h, i) => (
                    <button
                      key={`${h.text}-${i}`}
                      className="badge bg-accent-soft text-accent hover:bg-indigo-100"
                      title={`k=${h.k}${h.type ? ` · type=${h.type}` : ''}`}
                      onClick={() =>
                        void runSearch({ text: h.text, k: h.k, type: h.type })
                      }
                    >
                      {h.text}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          {error ? <Banner kind="error">{error}</Banner> : null}

          {loading ? (
            <div className="space-y-3">
              {[0, 1, 2].map((i) => (
                <div key={i} className="card space-y-2">
                  <Skeleton className="h-4 w-1/3" />
                  <Skeleton className="h-3 w-1/2" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ))}
            </div>
          ) : hits ? (
            hits.length === 0 ? (
              <Banner kind="info">
                No passages matched
                {lastQuery ? (
                  <>
                    {' '}
                    “<span className="font-medium">{lastQuery}</span>”
                  </>
                ) : null}
                . Try rephrasing or relaxing the type filter.
              </Banner>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-slate-400">
                  {hits.length} passage{hits.length === 1 ? '' : 's'}
                  {lastQuery ? (
                    <>
                      {' '}
                      for “<span className="font-medium">{lastQuery}</span>”
                    </>
                  ) : null}
                </p>
                {hits.map((hit, i) => (
                  <HitCard key={hit.chunk.id ?? i} hit={hit} rank={i + 1} />
                ))}
              </div>
            )
          ) : (
            <Banner kind="info">
              Ask a question above to retrieve cited passages from this space.
            </Banner>
          )}
        </>
      )}
    </div>
  );
}

/** A single ranked hit with full lineage (path/folder/type) + neighbor chunks. */
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
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="font-mono text-slate-600" title={source.path}>
            {source.path}
          </span>
          {source.folder ? (
            <span className="badge bg-slate-100 text-slate-600">
              📁 {source.folder}
            </span>
          ) : null}
          {source.type ? (
            <span className="badge bg-slate-100 text-slate-600">{source.type}</span>
          ) : null}
        </div>
      ) : (
        <p className="mb-2 text-xs text-slate-400">Source unknown</p>
      )}

      <p className="whitespace-pre-wrap text-sm text-ink">{hit.chunk.text}</p>

      {neighbors.length > 0 ? (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <h4 className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
            Neighboring passages
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
