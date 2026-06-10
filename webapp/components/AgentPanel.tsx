'use client';

import { useCallback, useEffect, useState } from 'react';
import { api, ApiError } from '@/lib/api';
import type {
  AgentSearchResults,
  DocumentDetail,
  FrameworkInfo,
  SnippetsResponse,
  SpaceOverview,
  ToolDef,
} from '@/lib/types';
import {
  Banner,
  CodeBlock,
  SectionHeader,
  Skeleton,
  StatCard,
} from '@/components/ui';

type SnippetKey = keyof SnippetsResponse;

const SNIPPET_TABS: { key: SnippetKey; label: string; lang: string }[] = [
  { key: 'python', label: 'Python', lang: 'python' },
  { key: 'cli', label: 'CLI', lang: 'bash' },
  { key: 'langchain', label: 'LangChain', lang: 'python' },
  { key: 'llamaindex', label: 'LlamaIndex', lang: 'python' },
  { key: 'mcp', label: 'MCP', lang: 'json' },
];

type Tool = 'overview' | 'search' | 'document';

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return e instanceof Error ? e.message : String(e);
}

/** Small pill noting whether a framework's extra is installed. */
function InstallBadge({ installed }: { installed: boolean }) {
  return installed ? (
    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">
      Installed
    </span>
  ) : (
    <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-500">
      Not installed
    </span>
  );
}

export function AgentPanel({ space }: { space: string | null }) {
  // --- reference data (frameworks / tools / snippets) -----------------------
  const [frameworks, setFrameworks] = useState<FrameworkInfo[] | null>(null);
  const [tools, setTools] = useState<ToolDef[] | null>(null);
  const [snippets, setSnippets] = useState<SnippetsResponse | null>(null);
  const [refLoading, setRefLoading] = useState(true);
  const [refError, setRefError] = useState<string | null>(null);

  const [snippetTab, setSnippetTab] = useState<SnippetKey>('python');

  useEffect(() => {
    let cancelled = false;
    setRefLoading(true);
    setRefError(null);
    Promise.all([api.agent.frameworks(), api.agent.tools(), api.agent.snippets()])
      .then(([fw, tl, sn]) => {
        if (cancelled) return;
        setFrameworks(fw);
        setTools(tl);
        setSnippets(sn);
      })
      .catch((e) => {
        if (!cancelled) setRefError(errMsg(e));
      })
      .finally(() => {
        if (!cancelled) setRefLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // --- playground -----------------------------------------------------------
  const [tool, setTool] = useState<Tool>('overview');
  const [text, setText] = useState('');
  const [k, setK] = useState('5');
  const [docType, setDocType] = useState('');
  const [pathOrId, setPathOrId] = useState('');

  const [running, setRunning] = useState(false);
  const [playError, setPlayError] = useState<string | null>(null);
  const [overview, setOverview] = useState<SpaceOverview | null>(null);
  const [results, setResults] = useState<AgentSearchResults | null>(null);
  const [detail, setDetail] = useState<DocumentDetail | null>(null);

  const clearOutputs = useCallback(() => {
    setOverview(null);
    setResults(null);
    setDetail(null);
    setPlayError(null);
  }, []);

  const onRun = useCallback(async () => {
    if (!space) {
      setPlayError('Build or open a space first.');
      return;
    }
    setRunning(true);
    clearOutputs();
    try {
      if (tool === 'overview') {
        const res = await api.agent.overview(space);
        setOverview(res);
      } else if (tool === 'search') {
        if (!text.trim()) {
          setPlayError('Enter a search query.');
          return;
        }
        const kNum = parseInt(k, 10);
        const res = await api.agent.search(
          space,
          text.trim(),
          Number.isNaN(kNum) || kNum < 1 ? 5 : kNum,
          docType.trim() || null,
        );
        setResults(res);
      } else {
        if (!pathOrId.trim()) {
          setPlayError('Enter a document path or id.');
          return;
        }
        const res = await api.agent.document(space, pathOrId.trim());
        if ('error' in res) {
          setPlayError(res.error);
        } else {
          setDetail(res);
        }
      }
    } catch (e) {
      setPlayError(errMsg(e));
    } finally {
      setRunning(false);
    }
  }, [space, tool, text, k, docType, pathOrId, clearOutputs]);

  // -------------------------------------------------------------------- render
  return (
    <div className="space-y-6">
      {/* ---- Frameworks --------------------------------------------------- */}
      <section className="card">
        <SectionHeader
          title="Agent frameworks"
          description="Plug this knowledge space into your agent stack. Install the matching extra to enable a framework."
        />
        {refError ? <Banner kind="error">{refError}</Banner> : null}
        {refLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-2/3" />
          </div>
        ) : frameworks && frameworks.length > 0 ? (
          <ul className="grid gap-2 sm:grid-cols-2">
            {frameworks.map((fw) => (
              <li
                key={fw.name}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-sm"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-ink">{fw.name}</div>
                  <code className="text-xs text-slate-400">indx[{fw.extra}]</code>
                </div>
                <InstallBadge installed={fw.installed} />
              </li>
            ))}
          </ul>
        ) : !refError ? (
          <p className="text-sm text-slate-400">No frameworks reported.</p>
        ) : null}
      </section>

      {/* ---- Tool contract ------------------------------------------------ */}
      <section className="card">
        <SectionHeader
          title="Tool contract"
          description="The tools your agent can call against this space, with their JSON-schema parameters."
        />
        {refLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : tools && tools.length > 0 ? (
          <ul className="space-y-3">
            {tools.map((t) => (
              <li key={t.name} className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                <div className="flex items-baseline gap-2">
                  <code className="text-sm font-semibold text-accent">{t.name}</code>
                </div>
                <p className="mt-1 text-sm text-slate-600">{t.description}</p>
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-600">
                    parameters
                  </summary>
                  <pre className="mt-2 overflow-auto rounded-md bg-slate-50 p-2 text-xs leading-relaxed text-slate-700">
                    <code>{JSON.stringify(t.parameters, null, 2)}</code>
                  </pre>
                </details>
              </li>
            ))}
          </ul>
        ) : !refError ? (
          <p className="text-sm text-slate-400">No tools reported.</p>
        ) : null}
      </section>

      {/* ---- Snippets ----------------------------------------------------- */}
      <section className="card">
        <SectionHeader
          title="Integration snippets"
          description="Copy-paste starters for each framework. They reference this space by path."
        />
        {refLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : snippets ? (
          <div className="space-y-3">
            <div role="tablist" aria-label="Snippet language" className="flex flex-wrap gap-2">
              {SNIPPET_TABS.map((tab) => (
                <button
                  key={tab.key}
                  role="tab"
                  aria-selected={snippetTab === tab.key}
                  type="button"
                  onClick={() => setSnippetTab(tab.key)}
                  className={`rounded-md border px-3 py-1 text-xs font-medium ${
                    snippetTab === tab.key
                      ? 'border-accent bg-accent text-white'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            {SNIPPET_TABS.filter((tab) => tab.key === snippetTab).map((tab) => (
              <CodeBlock key={tab.key} code={snippets[tab.key]} lang={tab.lang} />
            ))}
          </div>
        ) : !refError ? (
          <p className="text-sm text-slate-400">No snippets available.</p>
        ) : null}
      </section>

      {/* ---- Playground --------------------------------------------------- */}
      <section className="card">
        <SectionHeader
          title="Playground"
          description="Exercise the agent tools directly — no LLM required. Runs against the current space."
        />
        {!space ? (
          <Banner kind="info">Build or open a space to use the playground.</Banner>
        ) : (
          <div className="space-y-4">
            <div role="tablist" aria-label="Agent tool" className="flex flex-wrap gap-2">
              {(['overview', 'search', 'document'] as Tool[]).map((tl) => (
                <button
                  key={tl}
                  role="tab"
                  aria-selected={tool === tl}
                  type="button"
                  onClick={() => {
                    setTool(tl);
                    clearOutputs();
                  }}
                  className={`rounded-md border px-3 py-1 text-sm font-medium capitalize ${
                    tool === tl
                      ? 'border-accent bg-accent text-white'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {tl}
                </button>
              ))}
            </div>

            {/* tool-specific inputs */}
            {tool === 'search' ? (
              <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto]">
                <div>
                  <label className="label" htmlFor="ap-text">
                    Query
                  </label>
                  <input
                    id="ap-text"
                    className="field w-full"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="What do you want to find?"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') void onRun();
                    }}
                  />
                </div>
                <div>
                  <label className="label" htmlFor="ap-k">
                    Top k
                  </label>
                  <input
                    id="ap-k"
                    className="field w-20"
                    value={k}
                    onChange={(e) => setK(e.target.value)}
                    inputMode="numeric"
                  />
                </div>
                <div>
                  <label className="label" htmlFor="ap-doctype">
                    Doc type
                  </label>
                  <input
                    id="ap-doctype"
                    className="field w-32"
                    value={docType}
                    onChange={(e) => setDocType(e.target.value)}
                    placeholder="any"
                  />
                </div>
              </div>
            ) : null}

            {tool === 'document' ? (
              <div>
                <label className="label" htmlFor="ap-pathid">
                  Document path or id
                </label>
                <input
                  id="ap-pathid"
                  className="field w-full"
                  value={pathOrId}
                  onChange={(e) => setPathOrId(e.target.value)}
                  placeholder="notes/intro.md or a document id"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') void onRun();
                  }}
                />
              </div>
            ) : null}

            <div>
              <button className="btn-primary" type="button" onClick={onRun} disabled={running}>
                {running ? 'Running…' : 'Run'}
              </button>
            </div>

            {playError ? <Banner kind="error">{playError}</Banner> : null}

            {/* outputs */}
            {overview ? <OverviewOutput data={overview} /> : null}
            {results ? <SearchOutput data={results} /> : null}
            {detail ? <DocumentOutput data={detail} /> : null}
          </div>
        )}
      </section>
    </div>
  );
}

// --------------------------------------------------------------------- outputs

function OverviewOutput({ data }: { data: SpaceOverview }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Documents" value={data.documents} />
        <StatCard label="Chunks" value={data.chunks} />
        <StatCard label="Relations" value={data.relations} />
        <StatCard label="Embeddings" value={data.embeddings} />
      </div>
      <p className="text-xs text-slate-500">
        <span className="font-medium text-slate-700">{data.name}</span>
        {data.embedding_model ? (
          <>
            {' · '}
            {data.embedding_model}
            {data.embedding_dim ? ` (${data.embedding_dim}d)` : ''}
          </>
        ) : null}
      </p>
      {data.sample_documents.length > 0 ? (
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-slate-400">
            Sample documents
          </div>
          <ul className="space-y-2">
            {data.sample_documents.map((d) => (
              <li key={d.id} className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                <div className="flex items-center justify-between gap-2">
                  <code className="truncate text-sm text-ink" title={d.path}>
                    {d.path}
                  </code>
                  {d.doc_type ? (
                    <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                      {d.doc_type}
                    </span>
                  ) : null}
                </div>
                {d.summary ? (
                  <p className="mt-1 line-clamp-2 text-sm text-slate-600">{d.summary}</p>
                ) : null}
                <TagRow topics={d.topics} tags={d.tags} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function SearchOutput({ data }: { data: AgentSearchResults }) {
  if (data.count === 0) {
    return <Banner kind="info">No matches for “{data.query}”.</Banner>;
  }
  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-wide text-slate-400">
        {data.count} hit{data.count === 1 ? '' : 's'} for “{data.query}”
      </div>
      <ul className="space-y-2">
        {data.hits.map((h) => (
          <li key={h.chunk_id} className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
            <div className="flex items-center justify-between gap-2">
              {/* AgentHit.source is a FLAT path string (not a nested Source object). */}
              <code className="truncate text-xs text-slate-500" title={h.source ?? undefined}>
                {h.source ?? h.folder ?? '—'}
              </code>
              <span className="shrink-0 text-xs tabular-nums text-slate-400">
                {h.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-1 line-clamp-3 text-sm text-slate-700">{h.text}</p>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-400">
              {h.doc_type ? (
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-500">
                  {h.doc_type}
                </span>
              ) : null}
              <span className="truncate">{h.folder}</span>
            </div>
            <TagRow topics={h.topics} tags={h.tags} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function DocumentOutput({ data }: { data: DocumentDetail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <code className="truncate text-sm font-medium text-ink" title={data.path}>
          {data.path}
        </code>
        <div className="flex shrink-0 items-center gap-2">
          {data.doc_type ? (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
              {data.doc_type}
            </span>
          ) : null}
          <span className="text-xs text-slate-400">{data.chunk_count} chunks</span>
        </div>
      </div>
      <div className="mt-1 text-xs text-slate-400">{data.folder}</div>
      {data.summary ? (
        <p className="mt-2 text-sm text-slate-600">{data.summary}</p>
      ) : null}
      <TagRow topics={data.topics} tags={data.tags} />
      {data.text ? (
        <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-slate-50 p-3 text-xs leading-relaxed text-slate-700">
          {data.text}
        </pre>
      ) : null}
    </div>
  );
}

function TagRow({ topics, tags }: { topics: string[]; tags: string[] }) {
  if (topics.length === 0 && tags.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {topics.map((t) => (
        <span
          key={`topic-${t}`}
          className="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-xs text-sky-700"
        >
          {t}
        </span>
      ))}
      {tags.map((t) => (
        <span
          key={`tag-${t}`}
          className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-500"
        >
          #{t}
        </span>
      ))}
    </div>
  );
}
