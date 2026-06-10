'use client';

// OrganizeView — the centerpiece of the journey. Two modes:
//  (1) Build mode: when store.buildNonce flips to >0, stream a fresh build over
//      store.source, narrating each pipeline stage in plain "organizing" language with
//      a live timeline + counts. On `done` we inspect the produced space, register it
//      with the store, and fall through to the after view.
//  (2) Open mode: when store.currentSpace is already set with no pending build, just
//      inspect it and show the after view directly.
//  After view: a RelationGraph hero, type + topic/tag clusters via BarChart, and a
//  browsable document list. Selecting a node or a row opens the DocumentDrawer.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { streamBuild, api } from '@/lib/api';
import type {
  BuildErrorEvent,
  BuildRequest,
  BuildStageEvent,
  BuildStartEvent,
  BuildSummary,
  InspectDocument,
  InspectResponse,
} from '@/lib/types';
import { useStore } from '@/lib/store';
import type { GraphLink } from '@/lib/graph';
import { Banner, BarChart, SectionHeader, StatCard, Skeleton, useToast } from '@/components/ui';
import { RelationGraph } from '@/components/RelationGraph';
import { DocumentDrawer } from '@/components/DocumentDrawer';

// The DirectoryPipeline stages (src/indx app SSE `stage` events), mapped to the
// human "organizing" narration the spec asks for.
const STAGE_ORDER = [
  'walk',
  'parse',
  'chunk',
  'relate',
  'enrich',
  'embed-pack',
] as const;
type StageName = (typeof STAGE_ORDER)[number];
type StageState = 'pending' | 'active' | 'done';

const STAGE_LABEL: Record<StageName, string> = {
  walk: 'Reading your files',
  parse: 'Understanding each document',
  chunk: 'Splitting into passages',
  relate: 'Discovering relationships',
  enrich: 'Summarizing & tagging',
  'embed-pack': 'Making it searchable',
};

function initialStageStates(): Record<StageName, StageState> {
  return STAGE_ORDER.reduce<Record<StageName, StageState>>(
    (acc, s) => {
      acc[s] = 'pending';
      return acc;
    },
    {} as Record<StageName, StageState>,
  );
}

function allDoneStageStates(): Record<StageName, StageState> {
  return STAGE_ORDER.reduce<Record<StageName, StageState>>(
    (acc, s) => {
      acc[s] = 'done';
      return acc;
    },
    {} as Record<StageName, StageState>,
  );
}

/** Derive a `.indx` artifact name from the source folder/zip basename, so a corpus build is
 *  named `corpus.indx` rather than the generic `handbook.indx` default. Falls back to `space`
 *  when the source yields nothing usable; the result is filename-safe (alnum / - / _). */
function spaceNameFromSource(source: string | null): string {
  if (!source) return 'space';
  const base = source.replace(/[/\\]+$/, '').split(/[/\\]/).pop() ?? '';
  const stem = base.replace(/\.(indx|zip)$/i, '');
  const cleaned = stem.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
  return cleaned || 'space';
}

/** Aggregate document topics/tags into a name->count map for a BarChart. */
function clusterCounts(
  documents: InspectDocument[],
  pick: (d: InspectDocument) => string[],
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const doc of documents) {
    for (const name of pick(doc)) {
      if (!name) continue;
      counts[name] = (counts[name] ?? 0) + 1;
    }
  }
  return counts;
}

export function OrganizeView() {
  const store = useStore();
  const { push } = useToast();

  // ---- after-view (inspect) state ----
  const [inspect, setInspect] = useState<InspectResponse | null>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ---- build (mode 1) state ----
  const [building, setBuilding] = useState(false);
  const [stageStates, setStageStates] =
    useState<Record<StageName, StageState>>(initialStageStates);
  const [startInfo, setStartInfo] = useState<BuildStartEvent | null>(null);
  const [summary, setSummary] = useState<BuildSummary | null>(null);

  // ---- drawer + edge explainer ----
  const [selectedDoc, setSelectedDoc] = useState<InspectDocument | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphLink | null>(null);

  // Track which build/space we've already handled so effects don't re-fire.
  const handledNonce = useRef<number>(0);
  const inspectedSpace = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const runInspect = useCallback(async (space: string) => {
    setInspectLoading(true);
    setError(null);
    try {
      const res = await api.inspect(space);
      setInspect(res);
      inspectedSpace.current = space;
    } catch (e) {
      setInspect(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInspectLoading(false);
    }
  }, []);

  // -------------------------------------------------------------- mode 1: build
  useEffect(() => {
    // Only act on a fresh, not-yet-handled build trigger.
    if (store.buildNonce <= 0 || store.buildNonce === handledNonce.current) {
      return;
    }
    handledNonce.current = store.buildNonce;

    if (!store.source) {
      setError('No source folder selected. Go back to Ingest and pick a folder.');
      return;
    }

    const body: BuildRequest = {
      directory: store.source,
      // Name the artifact after the source folder (corpus.indx, not the generic handbook.indx).
      name: spaceNameFromSource(store.source),
      offline: store.preset === 'local',
      // Advanced builds use the indx.toml saved from the ConfigTab editor.
      config: store.preset === 'custom' ? store.configPath : null,
    };

    // Reset the live build UI (including any open drawer / edge explainer from a prior space —
    // their ids won't exist in the new space, so a stale explainer would show raw hash IDs).
    setInspect(null);
    setSummary(null);
    setStartInfo(null);
    setError(null);
    setSelectedDoc(null);
    setSelectedEdge(null);
    setStageStates(initialStageStates());
    setBuilding(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let producedOut: string | null = null;
    let producedCounts: BuildSummary['counts'] | null = null;

    void streamBuild(body, {
      signal: ctrl.signal,
      onEvent: (ev) => {
        switch (ev.event) {
          case 'start':
            setStartInfo(ev.data as BuildStartEvent);
            break;
          case 'stage': {
            const stageName = (ev.data as BuildStageEvent).name as StageName;
            setStageStates((prev) => {
              const next = { ...prev };
              for (const s of STAGE_ORDER) {
                if (s === stageName) {
                  next[s] = 'active';
                  break;
                }
                if (next[s] !== 'done') next[s] = 'done';
              }
              return next;
            });
            break;
          }
          case 'done': {
            const sum = ev.data as BuildSummary;
            producedOut = sum.out;
            producedCounts = sum.counts;
            setSummary(sum);
            setStageStates(allDoneStageStates());
            push('success', `Organized ${sum.counts.docs} documents in ${sum.elapsed_s.toFixed(1)}s.`);
            break;
          }
          case 'error': {
            const msg = (ev.data as BuildErrorEvent).message;
            setError(msg);
            push('error', `Organizing failed: ${msg}`);
            break;
          }
          default:
            break;
        }
      },
    })
      .catch((e) => {
        if (!ctrl.signal.aborted) {
          setError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        setBuilding(false);
        abortRef.current = null;
        if (producedOut) {
          const out = producedOut;
          // Claim the produced space synchronously so the mode-2 effect (which
          // re-runs as `building` flips false) short-circuits its stale guard
          // instead of racing this inspect with a re-inspect of the old space.
          inspectedSpace.current = out;
          // Inspect the freshly-built space, then register it with the store so the
          // rest of the journey (Ask, recents) sees it. openSpace keeps us on
          // 'organize' (its default goTo) so we land on the after view.
          void api
            .inspect(out)
            .then((res) => {
              setInspect(res);
              store.openSpace(
                {
                  path: out,
                  counts: producedCounts
                    ? {
                        docs: producedCounts.docs,
                        chunks: producedCounts.chunks,
                        relations: producedCounts.relations,
                      }
                    : undefined,
                },
                { goTo: 'organize' },
              );
              // The produced space is now inspected and registered; drop the
              // build-completed flag so the mode-2 effect resumes handling later
              // space switches (e.g. opening a recent from the Library).
              setSummary(null);
            })
            .catch((e) => {
              setError(e instanceof Error ? e.message : String(e));
              // Clear the build summary so showBuildTimeline drops to false and
              // the normal error/empty after view renders instead of getting
              // stuck on the spinner-less 'Opening your space…' card.
              setSummary(null);
            });
        }
      });

    return () => {
      ctrl.abort();
    };
    // We intentionally only depend on buildNonce — the other store values are read
    // once at trigger time and must not re-run the build.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.buildNonce]);

  // --------------------------------------------------------------- mode 2: open
  useEffect(() => {
    // Skip while a build is pending/running — mode 1 owns the inspect in that case.
    if (building) return;
    // Skip right after a build completes: `building` has flipped false but the
    // produced space hasn't been registered with the store yet (openSpace runs
    // after the post-build inspect resolves), so store.currentSpace is still the
    // OLD space here. Without this guard we'd race mode 1 by re-inspecting that
    // stale space. summary stays set until the new inspect lands (or is cleared
    // on its error path), so this hands ownership back to mode 1.
    if (summary !== null) return;
    const space = store.currentSpace;
    if (!space) return;
    if (inspectedSpace.current === space) return;
    void runInspect(space);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.currentSpace, building, summary]);

  // When the active space changes (e.g. opening a different recent space), drop any open drawer
  // or edge explainer so it never lingers showing a doc/edge that isn't in the new space.
  useEffect(() => {
    setSelectedDoc(null);
    setSelectedEdge(null);
  }, [store.currentSpace]);

  const onCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // ---- after-view derived data ----
  const documents = useMemo<InspectDocument[]>(
    () => inspect?.documents ?? [],
    [inspect],
  );
  const topicCounts = useMemo(
    () => clusterCounts(documents, (d) => d.topics),
    [documents],
  );
  const tagCounts = useMemo(
    () => clusterCounts(documents, (d) => d.tags),
    [documents],
  );

  const onSelectNode = useCallback(
    (id: string) => {
      const doc = documents.find((d) => d.id === id) ?? null;
      setSelectedDoc(doc);
    },
    [documents],
  );

  const onSelectEdge = useCallback((link: GraphLink) => setSelectedEdge(link), []);

  // Resolve an edge's endpoints to friendly labels for the relation explainer.
  const edgeExplainer = useMemo(() => {
    if (!selectedEdge) return null;
    const label = (id: string) => {
      const doc = documents.find((d) => d.id === id);
      if (!doc) return id;
      const parts = doc.path.split(/[\\/]+/).filter(Boolean);
      return parts.length ? parts[parts.length - 1] : doc.path;
    };
    return {
      source: label(selectedEdge.source),
      target: label(selectedEdge.target),
      type: selectedEdge.type,
    };
  }, [selectedEdge, documents]);

  // =====================================================================render
  // While building, show the live timeline (it takes priority over the after view).
  // `buildPending` covers the one frame between startBuild() (which bumps buildNonce and
  // flips the phase) and the mode-1 effect setting building=true, so the empty state
  // never flashes during the Ingest→Organize handoff.
  const buildPending = store.buildNonce > handledNonce.current;
  const showBuildTimeline = building || buildPending || (summary !== null && !inspect);

  return (
    <div className="space-y-5">
      {error ? <Banner kind="error">{error}</Banner> : null}

      {showBuildTimeline ? (
        <div className="card">
          <SectionHeader
            title="Organizing your knowledge"
            description="Reading your files, understanding them, and weaving the connections."
          />
          <ol className="space-y-2">
            {STAGE_ORDER.map((s) => {
              const st = stageStates[s];
              const dotStyle =
                st === 'done'
                  ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
                  : st === 'active'
                    ? 'border-accent bg-accent-soft text-accent animate-pulse'
                    : 'border-slate-200 bg-white text-slate-400';
              return (
                <li
                  key={s}
                  className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-sm font-medium ${dotStyle}`}
                >
                  <span aria-hidden="true">
                    {st === 'done' ? '✓' : st === 'active' ? '●' : '○'}
                  </span>
                  <span>{STAGE_LABEL[s]}</span>
                </li>
              );
            })}
          </ol>
          {startInfo ? (
            <p className="mt-3 text-xs text-slate-500">
              Organizing <span className="font-mono">{startInfo.directory}</span>
            </p>
          ) : null}
          {summary ? (
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatCard label="Documents" value={summary.counts.docs ?? 0} />
              <StatCard label="Passages" value={summary.counts.chunks ?? 0} />
              <StatCard label="Connections" value={summary.counts.relations ?? 0} />
              <StatCard label="Elapsed" value={`${summary.elapsed_s.toFixed(1)}s`} />
            </div>
          ) : null}
          {building ? (
            <div className="mt-4">
              <button className="btn-ghost" onClick={onCancel}>
                Cancel
              </button>
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">Opening your space…</p>
          )}
        </div>
      ) : null}

      {/* Loading state for the after view (mode 2 inspect). */}
      {!showBuildTimeline && inspectLoading && !inspect ? (
        <div className="card space-y-4">
          <Skeleton className="h-6 w-48" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
          </div>
          <Skeleton className="h-64 w-full" />
        </div>
      ) : null}

      {/* Empty state: nothing building, nothing inspected, no current space. */}
      {!showBuildTimeline && !inspectLoading && !inspect && !error ? (
        <div className="card">
          <SectionHeader
            title="Nothing organized yet"
            description="Pick a folder in Ingest and start organizing, or open a recent space from the Library."
          />
        </div>
      ) : null}

      {/* The after view. */}
      {!showBuildTimeline && inspect ? (
        <>
          {/* Space toolbar: identify the open space and offer a one-click export of the
              self-contained .indx (download via <a>, not fetch — see api.exportUrl). */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="min-w-0 truncate font-mono text-xs text-slate-500" title={inspect.path}>
              {inspect.path}
            </p>
            <a
              className="btn-ghost px-3 py-1.5 text-xs"
              href={api.exportUrl(inspect.path)}
              download
            >
              ⭳ Export .indx
            </a>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Documents" value={inspect.stats.documents} />
            <StatCard label="Passages" value={inspect.stats.chunks} />
            <StatCard label="Connections" value={inspect.stats.relations} />
            <StatCard label="Searchable" value={inspect.stats.embeddings} />
          </div>

          <div className="card">
            <SectionHeader
              title="How your knowledge connects"
              description="Documents are nodes; lines are the relationships discovered between them. Click a node to dive in."
            />
            <RelationGraph
              documents={inspect.documents}
              edges={inspect.edges}
              edgeTotal={inspect.edge_total}
              onSelectNode={onSelectNode}
              onSelectEdge={onSelectEdge}
            />
            {edgeExplainer ? (
              <div className="mt-3 flex items-start justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
                <p className="text-slate-600">
                  <span className="font-medium text-ink">{edgeExplainer.source}</span>{' '}
                  <span className="badge bg-accent-soft text-accent">{edgeExplainer.type}</span>{' '}
                  <span className="font-medium text-ink">{edgeExplainer.target}</span>
                </p>
                <button
                  className="shrink-0 text-slate-400 hover:text-slate-600"
                  aria-label="Dismiss relation"
                  onClick={() => setSelectedEdge(null)}
                >
                  ✕
                </button>
              </div>
            ) : null}
          </div>

          <div className="grid gap-5 lg:grid-cols-3">
            <div className="card">
              <SectionHeader title="Document types" />
              <BarChart data={inspect.types} emptyLabel="No typed documents" />
            </div>
            <div className="card">
              <SectionHeader title="Topics" />
              <BarChart data={topicCounts} emptyLabel="No topics yet" />
            </div>
            <div className="card">
              <SectionHeader title="Tags" />
              <BarChart data={tagCounts} emptyLabel="No tags yet" />
            </div>
          </div>

          <div className="card">
            <SectionHeader
              title="Documents"
              description={`${inspect.documents.length} document${
                inspect.documents.length === 1 ? '' : 's'
              } — click any row to open it.`}
            />
            {inspect.documents.length === 0 ? (
              <p className="text-sm text-slate-400">No documents in this space.</p>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase text-slate-400">
                    <tr>
                      <th className="py-1 pr-3">Path</th>
                      <th className="py-1 pr-3">Type</th>
                      <th className="py-1 pr-3">Folder</th>
                      <th className="py-1 pr-3">Topics</th>
                      <th className="py-1 pr-3">Tags</th>
                      <th className="py-1 pr-3 text-right">Passages</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspect.documents.map((d) => (
                      <tr
                        key={d.id}
                        className="cursor-pointer border-t border-slate-100 align-top hover:bg-slate-50"
                        onClick={() => setSelectedDoc(d)}
                        tabIndex={0}
                        role="button"
                        aria-label={`Open ${d.path}`}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            setSelectedDoc(d);
                          }
                        }}
                      >
                        <td className="py-1 pr-3 font-mono text-xs">{d.path}</td>
                        <td className="py-1 pr-3">{d.type ?? '—'}</td>
                        <td className="py-1 pr-3 font-mono text-xs">
                          {d.folder || '.'}
                        </td>
                        <td className="py-1 pr-3 text-xs text-slate-500">
                          {d.topics.length ? d.topics.join(', ') : '—'}
                        </td>
                        <td className="py-1 pr-3 text-xs text-slate-500">
                          {d.tags.length ? d.tags.join(', ') : '—'}
                        </td>
                        <td className="py-1 pr-3 text-right tabular-nums">
                          {d.chunks}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      ) : null}

      <DocumentDrawer
        space={inspect?.path ?? store.currentSpace}
        doc={selectedDoc}
        onClose={() => setSelectedDoc(null)}
      />
    </div>
  );
}
