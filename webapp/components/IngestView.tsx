'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/lib/api';
import { useStore } from '@/lib/store';
import type { BuildRequest, ComponentsResponse, DryRunResponse } from '@/lib/types';
import { Banner, BrowseModal, SectionHeader, Skeleton, StatCard } from '@/components/ui';
import { PresetPicker } from '@/components/PresetPicker';
import { ConfigTab } from '@/components/ConfigTab';

/** Human-readable byte sizes for the "before" picture. */
function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = bytes;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}

/** Last path segment, for a friendly source label. */
function basename(path: string): string {
  const trimmed = path.replace(/[/\\]+$/, '');
  const parts = trimmed.split(/[/\\]/);
  return parts[parts.length - 1] || trimmed || path;
}

/** Lowercased file extension (without the dot) for the "before" breakdown, or 'no ext'.
 *  The dry-run is walk-only, so documents have no parsed `type` yet — bucketing by extension
 *  gives an honest, mixed (md/txt/pdf…) preview instead of lumping everything into "other". */
function extOf(path: string): string {
  const name = path.replace(/[/\\]+$/, '').split(/[/\\]/).pop() ?? '';
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : 'no ext';
}

interface FolderRow {
  folder: string;
  files: number;
  bytes: number;
}

/**
 * IngestView — the "Ingest" phase of the journey (docs/app-spec.md §Ingest).
 *
 * Lets the user confirm the source folder/zip (BrowseModal), choose a quality
 * preset (<PresetPicker/>), optionally drop into the full <ConfigTab/> for
 * advanced control, and preview a live plan (api.dryRun) — the "before" picture
 * of what will be organized: a folder tree, file counts by type, total size and
 * the embed/enrich flags. "Start organizing" hands off to store.startBuild().
 *
 * No props: it reads/writes the shared store and calls /api directly.
 */
export function IngestView() {
  const store = useStore();
  const { source, preset, configPath, setSource, setPreset, startBuild } = store;

  const [browsing, setBrowsing] = useState(false);
  const [components, setComponents] = useState<ComponentsResponse | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(preset === 'custom');

  // Keep the Advanced disclosure and the "Advanced" preset chip in sync: selecting the
  // chip (or saving a custom config) opens the editor. Collapsing the editor leaves the
  // preset as-is, so `open` is driven by local state, not preset, to stay collapsible.
  useEffect(() => {
    if (preset === 'custom') setAdvancedOpen(true);
  }, [preset]);

  const [plan, setPlan] = useState<DryRunResponse | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  // Used to ignore stale dry-run responses when source/preset change quickly.
  const reqIdRef = useRef(0);

  // ---------------------------------------------------------------- components
  // Loaded once so PresetPicker can flag uninstalled cloud extras.
  useEffect(() => {
    let alive = true;
    api
      .components()
      .then((c) => {
        if (alive) setComponents(c);
      })
      .catch(() => {
        // PresetPicker degrades gracefully on null; the plan is the source of truth.
      });
    return () => {
      alive = false;
    };
  }, []);

  // -------------------------------------------------------------- live dry-run
  const dryRunBody = useMemo<BuildRequest | null>(() => {
    if (!source || !source.trim()) return null;
    return {
      directory: source.trim(),
      offline: preset === 'local',
      // Advanced builds plan against the saved indx.toml; offline/cloud use the defaults.
      config: preset === 'custom' ? configPath : null,
      dry_run: true,
    };
  }, [source, preset, configPath]);

  useEffect(() => {
    if (!dryRunBody) {
      setPlan(null);
      setPlanError(null);
      setPlanLoading(false);
      return;
    }
    const id = ++reqIdRef.current;
    setPlanLoading(true);
    setPlanError(null);
    api
      .dryRun(dryRunBody)
      .then((res) => {
        if (id === reqIdRef.current) setPlan(res);
      })
      .catch((e) => {
        if (id === reqIdRef.current) {
          setPlan(null);
          setPlanError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        if (id === reqIdRef.current) setPlanLoading(false);
      });
  }, [dryRunBody]);

  // -------------------------------------------------------- derived "before" view
  const typeCounts = useMemo<Record<string, number>>(() => {
    if (!plan) return {};
    const counts: Record<string, number> = {};
    for (const d of plan.documents) {
      // `type` is null pre-parse, so fall back to the file extension for the "before" picture.
      const key = d.type ?? extOf(d.path);
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }, [plan]);

  const totalBytes = useMemo<number>(() => {
    if (!plan) return 0;
    return plan.documents.reduce((sum, d) => sum + (d.size_bytes || 0), 0);
  }, [plan]);

  const folderRows = useMemo<FolderRow[]>(() => {
    if (!plan) return [];
    const map = new Map<string, FolderRow>();
    for (const d of plan.documents) {
      const folder = d.folder || '.';
      const row = map.get(folder) ?? { folder, files: 0, bytes: 0 };
      row.files += 1;
      row.bytes += d.size_bytes || 0;
      map.set(folder, row);
    }
    return [...map.values()].sort((a, b) => a.folder.localeCompare(b.folder));
  }, [plan]);

  // -------------------------------------------------------------------- actions
  const onSelectSource = useCallback(
    (p: string) => {
      setSource(p);
      setBrowsing(false);
    },
    [setSource],
  );

  // Block starting when the dry-run failed: the build instantiates the same stack, so it would
  // fail identically (e.g. a cloud preset whose parser/store extra isn't installed). Letting the
  // user click "Start organizing" into a guaranteed-to-fail build is worse than disabling it with
  // a hint pointing at the fix.
  const canStart = !!source && !!source.trim() && !planError;

  return (
    <div className="space-y-5">
      {/* -------------------------------------------------------- source picker */}
      <div className="card">
        <SectionHeader
          title="What should we organize?"
          description="Point INDX at a folder or a .zip of documents. We'll read them in place — nothing leaves your machine unless you pick a cloud preset."
        />
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm">
            {source ? (
              <span className="font-mono text-xs text-slate-700">{source}</span>
            ) : (
              <span className="text-slate-400">No source selected yet.</span>
            )}
          </div>
          <button className="btn-primary" onClick={() => setBrowsing(true)}>
            {source ? 'Change source' : 'Choose a folder or .zip'}
          </button>
          {source ? (
            <button className="btn-ghost" onClick={() => setSource(null)}>
              Clear
            </button>
          ) : null}
        </div>
        {source ? (
          <p className="mt-2 text-xs text-slate-500">
            Organizing <span className="font-medium text-ink">{basename(source)}</span>.
          </p>
        ) : null}
      </div>

      {/* --------------------------------------------------------- preset picker */}
      <div className="card">
        <SectionHeader
          title="How should we process it?"
          description="Pick a starting point. You can fine-tune every component under Advanced."
        />
        <PresetPicker value={preset} onChange={setPreset} components={components} />
      </div>

      {/* ----------------------------------------------------- advanced (ConfigTab) */}
      <details
        className="card"
        open={advancedOpen}
        onToggle={(e) => {
          const open = (e.target as HTMLDetailsElement).open;
          setAdvancedOpen(open);
          // Opening the editor selects the Advanced preset (mirrors the chip).
          if (open && preset !== 'custom') setPreset('custom');
        }}
      >
        <summary className="cursor-pointer text-sm font-medium text-slate-600">
          Advanced — configure the full stack
        </summary>
        <p className="mt-1 mb-4 text-xs text-slate-500">
          Saving a stack here switches the preset to <span className="font-medium">Advanced</span>{' '}
          and writes an <span className="font-mono">indx.toml</span> the organizer will use.
        </p>
        {advancedOpen ? <ConfigTab /> : null}
      </details>

      {/* ---------------------------------------------------------- live plan */}
      <div className="card">
        <SectionHeader
          title="Here's what we found"
          description="A preview of your documents before organizing — the raw, unstructured 'before'."
        />

        {!source ? (
          <p className="text-sm text-slate-400">
            Choose a source above to preview your documents.
          </p>
        ) : planLoading && !plan ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-1/3" />
          </div>
        ) : planError ? (
          <Banner kind="error">Couldn&apos;t scan that source: {planError}</Banner>
        ) : plan ? (
          plan.documents.length === 0 ? (
            <Banner kind="info">
              No supported documents found under{' '}
              <span className="font-mono">{plan.root}</span>. Try a different folder.
            </Banner>
          ) : (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard label="Documents" value={plan.documents.length} />
                <StatCard label="Folders" value={plan.folders.length} />
                <StatCard label="Total size" value={formatBytes(totalBytes)} />
                <StatCard label="File types" value={Object.keys(typeCounts).length} />
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-mono text-slate-500">{plan.root}</span>
                <span
                  className={`badge ${
                    plan.embed
                      ? 'bg-emerald-100 text-emerald-700'
                      : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  {plan.embed ? 'Will make searchable (embed)' : 'No embeddings'}
                </span>
                <span
                  className={`badge ${
                    plan.enrich
                      ? 'bg-emerald-100 text-emerald-700'
                      : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  {plan.enrich ? 'Will summarize & tag (enrich)' : 'No enrichment'}
                </span>
              </div>

              {/* file counts by type */}
              <div>
                <h3 className="mb-2 text-sm font-medium text-slate-600">By file type</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(typeCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, n]) => (
                      <span key={type} className="badge bg-slate-100 text-slate-700">
                        {type}
                        <span className="ml-1 font-mono tabular-nums">{n}</span>
                      </span>
                    ))}
                </div>
              </div>

              {/* folder tree */}
              <div>
                <h3 className="mb-2 text-sm font-medium text-slate-600">Folders</h3>
                <div className="overflow-auto rounded-lg border border-slate-200">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-50 text-xs uppercase text-slate-400">
                      <tr>
                        <th className="px-3 py-2">Folder</th>
                        <th className="px-3 py-2 text-right">Files</th>
                        <th className="px-3 py-2 text-right">Size</th>
                      </tr>
                    </thead>
                    <tbody>
                      {folderRows.map((row) => (
                        <tr key={row.folder} className="border-t border-slate-100">
                          <td className="px-3 py-1.5 font-mono text-xs text-slate-700">
                            {row.folder}
                          </td>
                          <td className="px-3 py-1.5 text-right tabular-nums">
                            {row.files}
                          </td>
                          <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">
                            {formatBytes(row.bytes)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )
        ) : null}
      </div>

      {/* ------------------------------------------------------- start organizing */}
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn-primary" onClick={startBuild} disabled={!canStart}>
          Start organizing →
        </button>
        {!source || !source.trim() ? (
          <span className="text-sm text-slate-400">Choose a source to begin.</span>
        ) : planError ? (
          <span className="text-sm text-rose-500">
            Resolve the issue above before organizing — or switch to the Local &amp; private preset.
          </span>
        ) : planLoading ? (
          <span className="text-sm text-slate-400">Still scanning — you can start anyway.</span>
        ) : null}
      </div>

      <BrowseModal
        open={browsing}
        initialPath={source}
        onClose={() => setBrowsing(false)}
        onSelect={onSelectSource}
      />
    </div>
  );
}
