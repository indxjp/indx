'use client';

import { useCallback, useRef, useState } from 'react';
import { api, streamBuild } from '@/lib/api';
import type {
  BuildRequest,
  BuildStartEvent,
  BuildSummary,
  DryRunResponse,
} from '@/lib/types';
import { Banner, BarChart, BrowseModal, SectionHeader, StatCard } from '@/components/ui';

// The pipeline stages the UI lights up as `stage` events arrive (docs/app-spec.md §3).
const STAGES = ['walk', 'parse', 'chunk', 'relate', 'enrich', 'embed-pack'] as const;
type StageName = (typeof STAGES)[number];
type StageState = 'pending' | 'active' | 'done';

export function BuildTab({ onInspect }: { onInspect: (path: string) => void }) {
  const [directory, setDirectory] = useState('');
  const [out, setOut] = useState('');
  const [name, setName] = useState('handbook');
  const [offline, setOffline] = useState(true);
  const [noEmbed, setNoEmbed] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [resume, setResume] = useState(false);
  const [jobs, setJobs] = useState('');

  const [running, setRunning] = useState(false);
  const [stageStates, setStageStates] = useState<Record<string, StageState>>({});
  const [startInfo, setStartInfo] = useState<BuildStartEvent | null>(null);
  const [summary, setSummary] = useState<BuildSummary | null>(null);
  const [plan, setPlan] = useState<DryRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [demoMsg, setDemoMsg] = useState<string | null>(null);

  const [browseFor, setBrowseFor] = useState<'directory' | 'out' | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const buildBody = useCallback((): BuildRequest => {
    const body: BuildRequest = {
      directory,
      name,
      offline,
      no_embed: noEmbed,
      resume,
      dry_run: dryRun,
    };
    if (out.trim()) body.out = out.trim();
    const j = parseInt(jobs, 10);
    if (!Number.isNaN(j) && j > 0) body.jobs = j;
    return body;
  }, [directory, name, offline, noEmbed, resume, dryRun, out, jobs]);

  const resetRun = useCallback(() => {
    setStageStates(
      STAGES.reduce<Record<string, StageState>>((acc, s) => {
        acc[s] = 'pending';
        return acc;
      }, {}),
    );
    setStartInfo(null);
    setSummary(null);
    setPlan(null);
    setError(null);
    setDemoMsg(null);
  }, []);

  const onRun = useCallback(async () => {
    if (!directory.trim()) {
      setError('Directory is required.');
      return;
    }
    resetRun();
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    await streamBuild(buildBody(), {
      signal: ctrl.signal,
      onEvent: (ev) => {
        switch (ev.event) {
          case 'start':
            setStartInfo(ev.data as BuildStartEvent);
            break;
          case 'stage': {
            const stageName = (ev.data as { name: StageName }).name;
            setStageStates((prev) => {
              const next = { ...prev };
              // mark prior active stages done, this one active
              for (const s of STAGES) {
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
          case 'done':
            setStageStates((prev) => {
              const next = { ...prev };
              for (const s of STAGES) next[s] = 'done';
              return next;
            });
            setSummary(ev.data as BuildSummary);
            break;
          case 'plan':
            setPlan(ev.data as DryRunResponse);
            break;
          case 'error':
            setError((ev.data as { message: string }).message);
            break;
          default:
            break;
        }
      },
    }).catch((e) => {
      if (!ctrl.signal.aborted) {
        setError(e instanceof Error ? e.message : String(e));
      }
    });

    setRunning(false);
    abortRef.current = null;
  }, [directory, buildBody, resetRun]);

  const onRunDemo = useCallback(async () => {
    resetRun();
    setRunning(true);
    try {
      const res = await api.demo();
      setSummary(res.summary);
      setStageStates(
        STAGES.reduce<Record<string, StageState>>((acc, s) => {
          acc[s] = 'done';
          return acc;
        }, {}),
      );
      setDemoMsg(`Demo built at ${res.out}`);
      setOut(res.out);
      onInspect(res.out);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [resetRun, onInspect]);

  const stageTimings: Record<string, number> = summary
    ? Object.fromEntries(summary.stages.map((s) => [s.name, s.seconds]))
    : {};

  return (
    <div className="space-y-5">
      <div className="card">
        <SectionHeader
          title="Build a knowledge space"
          description="Run DirectoryPipeline over a directory (or .zip) and watch the stages stream live."
        />
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <label className="label" htmlFor="bd-dir">
              Directory or .zip
            </label>
            <div className="flex gap-2">
              <input
                id="bd-dir"
                className="field"
                placeholder="/path/to/docs (use ./app for a folder literally named app)"
                value={directory}
                onChange={(e) => setDirectory(e.target.value)}
              />
              <button className="btn-ghost" onClick={() => setBrowseFor('directory')}>
                Browse
              </button>
            </div>
          </div>
          <div>
            <label className="label" htmlFor="bd-out">
              Output dir (optional)
            </label>
            <div className="flex gap-2">
              <input
                id="bd-out"
                className="field"
                placeholder="server makes a tempdir if blank"
                value={out}
                onChange={(e) => setOut(e.target.value)}
              />
              <button className="btn-ghost" onClick={() => setBrowseFor('out')}>
                Browse
              </button>
            </div>
          </div>
          <div>
            <label className="label" htmlFor="bd-name">
              Space name
            </label>
            <input
              id="bd-name"
              className="field"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
        </div>

        <fieldset className="mt-4">
          <legend className="label">Options</legend>
          <div className="flex flex-wrap items-center gap-4">
            {[
              { id: 'offline', label: 'offline', val: offline, set: setOffline },
              { id: 'no-embed', label: 'no-embed', val: noEmbed, set: setNoEmbed },
              { id: 'dry-run', label: 'dry-run', val: dryRun, set: setDryRun },
              { id: 'resume', label: 'resume', val: resume, set: setResume },
            ].map((c) => (
              <label key={c.id} className="inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-accent focus:ring-accent"
                  checked={c.val}
                  onChange={(e) => c.set(e.target.checked)}
                />
                {c.label}
              </label>
            ))}
            <label className="inline-flex items-center gap-2 text-sm">
              jobs
              <input
                type="number"
                min={1}
                className="field w-20"
                value={jobs}
                onChange={(e) => setJobs(e.target.value)}
              />
            </label>
          </div>
        </fieldset>

        <div className="mt-5 flex flex-wrap gap-2">
          <button className="btn-primary" onClick={onRun} disabled={running}>
            {running ? 'Running…' : 'Run build'}
          </button>
          <button className="btn-ghost" onClick={onRunDemo} disabled={running}>
            Run demo
          </button>
          {running && abortRef.current ? (
            <button
              className="btn-ghost"
              onClick={() => abortRef.current?.abort()}
            >
              Cancel
            </button>
          ) : null}
        </div>
      </div>

      {error ? <Banner kind="error">{error}</Banner> : null}
      {demoMsg ? <Banner kind="info">{demoMsg}</Banner> : null}

      {(running || summary || startInfo) && !dryRun ? (
        <div className="card">
          <SectionHeader title="Pipeline" />
          <ol className="flex flex-wrap gap-2">
            {STAGES.map((s) => {
              const st = stageStates[s] ?? 'pending';
              const style =
                st === 'done'
                  ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
                  : st === 'active'
                    ? 'border-accent bg-accent-soft text-accent animate-pulse'
                    : 'border-slate-200 bg-white text-slate-400';
              return (
                <li
                  key={s}
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium ${style}`}
                >
                  {st === 'done' ? '✓ ' : st === 'active' ? '● ' : '○ '}
                  {s}
                </li>
              );
            })}
          </ol>
          {startInfo ? (
            <p className="mt-3 text-xs text-slate-500">
              Building <span className="font-mono">{startInfo.directory}</span> →{' '}
              <span className="font-mono">{startInfo.out}</span>
            </p>
          ) : null}
        </div>
      ) : null}

      {plan ? (
        <div className="card">
          <SectionHeader
            title="Dry-run plan"
            description={`${plan.documents.length} documents across ${plan.folders.length} folders`}
          />
          <p className="mb-3 text-xs text-slate-500">
            root <span className="font-mono">{plan.root}</span> · embed:{' '}
            {String(plan.embed)} · enrich: {String(plan.enrich)}
          </p>
          <div className="overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-slate-400">
                <tr>
                  <th className="py-1 pr-3">Path</th>
                  <th className="py-1 pr-3">Type</th>
                  <th className="py-1 pr-3">Folder</th>
                  <th className="py-1 pr-3 text-right">Bytes</th>
                </tr>
              </thead>
              <tbody>
                {plan.documents.map((d) => (
                  <tr key={d.id} className="border-t border-slate-100">
                    <td className="py-1 pr-3 font-mono text-xs">{d.path}</td>
                    <td className="py-1 pr-3">{d.type ?? '—'}</td>
                    <td className="py-1 pr-3 font-mono text-xs">{d.folder || '.'}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">{d.size_bytes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {summary ? (
        <div className="card space-y-5">
          <SectionHeader title="Build summary" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Documents" value={summary.counts.docs ?? 0} />
            <StatCard label="Chunks" value={summary.counts.chunks ?? 0} />
            <StatCard label="Relations" value={summary.counts.relations ?? 0} />
            <StatCard label="Elapsed" value={`${summary.elapsed_s.toFixed(2)}s`} />
          </div>
          <div>
            <h3 className="mb-2 text-sm font-medium text-slate-600">Per-stage timing</h3>
            <BarChart data={stageTimings} />
          </div>
          <div>
            <h3 className="mb-2 text-sm font-medium text-slate-600">Components used</h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(summary.components).map(([slot, backend]) => (
                <span key={slot} className="badge bg-slate-100 text-slate-700">
                  {slot}: <span className="ml-1 font-mono">{backend}</span>
                </span>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-xs text-slate-500">
              out: <span className="font-mono">{summary.out}</span>
            </p>
            <button className="btn-ghost" onClick={() => onInspect(summary.out)}>
              Inspect this space →
            </button>
          </div>
        </div>
      ) : null}

      <BrowseModal
        open={browseFor !== null}
        initialPath={browseFor === 'directory' ? directory : out}
        onClose={() => setBrowseFor(null)}
        onSelect={(p) => {
          if (browseFor === 'directory') setDirectory(p);
          else if (browseFor === 'out') setOut(p);
          setBrowseFor(null);
        }}
      />
    </div>
  );
}
