'use client';

import { useCallback, useRef, useState } from 'react';

import { api, ApiError } from '@/lib/api';
import { useStore } from '@/lib/store';
import type { RecentSpace } from '@/lib/store';
import { Banner, BrowseModal, SectionHeader, useToast } from '@/components/ui';

/** Human-friendly "time ago" from an epoch-ms timestamp. */
function timeAgo(ms: number): string {
  const diff = Date.now() - ms;
  if (!Number.isFinite(diff) || diff < 0) return 'just now';
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return 'just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day}d ago`;
  const mon = Math.floor(day / 30);
  if (mon < 12) return `${mon}mo ago`;
  return `${Math.floor(mon / 12)}y ago`;
}

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

/** A single recent-space row that reopens the space on click, with a remove affordance. */
function RecentRow({
  space,
  onOpen,
  onRemove,
}: {
  space: RecentSpace;
  onOpen: (s: RecentSpace) => void;
  onRemove: (s: RecentSpace) => void;
}) {
  const counts = space.counts;
  return (
    <li className="group relative flex items-stretch rounded-lg border border-slate-200 bg-white shadow-sm transition hover:border-accent/50 hover:bg-slate-50 focus-within:ring-2 focus-within:ring-accent/30">
      <button
        type="button"
        onClick={() => onOpen(space)}
        className="flex min-w-0 flex-1 items-center justify-between gap-4 rounded-lg px-4 py-3 text-left focus:outline-none"
      >
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-ink">{space.name}</div>
          <div className="truncate font-mono text-xs text-slate-400" title={space.path}>
            {space.path}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3 text-right">
          {counts ? (
            <div className="hidden text-xs text-slate-500 tabular-nums sm:block">
              {counts.docs} docs · {counts.chunks} chunks · {counts.relations} links
            </div>
          ) : null}
          <span className="whitespace-nowrap text-xs text-slate-400">
            {timeAgo(space.builtAt)}
          </span>
        </div>
      </button>
      <button
        type="button"
        onClick={() => onRemove(space)}
        aria-label={`Remove ${space.name} from recent spaces`}
        title="Remove from recent spaces"
        className="shrink-0 px-3 text-slate-400 opacity-0 transition hover:text-rose-500 focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-accent/30 group-hover:opacity-100"
      >
        ✕
      </button>
    </li>
  );
}

export default function Library() {
  const store = useStore();
  const { push } = useToast();
  const [browseOpen, setBrowseOpen] = useState(false);
  const [demoBusy, setDemoBusy] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // --- Open a folder: pick a directory, then move to the Ingest phase. --------
  const onPickFolder = useCallback(
    (path: string) => {
      setBrowseOpen(false);
      setError(null);
      store.setSource(path);
      store.setPhase('ingest');
    },
    [store],
  );

  // --- Try the demo: build the bundled sample space, then open it. ------------
  const onDemo = useCallback(async () => {
    if (demoBusy) return;
    setDemoBusy(true);
    setError(null);
    try {
      const res = await api.demo();
      store.openSpace(
        {
          path: res.out,
          name: 'demo',
          counts: {
            docs: res.summary.counts.docs,
            chunks: res.summary.counts.chunks,
            relations: res.summary.counts.relations,
          },
        },
        { goTo: 'organize' },
      );
      push('success', 'Demo space built — explore the graph below.');
    } catch (e) {
      const m = errMsg(e);
      setError(m);
      push('error', `Demo failed: ${m}`);
    } finally {
      setDemoBusy(false);
    }
  }, [demoBusy, store, push]);

  // --- Import a .indx: a built space opens directly; raw sources go to Ingest. -
  const onImportClick = useCallback(() => {
    if (importBusy) return;
    fileRef.current?.click();
  }, [importBusy]);

  const onImportFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      // Allow re-selecting the same file later.
      e.target.value = '';
      if (!file) return;
      setImportBusy(true);
      setError(null);
      try {
        const res = await api.importSpace(file);
        if (res.kind === 'space') {
          store.openSpace({ path: res.path });
          push('success', 'Space imported — opened in Organize.');
        } else {
          store.setSource(res.path);
          store.setPhase('ingest');
          push('success', 'Upload ready — review the plan in Ingest.');
        }
      } catch (err) {
        const m = errMsg(err);
        setError(m);
        push('error', `Import failed: ${m}`);
      } finally {
        setImportBusy(false);
      }
    },
    [store, push],
  );

  const onOpenRecent = useCallback(
    (s: RecentSpace) => {
      setError(null);
      store.openSpace({ path: s.path, name: s.name, counts: s.counts });
    },
    [store],
  );

  const onRemoveRecent = useCallback(
    (s: RecentSpace) => {
      store.removeRecent(s.path);
      push('info', `Removed “${s.name}” from recent spaces.`);
    },
    [store, push],
  );

  const recents = store.recentSpaces;

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      {/* Hero ----------------------------------------------------------------- */}
      <section className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">
          Turn a folder of files into a searchable knowledge space
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-sm text-slate-500">
          Point INDX at your documents. It reads, organizes, and connects them so you
          can explore the map and ask questions in plain language.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            className="btn-primary"
            onClick={() => setBrowseOpen(true)}
          >
            Open a folder
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={onDemo}
            disabled={demoBusy}
            aria-busy={demoBusy}
          >
            {demoBusy ? 'Building demo…' : 'Try the demo'}
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={onImportClick}
            disabled={importBusy}
            aria-busy={importBusy}
          >
            {importBusy ? 'Importing…' : 'Import a .indx'}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".indx,.zip,application/zip"
            className="hidden"
            onChange={onImportFile}
          />
        </div>

        {error ? (
          <div className="mx-auto mt-6 max-w-xl text-left">
            <Banner kind="error">{error}</Banner>
          </div>
        ) : null}
      </section>

      {/* Recent spaces -------------------------------------------------------- */}
      <section className="mt-12">
        <SectionHeader
          title="Recent spaces"
          description="Spaces you have built or opened on this device."
        />
        {recents.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 bg-white/50 px-6 py-10 text-center">
            <p className="text-sm font-medium text-slate-600">No spaces yet</p>
            <p className="mx-auto mt-1 max-w-sm text-sm text-slate-400">
              Open a folder to organize your own files, or try the demo to see how it
              works. Built spaces will appear here.
            </p>
          </div>
        ) : (
          <ul className="space-y-2">
            {recents.map((s) => (
              <RecentRow
                key={s.path}
                space={s}
                onOpen={onOpenRecent}
                onRemove={onRemoveRecent}
              />
            ))}
          </ul>
        )}
      </section>

      <BrowseModal
        open={browseOpen}
        onClose={() => setBrowseOpen(false)}
        onSelect={onPickFolder}
      />
    </div>
  );
}
