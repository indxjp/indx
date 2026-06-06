'use client';

import { useEffect, useRef } from 'react';

/** Section heading + optional description. */
export function SectionHeader({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div className="mb-4">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      {description ? (
        <p className="mt-1 text-sm text-slate-500">{description}</p>
      ) : null}
    </div>
  );
}

/** Inline status / error banner. */
export function Banner({
  kind,
  children,
}: {
  kind: 'error' | 'success' | 'info';
  children: React.ReactNode;
}) {
  const styles: Record<string, string> = {
    error: 'border-rose-200 bg-rose-50 text-rose-700',
    success: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    info: 'border-sky-200 bg-sky-50 text-sky-700',
  };
  return (
    <div
      role={kind === 'error' ? 'alert' : 'status'}
      className={`rounded-md border px-3 py-2 text-sm ${styles[kind]}`}
    >
      {children}
    </div>
  );
}

/** A simple labelled stat card. */
export function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
    </div>
  );
}

/** Horizontal bar chart for a name->count map. */
export function BarChart({
  data,
  emptyLabel = 'No data',
}: {
  data: Record<string, number>;
  emptyLabel?: string;
}) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return <p className="text-sm text-slate-400">{emptyLabel}</p>;
  }
  const max = Math.max(...entries.map(([, n]) => n), 1);
  return (
    <ul className="space-y-2">
      {entries.map(([name, n]) => (
        <li key={name} className="flex items-center gap-3">
          <span className="w-32 shrink-0 truncate text-sm text-slate-600" title={name}>
            {name}
          </span>
          <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-accent"
              style={{ width: `${Math.max((n / max) * 100, 4)}%` }}
            />
          </div>
          <span className="w-10 shrink-0 text-right text-sm tabular-nums text-slate-700">
            {n}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Modal directory picker built on /api/browse. */
import { useState, useCallback } from 'react';
import { api } from '@/lib/api';
import type { BrowseResponse } from '@/lib/types';

export function BrowseModal({
  open,
  initialPath,
  onClose,
  onSelect,
}: {
  open: boolean;
  initialPath?: string | null;
  onClose: () => void;
  onSelect: (path: string) => void;
}) {
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async (path?: string) => {
    setError(null);
    try {
      const res = await api.browse(path);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    if (open) {
      void load(initialPath ?? undefined);
    }
  }, [open, initialPath, load]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    dialogRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Directory picker"
        tabIndex={-1}
        className="flex max-h-[80vh] w-full max-w-lg flex-col rounded-xl bg-white shadow-xl outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <h3 className="text-sm font-semibold">Pick a directory</h3>
          <button className="btn-ghost px-2 py-1" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="border-b border-slate-200 px-4 py-2 font-mono text-xs text-slate-500">
          {data?.path ?? '…'}
        </div>
        {error ? (
          <div className="p-4">
            <Banner kind="error">{error}</Banner>
          </div>
        ) : null}
        <ul className="flex-1 overflow-auto p-2">
          {data?.parent ? (
            <li>
              <button
                className="w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-100"
                onClick={() => load(data.parent ?? undefined)}
              >
                <span className="mr-2">↩</span>.. (parent)
              </button>
            </li>
          ) : null}
          {data?.entries.map((entry) => (
            <li key={entry.path}>
              <button
                className="w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-100"
                onClick={() => (entry.is_dir ? load(entry.path) : onSelect(entry.path))}
              >
                <span className="mr-2">{entry.is_dir ? '📁' : '📄'}</span>
                {entry.name}
              </button>
            </li>
          ))}
          {data && data.entries.length === 0 ? (
            <li className="px-3 py-2 text-sm text-slate-400">Empty directory</li>
          ) : null}
        </ul>
        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3">
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={() => data && onSelect(data.path)}
            disabled={!data}
          >
            Use this folder
          </button>
        </div>
      </div>
    </div>
  );
}
