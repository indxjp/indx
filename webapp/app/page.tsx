'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';
import { StoreProvider, useStore } from '@/lib/store';
import { ToastProvider } from '@/components/ui';
import { Shell } from '@/components/Shell';
import Library from '@/components/Library';
import { IngestView } from '@/components/IngestView';
import { OrganizeView } from '@/components/OrganizeView';
import { AskView } from '@/components/AskView';

/**
 * Inner app: reads the journey phase + theme from the store, fetches backend
 * health once, and keeps all four phase views mounted so their local state
 * survives navigation. Visibility is driven by `hidden` toggled on store.phase.
 */
function AppInner() {
  const { phase, setPhase, theme, toggleTheme } = useStore();
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <Shell
      phase={phase}
      setPhase={setPhase}
      health={health}
      theme={theme}
      onToggleTheme={toggleTheme}
    >
      <div className="mx-auto w-full max-w-5xl px-4 py-6">
        {/* The Library phase has its own hero h1; hide this header there to avoid two h1s. */}
        <header className="mb-6" hidden={phase === 'library'}>
          <h1 className="text-2xl font-bold tracking-tight text-ink">
            INDX
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Turn a pile of files into an organized, AI-ready knowledge base you
            can talk to.
          </p>
        </header>

        <div hidden={phase !== 'library'}>
          <Library />
        </div>
        <div hidden={phase !== 'ingest'}>
          <IngestView />
        </div>
        <div hidden={phase !== 'organize'}>
          <OrganizeView />
        </div>
        <div hidden={phase !== 'ask'}>
          <AskView />
        </div>

        <footer className="mt-10 text-center text-xs text-slate-400 dark:text-slate-500">
          INDX · your knowledge base, organized and searchable · runs
          locally, operates on local paths
        </footer>
      </div>
    </Shell>
  );
}

export default function Page() {
  return (
    <StoreProvider>
      <ToastProvider>
        <AppInner />
      </ToastProvider>
    </StoreProvider>
  );
}
