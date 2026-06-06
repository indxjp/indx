'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';
import { ConfigTab } from '@/components/ConfigTab';
import { BuildTab } from '@/components/BuildTab';
import { InspectTab } from '@/components/InspectTab';
import { QueryTab } from '@/components/QueryTab';

type TabId = 'config' | 'build' | 'inspect' | 'query';

const TABS: { id: TabId; label: string }[] = [
  { id: 'config', label: 'Config' },
  { id: 'build', label: 'Build' },
  { id: 'inspect', label: 'Inspect' },
  { id: 'query', label: 'Query' },
];

export default function Page() {
  const [tab, setTab] = useState<TabId>('config');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  // Cross-tab handoffs: Build "Run demo" / summaries seed Inspect + Query.
  const [spacePath, setSpacePath] = useState<string>('');

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const goInspect = useCallback((path: string) => {
    setSpacePath(path);
    setTab('inspect');
  }, []);

  return (
    <div className="mx-auto min-h-full max-w-5xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-ink">
            indx <span className="text-accent">web tester</span>
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Configure a stack, run a build, then inspect and query the knowledge space.
          </p>
        </div>
        <div className="text-right text-xs text-slate-400">
          {health ? (
            <>
              <div>
                v{health.version} ·{' '}
                <span className={health.static ? 'text-emerald-600' : 'text-amber-600'}>
                  bundle {health.static ? 'present' : 'absent'}
                </span>
              </div>
              <div className="text-emerald-600">backend ok</div>
            </>
          ) : (
            <div className="text-rose-500">backend unreachable</div>
          )}
        </div>
      </header>

      <nav
        className="mb-6 flex gap-1 rounded-lg border border-slate-200 bg-white p-1 shadow-sm"
        role="tablist"
        aria-label="Sections"
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            id={`tab-${t.id}`}
            aria-controls={`panel-${t.id}`}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition ${
              tab === t.id
                ? 'bg-accent text-white shadow'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        <div
          role="tabpanel"
          id="panel-config"
          aria-labelledby="tab-config"
          hidden={tab !== 'config'}
        >
          {tab === 'config' ? <ConfigTab /> : null}
        </div>
        <div
          role="tabpanel"
          id="panel-build"
          aria-labelledby="tab-build"
          hidden={tab !== 'build'}
        >
          {tab === 'build' ? <BuildTab onInspect={goInspect} /> : null}
        </div>
        <div
          role="tabpanel"
          id="panel-inspect"
          aria-labelledby="tab-inspect"
          hidden={tab !== 'inspect'}
        >
          {tab === 'inspect' ? (
            <InspectTab spacePath={spacePath} onSpaceChange={setSpacePath} />
          ) : null}
        </div>
        <div
          role="tabpanel"
          id="panel-query"
          aria-labelledby="tab-query"
          hidden={tab !== 'query'}
        >
          {tab === 'query' ? <QueryTab initialSpace={spacePath} /> : null}
        </div>
      </main>

      <footer className="mt-10 text-center text-xs text-slate-400">
        indx app · local web tester · operates on server-side paths
      </footer>
    </div>
  );
}
