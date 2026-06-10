'use client';

import type { Phase, Theme } from '@/lib/store';
import type { HealthResponse } from '@/lib/types';

/** Journey steps, in order. */
const NAV: { id: Phase; label: string; hint: string }[] = [
  { id: 'library', label: 'Library', hint: 'Open a folder or a recent space' },
  { id: 'ingest', label: 'Ingest', hint: 'Pick a source and preview the plan' },
  { id: 'organize', label: 'Organize', hint: 'Build & explore the knowledge graph' },
  { id: 'ask', label: 'Ask', hint: 'Search, query, and use as an agent' },
];

/** Compact backend health + version badge (green ok / amber bundle / rose unreachable). */
function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return (
      <div
        role="status"
        className="rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-right text-xs text-rose-600 dark:border-rose-900 dark:bg-rose-950"
      >
        backend unreachable
      </div>
    );
  }
  return (
    <div
      role="status"
      className="rounded-md border border-slate-200 bg-white px-2 py-1 text-right text-xs text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-900"
    >
      <div className="font-medium text-emerald-600 dark:text-emerald-400">backend ok</div>
      <div className="mt-0.5 text-slate-400">
        v{health.version} ·{' '}
        <span className={health.static ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
          bundle {health.static ? 'present' : 'absent'}
        </span>
      </div>
    </div>
  );
}

/** Light/dark theme toggle. */
function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const dark = theme === 'dark';
  return (
    <button
      type="button"
      className="btn-ghost px-2 py-1 text-sm"
      onClick={onToggle}
      aria-pressed={dark}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={dark ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      {dark ? '☀︎ Light' : '☾ Dark'}
    </button>
  );
}

/**
 * Persistent application frame: brand, journey nav (Library/Ingest/Organize/Ask),
 * a backend health + version badge, and a theme toggle. Renders {children} in a
 * <main>. Purely presentational — performs no data fetching.
 */
export function Shell({
  phase,
  setPhase,
  health,
  theme,
  onToggleTheme,
  children,
}: {
  phase: Phase;
  setPhase: (p: Phase) => void;
  health: HealthResponse | null;
  theme: Theme;
  onToggleTheme: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50 dark:bg-slate-950 md:flex-row">
      {/* Sidebar (becomes a top bar on small screens). */}
      <aside className="flex w-full shrink-0 flex-col gap-4 border-b border-slate-200 bg-white px-4 py-4 dark:border-slate-800 dark:bg-slate-900 md:w-60 md:border-b-0 md:border-r">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-sm font-bold text-white"
            >
              I
            </span>
            <span className="text-base font-bold tracking-tight text-ink">INDX</span>
          </div>
          <div className="md:hidden">
            <ThemeToggle theme={theme} onToggle={onToggleTheme} />
          </div>
        </div>

        <nav
          role="tablist"
          aria-label="Workflow"
          aria-orientation="vertical"
          className="flex gap-1 overflow-x-auto md:flex-col md:overflow-visible"
        >
          {NAV.map((item, i) => {
            const active = phase === item.id;
            return (
              <button
                key={item.id}
                role="tab"
                aria-selected={active}
                className={`flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-left text-sm font-medium transition md:w-full ${
                  active
                    ? 'bg-accent text-white shadow'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
                }`}
                title={item.hint}
                onClick={() => setPhase(item.id)}
              >
                <span
                  aria-hidden="true"
                  className={`flex h-5 w-5 items-center justify-center rounded-full text-xs tabular-nums ${
                    active
                      ? 'bg-white/20 text-white'
                      : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                  }`}
                >
                  {i + 1}
                </span>
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="mt-auto hidden flex-col gap-3 md:flex">
          <HealthBadge health={health} />
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Health badge for small screens (sidebar footer is hidden there). */}
        <div className="flex justify-end border-b border-slate-200 px-4 py-2 dark:border-slate-800 md:hidden">
          <HealthBadge health={health} />
        </div>
        <main className="min-w-0 flex-1">{children}</main>
      </div>
    </div>
  );
}

export default Shell;
