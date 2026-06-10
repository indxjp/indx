'use client';

import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';

// ---------------------------------------------------------------- public types

export type Phase = 'library' | 'ingest' | 'organize' | 'ask';
export type Preset = 'local' | 'cloud' | 'custom';
export type Theme = 'light' | 'dark';

export interface RecentSpace {
  path: string;
  name: string;
  builtAt: number;
  counts?: { docs: number; chunks: number; relations: number };
}

export interface StoreApi {
  phase: Phase;
  currentSpace: string | null;
  recentSpaces: RecentSpace[];
  preset: Preset;
  /** Path to the indx.toml saved from the Advanced (custom) editor; fed to build as
   *  BuildRequest.config when preset === 'custom'. */
  configPath: string | null;
  theme: Theme;
  source: string | null;
  buildNonce: number;
  setPhase(p: Phase): void;
  setSource(p: string | null): void;
  setPreset(p: Preset): void;
  setConfigPath(p: string | null): void;
  setCurrentSpace(path: string | null): void;
  addRecent(s: RecentSpace): void;
  /** Drop a space from the Recent list (e.g. a stale entry whose /tmp dir is gone). */
  removeRecent(path: string): void;
  openSpace(
    s: { path: string; name?: string; counts?: RecentSpace['counts'] },
    opts?: { goTo?: Phase },
  ): void;
  startBuild(): void;
  toggleTheme(): void;
}

// ----------------------------------------------------------------- persistence

const STORAGE_KEY = 'indx-app-store/v1';
const RECENTS_CAP = 12;

/** The subset of state we persist to localStorage. */
interface PersistedState {
  recentSpaces: RecentSpace[];
  theme: Theme;
  preset: Preset;
}

function basename(path: string): string {
  const trimmed = path.replace(/[/\\]+$/, '');
  const parts = trimmed.split(/[/\\]/);
  return parts[parts.length - 1] || trimmed || path;
}

/** Dedupe by path, most-recent-first, capped. */
function mergeRecent(list: RecentSpace[], entry: RecentSpace): RecentSpace[] {
  const filtered = list.filter((r) => r.path !== entry.path);
  return [entry, ...filtered].slice(0, RECENTS_CAP);
}

function loadPersisted(): PersistedState {
  const fallback: PersistedState = {
    recentSpaces: [],
    theme: 'light',
    preset: 'local',
  };
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    return {
      recentSpaces: Array.isArray(parsed.recentSpaces)
        ? parsed.recentSpaces.filter(
            (r): r is RecentSpace =>
              !!r &&
              typeof r.path === 'string' &&
              typeof r.name === 'string' &&
              typeof r.builtAt === 'number' &&
              Number.isFinite(r.builtAt),
          )
        : [],
      theme: parsed.theme === 'dark' ? 'dark' : 'light',
      preset:
        parsed.preset === 'cloud' || parsed.preset === 'custom'
          ? parsed.preset
          : 'local',
    };
  } catch {
    return fallback;
  }
}

function savePersisted(state: State): void {
  if (typeof window === 'undefined') return;
  try {
    const persisted: PersistedState = {
      recentSpaces: state.recentSpaces,
      theme: state.theme,
      preset: state.preset,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
  } catch {
    // ignore quota / serialization failures
  }
}

// ------------------------------------------------------------------- reducer

interface State {
  phase: Phase;
  currentSpace: string | null;
  recentSpaces: RecentSpace[];
  preset: Preset;
  configPath: string | null;
  theme: Theme;
  source: string | null;
  buildNonce: number;
}

type Action =
  | { type: 'setPhase'; phase: Phase }
  | { type: 'setSource'; source: string | null }
  | { type: 'setPreset'; preset: Preset }
  | { type: 'setConfigPath'; path: string | null }
  | { type: 'setCurrentSpace'; path: string | null }
  | { type: 'addRecent'; space: RecentSpace }
  | { type: 'removeRecent'; path: string }
  | {
      type: 'openSpace';
      space: { path: string; name?: string; counts?: RecentSpace['counts'] };
      goTo: Phase;
    }
  | { type: 'startBuild' }
  | { type: 'toggleTheme' }
  | { type: 'hydratePersisted'; persisted: PersistedState };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'setPhase':
      return { ...state, phase: action.phase };
    case 'setSource':
      return { ...state, source: action.source };
    case 'setPreset':
      return { ...state, preset: action.preset };
    case 'setConfigPath':
      return { ...state, configPath: action.path };
    case 'setCurrentSpace':
      return { ...state, currentSpace: action.path };
    case 'addRecent':
      return {
        ...state,
        recentSpaces: mergeRecent(state.recentSpaces, action.space),
      };
    case 'removeRecent':
      return {
        ...state,
        recentSpaces: state.recentSpaces.filter((r) => r.path !== action.path),
      };
    case 'openSpace': {
      const { path, name, counts } = action.space;
      const entry: RecentSpace = {
        path,
        name: name ?? basename(path),
        builtAt: Date.now(),
        counts,
      };
      return {
        ...state,
        currentSpace: path,
        recentSpaces: mergeRecent(state.recentSpaces, entry),
        phase: action.goTo,
      };
    }
    case 'startBuild':
      return {
        ...state,
        buildNonce: state.buildNonce + 1,
        phase: 'organize',
      };
    case 'toggleTheme':
      return { ...state, theme: state.theme === 'dark' ? 'light' : 'dark' };
    case 'hydratePersisted':
      return {
        ...state,
        recentSpaces: action.persisted.recentSpaces,
        theme: action.persisted.theme,
        preset: action.persisted.preset,
      };
    default:
      return state;
  }
}

// Deterministic seed — must NOT read localStorage, so the first client render
// matches the server-prerendered HTML (this is an output:'export' static build).
// Persisted state is merged in after mount via the hydratePersisted effect below.
function init(): State {
  return {
    phase: 'library',
    currentSpace: null,
    recentSpaces: [],
    preset: 'local',
    configPath: null,
    theme: 'light',
    source: null,
    buildNonce: 0,
  };
}

// ------------------------------------------------------------------- context

const StoreContext = createContext<StoreApi | null>(null);

export function StoreProvider(props: {
  children: React.ReactNode;
}): React.JSX.Element {
  const [state, dispatch] = useReducer(reducer, undefined, init);
  const hydrated = useRef(false);

  // Load persisted state from localStorage after mount (not during init), so the first
  // client render matches the server-prerendered HTML and React doesn't warn / flash.
  useEffect(() => {
    dispatch({ type: 'hydratePersisted', persisted: loadPersisted() });
    hydrated.current = true;
  }, []);

  // Persist the durable slice whenever it changes — but not before hydration, or the
  // initial fallback would clobber the stored value before we've read it back.
  useEffect(() => {
    if (!hydrated.current) return;
    savePersisted(state);
    // Depend only on the persisted slices — not the whole `state` object, which changes
    // identity on every dispatch and would re-write localStorage on unrelated updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.recentSpaces, state.theme, state.preset]);

  // Apply the theme to the document root.
  useEffect(() => {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;
    if (state.theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [state.theme]);

  const value = useMemo<StoreApi>(
    () => ({
      phase: state.phase,
      currentSpace: state.currentSpace,
      recentSpaces: state.recentSpaces,
      preset: state.preset,
      configPath: state.configPath,
      theme: state.theme,
      source: state.source,
      buildNonce: state.buildNonce,
      setPhase: (p) => dispatch({ type: 'setPhase', phase: p }),
      setSource: (p) => dispatch({ type: 'setSource', source: p }),
      setPreset: (p) => dispatch({ type: 'setPreset', preset: p }),
      setConfigPath: (p) => dispatch({ type: 'setConfigPath', path: p }),
      setCurrentSpace: (path) =>
        dispatch({ type: 'setCurrentSpace', path }),
      addRecent: (s) => dispatch({ type: 'addRecent', space: s }),
      removeRecent: (path) => dispatch({ type: 'removeRecent', path }),
      openSpace: (s, opts) =>
        dispatch({
          type: 'openSpace',
          space: s,
          goTo: opts?.goTo ?? 'organize',
        }),
      startBuild: () => dispatch({ type: 'startBuild' }),
      toggleTheme: () => dispatch({ type: 'toggleTheme' }),
    }),
    [state],
  );

  // Authored as a .ts file (per the shared contract), so we cannot use JSX
  // literal syntax here; createElement keeps this valid TypeScript.
  return React.createElement(
    StoreContext.Provider,
    { value },
    props.children,
  );
}

export function useStore(): StoreApi {
  const ctx = useContext(StoreContext);
  if (ctx === null) {
    throw new Error('useStore must be used within a <StoreProvider>');
  }
  return ctx;
}
