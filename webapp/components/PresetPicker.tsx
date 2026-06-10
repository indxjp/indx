'use client';

import { useMemo } from 'react';
import type { ComponentInfo, ComponentsResponse } from '@/lib/types';
import type { Preset } from '@/lib/store';
import { Banner, Skeleton } from '@/components/ui';

/**
 * Three selectable preset chips that steer the build stack:
 *   - 'Local & private'  -> 'local'  (offline preset; no cloud extras needed)
 *   - 'Best quality'     -> 'cloud'  (uses components.defaults; may need extras installed)
 *   - 'Advanced'         -> 'custom' (user hand-tunes the stack via <ConfigTab/>)
 *
 * For 'Best quality', we resolve each backend in `components.defaults` against the matching
 * slot's ComponentInfo list and surface any that are NOT installed (so the user knows which
 * `indx[<extra>]` to install before picking cloud).
 *
 * Controlled: render `value`, emit selections through `onChange`.
 */

interface PresetOption {
  value: Preset;
  title: string;
  blurb: string;
}

const OPTIONS: PresetOption[] = [
  {
    value: 'local',
    title: 'Local & private',
    blurb: 'Runs fully offline. Nothing leaves your machine — no API keys required.',
  },
  {
    value: 'cloud',
    title: 'Best quality',
    blurb: 'Cloud models for richer parsing, summaries, and embeddings.',
  },
  {
    value: 'custom',
    title: 'Advanced',
    blurb: 'Hand-pick a backend for every slot in the stack.',
  },
];

/** The components response keys the writer slot as `output`; defaults use the same keys. */
const SLOT_KEYS: (keyof Pick<
  ComponentsResponse,
  'parser' | 'llm' | 'vlm' | 'embedder' | 'store' | 'output'
>)[] = ['parser', 'llm', 'vlm', 'embedder', 'store', 'output'];

/** A cloud default backend that needs an uninstalled extra. */
interface MissingDefault {
  slot: string;
  name: string;
  extra: string | null;
}

/**
 * Resolve each backend named in `components.defaults` against its slot's option list and
 * collect the ones whose ComponentInfo.installed is false.
 */
function missingCloudDefaults(components: ComponentsResponse): MissingDefault[] {
  const missing: MissingDefault[] = [];
  for (const slot of SLOT_KEYS) {
    const name = components.defaults[slot];
    if (!name) continue;
    const options: ComponentInfo[] = components[slot];
    const info = options.find((o) => o.name === name);
    // Unknown selectors (provider-qualified / dotted) can't be checked — skip rather than
    // falsely flag them as missing.
    if (!info) continue;
    if (!info.installed) {
      missing.push({ slot, name, extra: info.extra });
    }
  }
  return missing;
}

export function PresetPicker({
  value,
  onChange,
  components,
}: {
  value: Preset;
  onChange: (p: Preset) => void;
  components: ComponentsResponse | null;
}) {
  const missing = useMemo<MissingDefault[]>(
    () => (components ? missingCloudDefaults(components) : []),
    [components],
  );

  return (
    <div>
      <div
        role="radiogroup"
        aria-label="Stack preset"
        className="grid grid-cols-1 gap-3 sm:grid-cols-3"
      >
        {OPTIONS.map((opt) => {
          const selected = value === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(opt.value)}
              className={`rounded-lg border px-4 py-3 text-left shadow-sm transition focus:outline-none focus:ring-2 focus:ring-accent/40 ${
                selected
                  ? 'border-accent bg-accent-soft'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-sm font-semibold ${
                    selected ? 'text-accent' : 'text-ink'
                  }`}
                >
                  {opt.title}
                </span>
                <span
                  aria-hidden="true"
                  className={`flex h-4 w-4 items-center justify-center rounded-full border ${
                    selected
                      ? 'border-accent bg-accent'
                      : 'border-slate-300 bg-white'
                  }`}
                >
                  {selected ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-white" />
                  ) : null}
                </span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-500">{opt.blurb}</p>
            </button>
          );
        })}
      </div>

      {/* Cloud-specific install guidance for the 'Best quality' preset. */}
      {value === 'cloud' ? (
        <div className="mt-3">
          {components === null ? (
            <div className="space-y-2" aria-busy="true">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-4 w-full" />
            </div>
          ) : missing.length === 0 ? (
            <Banner kind="success">
              All cloud default backends are installed and ready.
            </Banner>
          ) : (
            <Banner kind="info">
              <div className="font-medium">
                Some cloud defaults need extras installed:
              </div>
              <ul className="mt-1 space-y-1">
                {missing.map((m) => (
                  <li key={`${m.slot}:${m.name}`} className="text-sm">
                    <span className="font-mono">{m.name}</span>
                    <span className="text-slate-500"> ({m.slot})</span>
                    {m.extra ? (
                      <>
                        {' — '}
                        <span className="font-mono">pip install indx[{m.extra}]</span>
                      </>
                    ) : null}
                  </li>
                ))}
              </ul>
            </Banner>
          )}
        </div>
      ) : null}
    </div>
  );
}
