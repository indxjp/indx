'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/lib/api';
import { useStore } from '@/lib/store';
import type {
  ComponentInfo,
  ComponentsResponse,
  ConfigValidateResponse,
  SlotKey,
} from '@/lib/types';
import { Banner, SectionHeader } from '@/components/ui';

const SLOTS: { key: SlotKey; label: string }[] = [
  { key: 'parser', label: 'Parser' },
  { key: 'llm', label: 'LLM' },
  { key: 'vlm', label: 'VLM' },
  { key: 'embedder', label: 'Embedder' },
  { key: 'store', label: 'Store' },
  { key: 'output', label: 'Output writer' },
];

// The components map keys the writer slot as `output` (both defaults and offline). The
// config schema, however, is nested per slot (src/indx/config/schema.py):
//   parser.engine · enrich.llm · enrich.vlm · enrich.metadata · embed.model ·
//   store.backend · output.format
// plus opaque backend sub-tables (e.g. store.<backend> = {...}). We map between the flat
// UI slot values and that nested shape on load and on assemble.
type SlotValues = Record<SlotKey, string>;

function emptyValues(): SlotValues {
  return { parser: '', llm: '', vlm: '', embedder: '', store: '', output: '' };
}

/** Find the ComponentInfo for a chosen value within a slot's option list. */
function findOption(options: ComponentInfo[], value: string): ComponentInfo | undefined {
  return options.find((o) => o.name === value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function strField(table: Record<string, unknown>, key: string): string {
  const v = table[key];
  return typeof v === 'string' ? v : '';
}

/** Read the flat UI slot values out of a nested Config dict (Config.model_dump()). */
function valuesFromConfig(cfg: Record<string, unknown>): SlotValues {
  const parser = asRecord(cfg.parser);
  const enrich = asRecord(cfg.enrich);
  const embed = asRecord(cfg.embed);
  const store = asRecord(cfg.store);
  const output = asRecord(cfg.output);
  return {
    parser: strField(parser, 'engine'),
    llm: strField(enrich, 'llm'),
    vlm: strField(enrich, 'vlm'),
    embedder: strField(embed, 'model'),
    store: strField(store, 'backend'),
    output: strField(output, 'format'),
  };
}

/**
 * Pull the backend sub-table for the active store backend out of a loaded config so the
 * JSON editor round-trips it. e.g. for `store.backend = "qdrant"`, return `store.qdrant`.
 */
function storeSubTable(cfg: Record<string, unknown>): Record<string, unknown> {
  const store = asRecord(cfg.store);
  const backend = strField(store, 'backend');
  if (backend && store[backend] && typeof store[backend] === 'object') {
    return asRecord(store[backend]);
  }
  return {};
}

export function ConfigTab() {
  const { setPreset, setConfigPath } = useStore();
  const [components, setComponents] = useState<ComponentsResponse | null>(null);
  const [values, setValues] = useState<SlotValues>(emptyValues());
  const [backendJson, setBackendJson] = useState<string>('{}');
  const [backendJsonError, setBackendJsonError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<string[]>([]);
  const [metadataInput, setMetadataInput] = useState('');
  const [savePath, setSavePath] = useState('indx.toml');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [validation, setValidation] = useState<ConfigValidateResponse | null>(null);
  const [validating, setValidating] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ kind: 'success' | 'error'; text: string } | null>(
    null,
  );

  // ---------------------------------------------------------------- initial load
  useEffect(() => {
    Promise.all([api.components(), api.getConfig()])
      .then(([comp, cfg]) => {
        setComponents(comp);
        const cfgObj = cfg.config as Record<string, unknown>;
        setValues(valuesFromConfig(cfgObj));
        const sub = storeSubTable(cfgObj);
        setBackendJson(JSON.stringify(sub, null, 2));
        const enrich = asRecord(cfgObj.enrich);
        const meta = enrich.metadata;
        if (Array.isArray(meta)) {
          setMetadata(meta.filter((m): m is string => typeof m === 'string'));
        }
      })
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, []);

  // ----------------------------------------------------------- assembled config
  // Emits the nested schema shape so /config/validate and PUT /config accept it. Only
  // slots the user has chosen a value for are written; the rest fall back to defaults
  // server-side. The store backend sub-table (JSON editor) nests under store.<backend>.
  const assembledConfig = useMemo<Record<string, unknown>>(() => {
    const cfg: Record<string, unknown> = {};

    if (values.parser) cfg.parser = { engine: values.parser };

    const enrich: Record<string, unknown> = {};
    if (values.llm) enrich.llm = values.llm;
    if (values.vlm) enrich.vlm = values.vlm;
    if (metadata.length > 0) enrich.metadata = metadata;
    if (Object.keys(enrich).length > 0) cfg.enrich = enrich;

    if (values.embedder) cfg.embed = { model: values.embedder };

    let backendOpts: Record<string, unknown> | undefined;
    try {
      const parsed = backendJson.trim() ? JSON.parse(backendJson) : undefined;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        backendOpts = parsed as Record<string, unknown>;
      }
    } catch {
      backendOpts = undefined;
    }

    if (values.store) {
      const store: Record<string, unknown> = { backend: values.store };
      if (backendOpts && Object.keys(backendOpts).length > 0) {
        store[values.store] = backendOpts;
      }
      cfg.store = store;
    }

    if (values.output) cfg.output = { format: values.output };

    return cfg;
  }, [values, backendJson, metadata]);

  // --------------------------------------------------------- validate backend json
  useEffect(() => {
    if (!backendJson.trim()) {
      setBackendJsonError(null);
      return;
    }
    try {
      const parsed = JSON.parse(backendJson);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        setBackendJsonError('Backend options must be a JSON object.');
      } else {
        setBackendJsonError(null);
      }
    } catch (e) {
      setBackendJsonError(e instanceof Error ? e.message : 'Invalid JSON');
    }
  }, [backendJson]);

  // --------------------------------------------------------- debounced validation
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (backendJsonError) {
      setValidation(null);
      return;
    }
    setValidating(true);
    debounceRef.current = setTimeout(() => {
      api
        .validateConfig(assembledConfig)
        .then(setValidation)
        .catch((e) =>
          setValidation({
            valid: false,
            errors: [e instanceof Error ? e.message : String(e)],
          }),
        )
        .finally(() => setValidating(false));
    }, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [assembledConfig, backendJsonError]);

  // ----------------------------------------------------------------- presets
  const applyOffline = useCallback(() => {
    if (!components) return;
    const off = components.offline;
    setValues({
      parser: off.parser ?? 'plaintext',
      llm: off.llm ?? 'none',
      vlm: off.vlm ?? 'none',
      embedder: off.embedder ?? 'hash',
      store: off.store ?? 'jsonl',
      output: off.output ?? '.indx',
    });
  }, [components]);

  const applyDefaults = useCallback(() => {
    if (!components) return;
    const d = components.defaults;
    setValues({
      parser: d.parser ?? '',
      llm: d.llm ?? '',
      vlm: d.vlm ?? '',
      embedder: d.embedder ?? '',
      store: d.store ?? '',
      output: d.output ?? '',
    });
  }, [components]);

  // ----------------------------------------------------------------- metadata
  const addMetadata = useCallback(() => {
    const v = metadataInput.trim();
    if (v && !metadata.includes(v)) {
      setMetadata((m) => [...m, v]);
    }
    setMetadataInput('');
  }, [metadataInput, metadata]);

  // ----------------------------------------------------------------- save
  const onSave = useCallback(async () => {
    setSaveMsg(null);
    try {
      const res = await api.putConfig({ path: savePath, config: assembledConfig });
      // Saving a hand-tuned stack switches the active preset to "Advanced" and points the
      // build at this file (BuildRequest.config) instead of the offline/cloud defaults.
      setConfigPath(res.path);
      setPreset('custom');
      setSaveMsg({ kind: 'success', text: `Saved to ${res.path} — using this for Advanced builds.` });
    } catch (e) {
      setSaveMsg({ kind: 'error', text: e instanceof Error ? e.message : String(e) });
    }
  }, [savePath, assembledConfig, setConfigPath, setPreset]);

  if (loadError) {
    return <Banner kind="error">Failed to load config: {loadError}</Banner>;
  }
  if (!components) {
    return <p className="text-sm text-slate-500">Loading components…</p>;
  }

  const slotOptions: Record<SlotKey, ComponentInfo[]> = {
    parser: components.parser,
    llm: components.llm,
    vlm: components.vlm,
    embedder: components.embedder,
    store: components.store,
    output: components.output,
  };

  return (
    <div className="space-y-5">
      <div className="card">
        <SectionHeader
          title="Stack configuration"
          description="Pick a backend for each swappable slot. Options that need an uninstalled extra are disabled."
        />
        <div className="mb-4 flex flex-wrap gap-2">
          <button className="btn-ghost" onClick={applyOffline}>
            Offline preset
          </button>
          <button className="btn-ghost" onClick={applyDefaults}>
            Cloud defaults
          </button>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {SLOTS.map(({ key, label }) => {
            const options = slotOptions[key];
            const chosen = findOption(options, values[key]);
            // Some valid selectors are provider-qualified (`openai:gpt-5-mini`) or dotted
            // (`.indx`) and so are NOT bare option names. Surface the current value as its own
            // option so the <select> can display and round-trip it instead of falling back to
            // the first entry. (findOption matches on exact name, so `chosen` is undefined here.)
            const current = values[key];
            const isExtraValue =
              current !== '' && !options.some((o) => o.name === current);
            return (
              <div key={key}>
                <label className="label" htmlFor={`slot-${key}`}>
                  {label}
                </label>
                <select
                  id={`slot-${key}`}
                  className="field"
                  value={values[key]}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [key]: e.target.value }))
                  }
                >
                  <option value="">(default)</option>
                  {isExtraValue ? (
                    <option value={current}>{current}</option>
                  ) : null}
                  {options.map((o) => (
                    <option key={o.name} value={o.name} disabled={!o.installed}>
                      {o.name}
                      {o.installed || !o.extra ? '' : ` — needs indx[${o.extra}]`}
                    </option>
                  ))}
                </select>
                {chosen && !chosen.installed && chosen.extra ? (
                  <span className="badge mt-1 bg-amber-100 text-amber-800">
                    needs indx[{chosen.extra}]
                  </span>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <SectionHeader
          title="Store backend options"
          description={`Opaque sub-table for the selected store backend${
            values.store ? ` (store.${values.store})` : ''
          }. JSON; validated live.`}
        />
        <textarea
          className="field font-mono text-xs"
          rows={8}
          spellCheck={false}
          aria-label="Backend options JSON"
          value={backendJson}
          onChange={(e) => setBackendJson(e.target.value)}
        />
        {backendJsonError ? (
          <p className="mt-2 text-sm text-rose-600">{backendJsonError}</p>
        ) : (
          <p className="mt-2 text-xs text-slate-400">Valid JSON object.</p>
        )}
      </div>

      <div className="card">
        <SectionHeader
          title="Enrichment metadata"
          description="Metadata fields the enrich stage should populate per document."
        />
        <div className="mb-3 flex flex-wrap gap-2">
          {metadata.length === 0 ? (
            <span className="text-sm text-slate-400">None</span>
          ) : (
            metadata.map((m) => (
              <span
                key={m}
                className="badge gap-1 bg-accent-soft text-accent"
              >
                {m}
                <button
                  className="ml-1 text-accent/70 hover:text-accent"
                  aria-label={`Remove ${m}`}
                  onClick={() => setMetadata((arr) => arr.filter((x) => x !== m))}
                >
                  ✕
                </button>
              </span>
            ))
          )}
        </div>
        <div className="flex gap-2">
          <input
            className="field"
            placeholder="e.g. topics"
            aria-label="New metadata field"
            value={metadataInput}
            onChange={(e) => setMetadataInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addMetadata();
              }
            }}
          />
          <button className="btn-ghost" onClick={addMetadata}>
            Add
          </button>
        </div>
      </div>

      <div className="card">
        <SectionHeader title="Validation & save" />
        <div className="mb-4">
          {validating ? (
            <Banner kind="info">Validating…</Banner>
          ) : validation ? (
            validation.valid ? (
              <Banner kind="success">Configuration is valid.</Banner>
            ) : (
              <Banner kind="error">
                <div className="font-medium">Invalid configuration</div>
                <ul className="mt-1 list-disc pl-5">
                  {validation.errors.map((err, i) => (
                    <li key={i} className="font-mono text-xs">
                      {err}
                    </li>
                  ))}
                </ul>
              </Banner>
            )
          ) : null}
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1">
            <label className="label" htmlFor="save-path">
              Write path (under server CWD)
            </label>
            <input
              id="save-path"
              className="field"
              value={savePath}
              onChange={(e) => setSavePath(e.target.value)}
            />
          </div>
          <button
            className="btn-primary"
            onClick={onSave}
            disabled={!!backendJsonError || (validation ? !validation.valid : false)}
          >
            Save indx.toml
          </button>
        </div>
        {saveMsg ? (
          <div className="mt-3">
            <Banner kind={saveMsg.kind}>{saveMsg.text}</Banner>
          </div>
        ) : null}
      </div>

      <details className="card">
        <summary className="cursor-pointer text-sm font-medium text-slate-600">
          Preview assembled config
        </summary>
        <pre className="mt-3 overflow-auto rounded-md bg-slate-900 p-3 text-xs text-slate-100">
          {JSON.stringify(assembledConfig, null, 2)}
        </pre>
      </details>
    </div>
  );
}
