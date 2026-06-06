// TypeScript interfaces mirroring src/indx/app/models.py and the embedded core models
// (src/indx/store/base.py SearchHit, src/indx/core/stats.py SpaceStats,
//  src/indx/core/knowledge_space.py Manifest, src/indx/core/chunk.py Chunk,
//  src/indx/core/source.py Source). Field names match the Pydantic models exactly.

// ----------------------------------------------------------------------- health
export interface HealthResponse {
  status: string;
  version: string;
  static: boolean;
}

// ------------------------------------------------------------------- components
export interface ComponentInfo {
  name: string;
  builtin: boolean;
  extra: string | null;
  installed: boolean;
}

/** Slot key used in the components response and the build `components` map. */
export type SlotKey =
  | 'parser'
  | 'llm'
  | 'vlm'
  | 'embedder'
  | 'store'
  | 'output';

export interface ComponentsResponse {
  parser: ComponentInfo[];
  llm: ComponentInfo[];
  vlm: ComponentInfo[];
  embedder: ComponentInfo[];
  store: ComponentInfo[];
  output: ComponentInfo[];
  // defaults (cloud) + offline preset; both key the writer slot as `output`.
  defaults: Record<string, string>;
  offline: Record<string, string>;
}

// ----------------------------------------------------------------------- config
export interface ConfigGetResponse {
  path: string | null;
  config: Record<string, unknown>;
}

export interface ConfigValidateResponse {
  valid: boolean;
  errors: string[];
}

export interface ConfigPutRequest {
  path?: string | null;
  config: Record<string, unknown>;
}

// ------------------------------------------------------------------------ build
export interface BuildRequest {
  directory: string;
  out?: string | null;
  name?: string;
  parser?: string | null;
  llm?: string | null;
  vlm?: string | null;
  embedder?: string | null;
  store?: string | null;
  // NOTE: output format field is `format`, NOT `fmt`.
  format?: string | null;
  config?: string | null;
  offline?: boolean;
  no_embed?: boolean;
  resume?: boolean;
  dry_run?: boolean;
  jobs?: number | null;
}

export interface StageTiming {
  name: string;
  seconds: number;
}

/** `start` SSE event payload. */
export interface BuildStartEvent {
  // components map uses key `format` (not `output`) to match the build summary shape.
  components: {
    parser: string;
    llm: string;
    vlm: string;
    embedder: string;
    store: string;
    format: string;
  };
  directory: string;
  out: string;
}

/** `stage` SSE event payload. */
export interface BuildStageEvent {
  name: string; // walk | parse | chunk | relate | enrich | embed-pack
}

/** `done` SSE event payload (also the /api/demo summary). */
export interface BuildSummary {
  out: string;
  counts: { docs: number; chunks: number; relations: number } & Record<string, number>;
  elapsed_s: number;
  stages: StageTiming[]; // includes a final "write" entry
  components: Record<string, string>;
}

/** `error` SSE event payload. */
export interface BuildErrorEvent {
  message: string;
}

// ---------------------------------------------------------------------- dry-run
export interface DryRunDocument {
  id: string;
  path: string;
  type: string | null;
  folder: string;
  size_bytes: number;
}

export interface DryRunResponse {
  root: string;
  documents: DryRunDocument[];
  folders: string[];
  components: Record<string, string>;
  embed: boolean;
  enrich: boolean;
}

// ---------------------------------------------------------------------- inspect
export interface Manifest {
  schema_version: string;
  indx_version: string;
  source_root: string;
  components: Record<string, string>;
  embedding_model: string | null;
  embedding_dim: number | null;
}

export interface SpaceStats {
  documents: number;
  chunks: number;
  relations: number;
  embeddings: number;
  embed_dim: number | null;
  types: Record<string, number>;
  bytes_source: number;
}

export interface InspectDocument {
  id: string;
  type: string | null;
  path: string;
  folder: string;
  topics: string[];
  tags: string[];
  chunks: number;
}

export interface InspectResponse {
  path: string;
  manifest: Manifest;
  stats: SpaceStats;
  types: Record<string, number>;
  relations: Record<string, number>;
  documents: InspectDocument[];
}

// ------------------------------------------------------------------------ query
export interface Source {
  path: string;
  folder: string;
  type: string | null;
}

export interface Chunk {
  id: string;
  doc_id: string;
  position: number;
  text: string;
  prev_id: string | null;
  next_id: string | null;
  embedding: number[] | null;
  source: Source | null;
  metadata: Record<string, unknown>;
  // `index` and `neighbors` are plain Python @property (not @computed_field), so pydantic's
  // model_dump() does NOT emit them. Derive client-side from position / prev_id+next_id.
  index?: number;
  neighbors?: string[];
}

export interface SearchHit {
  chunk: Chunk;
  score: number;
  neighbors: Chunk[];
  // NB: there is no top-level `source` key. SearchHit.source is a Python @property shorthand
  // for chunk.source and is NOT serialized — read provenance from `hit.chunk.source`.
}

export interface QueryRequest {
  space: string;
  text: string;
  k?: number;
  type?: string | null;
}

export interface QueryResponse {
  hits: SearchHit[];
}

// ------------------------------------------------------------------------- demo
export interface DemoResponse {
  out: string;
  summary: BuildSummary;
}

// ----------------------------------------------------------------------- browse
export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BrowseResponse {
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
}
