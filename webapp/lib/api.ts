// Typed client for the indx app HTTP API (docs/app-spec.md §3). Same-origin in production
// (FastAPI serves the SPA + /api); proxied to :8000 in dev via next.config rewrites.

import type {
  BrowseResponse,
  BuildRequest,
  ComponentsResponse,
  ConfigGetResponse,
  ConfigPutRequest,
  ConfigValidateResponse,
  DemoResponse,
  DryRunResponse,
  HealthResponse,
  InspectResponse,
  QueryRequest,
  QueryResponse,
} from './types';

const BASE = '/api';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') {
        detail = body.detail;
      }
    } catch {
      // non-JSON error body; keep the status line
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (): Promise<HealthResponse> => request('/health'),

  components: (): Promise<ComponentsResponse> => request('/components'),

  getConfig: (path?: string): Promise<ConfigGetResponse> =>
    request(`/config${path ? `?path=${encodeURIComponent(path)}` : ''}`),

  validateConfig: (config: Record<string, unknown>): Promise<ConfigValidateResponse> =>
    request('/config/validate', { method: 'POST', body: JSON.stringify(config) }),

  putConfig: (payload: ConfigPutRequest): Promise<ConfigGetResponse> =>
    request('/config', { method: 'PUT', body: JSON.stringify(payload) }),

  dryRun: (body: BuildRequest): Promise<DryRunResponse> =>
    request('/dry-run', { method: 'POST', body: JSON.stringify(body) }),

  inspect: (space: string): Promise<InspectResponse> =>
    request(`/inspect?space=${encodeURIComponent(space)}`),

  query: (body: QueryRequest): Promise<QueryResponse> =>
    request('/query', { method: 'POST', body: JSON.stringify(body) }),

  demo: (): Promise<DemoResponse> => request('/demo', { method: 'POST', body: '{}' }),

  browse: (path?: string): Promise<BrowseResponse> =>
    request(`/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`),
};

// --------------------------------------------------------------------------- SSE

/** A parsed Server-Sent Event frame from POST /api/build. */
export interface SseEvent {
  event: string;
  data: unknown;
}

export interface BuildStreamHandlers {
  onEvent: (ev: SseEvent) => void;
  signal?: AbortSignal;
}

/**
 * POST /api/build and parse the text/event-stream response into discrete
 * `event:` / `data:` frames. We use fetch + a ReadableStream reader because
 * EventSource cannot issue POST requests with a JSON body.
 *
 * Terminal events: `done`, `plan` (dry-run), and `error`.
 */
export async function streamBuild(
  body: BuildRequest,
  handlers: BuildStreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/build`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal: handlers.signal,
  });

  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = await res.json();
      if (errBody && typeof errBody.detail === 'string') {
        detail = errBody.detail;
      }
    } catch {
      // ignore
    }
    handlers.onEvent({ event: 'error', data: { message: `error: ${detail}` } });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const flushFrame = (raw: string) => {
    const lines = raw.split('\n');
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      // lines starting with ':' are comments/heartbeats — ignored
    }
    if (dataLines.length === 0 && eventName === 'message') {
      return;
    }
    const dataStr = dataLines.join('\n');
    let data: unknown = dataStr;
    if (dataStr) {
      try {
        data = JSON.parse(dataStr);
      } catch {
        data = dataStr;
      }
    }
    handlers.onEvent({ event: eventName, data });
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line. Normalize CRLF first.
    buffer = buffer.replace(/\r\n/g, '\n');
    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (frame.trim()) flushFrame(frame);
    }
  }
  // flush any trailing frame without a final blank line
  if (buffer.trim()) flushFrame(buffer.replace(/\r\n/g, '\n'));
}

export { ApiError };
