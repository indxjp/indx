'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { DocumentDetail, InspectDocument } from '@/lib/types';
import { Banner, Drawer, Skeleton } from '@/components/ui';

/** Type guard for the `{ error }` sentinel returned by api.agent.document. */
function isError(
  v: DocumentDetail | { error: string },
): v is { error: string } {
  return 'error' in v;
}

/** A small label/value metadata row. */
function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-3">
      <dt className="w-24 shrink-0 text-xs uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="min-w-0 break-words text-sm text-slate-700">{children}</dd>
    </div>
  );
}

/** A row of pill-style chips for topics/tags. */
function Chips({ items, empty }: { items: string[]; empty: string }) {
  if (items.length === 0) {
    return <span className="text-sm text-slate-400">{empty}</span>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((it) => (
        <span
          key={it}
          className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600"
        >
          {it}
        </span>
      ))}
    </div>
  );
}

/**
 * Right-side slide-over showing a selected document's metadata plus its
 * richer summary + full text (fetched lazily via api.agent.document).
 *
 * Props:
 *  - space:   the open space path (null disables the detail fetch)
 *  - doc:     the selected InspectDocument, or null to keep the drawer closed
 *  - onClose: invoked when the drawer requests dismissal
 */
export function DocumentDrawer({
  space,
  doc,
  onClose,
}: {
  space: string | null;
  doc: InspectDocument | null;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (spacePath: string, id: string) => {
      setLoading(true);
      setError(null);
      setDetail(null);
      try {
        const res = await api.agent.document(spacePath, id);
        if (isError(res)) {
          setError(res.error);
        } else {
          setDetail(res);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!doc) {
      // Reset when the drawer is dismissed so a re-open starts clean.
      setDetail(null);
      setError(null);
      setLoading(false);
      return;
    }
    if (!space) {
      setDetail(null);
      setError(null);
      setLoading(false);
      return;
    }
    void load(space, doc.id);
  }, [space, doc, load]);

  const basename = doc ? doc.path.split('/').pop() || doc.path : '';

  return (
    <Drawer open={doc !== null} title={basename} onClose={onClose}>
      {doc ? (
        <div className="space-y-6">
          {/* Metadata pulled straight from the InspectDocument (always available). */}
          <dl className="space-y-3">
            <MetaRow label="Type">
              <span className="rounded-md bg-accent/10 px-2 py-0.5 font-medium text-accent">
                {doc.type ?? 'unknown'}
              </span>
            </MetaRow>
            <MetaRow label="Path">
              <span className="font-mono text-xs text-slate-600">{doc.path}</span>
            </MetaRow>
            <MetaRow label="Folder">
              <span className="font-mono text-xs text-slate-600">
                {doc.folder || '/'}
              </span>
            </MetaRow>
            <MetaRow label="Chunks">
              <span className="tabular-nums">
                {detail?.chunk_count ?? doc.chunks}
              </span>
            </MetaRow>
            <MetaRow label="Topics">
              <Chips items={doc.topics} empty="No topics" />
            </MetaRow>
            <MetaRow label="Tags">
              <Chips items={doc.tags} empty="No tags" />
            </MetaRow>
          </dl>

          {/* Richer detail: summary + full text, fetched via the agent endpoint. */}
          <section className="space-y-3 border-t border-slate-200 pt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Summary
            </h4>

            {!space ? (
              <Banner kind="info">
                Open a space to load this document&apos;s summary and text.
              </Banner>
            ) : loading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
              </div>
            ) : error ? (
              <Banner kind="error">{error}</Banner>
            ) : detail ? (
              <>
                {detail.summary ? (
                  <p className="text-sm leading-relaxed text-slate-700">
                    {detail.summary}
                  </p>
                ) : (
                  <p className="text-sm text-slate-400">No summary available.</p>
                )}

                <h4 className="pt-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Document text
                </h4>
                {detail.text ? (
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-700">
                    {detail.text}
                  </pre>
                ) : (
                  <p className="text-sm text-slate-400">No text available.</p>
                )}
              </>
            ) : (
              <p className="text-sm text-slate-400">No detail available.</p>
            )}
          </section>
        </div>
      ) : null}
    </Drawer>
  );
}
