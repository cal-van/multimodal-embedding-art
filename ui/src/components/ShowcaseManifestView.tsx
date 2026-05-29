import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type {
    ShowcaseManifest,
    ShowcaseModalityRender,
    ShowcaseTrackSummary,
} from '../services/api';

interface Props {
    manifestUrl: string;
}

/**
 * Renders the contents of ``outputs/showcase/<id>/manifest.json``.
 *
 * Supports both layouts produced by the showcase command:
 * - Single-track flat layout: ``renders`` map of modality -> render entry.
 * - Dual-track layout: ``per_track`` map of track name -> per-track summary
 *   pointing at its own subdirectory.
 *
 * Interpretation and evaluation panels render whatever fields are present in
 * the manifest, gracefully degrading when fields are missing — both pieces are
 * still under active development backend-side.
 */
export function ShowcaseManifestView({ manifestUrl }: Props) {
    const [manifest, setManifest] = useState<ShowcaseManifest | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        setManifest(null);
        setError(null);
        api.fetchShowcaseManifest(manifestUrl)
            .then((data) => {
                if (!cancelled) setManifest(data);
            })
            .catch((err) => {
                if (!cancelled) setError(err instanceof Error ? err.message : String(err));
            });
        return () => {
            cancelled = true;
        };
    }, [manifestUrl]);

    if (error) {
        return (
            <div className="panel flex flex-col gap-1">
                <span className="eyebrow" style={{ color: 'var(--err)' }}>
                    Manifest error
                </span>
                <span className="text-sm" style={{ color: 'var(--err)' }}>
                    Failed to load showcase manifest: {error}
                </span>
            </div>
        );
    }
    if (!manifest) {
        return (
            <div className="panel flex items-center gap-3">
                <span
                    style={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: 'var(--aqua)',
                        boxShadow: '0 0 8px var(--aqua)',
                        animation: 'pulse 2.4s ease-in-out infinite',
                        flexShrink: 0,
                    }}
                />
                <span className="eyebrow">Loading showcase manifest…</span>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-6">
            <div className="panel">
                <span className="eyebrow">— Concept</span>
                <h2 className="display mt-1" style={{ fontSize: '2rem' }}>
                    {manifest.target_text}
                </h2>
                <div className="divider" />
                <div className="flex flex-wrap gap-3 items-center text-sm">
                    <span className="field-label">Encoder</span>
                    <span className="mono dim">{manifest.encoder}</span>
                    <span className="field-label">Modalities</span>
                    <span className="mono dim">{manifest.modalities.join(', ')}</span>
                    {manifest.tracks && manifest.tracks.length > 0 && (
                        <>
                            <span className="field-label">Tracks</span>
                            <span className="mono dim">{manifest.tracks.join(', ')}</span>
                        </>
                    )}
                </div>
            </div>

            {manifest.per_track && Object.keys(manifest.per_track).length > 0 ? (
                <DualTrackView perTrack={manifest.per_track} />
            ) : manifest.renders ? (
                <SingleTrackView renders={manifest.renders} />
            ) : (
                <div className="panel dim text-sm">Manifest contains no renders yet.</div>
            )}

            {manifest.evaluation && Object.keys(manifest.evaluation).length > 0 && (
                <EvaluationCard evaluation={manifest.evaluation} />
            )}
        </div>
    );
}

function DualTrackView({
    perTrack,
}: {
    perTrack: Record<string, ShowcaseTrackSummary>;
}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(perTrack).map(([track, summary]) => {
                const isHonest = track.toLowerCase().includes('honest');
                const accent = isHonest ? 'var(--aqua)' : 'var(--amber)';
                return (
                    <div key={track} className="panel flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                            <div className="flex flex-col gap-1">
                                <span className="eyebrow">
                                    {isHonest ? '— Machine-legible' : '— Human-legible'}
                                </span>
                                <h3 className="display" style={{ fontSize: '1.6rem', color: accent }}>
                                    {track.toUpperCase()}
                                </h3>
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-1">
                            {summary.modalities.map((m) => (
                                <span
                                    key={m}
                                    className="mono text-xs"
                                    style={{
                                        color: 'var(--text-dim)',
                                        border: '1px solid var(--line)',
                                        borderRadius: 'var(--radius)',
                                        padding: '0.12rem 0.45rem',
                                    }}
                                >
                                    {m}
                                </span>
                            ))}
                        </div>
                        {summary.similarity_summary && (
                            <div className="flex flex-col gap-1">
                                <span className="field-label">Similarity</span>
                                <div className="flex flex-wrap gap-1">
                                    {Object.entries(summary.similarity_summary).map(([k, v]) => (
                                        <span
                                            key={k}
                                            className={`readout ${isHonest ? 'cool' : ''}`}
                                        >
                                            {k} {v.toFixed(3)}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                        {summary.output_dir && (
                            <div className="mono text-xs faint" style={{ wordBreak: 'break-all' }}>
                                {summary.output_dir}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

function SingleTrackView({
    renders,
}: {
    renders: Record<string, ShowcaseModalityRender>;
}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(renders).map(([modality, render]) => (
                <ModalityCard key={modality} modality={modality} render={render} />
            ))}
        </div>
    );
}

function ModalityCard({
    modality,
    render,
}: {
    modality: string;
    render: ShowcaseModalityRender;
}) {
    const fileUrl = render.output_file ? api.absoluteUrl(render.output_file) : null;
    return (
        <div className="panel flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                    <span className="eyebrow">— Work</span>
                    <h3 className="display" style={{ fontSize: '1.6rem' }}>
                        {modality}
                    </h3>
                </div>
                {typeof render.similarity === 'number' && (
                    <span className="readout cool">sim {render.similarity.toFixed(3)}</span>
                )}
            </div>
            {fileUrl && <RenderPreview modality={modality} url={fileUrl} />}
            {render.interpretation && <InterpretationBlock data={render.interpretation} />}
        </div>
    );
}

function RenderPreview({ modality, url }: { modality: string; url: string }) {
    const framed: React.CSSProperties = {
        maxWidth: '100%',
        maxHeight: 256,
        objectFit: 'contain',
        borderRadius: 'var(--radius)',
        border: '1px solid var(--line)',
        background: 'var(--ink-deep)',
        margin: '0 auto',
        display: 'block',
    };
    if (modality === 'image') {
        return <img src={url} alt={`${modality} render`} style={framed} />;
    }
    if (modality === 'audio') {
        return <audio controls src={url} style={{ width: '100%' }} />;
    }
    if (modality === 'video') {
        return <video controls src={url} style={{ ...framed, width: '100%' }} />;
    }
    return (
        <a href={url} target="_blank" rel="noreferrer" className="preset">
            Open {modality} output ↗
        </a>
    );
}

function InterpretationBlock({ data }: { data: Record<string, unknown> }) {
    const textAnchor = data['text_anchor'];
    const sae = data['sae_decomposition'];
    const probes = data['linear_probes'] as
        | Record<string, number | Record<string, number>>
        | undefined;
    return (
        <div className="flex flex-col gap-2 pt-2" style={{ borderTop: '1px solid var(--line)' }}>
            {Array.isArray(textAnchor) && textAnchor.length > 0 && (
                <div className="flex flex-col gap-1">
                    <span className="field-label">text-anchor</span>
                    <span className="mono text-xs dim">
                        {(textAnchor as Array<{ word: string; similarity: number }>)
                            .slice(0, 8)
                            .map((t) => `${t.word} (${t.similarity?.toFixed?.(2) ?? '?'})`)
                            .join(', ')}
                    </span>
                </div>
            )}
            {Array.isArray(sae) && sae.length > 0 && (
                <div className="flex flex-col gap-1">
                    <span className="field-label">SAE features</span>
                    <span className="mono text-xs dim">
                        {(sae as Array<{ index: number; label?: string; activation?: number }>)
                            .slice(0, 5)
                            .map((f) =>
                                f.label
                                    ? `${f.label} (${f.activation?.toFixed?.(2) ?? '?'})`
                                    : `#${f.index}`,
                            )
                            .join(', ')}
                    </span>
                </div>
            )}
            {probes && Object.keys(probes).length > 0 && (
                <div className="flex flex-col gap-1">
                    <span className="field-label">linear probes</span>
                    <span className="mono text-xs dim">
                        {Object.entries(probes)
                            .map(([name, value]) => formatProbeReading(name, value))
                            .join(' · ')}
                    </span>
                </div>
            )}
        </div>
    );
}

function formatProbeReading(
    name: string,
    value: number | Record<string, number>,
): string {
    if (typeof value === 'number') {
        return `${name}=${value.toFixed(2)}`;
    }
    // Multi-class: surface the argmax label + its probability.
    const entries = Object.entries(value);
    if (entries.length === 0) return `${name}=?`;
    const [topLabel, topProb] = entries.reduce(
        (best, cur) => (cur[1] > best[1] ? cur : best),
        entries[0],
    );
    return `${name}: ${topLabel} (${topProb.toFixed(2)})`;
}

function EvaluationCard({ evaluation }: { evaluation: Record<string, unknown> }) {
    const perModality = evaluation.per_modality_similarity as
        | Record<string, number>
        | undefined;
    const jaccard = evaluation.cross_modal_text_anchor_agreement_jaccard as
        | Record<string, Record<string, number>>
        | undefined;
    const probes = evaluation.cross_encoder_probes as
        | { table?: Record<string, Record<string, number | null>>; reports?: unknown[] }
        | undefined;
    const stability = evaluation.seed_stability as
        | {
              n_seeds?: number;
              mean_similarity?: number;
              std_similarity?: number;
              min_similarity?: number;
              max_similarity?: number;
              feature_overlap_jaccard?: number | null;
          }
        | undefined;

    return (
        <div className="panel flex flex-col gap-4">
            <span className="eyebrow">Evaluation</span>

            {perModality && Object.keys(perModality).length > 0 && (
                <div className="flex flex-col gap-2">
                    <span className="field-label">Per-modality similarity</span>
                    <div className="flex flex-wrap gap-1">
                        {Object.entries(perModality).map(([mod, sim]) => (
                            <span key={mod} className="readout cool">
                                {mod} {Number(sim).toFixed(3)}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {jaccard && Object.keys(jaccard).length > 0 && (
                <SimilarityMatrix
                    title="Cross-modal text-anchor agreement (Jaccard)"
                    matrix={jaccard}
                />
            )}

            {probes?.table && Object.keys(probes.table).length > 0 && (
                <div className="flex flex-col gap-2">
                    <span className="field-label">Cross-encoder probes</span>
                    <ProbeTable table={probes.table} />
                </div>
            )}

            {stability && stability.n_seeds !== undefined && (
                <div className="flex flex-col gap-2">
                    <span className="field-label">Seed stability ({stability.n_seeds} seeds)</span>
                    <div className="flex flex-wrap gap-1">
                        <span className="readout">mean {stability.mean_similarity?.toFixed(3)}</span>
                        <span className="readout">std {stability.std_similarity?.toFixed(3)}</span>
                        <span className="readout">
                            min/max {stability.min_similarity?.toFixed(3)} /{' '}
                            {stability.max_similarity?.toFixed(3)}
                        </span>
                        {stability.feature_overlap_jaccard !== null &&
                            stability.feature_overlap_jaccard !== undefined && (
                                <span className="readout cool">
                                    feature overlap J {stability.feature_overlap_jaccard.toFixed(3)}
                                </span>
                            )}
                    </div>
                </div>
            )}
        </div>
    );
}

function jaccardHeatColor(v: number): string {
    // amber heat — agreement intensity
    const clamped = Math.max(0, Math.min(1, v));
    return `rgba(242, 168, 59, ${(clamped * 0.55).toFixed(3)})`;
}

function cosineHeatColor(v: number): string {
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped >= 0) {
        // positive agreement → amber
        return `rgba(242, 168, 59, ${(clamped * 0.55).toFixed(3)})`;
    }
    // disagreement → aqua
    return `rgba(111, 227, 224, ${(-clamped * 0.4).toFixed(3)})`;
}

function SimilarityMatrix({
    title,
    matrix,
}: {
    title: string;
    matrix: Record<string, Record<string, number>>;
}) {
    const rowKeys = Object.keys(matrix).sort();
    const colKeys = Array.from(
        new Set(rowKeys.flatMap((r) => Object.keys(matrix[r] ?? {}))),
    ).sort();
    return (
        <div className="flex flex-col gap-2 mt-2">
            <span className="field-label">{title}</span>
            <div
                style={{
                    overflowX: 'auto',
                    border: '1px solid var(--line)',
                    borderRadius: 'var(--radius-lg)',
                    background: 'var(--ink-deep)',
                    padding: '0.75rem',
                }}
            >
                <table
                    className="w-full text-xs mono"
                    style={{ borderCollapse: 'collapse' }}
                >
                    <thead>
                        <tr>
                            <th className="faint" style={{ padding: '0.5rem', textAlign: 'left', fontWeight: 400 }}></th>
                            {colKeys.map((c) => (
                                <th
                                    key={c}
                                    className="faint"
                                    style={{ padding: '0.5rem', textAlign: 'center', fontWeight: 400 }}
                                >
                                    {c}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rowKeys.map((r) => (
                            <tr key={r}>
                                <th
                                    className="faint"
                                    style={{
                                        padding: '0.5rem',
                                        textAlign: 'left',
                                        fontWeight: 400,
                                        borderTop: '1px solid var(--line)',
                                    }}
                                >
                                    {r}
                                </th>
                                {colKeys.map((c) => {
                                    const v = matrix[r]?.[c];
                                    const bg = v !== undefined ? jaccardHeatColor(v) : 'transparent';
                                    return (
                                        <td
                                            key={c}
                                            style={{
                                                padding: '0.5rem',
                                                textAlign: 'center',
                                                border: '1px solid var(--line)',
                                                backgroundColor: bg,
                                                color: 'var(--text)',
                                            }}
                                            title={`${r} ↔ ${c}: ${v !== undefined ? v.toFixed(3) : 'N/A'}`}
                                        >
                                            {v === undefined ? '—' : v.toFixed(3)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function ProbeTable({
    table,
}: {
    table: Record<string, Record<string, number | null>>;
}) {
    const modalities = Object.keys(table).sort();
    const probeNames = Array.from(
        new Set(modalities.flatMap((m) => Object.keys(table[m] ?? {}))),
    ).sort();
    return (
        <div
            style={{
                overflowX: 'auto',
                border: '1px solid var(--line)',
                borderRadius: 'var(--radius-lg)',
                background: 'var(--ink-deep)',
                padding: '0.75rem',
                marginTop: '0.5rem',
            }}
        >
            <table className="w-full text-xs mono" style={{ borderCollapse: 'collapse' }}>
                <thead>
                    <tr>
                        <th className="faint" style={{ padding: '0.5rem', textAlign: 'left', fontWeight: 400 }}></th>
                        {probeNames.map((p) => (
                            <th
                                key={p}
                                className="faint"
                                style={{ padding: '0.5rem', textAlign: 'center', fontWeight: 400 }}
                            >
                                {p}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {modalities.map((m) => (
                        <tr key={m}>
                            <th
                                className="faint"
                                style={{
                                    padding: '0.5rem',
                                    textAlign: 'left',
                                    fontWeight: 400,
                                    borderTop: '1px solid var(--line)',
                                }}
                            >
                                {m}
                            </th>
                            {probeNames.map((p) => {
                                const v = table[m]?.[p];
                                const bg = v !== undefined && v !== null ? cosineHeatColor(v) : 'transparent';
                                return (
                                    <td
                                        key={p}
                                        style={{
                                            padding: '0.5rem',
                                            textAlign: 'center',
                                            border: '1px solid var(--line)',
                                            backgroundColor: bg,
                                            color: 'var(--text)',
                                        }}
                                        title={`${m} ↔ ${p}: ${v !== undefined && v !== null ? v.toFixed(3) : 'N/A'}`}
                                    >
                                        {v === undefined || v === null ? '—' : v.toFixed(3)}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
