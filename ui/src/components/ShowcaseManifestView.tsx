import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type {
    ShowcaseManifest,
    ShowcaseModalityRender,
    ShowcasePerTrack,
    ShowcaseTrackModalityEntry,
} from '../services/api';

interface Props {
    manifestUrl: string;
}

/** Directory the manifest lives in, e.g. "/outputs/showcase/<id>". Render `path`s are relative to it. */
function manifestDir(manifestUrl: string): string {
    const i = manifestUrl.lastIndexOf('/');
    return i >= 0 ? manifestUrl.slice(0, i) : manifestUrl;
}

function joinUrl(dir: string, ...parts: string[]): string {
    return api.absoluteUrl([dir, ...parts].join('/'));
}

/**
 * Renders the contents of ``outputs/showcase/<id>/manifest.json``.
 *
 * Matches the schema `embed-art showcase` actually writes:
 * - Single-track: `concept`, `modalities` (object keyed by modality), `evaluation`.
 * - Dual-track: `concept`, `tracks`, `per_track` (object keyed by track name).
 *
 * Per-modality file paths are relative to the manifest's directory, so they are
 * resolved against `manifestDir(manifestUrl)`.
 */
export function ShowcaseManifestView({ manifestUrl }: Props) {
    const [manifest, setManifest] = useState<ShowcaseManifest | null>(null);
    const [error, setError] = useState<string | null>(null);
    const dir = manifestDir(manifestUrl);

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
                <span className="pulse-dot" style={{ flexShrink: 0 }} />
                <span className="eyebrow">Loading showcase manifest…</span>
            </div>
        );
    }

    const concept = manifest.concept?.text ?? '(untitled)';
    const modalityNames = manifest.modalities
        ? Object.keys(manifest.modalities)
        : manifest.tracks ?? [];
    const realismValue = typeof manifest.realism === 'number' ? manifest.realism : null;

    return (
        <div className="flex flex-col gap-6">
            <div className="panel">
                <span className="eyebrow">— Concept</span>
                <h2 className="display mt-1" style={{ fontSize: '2rem' }}>
                    {concept}
                </h2>
                <div className="divider" />
                <div className="flex flex-wrap gap-3 items-center text-sm">
                    <span className="field-label">Encoder</span>
                    <span className="mono dim">{manifest.encoder}</span>
                    <span className="field-label">{manifest.modalities ? 'Modalities' : 'Tracks'}</span>
                    <span className="mono dim">{modalityNames.join(', ')}</span>
                    {realismValue !== null && (
                        <span className={`readout ${realismValue < 0.5 ? 'cool' : ''}`}>
                            realism {realismValue.toFixed(2)}
                        </span>
                    )}
                    {manifest.track && !realismValue && (
                        <span className="readout">{manifest.track}</span>
                    )}
                </div>
            </div>

            {manifest.per_track && Object.keys(manifest.per_track).length > 0 ? (
                <DualTrackView perTrack={manifest.per_track} dir={dir} concept={concept} />
            ) : manifest.modalities && Object.keys(manifest.modalities).length > 0 ? (
                <SingleTrackView modalities={manifest.modalities} dir={dir} concept={concept} />
            ) : (
                <div className="panel dim text-sm">Renders will appear here once the run completes.</div>
            )}

            {manifest.evaluation && Object.keys(manifest.evaluation).length > 0 && (
                <EvaluationCard evaluation={manifest.evaluation} />
            )}
        </div>
    );
}

function SingleTrackView({
    modalities,
    dir,
    concept,
}: {
    modalities: Record<string, ShowcaseModalityRender>;
    dir: string;
    concept: string;
}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(modalities).map(([modality, render]) => (
                <ModalityCard
                    key={modality}
                    modality={modality}
                    url={render.path ? joinUrl(dir, render.path) : null}
                    similarity={render.final_similarity}
                    interpretation={render.interpretation ?? null}
                    concept={concept}
                />
            ))}
        </div>
    );
}

function DualTrackView({
    perTrack,
    dir,
    concept,
}: {
    perTrack: Record<string, ShowcasePerTrack>;
    dir: string;
    concept: string;
}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(perTrack).map(([track, entry]) => {
                const isHonest = track.toLowerCase().includes('honest');
                const isNatural = track.toLowerCase().includes('natural');
                const accent = isHonest ? 'var(--aqua)' : isNatural ? 'var(--amber)' : 'var(--text)';
                const eyebrow = isHonest
                    ? '— Machine-legible'
                    : isNatural
                      ? '— Human-legible'
                      : '— Track';
                // summary maps modality -> {path, final_similarity, backbone}, plus a _evaluation block.
                const mods = Object.entries(entry.summary).filter(
                    ([k]) => k !== '_evaluation',
                ) as [string, ShowcaseTrackModalityEntry][];
                return (
                    <div key={track} className="panel flex flex-col gap-3">
                        <div className="flex flex-col gap-1">
                            <span className="eyebrow">{eyebrow}</span>
                            <h3 className="display" style={{ fontSize: '1.6rem', color: accent }}>
                                {track.toUpperCase()}
                            </h3>
                        </div>
                        {mods.map(([modality, m]) => (
                            <div key={modality} className="flex flex-col gap-2">
                                <div className="flex items-center justify-between">
                                    <span className="field-label">{modality}</span>
                                    {typeof m.final_similarity === 'number' && (
                                        <span className={`readout ${isHonest ? 'cool' : ''}`}>
                                            sim {m.final_similarity.toFixed(3)}
                                        </span>
                                    )}
                                </div>
                                {m.path && (
                                    <RenderPreview
                                        modality={modality}
                                        url={joinUrl(dir, entry.path, m.path)}
                                        concept={concept}
                                        track={track}
                                    />
                                )}
                            </div>
                        ))}
                    </div>
                );
            })}
        </div>
    );
}

function ModalityCard({
    modality,
    url,
    similarity,
    interpretation,
    concept,
}: {
    modality: string;
    url: string | null;
    similarity?: number;
    interpretation: Record<string, unknown> | null;
    concept: string;
}) {
    return (
        <div className="panel flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                    <span className="eyebrow">— Work</span>
                    <h3 className="display" style={{ fontSize: '1.6rem' }}>
                        {modality}
                    </h3>
                </div>
                {typeof similarity === 'number' && (
                    <span className="readout cool">sim {similarity.toFixed(3)}</span>
                )}
            </div>
            {url && <RenderPreview modality={modality} url={url} concept={concept} />}
            {interpretation && <InterpretationBlock data={interpretation} />}
        </div>
    );
}

function RenderPreview({
    modality,
    url,
    concept,
    track,
}: {
    modality: string;
    url: string;
    concept: string;
    track?: string;
}) {
    const alt = `${modality} render of "${concept}"${track ? ` (${track})` : ''}`;
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
        return <img src={url} alt={alt} style={framed} />;
    }
    if (modality === 'audio') {
        return <audio controls src={url} style={{ width: '100%' }} aria-label={alt} />;
    }
    if (modality === 'video') {
        return <video controls src={url} style={{ ...framed, width: '100%' }} aria-label={alt} />;
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

function formatProbeReading(name: string, value: number | Record<string, number>): string {
    if (typeof value === 'number') {
        return `${name}=${value.toFixed(2)}`;
    }
    const entries = Object.entries(value);
    if (entries.length === 0) return `${name}=?`;
    const [topLabel, topProb] = entries.reduce(
        (best, cur) => (cur[1] > best[1] ? cur : best),
        entries[0],
    );
    return `${name}: ${topLabel} (${topProb.toFixed(2)})`;
}

function EvaluationCard({ evaluation }: { evaluation: Record<string, unknown> }) {
    const perModality = evaluation.per_modality_similarity as Record<string, number> | undefined;
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

function heatColor(v: number): string {
    // Two-branch: positive agreement → amber, negative/disagreement → aqua. Opacity ∝ |v|.
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped >= 0) {
        return `rgba(242, 168, 59, ${(clamped * 0.55).toFixed(3)})`;
    }
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
                <table className="w-full text-xs mono" style={{ borderCollapse: 'collapse' }}>
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
                                    const bg = v !== undefined ? heatColor(v) : 'transparent';
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

function ProbeTable({ table }: { table: Record<string, Record<string, number | null>> }) {
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
                                const bg = v !== undefined && v !== null ? heatColor(v) : 'transparent';
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
