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
            <div className="card text-sm text-red-400">
                Failed to load showcase manifest: {error}
            </div>
        );
    }
    if (!manifest) {
        return (
            <div className="card text-sm text-dim animate-pulse">Loading showcase manifest…</div>
        );
    }

    return (
        <div className="flex flex-col gap-6">
            <div className="card">
                <h2 className="text-lg font-bold">{manifest.target_text}</h2>
                <div className="text-sm text-dim mt-1">
                    Encoder: <span className="font-mono">{manifest.encoder}</span> · Modalities:{' '}
                    <span className="font-mono">{manifest.modalities.join(', ')}</span>
                    {manifest.tracks && manifest.tracks.length > 0 && (
                        <>
                            {' '}
                            · Tracks: <span className="font-mono">{manifest.tracks.join(', ')}</span>
                        </>
                    )}
                </div>
            </div>

            {manifest.per_track && Object.keys(manifest.per_track).length > 0 ? (
                <DualTrackView perTrack={manifest.per_track} />
            ) : manifest.renders ? (
                <SingleTrackView renders={manifest.renders} />
            ) : (
                <div className="card text-sm text-dim">Manifest contains no renders yet.</div>
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
            {Object.entries(perTrack).map(([track, summary]) => (
                <div key={track} className="card flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold uppercase tracking-wide">{track}</h3>
                        <span className="badge text-xs">{summary.modalities.join(' · ')}</span>
                    </div>
                    {summary.similarity_summary && (
                        <div className="text-xs text-dim font-mono">
                            {Object.entries(summary.similarity_summary).map(([k, v]) => (
                                <div key={k}>
                                    {k}: {v.toFixed(3)}
                                </div>
                            ))}
                        </div>
                    )}
                    {summary.output_dir && (
                        <div className="text-xs text-dim">
                            <span className="font-mono">{summary.output_dir}</span>
                        </div>
                    )}
                </div>
            ))}
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
        <div className="card flex flex-col gap-2">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold uppercase tracking-wide">{modality}</h3>
                {typeof render.similarity === 'number' && (
                    <span className="text-xs text-dim font-mono">
                        sim={render.similarity.toFixed(3)}
                    </span>
                )}
            </div>
            {fileUrl && (
                <RenderPreview modality={modality} url={fileUrl} />
            )}
            {render.interpretation && (
                <InterpretationBlock data={render.interpretation} />
            )}
        </div>
    );
}

function RenderPreview({ modality, url }: { modality: string; url: string }) {
    if (modality === 'image') {
        return (
            <img
                src={url}
                alt={`${modality} render`}
                className="rounded-md max-h-64 object-contain bg-black/30"
            />
        );
    }
    if (modality === 'audio') {
        return <audio controls src={url} className="w-full" />;
    }
    if (modality === 'video') {
        return (
            <video
                controls
                src={url}
                className="rounded-md max-h-64 object-contain bg-black/30 w-full"
            />
        );
    }
    return (
        <a href={url} target="_blank" rel="noreferrer" className="text-xs underline text-dim">
            Open {modality} output
        </a>
    );
}

function InterpretationBlock({ data }: { data: Record<string, unknown> }) {
    const textAnchor = data['text_anchor'];
    const sae = data['sae_decomposition'];
    return (
        <div className="text-xs text-dim flex flex-col gap-1">
            {Array.isArray(textAnchor) && textAnchor.length > 0 && (
                <div>
                    <span className="text-[#a3a3a3]">text-anchor:</span>{' '}
                    {(textAnchor as Array<{ word: string; similarity: number }>)
                        .slice(0, 8)
                        .map((t) => `${t.word} (${t.similarity?.toFixed?.(2) ?? '?'})`)
                        .join(', ')}
                </div>
            )}
            {Array.isArray(sae) && sae.length > 0 && (
                <div>
                    <span className="text-[#a3a3a3]">SAE features:</span>{' '}
                    {(sae as Array<{ index: number; label?: string; activation?: number }>)
                        .slice(0, 5)
                        .map((f) =>
                            f.label
                                ? `${f.label} (${f.activation?.toFixed?.(2) ?? '?'})`
                                : `#${f.index}`,
                        )
                        .join(', ')}
                </div>
            )}
        </div>
    );
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
        <div className="card flex flex-col gap-4">
            <h3 className="text-sm font-bold uppercase tracking-wide">Evaluation</h3>

            {perModality && Object.keys(perModality).length > 0 && (
                <div>
                    <div className="text-xs uppercase tracking-wide text-dim mb-1">
                        Per-modality similarity
                    </div>
                    <table className="text-xs font-mono">
                        <tbody>
                            {Object.entries(perModality).map(([mod, sim]) => (
                                <tr key={mod}>
                                    <td className="pr-3">{mod}</td>
                                    <td>{Number(sim).toFixed(3)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {jaccard && Object.keys(jaccard).length > 0 && (
                <SimilarityMatrix
                    title="Cross-modal text-anchor agreement (Jaccard)"
                    matrix={jaccard}
                />
            )}

            {probes?.table && Object.keys(probes.table).length > 0 && (
                <div>
                    <div className="text-xs uppercase tracking-wide text-dim mb-1">
                        Cross-encoder probes
                    </div>
                    <ProbeTable table={probes.table} />
                </div>
            )}

            {stability && stability.n_seeds !== undefined && (
                <div>
                    <div className="text-xs uppercase tracking-wide text-dim mb-1">
                        Seed stability ({stability.n_seeds} seeds)
                    </div>
                    <table className="text-xs font-mono">
                        <tbody>
                            <tr>
                                <td className="pr-3">mean</td>
                                <td>{stability.mean_similarity?.toFixed(3)}</td>
                            </tr>
                            <tr>
                                <td className="pr-3">std</td>
                                <td>{stability.std_similarity?.toFixed(3)}</td>
                            </tr>
                            <tr>
                                <td className="pr-3">min / max</td>
                                <td>
                                    {stability.min_similarity?.toFixed(3)} /{' '}
                                    {stability.max_similarity?.toFixed(3)}
                                </td>
                            </tr>
                            {stability.feature_overlap_jaccard !== null &&
                                stability.feature_overlap_jaccard !== undefined && (
                                    <tr>
                                        <td className="pr-3">feature overlap (J)</td>
                                        <td>
                                            {stability.feature_overlap_jaccard.toFixed(3)}
                                        </td>
                                    </tr>
                                )}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
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
        <div>
            <div className="text-xs uppercase tracking-wide text-dim mb-1">{title}</div>
            <table className="text-xs font-mono">
                <thead>
                    <tr>
                        <th></th>
                        {colKeys.map((c) => (
                            <th key={c} className="px-2 text-left">
                                {c}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rowKeys.map((r) => (
                        <tr key={r}>
                            <td className="pr-2 font-bold">{r}</td>
                            {colKeys.map((c) => {
                                const v = matrix[r]?.[c];
                                return (
                                    <td key={c} className="px-2">
                                        {v === undefined ? '—' : v.toFixed(3)}
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
        <table className="text-xs font-mono">
            <thead>
                <tr>
                    <th></th>
                    {probeNames.map((p) => (
                        <th key={p} className="px-2 text-left">
                            {p}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {modalities.map((m) => (
                    <tr key={m}>
                        <td className="pr-2 font-bold">{m}</td>
                        {probeNames.map((p) => {
                            const v = table[m]?.[p];
                            return (
                                <td key={p} className="px-2">
                                    {v === undefined || v === null ? '—' : v.toFixed(3)}
                                </td>
                            );
                        })}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
