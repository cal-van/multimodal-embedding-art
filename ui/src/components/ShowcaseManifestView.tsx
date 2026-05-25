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
    return (
        <div className="card flex flex-col gap-2">
            <h3 className="text-sm font-bold uppercase tracking-wide">Evaluation</h3>
            <pre className="text-xs text-dim font-mono whitespace-pre-wrap">
                {JSON.stringify(evaluation, null, 2)}
            </pre>
        </div>
    );
}
