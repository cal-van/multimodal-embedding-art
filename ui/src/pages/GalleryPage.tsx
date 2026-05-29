import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import type { Job } from '../services/api';

const API_BASE = 'http://127.0.0.1:8000';

const MODALITY_GLYPH: Record<string, string> = {
    image: '◳',
    audio: '◌',
    video: '▷',
    text: '¶',
};

// Cycle the stagger classes across the first few cards for a sequenced load.
const RISE_CYCLE = ['rise-1', 'rise-2', 'rise-3', 'rise-4', 'rise-5'];

function isImage(modality?: string): boolean {
    return (modality ?? 'image') === 'image';
}

export function GalleryPage() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.listJobs()
            .then(setJobs)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return <div className="eyebrow">Loading catalogue…</div>;
    }

    return (
        <div className="flex flex-col gap-6">
            <header className="page-head rise rise-1">
                <span className="eyebrow">04 — Catalogue</span>
                <h1 className="display">
                    Gall<em>ery</em>
                </h1>
                <p className="lede">Every concept pulled out of the embedding space.</p>
            </header>

            {jobs.length === 0 ? (
                <div className="panel rise rise-2">
                    <p className="display" style={{ fontSize: '1.5rem', marginBottom: '0.6rem' }}>
                        The wall is empty.
                    </p>
                    <p className="dim text-sm">
                        Nothing has been rendered yet.{' '}
                        <Link to="/showcase" style={{ color: 'var(--accent)' }}>
                            Run a showcase →
                        </Link>
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {jobs.map((job, i) => {
                        const title = (job.target_text || []).join(', ') || 'Untitled';
                        const modality = job.output_modality ?? 'image';
                        const glyph = MODALITY_GLYPH[modality] ?? '◳';
                        const showImage = job.result_url && isImage(job.output_modality);
                        const riseClass = i < 6 ? `rise ${RISE_CYCLE[i % RISE_CYCLE.length]}` : '';

                        return (
                            <Link
                                key={job.id}
                                to={`/jobs/${job.id}`}
                                className={`panel ${riseClass}`}
                                style={{ padding: '1rem', display: 'block' }}
                            >
                                {/* square thumbnail */}
                                <div
                                    className="relative rounded-lg"
                                    style={{
                                        aspectRatio: '1',
                                        overflow: 'hidden',
                                        background: 'var(--ink-deep)',
                                        border: '1px solid var(--line)',
                                        marginBottom: '0.9rem',
                                    }}
                                >
                                    {showImage ? (
                                        <img
                                            src={`${API_BASE}${job.result_url}`}
                                            alt={title}
                                            style={{
                                                width: '100%',
                                                height: '100%',
                                                objectFit: 'cover',
                                            }}
                                        />
                                    ) : (
                                        <div
                                            className="flex flex-col items-center justify-center h-full w-full gap-2"
                                            style={{ textAlign: 'center' }}
                                        >
                                            <span
                                                className="mono"
                                                style={{
                                                    fontSize: '2.6rem',
                                                    lineHeight: 1,
                                                    color: 'var(--text-faint)',
                                                }}
                                            >
                                                {glyph}
                                            </span>
                                            <span className="eyebrow">{job.status}</span>
                                        </div>
                                    )}

                                    {/* modality tag — corner chip */}
                                    <span
                                        className="mono absolute"
                                        style={{
                                            top: '0.5rem',
                                            right: '0.5rem',
                                            fontSize: '0.6rem',
                                            letterSpacing: '0.14em',
                                            textTransform: 'uppercase',
                                            color: 'var(--text-dim)',
                                            background: 'var(--panel-2)',
                                            border: '1px solid var(--line-bright)',
                                            borderRadius: 'var(--radius)',
                                            padding: '0.18rem 0.45rem',
                                        }}
                                    >
                                        {modality}
                                    </span>
                                </div>

                                {/* wall label */}
                                <h3
                                    className="display"
                                    title={title}
                                    style={{
                                        fontSize: '1.35rem',
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                    }}
                                >
                                    {title}
                                </h3>
                                <div className="flex justify-between items-center mt-2">
                                    <span className={`badge ${job.status}`}>{job.status}</span>
                                    <span className="mono text-xs faint">{job.id.slice(0, 8)}</span>
                                </div>
                            </Link>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
