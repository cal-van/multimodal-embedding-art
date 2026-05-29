import { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../services/api';
import type { Job } from '../services/api';
import { ShowcaseManifestView } from '../components/ShowcaseManifestView';

interface ShowcaseEvent {
    type: string;
    track?: string;
    modality?: string;
    similarity?: number;
    text_anchor?: Array<{ word: string; similarity?: number }> | null;
    [key: string]: unknown;
}

export function JobPage() {
    const { id } = useParams<{ id: string }>();
    const [job, setJob] = useState<Job | null>(null);
    const [events, setEvents] = useState<ShowcaseEvent[]>([]);
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const logsRef = useRef<HTMLDivElement>(null);

    // Initial fetch
    useEffect(() => {
        if (!id) return;

        setError(null);
        api.getJob(id)
            .then(setJob)
            .catch(err => {
                console.error(err);
                setError("Job not found. The server may have restarted.");
            });
    }, [id]);

    // WebSocket connection
    useEffect(() => {
        if (!id || error) return;

        let reconnectTimer: number;

        const connect = () => {
            // If already connected/connecting, skip
            if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return;

            // Clean up existing closed/closing connection if any
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
            }

            const wsUrl = `ws://127.0.0.1:8000/jobs/${id}/ws`;
            console.log("Connecting WS:", wsUrl);
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WS Connected');
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    setJob((prev) => {
                        // If we don't have job state yet, wait for init or fetch
                        // But init usually comes first or close to it.
                        const current = prev || {} as Job;

                        if (msg.type === 'init') {
                            return {
                                ...current,
                                id: id, // Ensure ID is present
                                status: msg.status,
                                progress: msg.progress,
                                logs: msg.logs,
                                error: msg.error,
                                result_url: msg.result_url || current.result_url,
                                target_text: current.target_text || [],
                                output_modality: current.output_modality || 'image'
                            } as Job;
                        }

                        if (!prev) return null; // Ignore other updates if not initialized

                        if (msg.type === 'log') {
                            return { ...prev, logs: [...prev.logs, msg.message] };
                        }
                        if (msg.type === 'result') {
                            const next = { ...prev, result_url: msg.url };
                            if (msg.kind === 'showcase') {
                                next.manifest_url = msg.url;
                                next.kind = 'showcase';
                            }
                            return next;
                        }
                        if (msg.type === 'progress') {
                            return {
                                ...prev,
                                progress: msg.progress,
                                status: msg.status || prev.status,
                                result_url: msg.result_url || prev.result_url
                            };
                        }
                        if (msg.type === 'status') {
                            return { ...prev, status: msg.status };
                        }
                        if (msg.type === 'error') {
                            return { ...prev, error: msg.message, status: 'failed' };
                        }
                        if (msg.type === 'showcase_event') {
                            // Stored in a sibling state slot, not on the job
                            // itself, so it doesn't shadow per-modality results.
                            setEvents((prev) => [...prev, msg.event as ShowcaseEvent]);
                            return prev;
                        }
                        return prev;
                    });
                } catch (e) {
                    console.error("Failed to parse WS message", e);
                }
            };

            ws.onerror = (e) => {
                // Don't log too aggressively as some errors are expected during dev/reload
                console.error("WebSocket error observed", e);
            };

            ws.onclose = () => {
                console.log("WS Closed");
                reconnectTimer = window.setTimeout(() => {
                    console.log("Attempting reconnect...");
                    connect();
                }, 3000);
            };
        };

        connect();

        return () => {
            console.log("Cleaning up WS");
            clearTimeout(reconnectTimer);
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [id, error]);

    // Auto-scroll logs
    useEffect(() => {
        if (logsRef.current) {
            logsRef.current.scrollTop = logsRef.current.scrollHeight;
        }
    }, [job?.logs]);

    if (error) {
        return (
            <div className="max-w-2xl mx-auto">
                <div className="panel flex flex-col gap-4 rise rise-1" style={{ textAlign: 'center' }}>
                    <span className="eyebrow" style={{ color: 'var(--err)' }}>
                        Signal lost
                    </span>
                    <h1 className="display" style={{ fontSize: '2.4rem' }}>
                        Job not <em style={{ fontStyle: 'italic', color: 'var(--accent)' }}>found</em>
                    </h1>
                    <p className="dim text-sm">
                        {error}
                        <br />
                        <span className="faint text-xs mt-1" style={{ display: 'block' }}>
                            Create a new job to continue.
                        </span>
                    </p>
                    <div className="flex justify-center mt-1">
                        <a href="/" className="preset">
                            ← Go home
                        </a>
                    </div>
                </div>
            </div>
        );
    }

    if (!job) {
        return (
            <div className="flex items-center gap-3" style={{ padding: '2rem 0' }}>
                <span className="pulse-dot" />
                <span className="eyebrow">Loading run…</span>
            </div>
        );
    }

    const isShowcase = job.kind === 'showcase';
    const progress = job.progress || 0;
    const shortId = (job.id || id || '').slice(0, 8);

    const logsPanel = (
        <div
            className="panel"
            ref={logsRef}
            style={{
                height: '24rem',
                overflow: 'auto',
                fontFamily: 'var(--font-mono)',
                padding: '0',
            }}
        >
            <div
                className="eyebrow"
                style={{
                    position: 'sticky',
                    top: 0,
                    background: 'var(--panel)',
                    padding: '1rem 1.25rem 0.75rem',
                    borderBottom: '1px solid var(--line)',
                    zIndex: 1,
                }}
            >
                Logs
            </div>
            <div className="flex flex-col gap-1 text-xs" style={{ padding: '0.75rem 1.25rem 1rem' }}>
                {(job.logs || []).map((log, i) => {
                    const isErr = /error|failed/i.test(log);
                    const isOk = /completed|sim=/i.test(log);
                    const color = isErr
                        ? 'var(--err)'
                        : isOk
                          ? 'var(--aqua)'
                          : 'var(--text-dim)';
                    return (
                        <div key={i} style={{ color, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {log}
                        </div>
                    );
                })}
            </div>
        </div>
    );

    return (
        <div className="flex flex-col gap-6">
            <header className="page-head rise rise-1" style={{ marginBottom: '0.4rem' }}>
                <div className="flex justify-between items-center">
                    <div className="flex flex-col gap-1">
                        <span className="eyebrow">
                            {isShowcase ? '— Showcase run' : '— Single-modality run'} ·{' '}
                            <span className="mono">{shortId}</span>
                        </span>
                        <h1 className="display" style={{ fontSize: '3rem' }}>
                            {isShowcase ? (
                                <>
                                    Show<em style={{ fontStyle: 'italic', color: 'var(--accent)' }}>case</em>
                                </>
                            ) : (
                                <>
                                    Ren<em style={{ fontStyle: 'italic', color: 'var(--accent)' }}>der</em>
                                </>
                            )}
                        </h1>
                    </div>
                    <div className={`badge ${job.status}`}>{job.status?.toUpperCase() || 'UNKNOWN'}</div>
                </div>
            </header>

            <div className="panel rise rise-2">
                <div className="flex justify-between items-center mb-2">
                    <span className="field-label">Progress</span>
                    <span className="readout">{Math.round(progress * 100)}%</span>
                </div>
                <div
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round((job.progress || 0) * 100)}
                    style={{
                        width: '100%',
                        height: 3,
                        borderRadius: 2,
                        background: 'var(--line)',
                        overflow: 'hidden',
                    }}
                >
                    <div
                        style={{
                            height: '100%',
                            width: `${progress * 100}%`,
                            background: 'var(--accent)',
                            borderRadius: 2,
                            transition: 'width 0.3s cubic-bezier(0.22, 1, 0.36, 1)',
                        }}
                    />
                </div>
            </div>

            {isShowcase ? (
                <div
                    className="grid gap-6 rise rise-3"
                    style={{ gridTemplateColumns: 'minmax(0, 1fr) 320px' }}
                >
                    <div className="flex flex-col gap-4">
                        <ShowcaseEventStream events={events} />
                        {job.manifest_url ? (
                            <ShowcaseManifestView manifestUrl={job.manifest_url} />
                        ) : (
                            <div className="panel flex items-center gap-3">
                                <span className="pulse-dot" style={{ flexShrink: 0 }} />
                                <span className="dim text-sm">
                                    Showcase running — the manifest will appear once renders are
                                    complete.
                                </span>
                            </div>
                        )}
                    </div>
                    {logsPanel}
                </div>
            ) : (
                <div className="grid gap-6 rise rise-3" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
                    {logsPanel}

                    <div
                        className="panel flex items-center justify-center"
                        style={{ minHeight: 320, overflow: 'hidden' }}
                    >
                        {job.result_url ? (
                            <div
                                className="flex flex-col items-center justify-center w-full"
                                style={{ position: 'relative' }}
                            >
                                {job.output_modality === 'audio' || job.result_url.endsWith('.wav') ? (
                                    <div className="flex flex-col gap-3 w-full">
                                        <span className="field-label">Audio · work</span>
                                        <audio
                                            src={`http://127.0.0.1:8000${job.result_url}`}
                                            controls
                                            style={{ width: '100%' }}
                                        />
                                    </div>
                                ) : job.output_modality === 'video' ||
                                  job.result_url.endsWith('.mp4') ||
                                  job.result_url.endsWith('.gif') ? (
                                    <video
                                        src={`http://127.0.0.1:8000${job.result_url}`}
                                        controls
                                        autoPlay
                                        loop
                                        style={{
                                            maxWidth: '100%',
                                            maxHeight: 420,
                                            objectFit: 'contain',
                                            borderRadius: 'var(--radius)',
                                            border: '1px solid var(--line)',
                                        }}
                                    />
                                ) : (
                                    <img
                                        src={`http://127.0.0.1:8000${job.result_url}`}
                                        alt={(job.target_text || []).join(', ') || 'Optimization result'}
                                        style={{
                                            maxWidth: '100%',
                                            maxHeight: 420,
                                            objectFit: 'contain',
                                            borderRadius: 'var(--radius)',
                                            border: '1px solid var(--line)',
                                        }}
                                    />
                                )}
                                <a
                                    href={`http://127.0.0.1:8000${job.result_url}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="preset"
                                    style={{ position: 'absolute', bottom: 0, right: 0 }}
                                >
                                    Open full ↗
                                </a>
                            </div>
                        ) : (
                            <div className="flex items-center gap-3">
                                <span className="pulse-dot" />
                                <span className="eyebrow">Awaiting render…</span>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}


function ShowcaseEventStream({ events }: { events: ShowcaseEvent[] }) {
    if (events.length === 0) {
        return (
            <div className="panel flex items-center gap-3">
                <span className="pulse-dot" style={{ flexShrink: 0 }} />
                <span className="dim text-sm">Waiting for showcase events…</span>
            </div>
        );
    }

    // Build a compact view: latest modality activity, plus the most recent
    // text-anchor readout from any modality_complete event.
    const last = events[events.length - 1];
    const lastComplete = [...events].reverse().find((e) => e.type === 'modality_complete');

    return (
        <div className="panel flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <span className="eyebrow">Live stream</span>
                <span className="readout cool">
                    {last.type}
                    {last.modality ? ` · ${last.modality}` : ''}
                    {last.track ? ` · ${last.track}` : ''}
                </span>
            </div>
            {lastComplete && (
                <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="field-label">Last complete</span>
                        <span className="readout cool">
                            {lastComplete.modality} · sim=
                            {typeof lastComplete.similarity === 'number'
                                ? lastComplete.similarity.toFixed(3)
                                : '?'}
                        </span>
                    </div>
                    {Array.isArray(lastComplete.text_anchor) && lastComplete.text_anchor.length > 0 && (
                        <div className="flex flex-col gap-1">
                            <span className="field-label">Text-anchor</span>
                            <div className="flex flex-wrap gap-1">
                                {lastComplete.text_anchor.slice(0, 8).map((t, i) => (
                                    <span
                                        key={`${t.word}-${i}`}
                                        className="mono text-xs"
                                        style={{
                                            color: 'var(--text-dim)',
                                            border: '1px solid var(--line)',
                                            borderRadius: 'var(--radius)',
                                            padding: '0.12rem 0.45rem',
                                            letterSpacing: '0.04em',
                                        }}
                                    >
                                        {t.word}
                                        {typeof t.similarity === 'number' && (
                                            <span className="faint"> {t.similarity.toFixed(2)}</span>
                                        )}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
