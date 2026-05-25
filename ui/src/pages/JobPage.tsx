import { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../services/api';
import type { Job } from '../services/api';
import { ShowcaseManifestView } from '../components/ShowcaseManifestView';

export function JobPage() {
    const { id } = useParams<{ id: string }>();
    const [job, setJob] = useState<Job | null>(null);
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
            <div className="flex flex-col items-center justify-center h-64 gap-4">
                <div className="text-red-500 font-bold text-xl">Job Not Found</div>
                <div className="text-dim text-center">
                    {error} <br />
                    <span className="text-xs mt-2 block">Create a new job to continue.</span>
                </div>
                <div className="flex gap-4">
                    <a href="/" className="px-4 py-2 bg-[#27272a] hover:bg-[#3f3f46] rounded text-white transition-colors">
                        Go Home
                    </a>
                </div>
            </div>
        );
    }

    if (!job) return <div>Loading...</div>;

    const isShowcase = job.kind === 'showcase';
    const heading = isShowcase ? 'Showcase Job' : 'Optimization Job';

    return (
        <div className="flex flex-col gap-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold">{heading}</h1>
                <div className={`badge ${job.status}`}>{job.status?.toUpperCase() || 'UNKNOWN'}</div>
            </div>

            <div className="card">
                <div className="flex justify-between text-sm mb-2 text-dim">
                    <span>Progress</span>
                    <span>{Math.round((job.progress || 0) * 100)}%</span>
                </div>
                <div className="w-full bg-[#27272a] h-2 rounded-full overflow-hidden">
                    <div
                        className="bg-[#8b5cf6] h-full transition-all duration-300"
                        style={{ width: `${(job.progress || 0) * 100}%` }}
                    />
                </div>
            </div>

            {isShowcase ? (
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
                    <div>
                        {job.manifest_url ? (
                            <ShowcaseManifestView manifestUrl={job.manifest_url} />
                        ) : (
                            <div className="card text-sm text-dim">
                                Showcase running… the manifest will appear once renders are
                                complete.
                            </div>
                        )}
                    </div>
                    <div
                        className="card flex flex-col gap-2 h-96 overflow-auto font-mono text-xs"
                        ref={logsRef}
                    >
                        <h3 className="text-sm font-bold sticky top-0 bg-[#18181b] py-2 border-b border-[#27272a]">
                            Logs
                        </h3>
                        {(job.logs || []).map((log, i) => (
                            <div key={i}>{log}</div>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-2 gap-6">
                    <div
                        className="card flex flex-col gap-2 h-96 overflow-auto font-mono text-xs"
                        ref={logsRef}
                    >
                        <h3 className="text-sm font-bold sticky top-0 bg-[#18181b] py-2 border-b border-[#27272a]">
                            Logs
                        </h3>
                        {(job.logs || []).map((log, i) => (
                            <div key={i}>{log}</div>
                        ))}
                    </div>

                    <div className="card flex items-center justify-center text-dim bg-black/20 overflow-hidden relative">
                        {job.result_url ? (
                            <div className="relative w-full h-full flex items-center justify-center">
                                <img
                                    src={`http://127.0.0.1:8000${job.result_url}`}
                                    alt="Optimization Result"
                                    className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                                />
                                <a
                                    href={`http://127.0.0.1:8000${job.result_url}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="absolute bottom-4 right-4 bg-white/10 hover:bg-white/20 text-white px-3 py-1 rounded-full text-xs backdrop-blur-sm transition-colors"
                                >
                                    Open Full
                                </a>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center gap-2">
                                <div className="animate-pulse">
                                    Waiting for result…
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
