import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import type { Job } from '../services/api';

export function GalleryPage() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.listJobs()
            .then(setJobs)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div>Loading jobs...</div>;

    return (
        <div className="flex flex-col gap-6">
            <h1 className="text-2xl font-bold">Gallery</h1>

            {jobs.length === 0 ? (
                <div className="text-dim">No jobs yet. <Link to="/" className="text-violet-400 hover:text-violet-300">Create one!</Link></div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {jobs.map(job => (
                        <Link key={job.id} to={`/jobs/${job.id}`} className="card hover:bg-[#3f3f46] transition-colors group">
                            <div className="aspect-square bg-black/20 rounded-lg mb-3 overflow-hidden flex items-center justify-center relative">
                                {job.result_url ? (
                                    <img
                                        src={`http://127.0.0.1:8000${job.result_url}`}
                                        alt={(job.target_text || []).join(", ")}
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <div className="text-xs text-dim uppercase tracking-wider">{job.status}</div>
                                )}

                                {/* Status overlay */}
                                <div className="absolute top-2 right-2 badge text-xs backdrop-blur-md bg-black/50">
                                    {job.output_modality}
                                </div>
                            </div>

                            <h3 className="font-bold truncate text-sm" title={(job.target_text || []).join(", ")}>
                                {(job.target_text || []).join(", ") || "Untitled Job"}
                            </h3>
                            <div className="flex justify-between items-center mt-2">
                                <span className={`text-xs badge ${job.status}`}>{job.status}</span>
                                <span className="text-xs text-dim">
                                    {/* Format date if available, or just ID */}
                                    {job.id.slice(0, 8)}
                                </span>
                            </div>
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
}
