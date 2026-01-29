import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

export function CreatePage() {
    const [target, setTarget] = useState('goldfish');
    const [modality, setModality] = useState('image');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!target.trim()) return;

        setLoading(true);
        try {
            const job = await api.createJob([target], modality);
            // Correct logic to navigate to /jobs/:id (plural)
            navigate(`/jobs/${job.id}`);
        } catch (err) {
            console.error(err);
            alert('Failed to create job');
        } finally {
            setLoading(false);
        }
    };

    const isValid = target.trim().length > 0;

    return (
        <div className="max-w-xl mx-auto flex flex-col gap-6">
            <h1 className="text-2xl font-bold">Create New Optimization</h1>

            <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
                <div>
                    <label className="block text-sm text-dim mb-2">Target Concept</label>
                    <input
                        type="text"
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                        placeholder="e.g. goldfish, fire + water"
                        className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-3 rounded-md focus:border-[#8b5cf6] focus:outline-none transition-colors"
                        autoFocus
                    />
                    <p className="text-xs text-dim mt-2">Enter text descriptions to guide the optimization.</p>
                </div>

                <div>
                    <label className="block text-sm text-dim mb-2">Output Modality</label>
                    <div className="relative">
                        <select
                            className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-3 rounded-md appearance-none focus:border-[#8b5cf6] focus:outline-none transition-colors"
                            id="modality"
                            name="modality"
                            value={modality}
                            onChange={(e) => setModality(e.target.value)}
                        >
                            <option value="image">Image 🖼️</option>
                            <option value="video">Video 🎥</option>
                            <option value="audio">Audio 🎵</option>
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-dim">
                            ▼
                        </div>
                    </div>
                </div>

                <div className="flex justify-end pt-4">
                    <button
                        type="submit"
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${isValid && !loading
                                ? 'bg-[#8b5cf6] hover:bg-[#7c3aed] text-white shadow-[0_0_15px_rgba(139,92,246,0.5)]'
                                : 'bg-[#27272a] text-dim cursor-not-allowed'
                            }`}
                        disabled={!isValid || loading}
                    >
                        {loading ? 'Starting Engine...' : 'Start Optimization'}
                    </button>
                </div>
            </form>
        </div>
    );
}
