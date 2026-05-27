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
                        className="w-full"
                        autoFocus
                    />
                    <div className="preset-container">
                        {[
                            { label: 'goldfish 🐠', value: 'goldfish' },
                            { label: 'fire + water 🔥💧', value: 'fire + water' },
                            { label: 'sunset 🌅', value: 'sunset' },
                            { label: 'pine forest 🌲', value: 'pine forest' },
                            { label: 'electricity ⚡', value: 'electricity' }
                        ].map((p) => (
                            <button
                                type="button"
                                key={p.value}
                                onClick={() => setTarget(p.value)}
                                className="preset-btn"
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>
                    <p className="text-xs text-dim mt-2">Enter text descriptions to guide the optimization.</p>
                </div>

                <div>
                    <label className="block text-sm text-dim mb-2">Output Modality</label>
                    <div className="relative">
                        <select
                            className="w-full appearance-none pr-8"
                            style={{ appearance: 'none', WebkitAppearance: 'none' }}
                            id="modality"
                            name="modality"
                            value={modality}
                            onChange={(e) => setModality(e.target.value)}
                        >
                            <option value="image">Image 🖼️</option>
                            <option value="video">Video 🎥</option>
                            <option value="audio">Audio 🎵</option>
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-dim" style={{ fontSize: '0.75rem' }}>
                            ▼
                        </div>
                    </div>
                </div>

                <div className="flex justify-end pt-4">
                    <button
                        type="submit"
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${isValid && !loading
                                ? 'primary'
                                : 'cursor-not-allowed'
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
