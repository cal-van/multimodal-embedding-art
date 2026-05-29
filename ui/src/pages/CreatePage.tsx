import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const PRESETS = ['goldfish', 'fire + water', 'sunset', 'pine forest', 'electricity'];

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
        <div className="max-w-xl mx-auto">
            <header className="page-head rise rise-1">
                <span className="eyebrow">02 — Single-modality optimisation</span>
                <h1 className="display">
                    Single<em>-modality</em>
                </h1>
                <p className="lede">
                    The legacy ImageBind single-generator path — one concept, one output modality.
                    The four-modality <em>Showcase</em> is the canonical flow; this route is kept for
                    back-compat and ablation.
                </p>
            </header>

            <form onSubmit={handleSubmit} className="panel rise rise-2 flex flex-col gap-4">
                <div className="field">
                    <label htmlFor="target">Target concept</label>
                    <input
                        id="target"
                        type="text"
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                        placeholder="e.g. goldfish, fire + water"
                        autoFocus
                    />
                    <div className="flex flex-wrap gap-2 mt-1">
                        {PRESETS.map((p) => (
                            <button
                                type="button"
                                key={p}
                                onClick={() => setTarget(p)}
                                className="preset"
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                    <p className="text-xs dim mt-2">
                        Enter a text description to guide the optimisation.
                    </p>
                </div>

                <div className="divider" />

                <div className="field">
                    <label htmlFor="modality">Output modality</label>
                    <select
                        id="modality"
                        name="modality"
                        value={modality}
                        onChange={(e) => setModality(e.target.value)}
                    >
                        <option value="image">Image</option>
                        <option value="video">Video</option>
                        <option value="audio">Audio</option>
                    </select>
                </div>

                <div className="flex justify-end pt-2">
                    <button type="submit" className="btn-primary" disabled={!isValid || loading}>
                        {loading ? 'Initialising…' : 'Start optimisation →'}
                    </button>
                </div>
            </form>
        </div>
    );
}
