import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import type { ShowcaseRequest } from '../services/api';

type Modality = 'image' | 'audio' | 'video' | 'text';
type Track = 'honest' | 'natural';

const ALL_MODALITIES: Modality[] = ['image', 'audio', 'video', 'text'];

/**
 * v3 four-modality showcase form. POSTs to /jobs/showcase and navigates to
 * /jobs/:id where the JobPage renders the manifest bundle.
 */
export function ShowcasePage() {
    const [target, setTarget] = useState('thunder');
    const [modalities, setModalities] = useState<Set<Modality>>(new Set(ALL_MODALITIES));
    const [tracks, setTracks] = useState<Set<Track>>(new Set(['honest']));
    const [imageBackbone, setImageBackbone] = useState<'sd35' | 'sdxl'>('sd35');
    const [audioBackbone, setAudioBackbone] = useState<'stable-audio-open' | 'audioldm2'>(
        'stable-audio-open',
    );
    const [videoBackbone, setVideoBackbone] = useState<'ltx-video' | 'svd'>('ltx-video');
    const [autocastDtype, setAutocastDtype] = useState<'fp32' | 'fp16' | 'bf16'>('fp32');
    const [steps, setSteps] = useState(200);
    const [seed, setSeed] = useState<string>('');
    const [interpret, setInterpret] = useState(true);
    const [evaluate, setEvaluate] = useState(true);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const toggle = <T extends string>(set: Set<T>, value: T): Set<T> => {
        const next = new Set(set);
        if (next.has(value)) {
            next.delete(value);
        } else {
            next.add(value);
        }
        return next;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!target.trim()) return;
        if (modalities.size === 0) {
            alert('Select at least one modality');
            return;
        }
        if (tracks.size === 0) {
            alert('Select at least one track');
            return;
        }

        const request: ShowcaseRequest = {
            target_text: target,
            modalities: Array.from(modalities),
            tracks: Array.from(tracks),
            image_backbone: imageBackbone,
            audio_backbone: audioBackbone,
            video_backbone: videoBackbone,
            autocast_dtype: autocastDtype,
            steps,
            seed: seed.trim() ? Number(seed) : null,
            interpret,
            evaluate,
        };

        setLoading(true);
        try {
            const job = await api.createShowcaseJob(request);
            navigate(`/jobs/${job.id}`);
        } catch (err) {
            console.error(err);
            const msg = err instanceof Error ? err.message : 'Failed to start showcase';
            alert(msg);
        } finally {
            setLoading(false);
        }
    };

    const isValid = target.trim().length > 0 && modalities.size > 0 && tracks.size > 0;

    return (
        <div className="max-w-3xl mx-auto flex flex-col gap-6">
            <div>
                <h1 className="text-2xl font-bold">Four-Modality Showcase</h1>
                <p className="text-sm text-dim mt-1">
                    Render one concept across image, audio, video and text in a single shared
                    LanguageBind embedding space. Optionally rendered twice — honest (high
                    alignment) and natural (heavy regularisation) — to surface the difference
                    between the model's representation and an aesthetic version of the same
                    concept.
                </p>
            </div>

            <form onSubmit={handleSubmit} className="card flex flex-col gap-5">
                <div>
                    <label className="block text-sm text-dim mb-2" htmlFor="target">
                        Target Concept
                    </label>
                    <input
                        id="target"
                        type="text"
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                        placeholder="e.g. thunder, deep sea, electric storm"
                        className="w-full"
                        autoFocus
                    />
                    <div className="preset-container">
                        {[
                            { label: 'thunder ⚡', value: 'thunder' },
                            { label: 'deep sea 🌊', value: 'deep sea' },
                            { label: 'goldfish 🐠', value: 'goldfish' },
                            { label: 'electric storm ⛈️', value: 'electric storm' },
                            { label: 'nebula 🌌', value: 'nebula' },
                            { label: 'volcano 🔥', value: 'volcano' }
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
                </div>

                <div>
                    <div className="text-sm text-dim mb-2">Modalities</div>
                    <div className="flex flex-wrap gap-2">
                        {ALL_MODALITIES.map((m) => {
                            const active = modalities.has(m);
                            return (
                                <button
                                    type="button"
                                    key={m}
                                    onClick={() => setModalities(toggle(modalities, m))}
                                    style={{
                                        background: active ? 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-hover) 100%)' : 'rgba(255,255,255,0.03)',
                                        borderColor: active ? 'transparent' : 'rgba(255,255,255,0.06)',
                                        color: active ? '#ffffff' : 'var(--color-text-dim)',
                                        boxShadow: active ? '0 4px 12px rgba(139, 92, 246, 0.25)' : 'none',
                                    }}
                                    className="px-4 py-1.5 rounded-full text-sm font-medium transition-all"
                                >
                                    {m === 'image' && '🖼️ '}
                                    {m === 'audio' && '🎵 '}
                                    {m === 'video' && '🎥 '}
                                    {m === 'text' && '✍️ '}
                                    {m.charAt(0).toUpperCase() + m.slice(1)}
                                </button>
                            );
                        })}
                    </div>
                </div>

                <div>
                    <div className="text-sm text-dim mb-2">Tracks</div>
                    <div className="flex flex-wrap gap-2">
                        {(['honest', 'natural'] as Track[]).map((t) => {
                            const active = tracks.has(t);
                            return (
                                <button
                                    type="button"
                                    key={t}
                                    onClick={() => setTracks(toggle(tracks, t))}
                                    style={{
                                        background: active ? 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-hover) 100%)' : 'rgba(255,255,255,0.03)',
                                        borderColor: active ? 'transparent' : 'rgba(255,255,255,0.06)',
                                        color: active ? '#ffffff' : 'var(--color-text-dim)',
                                        boxShadow: active ? '0 4px 12px rgba(139, 92, 246, 0.25)' : 'none',
                                    }}
                                    className="px-4 py-1.5 rounded-full text-sm font-medium transition-all"
                                >
                                    {t === 'honest' ? '😇 Honest' : '🎨 Natural'}
                                </button>
                            );
                        })}
                    </div>
                    <p className="text-xs text-dim mt-2">
                        Honest favours embedding alignment; natural favours aesthetic
                        regularisation. Selecting both renders each into its own subdirectory
                        and writes a top-level manifest covering both.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="image-backbone">
                            Image backbone
                        </label>
                        <div className="relative">
                            <select
                                id="image-backbone"
                                value={imageBackbone}
                                onChange={(e) =>
                                    setImageBackbone(e.target.value as 'sd35' | 'sdxl')
                                }
                                className="w-full pr-8"
                                style={{ appearance: 'none', WebkitAppearance: 'none' }}
                            >
                                <option value="sd35">SD 3.5 Medium (canonical)</option>
                                <option value="sdxl">SDXL (ablation)</option>
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-xs text-dim">
                                ▼
                            </div>
                        </div>
                    </div>
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="audio-backbone">
                            Audio backbone
                        </label>
                        <div className="relative">
                            <select
                                id="audio-backbone"
                                value={audioBackbone}
                                onChange={(e) =>
                                    setAudioBackbone(
                                        e.target.value as 'stable-audio-open' | 'audioldm2',
                                    )
                                }
                                className="w-full pr-8"
                                style={{ appearance: 'none', WebkitAppearance: 'none' }}
                            >
                                <option value="stable-audio-open">Stable Audio Open (canonical)</option>
                                <option value="audioldm2">AudioLDM 2 (ablation)</option>
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-xs text-dim">
                                ▼
                            </div>
                        </div>
                    </div>
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="video-backbone">
                            Video backbone
                        </label>
                        <div className="relative">
                            <select
                                id="video-backbone"
                                value={videoBackbone}
                                onChange={(e) =>
                                    setVideoBackbone(e.target.value as 'ltx-video' | 'svd')
                                }
                                className="w-full pr-8"
                                style={{ appearance: 'none', WebkitAppearance: 'none' }}
                            >
                                <option value="ltx-video">LTX-Video (canonical)</option>
                                <option value="svd">SVD (ablation)</option>
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-xs text-dim">
                                ▼
                            </div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="autocast">
                            Autocast dtype
                        </label>
                        <div className="relative">
                            <select
                                id="autocast"
                                value={autocastDtype}
                                onChange={(e) =>
                                    setAutocastDtype(e.target.value as 'fp32' | 'fp16' | 'bf16')
                                }
                                className="w-full pr-8"
                                style={{ appearance: 'none', WebkitAppearance: 'none' }}
                            >
                                <option value="fp32">fp32 (safe default)</option>
                                <option value="bf16">bf16 (M1 Max: ~2x faster)</option>
                                <option value="fp16">fp16</option>
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-xs text-dim">
                                ▼
                            </div>
                        </div>
                    </div>
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <label className="block text-sm text-dim" htmlFor="steps">
                                Steps
                            </label>
                            <span className="text-xs text-dim" style={{ fontSize: '0.75rem', padding: '0.125rem 0.5rem', background: 'rgba(255,255,255,0.06)', borderRadius: '9999px', lineHeight: 1 }}>{steps}</span>
                        </div>
                        <div className="flex items-center gap-4">
                            <input
                                id="steps-slider"
                                type="range"
                                min={10}
                                max={1000}
                                step={10}
                                value={steps > 1000 ? 1000 : steps}
                                onChange={(e) => setSteps(Number(e.target.value))}
                                className="flex-1"
                            />
                            <input
                                id="steps"
                                type="number"
                                min={10}
                                max={5000}
                                value={steps}
                                onChange={(e) => setSteps(Number(e.target.value))}
                                className="w-24 text-center"
                                style={{ width: '5.5rem', textAlign: 'center', padding: '0.5rem 0.25rem' }}
                            />
                        </div>
                    </div>
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="seed">
                            Seed (optional)
                        </label>
                        <div className="flex gap-2">
                            <input
                                id="seed"
                                type="number"
                                value={seed}
                                placeholder="random"
                                onChange={(e) => setSeed(e.target.value)}
                                className="flex-1"
                            />
                            <button
                                type="button"
                                onClick={() => setSeed(Math.floor(Math.random() * 1000000).toString())}
                                className="px-3"
                                style={{ minWidth: '2.5rem', padding: '0.5rem' }}
                                title="Generate random seed"
                            >
                                🎲
                            </button>
                            {seed && (
                                <button
                                    type="button"
                                    onClick={() => setSeed('')}
                                    className="px-3"
                                    style={{ minWidth: '2.5rem', padding: '0.5rem', color: 'var(--color-error)' }}
                                    title="Clear seed"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex gap-6 text-sm text-dim">
                    <label className="flex items-center gap-2 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={interpret}
                            onChange={(e) => setInterpret(e.target.checked)}
                            style={{ accentColor: 'var(--color-primary)', cursor: 'pointer' }}
                        />
                        Compute interpretation bundle
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={evaluate}
                            onChange={(e) => setEvaluate(e.target.checked)}
                            style={{ accentColor: 'var(--color-primary)', cursor: 'pointer' }}
                        />
                        Compute evaluation card
                    </label>
                </div>

                <div className="flex justify-end pt-2">
                    <button
                        type="submit"
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${isValid && !loading
                                ? 'primary'
                                : 'cursor-not-allowed'
                            }`}
                        disabled={!isValid || loading}
                    >
                        {loading ? 'Starting…' : 'Start Showcase'}
                    </button>
                </div>
            </form>
        </div>
    );
}
