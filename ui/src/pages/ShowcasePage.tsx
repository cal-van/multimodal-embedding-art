import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import type { ShowcaseRequest } from '../services/api';

type Modality = 'image' | 'audio' | 'video' | 'text';
type CompileMode = 'none' | 'default' | 'reduce-overhead' | 'max-autotune';

const ALL_MODALITIES: Modality[] = ['image', 'audio', 'video', 'text'];
const MODALITY_GLYPH: Record<Modality, string> = {
    image: '◳',
    audio: '◌',
    video: '▷',
    text: '¶',
};

const PRESETS = ['thunder', 'deep sea', 'goldfish', 'electric storm', 'nebula', 'volcano'];

/** Human-readable framing for a point on the honest<->natural dial. */
function realismCaption(realism: number): { title: string; blurb: string } {
    if (realism <= 0.1) {
        return {
            title: 'Honest · what the model sees',
            blurb:
                "Maximum embedding alignment, almost no regularisation. The generator's raw attempt to occupy the concept's coordinate — the machine-legible artefact. With minimal regularisation this raw maximiser can lean adversarial: it may look alien rather than faithfully representative of the concept.",
        };
    }
    if (realism >= 0.9) {
        return {
            title: 'Natural · human-legible',
            blurb:
                'Heavy regularisation, relaxed alignment. A conventionally photographic, smooth rendering — easy for a person to read, further from the raw representation.',
        };
    }
    return {
        title: `Blend · ${Math.round((1 - realism) * 100)}% honest / ${Math.round(realism * 100)}% natural`,
        blurb:
            'A point between the raw representation and a human-legible image. Alignment and regularisation are interpolated continuously.',
    };
}

/** Honest (aqua) -> Natural (amber) temperature for the dial. */
function temperature(realism: number): string {
    const honest = [111, 227, 224];
    const natural = [242, 168, 59];
    const c = honest.map((h, i) => Math.round(h + (natural[i] - h) * realism));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function ShowcasePage() {
    const [target, setTarget] = useState('thunder');
    const [modalities, setModalities] = useState<Set<Modality>>(new Set(ALL_MODALITIES));
    // Continuous honest<->natural dial: 0 = honest (machine-legible), 1 = natural (human-legible).
    const [realism, setRealism] = useState(0);
    // When on, render both extremes side-by-side instead of the single dial point.
    const [compareExtremes, setCompareExtremes] = useState(false);
    const [imageBackbone, setImageBackbone] = useState<'sd35' | 'sdxl'>('sd35');
    const [audioBackbone, setAudioBackbone] = useState<'stable-audio-open' | 'audioldm2'>(
        'stable-audio-open',
    );
    const [videoBackbone, setVideoBackbone] = useState<'ltx-video' | 'svd'>('ltx-video');
    const [autocastDtype, setAutocastDtype] = useState<'fp32' | 'fp16' | 'bf16'>('bf16');
    const [compileMode, setCompileMode] = useState<CompileMode>('reduce-overhead');
    const [steps, setSteps] = useState(200);
    const [seed, setSeed] = useState<string>('');
    const [interpret, setInterpret] = useState(true);
    const [evaluate, setEvaluate] = useState(true);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const toggle = <T extends string>(set: Set<T>, value: T): Set<T> => {
        const next = new Set(set);
        if (next.has(value)) next.delete(value);
        else next.add(value);
        return next;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!target.trim()) return;
        if (modalities.size === 0) {
            alert('Select at least one modality');
            return;
        }

        const request: ShowcaseRequest = {
            target_text: target,
            modalities: Array.from(modalities),
            // Compare mode renders both named endpoints; otherwise a single
            // bundle at the dial position.
            ...(compareExtremes ? { tracks: ['honest', 'natural'] } : { realism }),
            image_backbone: imageBackbone,
            audio_backbone: audioBackbone,
            video_backbone: videoBackbone,
            autocast_dtype: autocastDtype,
            compile_mode: compileMode,
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

    const isValid = target.trim().length > 0 && modalities.size > 0;
    const caption = realismCaption(realism);
    const temp = temperature(realism);
    const dialStyle = compareExtremes
        ? { opacity: 0.3, pointerEvents: 'none' as const }
        : ({
              ['--accent']: temp,
              ['--accent-bright']: temp,
              ['--accent-glow']: 'transparent',
          } as React.CSSProperties);

    return (
        <div className="max-w-3xl mx-auto">
            <header className="page-head rise rise-1">
                <span className="eyebrow">01 — Four-modality synthesis</span>
                <h1 className="display">
                    Show<em>case</em>
                </h1>
                <p className="lede">
                    See what the model actually thinks a concept <em>looks</em>, <em>sounds</em>,
                    and <em>moves</em> like — not generated from training images, but the coordinate
                    pulled straight out of its representation. The realism dial sweeps from{' '}
                    <em>honest</em> (the raw representation) to <em>natural</em> (a human-legible
                    image).
                </p>
            </header>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                {/* concept */}
                <div className="panel rise rise-2 field">
                    <label htmlFor="target">Target concept</label>
                    <input
                        id="target"
                        type="text"
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                        placeholder="thunder, deep sea, electric storm…"
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

                    <div className="divider" />

                    <span className="field-label">Modalities</span>
                    <div className="flex flex-wrap gap-2 mt-1">
                        {ALL_MODALITIES.map((m) => (
                            <button
                                type="button"
                                key={m}
                                onClick={() => setModalities(toggle(modalities, m))}
                                aria-pressed={modalities.has(m)}
                                className={`chip ${modalities.has(m) ? 'on' : ''}`}
                            >
                                <span className="mono" style={{ marginRight: '0.4rem' }}>
                                    {MODALITY_GLYPH[m]}
                                </span>
                                {m}
                            </button>
                        ))}
                    </div>
                </div>

                {/* the realism dial — centrepiece */}
                <div className="panel rise rise-3">
                    <div className="flex justify-between items-baseline mb-3">
                        <span id="realism-label" className="field-label">
                            Realism ↔ Honesty
                        </span>
                        <span
                            className={`readout ${realism < 0.5 ? 'cool' : ''}`}
                            style={{ opacity: compareExtremes ? 0.3 : 1 }}
                        >
                            r = {realism.toFixed(2)}
                        </span>
                    </div>

                    <div style={dialStyle}>
                        <div
                            className="flex justify-between mono"
                            style={{ fontSize: '0.68rem', letterSpacing: '0.1em', marginBottom: '0.5rem' }}
                        >
                            <span style={{ color: 'var(--aqua)' }}>HONEST · machine</span>
                            <span style={{ color: 'var(--amber)' }}>human · NATURAL</span>
                        </div>
                        <div
                            style={{
                                height: 3,
                                borderRadius: 2,
                                marginBottom: '-3px',
                                background:
                                    'linear-gradient(90deg, var(--aqua) 0%, var(--amber) 100%)',
                                opacity: 0.45,
                            }}
                        />
                        <input
                            id="realism"
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={realism}
                            disabled={compareExtremes}
                            onChange={(e) => setRealism(Number(e.target.value))}
                            aria-label="Realism dial: honest to natural"
                            aria-labelledby="realism-label"
                            aria-valuemin={0}
                            aria-valuemax={1}
                            aria-valuenow={realism}
                            aria-valuetext={caption.title}
                        />
                        <div className="mt-3">
                            <div
                                className="display"
                                style={{ fontSize: '1.5rem', color: temp }}
                            >
                                {caption.title}
                            </div>
                            <p className="dim text-sm mt-1" style={{ maxWidth: '58ch' }}>
                                {caption.blurb}
                            </p>
                        </div>
                    </div>

                    <label
                        className="flex items-center gap-2 cursor-pointer select-none text-sm dim mt-4 pt-3"
                        style={{ borderTop: '1px solid var(--line)' }}
                    >
                        <input
                            type="checkbox"
                            checked={compareExtremes}
                            onChange={(e) => setCompareExtremes(e.target.checked)}
                        />
                        Compare both extremes side-by-side — render honest <em>and</em> natural
                    </label>
                </div>

                {/* backbones */}
                <div className="panel rise rise-4">
                    <span className="field-label">Generators</span>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-3">
                        <div className="field">
                            <label htmlFor="image-backbone">Image</label>
                            <select
                                id="image-backbone"
                                value={imageBackbone}
                                onChange={(e) => setImageBackbone(e.target.value as 'sd35' | 'sdxl')}
                            >
                                <option value="sd35">SD 3.5 Medium</option>
                                <option value="sdxl">SDXL (comparison)</option>
                            </select>
                        </div>
                        <div className="field">
                            <label htmlFor="audio-backbone">Audio</label>
                            <select
                                id="audio-backbone"
                                value={audioBackbone}
                                onChange={(e) =>
                                    setAudioBackbone(
                                        e.target.value as 'stable-audio-open' | 'audioldm2',
                                    )
                                }
                            >
                                <option value="stable-audio-open">Stable Audio Open</option>
                                <option value="audioldm2">AudioLDM 2 (comparison)</option>
                            </select>
                        </div>
                        <div className="field">
                            <label htmlFor="video-backbone">Video</label>
                            <select
                                id="video-backbone"
                                value={videoBackbone}
                                onChange={(e) => setVideoBackbone(e.target.value as 'ltx-video' | 'svd')}
                            >
                                <option value="ltx-video">LTX-Video</option>
                                <option value="svd">SVD (comparison)</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* quality + run params */}
                <div className="panel rise rise-5">
                    <div className="field">
                        <div className="flex justify-between items-baseline">
                            <label htmlFor="steps">Quality · steps per modality</label>
                            <span className="readout">{steps}</span>
                        </div>
                        <div className="text-xs dim">
                            200 = quick preview · 1000+ = full render
                        </div>
                        <div className="flex items-center gap-3 mt-1">
                            <input
                                type="range"
                                min={10}
                                max={1000}
                                step={10}
                                value={steps > 1000 ? 1000 : steps}
                                onChange={(e) => setSteps(Number(e.target.value))}
                                className="flex-1"
                                aria-label="Steps per modality"
                                aria-valuetext={String(steps)}
                            />
                            <input
                                id="steps"
                                type="number"
                                min={10}
                                max={5000}
                                value={steps}
                                onChange={(e) => setSteps(Number(e.target.value))}
                                style={{ width: '5rem', textAlign: 'center' }}
                            />
                        </div>
                    </div>

                    <div className="divider" />

                    <details>
                        <summary
                            className="field-label"
                            style={{ cursor: 'pointer', listStyle: 'none' }}
                        >
                            Advanced · compute &amp; reproducibility
                        </summary>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                            <div className="field">
                                <label htmlFor="autocast">Autocast dtype</label>
                                <select
                                    id="autocast"
                                    value={autocastDtype}
                                    onChange={(e) =>
                                        setAutocastDtype(
                                            e.target.value as 'fp32' | 'fp16' | 'bf16',
                                        )
                                    }
                                >
                                    <option value="bf16">bf16 — M1 Max, ~2× faster</option>
                                    <option value="fp32">fp32 — safe / reproducible</option>
                                    <option value="fp16">fp16</option>
                                </select>
                            </div>
                            <div className="field">
                                <label htmlFor="compile">torch.compile</label>
                                <select
                                    id="compile"
                                    value={compileMode}
                                    onChange={(e) => setCompileMode(e.target.value as CompileMode)}
                                >
                                    <option value="reduce-overhead">
                                        reduce-overhead — 1.5–2.5×
                                    </option>
                                    <option value="none">none</option>
                                    <option value="default">default</option>
                                    <option value="max-autotune">max-autotune — aggressive</option>
                                </select>
                            </div>
                            <div className="field">
                                <label htmlFor="seed">Seed</label>
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
                                        onClick={() =>
                                            setSeed(Math.floor(Math.random() * 1000000).toString())
                                        }
                                        title="Random seed"
                                    >
                                        ⟳
                                    </button>
                                    {seed && (
                                        <button
                                            type="button"
                                            onClick={() => setSeed('')}
                                            title="Clear seed"
                                            style={{ color: 'var(--err)' }}
                                        >
                                            ✕
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </details>

                    <div className="divider" />

                    <div className="flex gap-6 text-sm dim flex-wrap">
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={interpret}
                                onChange={(e) => setInterpret(e.target.checked)}
                            />
                            Interpretation bundle
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={evaluate}
                                onChange={(e) => setEvaluate(e.target.checked)}
                            />
                            Evaluation card
                        </label>
                    </div>
                </div>

                <div className="flex justify-between items-center gap-3 flex-wrap rise rise-5">
                    <div className="flex items-center gap-2 flex-wrap text-xs dim">
                        <span>
                            Renders run locally and take minutes — longer with more steps and
                            modalities.
                        </span>
                        <span className="readout">
                            {modalities.size} modalities × {steps} steps
                        </span>
                    </div>
                    <button type="submit" className="btn-primary" disabled={!isValid || loading}>
                        {loading ? 'Initialising…' : 'Run showcase →'}
                    </button>
                </div>
            </form>
        </div>
    );
}
