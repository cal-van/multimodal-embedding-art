import { useState } from 'react';
import { api } from '../services/api';
import type {
    AnchorCompareMultimodalResponse,
    AnchorCompareResponse,
} from '../services/api';

/**
 * Platonic-representation probe.
 *
 * Two modes:
 * * Text — compare two or more text references in the canonical
 *   LanguageBind embedding space.
 * * Multimodal — compare the *same* concept across text + image +
 *   audio + video references. The cosine matrix is the
 *   cross-modal-agreement readout.
 *
 * In both modes the response also includes a per-reference (or per-
 * modality) top-K text-anchor readout — what the model thinks the
 * encoding "means" in language space.
 */
export function AnchorComparePage() {
    const [mode, setMode] = useState<'text' | 'multimodal'>('text');
    return (
        <div className="max-w-3xl mx-auto">
            <header className="page-head rise rise-1">
                <span className="eyebrow">03 — Cross-modal alignment</span>
                <h1 className="display">
                    Anchor <em>Compare</em>
                </h1>
                <p className="lede">
                    Project text, image, audio and video references into the shared
                    LanguageBind embedding space, then read their pairwise{' '}
                    <em>cosine geometry</em> and the top text-anchor words each encoding
                    resolves to in language space.
                </p>
            </header>

            <div className="flex flex-col gap-4">
                <ModeToggle mode={mode} setMode={setMode} />
                {mode === 'text' ? <TextOnlyForm /> : <MultimodalForm />}
            </div>
        </div>
    );
}

function ModeToggle({
    mode,
    setMode,
}: {
    mode: 'text' | 'multimodal';
    setMode: (m: 'text' | 'multimodal') => void;
}) {
    return (
        <div className="flex gap-2 rise rise-2">
            {(['text', 'multimodal'] as const).map((m) => (
                <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`chip ${mode === m ? 'on' : ''}`}
                >
                    {m === 'text' ? 'Text' : 'Multimodal'}
                </button>
            ))}
        </div>
    );
}

// ---------------------------------------------------------------------------
// Text-only form (the original v1 path).
// ---------------------------------------------------------------------------

function TextOnlyForm() {
    const [conceptLabel, setConceptLabel] = useState('thunder');
    const [textsRaw, setTextsRaw] = useState('thunder\nelectric storm\ncrackling sound');
    const [topK, setTopK] = useState(12);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AnchorCompareResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const texts = textsRaw
            .split('\n')
            .map((t) => t.trim())
            .filter((t) => t.length > 0);
        if (texts.length < 2) {
            setError('Provide at least two text references (one per line).');
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const response = await api.anchorCompare({
                concept_label: conceptLabel,
                texts,
                top_k_text: topK,
            });
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <form onSubmit={handleSubmit} className="panel rise rise-3 flex flex-col gap-4">
                <div className="field">
                    <label htmlFor="concept">Concept label</label>
                    <input
                        id="concept"
                        type="text"
                        value={conceptLabel}
                        onChange={(e) => setConceptLabel(e.target.value)}
                    />
                </div>
                <div className="field">
                    <label htmlFor="texts">Text references (one per line, at least two)</label>
                    <textarea
                        id="texts"
                        value={textsRaw}
                        onChange={(e) => setTextsRaw(e.target.value)}
                        rows={6}
                        className="mono text-sm"
                    />
                </div>

                <div className="divider" />

                <div className="flex gap-4 items-end flex-wrap">
                    <div className="field">
                        <label htmlFor="topk">Top-K text-anchor words</label>
                        <input
                            id="topk"
                            type="number"
                            min={1}
                            max={50}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            style={{ width: '6rem', textAlign: 'center' }}
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="btn-primary"
                        style={{ marginLeft: 'auto' }}
                    >
                        {loading ? 'Comparing…' : 'Run comparison →'}
                    </button>
                </div>
            </form>
            {error && <ErrorPanel message={error} />}
            {result && <TextResultView result={result} />}
        </>
    );
}

function TextResultView({ result }: { result: AnchorCompareResponse }) {
    const labels = result.entries.map((e) => e.label);
    return (
        <div className="flex flex-col gap-4">
            <CosineMatrix labels={labels} matrix={result.cosine_matrix} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.entries.map((entry) => (
                    <AnchorReadoutCard
                        key={entry.label}
                        title={entry.label}
                        subtitle={`"${entry.text}" · dim=${entry.embedding_dim}`}
                        anchors={entry.text_anchor}
                    />
                ))}
            </div>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Multimodal form: text + image + audio + video uploads.
// ---------------------------------------------------------------------------

function MultimodalForm() {
    const [conceptLabel, setConceptLabel] = useState('thunder');
    const [text, setText] = useState('thunder');
    const [image, setImage] = useState<File | null>(null);
    const [audio, setAudio] = useState<File | null>(null);
    const [video, setVideo] = useState<File | null>(null);
    const [topK, setTopK] = useState(12);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AnchorCompareMultimodalResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const refsPresent = (text.trim() ? 1 : 0) + (image ? 1 : 0) + (audio ? 1 : 0) + (video ? 1 : 0);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (refsPresent === 0) {
            setError('Provide at least one of text / image / audio / video.');
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const response = await api.anchorCompareMultimodal({
                concept_label: conceptLabel,
                top_k_text: topK,
                text: text.trim() || null,
                image,
                audio,
                video,
            });
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <form onSubmit={handleSubmit} className="panel rise rise-3 flex flex-col gap-4">
                <div className="field">
                    <label htmlFor="concept-multi">Concept label</label>
                    <input
                        id="concept-multi"
                        type="text"
                        value={conceptLabel}
                        onChange={(e) => setConceptLabel(e.target.value)}
                    />
                </div>
                <div className="field">
                    <label htmlFor="text-multi">Text reference (optional)</label>
                    <input
                        id="text-multi"
                        type="text"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                </div>

                <div className="field">
                    <span className="field-label">Modal references</span>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-1">
                        <FileSlot label="Image" accept="image/*" file={image} onChange={setImage} />
                        <FileSlot label="Audio" accept="audio/*" file={audio} onChange={setAudio} />
                        <FileSlot label="Video" accept="video/*" file={video} onChange={setVideo} />
                    </div>
                </div>

                <div className="divider" />

                <div className="flex gap-4 items-end flex-wrap">
                    <div className="field">
                        <label htmlFor="topk-multi">Top-K text-anchor words</label>
                        <input
                            id="topk-multi"
                            type="number"
                            min={1}
                            max={50}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            style={{ width: '6rem', textAlign: 'center' }}
                        />
                    </div>
                    <span className="readout cool" style={{ alignSelf: 'center' }}>
                        {refsPresent} ref{refsPresent === 1 ? '' : 's'} attached
                    </span>
                    <button
                        type="submit"
                        disabled={loading || refsPresent === 0}
                        className="btn-primary"
                        style={{ marginLeft: 'auto' }}
                    >
                        {loading ? 'Comparing…' : 'Run comparison →'}
                    </button>
                </div>
            </form>
            {error && <ErrorPanel message={error} />}
            {result && <MultimodalResultView result={result} />}
        </>
    );
}

function FileSlot({
    label,
    accept,
    file,
    onChange,
}: {
    label: string;
    accept: string;
    file: File | null;
    onChange: (f: File | null) => void;
}) {
    const inputId = `file-${label.toLowerCase()}`;
    return (
        <div className="field">
            <label htmlFor={inputId}>{label}</label>
            <label
                htmlFor={inputId}
                className="flex items-center justify-between gap-2 cursor-pointer mono text-xs"
                style={{
                    padding: '0.62rem 0.8rem',
                    borderRadius: 'var(--radius)',
                    border: `1px dashed ${file ? 'var(--aqua-deep)' : 'var(--line-bright)'}`,
                    background: 'var(--ink-deep)',
                    color: file ? 'var(--aqua-bright)' : 'var(--text-faint)',
                }}
            >
                <span
                    style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                    }}
                    title={file ? file.name : undefined}
                >
                    {file ? file.name : 'No file — click to select'}
                </span>
                <span aria-hidden style={{ color: 'var(--text-faint)' }}>
                    {file ? '◉' : '＋'}
                </span>
            </label>
            <input
                id={inputId}
                type="file"
                accept={accept}
                onChange={(e) => onChange(e.target.files?.[0] ?? null)}
                style={{ display: 'none' }}
            />
        </div>
    );
}

function MultimodalResultView({ result }: { result: AnchorCompareMultimodalResponse }) {
    const labels = result.entries.map((e) => e.modality);
    return (
        <div className="flex flex-col gap-4">
            <CosineMatrix labels={labels} matrix={result.cosine_matrix} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.entries.map((entry) => (
                    <AnchorReadoutCard
                        key={entry.modality}
                        title={entry.modality}
                        subtitle={`dim=${entry.embedding_dim}${entry.source ? ` · ${entry.source.split('/').slice(-1)[0]}` : ''}`}
                        anchors={entry.text_anchor}
                    />
                ))}
            </div>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Shared sub-components (cosine matrix + anchor card).
// ---------------------------------------------------------------------------

function ErrorPanel({ message }: { message: string }) {
    return (
        <div
            className="panel rise rise-3 mono text-sm"
            style={{ color: 'var(--err)', borderColor: 'rgba(240, 120, 106, 0.4)' }}
        >
            {message}
        </div>
    );
}

function CosineMatrix({
    labels,
    matrix,
}: {
    labels: string[];
    matrix: Record<string, Record<string, number>>;
}) {
    return (
        <div className="panel rise rise-4">
            <div className="flex items-center justify-between mb-3">
                <span className="field-label">Cosine matrix</span>
                <span className="eyebrow">cross-modal agreement</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
                <table
                    className="mono text-xs"
                    style={{
                        width: '100%',
                        borderCollapse: 'separate',
                        borderSpacing: '2px',
                    }}
                >
                    <thead>
                        <tr>
                            <th style={{ padding: '0.5rem' }} />
                            {labels.map((l) => (
                                <th
                                    key={l}
                                    className="eyebrow"
                                    style={{
                                        padding: '0.5rem',
                                        textAlign: 'center',
                                        color: 'var(--text-dim)',
                                        fontWeight: 400,
                                    }}
                                >
                                    {l}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {labels.map((row) => (
                            <tr key={row}>
                                <th
                                    className="eyebrow"
                                    style={{
                                        padding: '0.5rem 0.75rem',
                                        textAlign: 'right',
                                        color: 'var(--text-dim)',
                                        fontWeight: 400,
                                        whiteSpace: 'nowrap',
                                    }}
                                >
                                    {row}
                                </th>
                                {labels.map((col) => {
                                    const v = matrix[row]?.[col] ?? 0;
                                    const isDiagonal = row === col;
                                    // Heatmap: aqua opacity scales with the cosine value.
                                    const fill = Math.max(0, Math.min(1, v));
                                    return (
                                        <td
                                            key={col}
                                            style={{
                                                textAlign: 'center',
                                                padding: '0.55rem 0.4rem',
                                                borderRadius: 'var(--radius)',
                                                border: isDiagonal
                                                    ? '1px solid var(--aqua-deep)'
                                                    : '1px solid var(--line)',
                                                background: `rgba(111, 227, 224, ${fill})`,
                                                color: v > 0.6 ? 'var(--ink)' : 'var(--text)',
                                                fontWeight: isDiagonal ? 700 : 500,
                                            }}
                                            title={`${row} · ${col} = ${v.toFixed(4)}`}
                                        >
                                            {v.toFixed(3)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <p className="faint text-xs mono mt-3" style={{ letterSpacing: '0.04em' }}>
                cell opacity ∝ cosine · diagonal = self (1.000)
            </p>
        </div>
    );
}

function AnchorReadoutCard({
    title,
    subtitle,
    anchors,
}: {
    title: string;
    subtitle: string;
    anchors: Array<{ word: string; similarity: number }>;
}) {
    return (
        <div className="panel rise rise-5 flex flex-col gap-3">
            <div>
                <div className="display" style={{ fontSize: '1.5rem' }}>
                    {title}
                </div>
                <div className="mono text-xs faint mt-1">{subtitle}</div>
            </div>
            {anchors.length > 0 ? (
                <div className="flex flex-col gap-1">
                    {anchors.map((a) => {
                        const pct = Math.max(0, Math.min(100, a.similarity * 100));
                        return (
                            <div
                                key={a.word}
                                className="relative flex items-center justify-between mono text-xs"
                                style={{
                                    overflow: 'hidden',
                                    padding: '0.4rem 0.7rem',
                                    borderRadius: 'var(--radius)',
                                    border: '1px solid var(--line)',
                                    background: 'var(--ink-deep)',
                                }}
                            >
                                <div
                                    className="absolute pointer-events-none"
                                    style={{
                                        left: 0,
                                        top: 0,
                                        bottom: 0,
                                        width: `${pct}%`,
                                        background: 'rgba(111, 227, 224, 0.14)',
                                        borderRight: '1px solid rgba(111, 227, 224, 0.35)',
                                    }}
                                />
                                <span className="relative" style={{ color: 'var(--text)' }}>
                                    {a.word}
                                </span>
                                <span
                                    className="relative"
                                    style={{ color: 'var(--aqua-bright)', fontWeight: 700 }}
                                >
                                    {a.similarity.toFixed(3)}
                                </span>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="mono text-xs faint">No anchor words available.</div>
            )}
        </div>
    );
}
