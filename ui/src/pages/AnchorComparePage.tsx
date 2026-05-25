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
        <div className="max-w-5xl mx-auto flex flex-col gap-6">
            <div>
                <h1 className="text-2xl font-bold">Anchor Comparison</h1>
                <p className="text-sm text-dim mt-1">
                    Probe whether semantically equivalent inputs land at the same point
                    in the canonical LanguageBind embedding space. Use <strong>Text</strong> for
                    a pure-text comparison or <strong>Multimodal</strong> to bring image / audio /
                    video references into the same space.
                </p>
            </div>
            <ModeToggle mode={mode} setMode={setMode} />
            {mode === 'text' ? <TextOnlyForm /> : <MultimodalForm />}
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
        <div className="flex gap-2">
            {(['text', 'multimodal'] as const).map((m) => (
                <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${mode === m
                        ? 'bg-[#8b5cf6] text-white shadow-[0_0_15px_rgba(139,92,246,0.5)]'
                        : 'bg-[#27272a] text-dim hover:bg-[#3f3f46]'
                        }`}
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
            <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
                <div>
                    <label className="block text-sm text-dim mb-2" htmlFor="concept">
                        Concept label
                    </label>
                    <input
                        id="concept"
                        type="text"
                        value={conceptLabel}
                        onChange={(e) => setConceptLabel(e.target.value)}
                        className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md"
                    />
                </div>
                <div>
                    <label className="block text-sm text-dim mb-2" htmlFor="texts">
                        Text references (one per line, at least two)
                    </label>
                    <textarea
                        id="texts"
                        value={textsRaw}
                        onChange={(e) => setTextsRaw(e.target.value)}
                        rows={6}
                        className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md font-mono text-sm"
                    />
                </div>
                <div className="flex gap-4 items-end">
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="topk">
                            Top-K text-anchor words
                        </label>
                        <input
                            id="topk"
                            type="number"
                            min={1}
                            max={50}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            className="bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md w-24"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ml-auto ${loading
                                ? 'bg-[#27272a] text-dim cursor-not-allowed'
                                : 'bg-[#8b5cf6] hover:bg-[#7c3aed] text-white shadow-[0_0_15px_rgba(139,92,246,0.5)]'
                            }`}
                    >
                        {loading ? 'Comparing…' : 'Run comparison'}
                    </button>
                </div>
            </form>
            {error && <div className="card text-sm text-red-400">{error}</div>}
            {result && <TextResultView result={result} />}
        </>
    );
}

function TextResultView({ result }: { result: AnchorCompareResponse }) {
    const labels = result.entries.map((e) => e.label);
    return (
        <div className="flex flex-col gap-6">
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
            <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
                <div>
                    <label className="block text-sm text-dim mb-2" htmlFor="concept-multi">
                        Concept label
                    </label>
                    <input
                        id="concept-multi"
                        type="text"
                        value={conceptLabel}
                        onChange={(e) => setConceptLabel(e.target.value)}
                        className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md"
                    />
                </div>
                <div>
                    <label className="block text-sm text-dim mb-2" htmlFor="text-multi">
                        Text reference (optional)
                    </label>
                    <input
                        id="text-multi"
                        type="text"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        className="w-full bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md"
                    />
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <FileSlot label="Image" accept="image/*" file={image} onChange={setImage} />
                    <FileSlot label="Audio" accept="audio/*" file={audio} onChange={setAudio} />
                    <FileSlot label="Video" accept="video/*" file={video} onChange={setVideo} />
                </div>
                <div className="flex gap-4 items-end">
                    <div>
                        <label className="block text-sm text-dim mb-2" htmlFor="topk-multi">
                            Top-K text-anchor words
                        </label>
                        <input
                            id="topk-multi"
                            type="number"
                            min={1}
                            max={50}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            className="bg-[#18181b] border border-[#27272a] text-[#f4f4f5] p-2 rounded-md w-24"
                        />
                    </div>
                    <span className="text-xs text-dim self-center">
                        {refsPresent} reference{refsPresent === 1 ? '' : 's'} attached
                    </span>
                    <button
                        type="submit"
                        disabled={loading || refsPresent === 0}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ml-auto ${loading || refsPresent === 0
                            ? 'bg-[#27272a] text-dim cursor-not-allowed'
                            : 'bg-[#8b5cf6] hover:bg-[#7c3aed] text-white shadow-[0_0_15px_rgba(139,92,246,0.5)]'
                            }`}
                    >
                        {loading ? 'Comparing…' : 'Run comparison'}
                    </button>
                </div>
            </form>
            {error && <div className="card text-sm text-red-400">{error}</div>}
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
    return (
        <div>
            <label className="block text-sm text-dim mb-2">{label}</label>
            <input
                type="file"
                accept={accept}
                onChange={(e) => onChange(e.target.files?.[0] ?? null)}
                className="block w-full text-xs text-dim file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-[#27272a] file:text-[#f4f4f5]"
            />
            {file && (
                <div className="text-xs text-dim mt-1 truncate" title={file.name}>
                    {file.name}
                </div>
            )}
        </div>
    );
}

function MultimodalResultView({ result }: { result: AnchorCompareMultimodalResponse }) {
    const labels = result.entries.map((e) => e.modality);
    return (
        <div className="flex flex-col gap-6">
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

function CosineMatrix({
    labels,
    matrix,
}: {
    labels: string[];
    matrix: Record<string, Record<string, number>>;
}) {
    return (
        <div className="card">
            <h3 className="text-sm font-bold uppercase tracking-wide mb-3">Cosine Matrix</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                    <thead>
                        <tr>
                            <th className="text-left p-2 text-dim font-normal"></th>
                            {labels.map((l) => (
                                <th key={l} className="text-center p-2 text-dim font-normal">
                                    {l}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {labels.map((row) => (
                            <tr key={row}>
                                <th className="text-left p-2 text-dim font-normal">{row}</th>
                                {labels.map((col) => {
                                    const v = matrix[row]?.[col] ?? 0;
                                    const bg = cosineHeatColor(v);
                                    return (
                                        <td
                                            key={col}
                                            className="text-center p-2 rounded"
                                            style={{ backgroundColor: bg }}
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
        <div className="card flex flex-col gap-2">
            <div>
                <div className="text-sm font-bold">{title}</div>
                <div className="text-xs text-dim font-mono">{subtitle}</div>
            </div>
            {anchors.length > 0 ? (
                <ul className="text-xs text-dim font-mono space-y-1">
                    {anchors.map((a) => (
                        <li key={a.word} className="flex justify-between">
                            <span>{a.word}</span>
                            <span>{a.similarity.toFixed(3)}</span>
                        </li>
                    ))}
                </ul>
            ) : (
                <div className="text-xs text-dim">No anchor words available.</div>
            )}
        </div>
    );
}

/** Map cosine [-1, 1] to a faint purple gradient for the heatmap.
 * 1.0 → strong purple, 0.0 → near-neutral, < 0 → faded blue. */
function cosineHeatColor(v: number): string {
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped >= 0) {
        const alpha = Math.round(clamped * 0.7 * 255);
        return `rgba(139, 92, 246, ${alpha / 255})`;
    }
    const alpha = Math.round(-clamped * 0.4 * 255);
    return `rgba(59, 130, 246, ${alpha / 255})`;
}
