import { useState } from 'react';
import { api } from '../services/api';
import type { AnchorCompareResponse } from '../services/api';

/**
 * The text-only Platonic-representation probe.
 *
 * Encode 2+ text references through the canonical LanguageBind space
 * and display:
 * * A pairwise cosine matrix (heatmap-ish).
 * * Per-reference top-K text-anchor readouts side by side.
 *
 * Multipart upload for image / audio / video references is a planned
 * follow-up.
 */
export function AnchorComparePage() {
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
        <div className="max-w-5xl mx-auto flex flex-col gap-6">
            <div>
                <h1 className="text-2xl font-bold">Anchor Comparison</h1>
                <p className="text-sm text-dim mt-1">
                    Encode multiple text references through the canonical LanguageBind embedding
                    space and compare them. Cosine matrix tells you whether semantically similar
                    references land at the same point; per-reference text-anchor readouts show
                    what the model thinks each text "means" in language space.
                </p>
            </div>

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

            {error && (
                <div className="card text-sm text-red-400">{error}</div>
            )}

            {result && <AnchorCompareResultView result={result} />}
        </div>
    );
}

function AnchorCompareResultView({ result }: { result: AnchorCompareResponse }) {
    const labels = result.entries.map((e) => e.label);
    return (
        <div className="flex flex-col gap-6">
            <CosineMatrix labels={labels} matrix={result.cosine_matrix} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.entries.map((entry) => (
                    <AnchorReadoutCard key={entry.label} entry={entry} />
                ))}
            </div>
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
    entry,
}: {
    entry: AnchorCompareResponse['entries'][number];
}) {
    return (
        <div className="card flex flex-col gap-2">
            <div>
                <div className="text-sm font-bold">{entry.label}</div>
                <div className="text-xs text-dim font-mono">
                    "{entry.text}" · dim={entry.embedding_dim}
                </div>
            </div>
            {entry.text_anchor.length > 0 ? (
                <ul className="text-xs text-dim font-mono space-y-1">
                    {entry.text_anchor.map((a) => (
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
