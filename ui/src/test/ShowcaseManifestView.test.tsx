import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ShowcaseManifestView } from '../components/ShowcaseManifestView';
import { api } from '../services/api';

// Mock the api module: the view only needs fetchShowcaseManifest + absoluteUrl.
vi.mock('../services/api', () => ({
    api: {
        fetchShowcaseManifest: vi.fn(),
        absoluteUrl: (p: string) => `http://127.0.0.1:8000${p}`,
    },
}));

// A real single-track manifest, matching what `embed-art showcase` writes:
// `concept` object, `modalities` as an OBJECT (not array), dir-relative `path`.
const SINGLE_TRACK = {
    concept: { text: 'thunder', description: 'thunder', embedding_dim: 768 },
    encoder: 'languagebind',
    device: 'mps',
    steps: 200,
    seed: 42,
    track: 'realism-0.30',
    realism: 0.3,
    image_backbone: 'sd35',
    modalities: {
        image: {
            path: 'image.png',
            final_similarity: 0.1388,
            backbone: 'sd35',
            interpretation: {
                text_anchor: [{ word: 'chaos', similarity: 0.18 }],
            },
        },
    },
    evaluation: {
        per_modality_similarity: { image: 0.1388 },
        cross_modal_text_anchor_agreement_jaccard: {},
        encoder: 'languagebind',
    },
};

// A real dual-track top manifest: NO top-level `modalities`; `per_track` with summaries.
const DUAL_TRACK = {
    concept: { text: 'ocean', description: 'ocean', embedding_dim: 768 },
    encoder: 'languagebind',
    steps: 200,
    seed: 7,
    tracks: ['honest', 'natural'],
    per_track: {
        honest: {
            path: 'honest',
            manifest: 'honest/manifest.json',
            summary: { image: { path: 'image.png', final_similarity: 0.42, backbone: 'sd35' } },
        },
        natural: {
            path: 'natural',
            manifest: 'natural/manifest.json',
            summary: { image: { path: 'image.png', final_similarity: 0.31, backbone: 'sd35' } },
        },
    },
};

describe('ShowcaseManifestView', () => {
    beforeEach(() => {
        vi.mocked(api.fetchShowcaseManifest).mockReset();
    });

    it('renders a single-track manifest without throwing on the modalities object', async () => {
        vi.mocked(api.fetchShowcaseManifest).mockResolvedValue(SINGLE_TRACK as never);
        render(<ShowcaseManifestView manifestUrl="/outputs/showcase/abc/manifest.json" />);

        // Concept comes from concept.text (not a top-level target_text).
        await waitFor(() => expect(screen.getByText('thunder')).toBeInTheDocument());
        // The per-modality render under `modalities` must actually display (as a card heading).
        expect(screen.getByRole('heading', { name: 'image' })).toBeInTheDocument();
        // Similarity readout from final_similarity.
        expect(screen.getByText(/sim 0\.139/)).toBeInTheDocument();
        // Image src resolved relative to the manifest directory.
        const img = screen.getByRole('img') as HTMLImageElement;
        expect(img.src).toContain('/outputs/showcase/abc/image.png');
    });

    it('renders a dual-track manifest with both endpoints', async () => {
        vi.mocked(api.fetchShowcaseManifest).mockResolvedValue(DUAL_TRACK as never);
        render(<ShowcaseManifestView manifestUrl="/outputs/showcase/xyz/manifest.json" />);

        await waitFor(() => expect(screen.getByText('ocean')).toBeInTheDocument());
        expect(screen.getByText('HONEST')).toBeInTheDocument();
        expect(screen.getByText('NATURAL')).toBeInTheDocument();
    });
});
