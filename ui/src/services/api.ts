const API_BASE = 'http://127.0.0.1:8000';

export interface Job {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  error: string | null;
  logs: string[];
  result_url?: string;
  encoder_name?: string;
  similarity_weight?: number;
  feature_matching_weight?: number;
  kind?: 'single' | 'compare' | 'showcase';
  manifest_url?: string | null;
  tracks?: string[];
  modalities?: string[];
  target_text?: string[];
  output_modality?: string;
}

export interface ShowcaseRequest {
  target_text: string;
  modalities?: string[];
  encoder_name?: string;
  image_backbone?: 'sd35' | 'sdxl';
  audio_backbone?: 'stable-audio-open' | 'audioldm2';
  video_backbone?: 'ltx-video' | 'svd';
  tracks?: Array<'honest' | 'natural'>;
  autocast_dtype?: 'fp32' | 'fp16' | 'bf16';
  interpret?: boolean;
  evaluate?: boolean;
  sae_path?: string | null;
  steps?: number;
  seed?: number | null;
}

/** Top-level structure of the JSON written to outputs/showcase/<id>/manifest.json. */
export interface ShowcaseManifest {
  target_text: string;
  encoder: string;
  modalities: string[];
  tracks?: string[];
  // Single-track layout: per-modality renders sit directly here.
  renders?: Record<string, ShowcaseModalityRender>;
  // Dual-track layout: track-name -> per-track manifest summary.
  per_track?: Record<string, ShowcaseTrackSummary>;
  evaluation?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface ShowcaseTrackSummary {
  modalities: string[];
  output_dir?: string;
  similarity_summary?: Record<string, number>;
  [key: string]: unknown;
}

export interface ShowcaseModalityRender {
  output_file?: string;
  similarity?: number;
  steps?: number;
  interpretation?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface AnchorCompareTextEntry {
  label: string;
  text: string;
  embedding_dim: number;
  text_anchor: Array<{ word: string; similarity: number }>;
}

export interface AnchorCompareResponse {
  concept_label: string;
  encoder: string;
  entries: AnchorCompareTextEntry[];
  cosine_matrix: Record<string, Record<string, number>>;
}

export interface AnchorCompareMultimodalEntry {
  modality: string;
  embedding_dim: number;
  text_anchor: Array<{ word: string; similarity: number }>;
  source: string | null;
}

export interface AnchorCompareMultimodalResponse {
  concept_label: string;
  encoder: string;
  entries: AnchorCompareMultimodalEntry[];
  cosine_matrix: Record<string, Record<string, number>>;
}

export const api = {
  createJob: async (targetText: string[], outputModality: string = 'image'): Promise<Job> => {
    const res = await fetch(`${API_BASE}/jobs/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_text: targetText, output_modality: outputModality }),
    });
    if (!res.ok) throw new Error('Failed to create job');
    return res.json();
  },

  createShowcaseJob: async (request: ShowcaseRequest): Promise<Job> => {
    const res = await fetch(`${API_BASE}/jobs/showcase`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Failed to create showcase job: ${detail}`);
    }
    return res.json();
  },

  listJobs: async (): Promise<Job[]> => {
    const res = await fetch(`${API_BASE}/jobs/`);
    if (!res.ok) throw new Error('Failed to list jobs');
    return res.json();
  },

  getJob: async (id: string): Promise<Job> => {
    const res = await fetch(`${API_BASE}/jobs/${id}`);
    if (!res.ok) throw new Error('Failed to get job');
    return res.json();
  },

  fetchShowcaseManifest: async (manifestUrl: string): Promise<ShowcaseManifest> => {
    const res = await fetch(`${API_BASE}${manifestUrl}`);
    if (!res.ok) throw new Error(`Failed to fetch manifest at ${manifestUrl}`);
    return res.json();
  },

  anchorCompare: async (request: {
    concept_label: string;
    texts: string[];
    encoder?: string;
    top_k_text?: number;
  }): Promise<AnchorCompareResponse> => {
    const res = await fetch(`${API_BASE}/experiments/anchor-compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`anchor-compare failed: ${detail}`);
    }
    return res.json();
  },

  anchorCompareMultimodal: async (request: {
    concept_label: string;
    encoder?: string;
    top_k_text?: number;
    text?: string | null;
    image?: File | null;
    audio?: File | null;
    video?: File | null;
  }): Promise<AnchorCompareMultimodalResponse> => {
    const form = new FormData();
    form.append('concept_label', request.concept_label);
    if (request.encoder) form.append('encoder', request.encoder);
    if (request.top_k_text !== undefined) {
      form.append('top_k_text', String(request.top_k_text));
    }
    if (request.text) form.append('text', request.text);
    if (request.image) form.append('image', request.image);
    if (request.audio) form.append('audio', request.audio);
    if (request.video) form.append('video', request.video);
    const res = await fetch(`${API_BASE}/experiments/anchor-compare-multimodal`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`anchor-compare-multimodal failed: ${detail}`);
    }
    return res.json();
  },

  absoluteUrl: (path: string): string => `${API_BASE}${path}`,
};
