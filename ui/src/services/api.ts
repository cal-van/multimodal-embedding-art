const API_BASE = 'http://127.0.0.1:8000';

export interface Job {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  error: string | null;
  logs: string[];
  result_url?: string;
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
};
