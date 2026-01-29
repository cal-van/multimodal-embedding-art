import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { CreatePage } from '../pages/CreatePage';
import { JobPage } from '../pages/JobPage';
import { api } from '../services/api';

// Mock API
vi.mock('../services/api', () => ({
    api: {
        createJob: vi.fn(),
        getJob: vi.fn(),
    },
}));

describe('CreatePage', () => {
    it('renders correctly', () => {
        render(
            <MemoryRouter>
                <CreatePage />
            </MemoryRouter>
        );
        expect(screen.getByText('Create New Optimization')).toBeInTheDocument();
        expect(screen.getByText('Start Optimization')).toBeInTheDocument();
    });

    it('submits form', async () => {
        const mockCreateJob = vi.mocked(api.createJob);
        mockCreateJob.mockResolvedValue({ id: '123', status: 'queued', progress: 0, error: null, logs: [] });

        render(
            <MemoryRouter>
                <CreatePage />
            </MemoryRouter>
        );

        const input = screen.getByPlaceholderText('e.g. goldfish, fire + water');
        fireEvent.change(input, { target: { value: 'test concept' } });

        const button = screen.getByText('Start Optimization');
        fireEvent.click(button);

        await waitFor(() => {
            expect(mockCreateJob).toHaveBeenCalledWith(['test concept'], 'image');
        });
    });
});

describe('JobPage', () => {
    it('polls for status', async () => {
        const mockGetJob = vi.mocked(api.getJob);
        mockGetJob.mockResolvedValue({
            id: '123',
            status: 'running',
            progress: 0.5,
            error: null,
            logs: ['Starting...']
        });

        render(
            <MemoryRouter initialEntries={['/job/123']}>
                <Routes>
                    <Route path="/job/:id" element={<JobPage />} />
                </Routes>
            </MemoryRouter>
        );

        await waitFor(() => {
            expect(screen.getByText('RUNNING')).toBeInTheDocument();
            expect(screen.getByText('50%')).toBeInTheDocument();
        });

        expect(mockGetJob).toHaveBeenCalledWith('123');
    });
});
