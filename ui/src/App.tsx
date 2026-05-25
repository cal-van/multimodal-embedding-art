import { BrowserRouter, Route, Routes, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { CreatePage } from './pages/CreatePage';
import { JobPage } from './pages/JobPage';
import { GalleryPage } from './pages/GalleryPage';
import { ShowcasePage } from './pages/ShowcasePage';
import { AnchorComparePage } from './pages/AnchorComparePage';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/showcase" replace />} />
          <Route path="/showcase" element={<ShowcasePage />} />
          <Route path="/anchor-compare" element={<AnchorComparePage />} />
          <Route path="/create" element={<CreatePage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/gallery" element={<GalleryPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
