import { Link, useLocation } from 'react-router-dom';

const NAV = [
    { idx: '01', to: '/showcase', label: 'Showcase' },
    { idx: '02', to: '/create', label: 'Single-modality' },
    { idx: '03', to: '/anchor-compare', label: 'Anchor Compare' },
];

export function Sidebar() {
    const location = useLocation();
    const isActive = (path: string) => location.pathname.startsWith(path);

    return (
        <aside className="rail">
            <Link to="/showcase" className="rail-mark">
                <span className="glyph">
                    Latent<br />
                    <em>Instrument</em>
                </span>
                <span className="sub">Embedding Art</span>
            </Link>

            <nav className="rail-nav">
                {NAV.map((n) => (
                    <Link
                        key={n.to}
                        to={n.to}
                        className={`nav-item ${isActive(n.to) ? 'active' : ''}`}
                    >
                        <span className="idx">{n.idx}</span>
                        <span>{n.label}</span>
                    </Link>
                ))}

                <div className="rail-rule" />

                <Link
                    to="/gallery"
                    className={`nav-item ${isActive('/gallery') ? 'active' : ''}`}
                >
                    <span className="idx">04</span>
                    <span>Gallery</span>
                </Link>
                <button className="nav-item" disabled>
                    <span className="idx">05</span>
                    <span>Settings</span>
                </button>
            </nav>

            <div className="rail-foot">
                <div className="status">
                    <span className="dot" />
                    Engine ready
                </div>
                <div className="status" style={{ color: 'var(--text-faint)' }}>
                    v0.1.0 · 768-d
                </div>
            </div>
        </aside>
    );
}
