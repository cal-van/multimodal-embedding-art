import { Link, useLocation } from 'react-router-dom';

export function Sidebar() {
    const location = useLocation();

    const isActive = (path: string) => location.pathname.startsWith(path);

    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <span>🎨</span> Embedding Art
            </div>

            <nav className="sidebar-nav">
                <Link 
                    to="/showcase" 
                    className={`sidebar-link ${isActive('/showcase') ? 'active' : ''}`}
                >
                    <span>✨ Showcase</span>
                </Link>
                <Link 
                    to="/create" 
                    className={`sidebar-link ${isActive('/create') ? 'active' : ''}`}
                >
                    <span>➕ Single-modality</span>
                </Link>
                <Link 
                    to="/anchor-compare" 
                    className={`sidebar-link ${isActive('/anchor-compare') ? 'active' : ''}`}
                >
                    <span>🧭 Anchor Compare</span>
                </Link>

                <div className="sidebar-divider" />

                <Link 
                    to="/gallery" 
                    className={`sidebar-link ${isActive('/gallery') ? 'active' : ''}`}
                >
                    <span>🖼️ Gallery</span>
                </Link>
                <button className="sidebar-link" disabled>
                    ⚙️ Settings
                </button>
            </nav>

            <div style={{ flex: 1 }} />

            <div className="sidebar-footer">
                v0.1.0-alpha
            </div>
        </aside>
    );
}

