import { Link, useLocation } from 'react-router-dom';

export function Sidebar() {
    const location = useLocation();

    const isActive = (path: string) => location.pathname.startsWith(path);

    return (
        <aside className="w-64 border-r border-[#27272a] bg-[#18181b] flex flex-col p-4 gap-4">
            <div className="text-xl font-bold flex items-center gap-2">
                <span>🎨</span> Embedding Art
            </div>

            <nav className="flex flex-col gap-2">
                <Link to="/create">
                    <button className={`w-full text-left ${isActive('/create') ? 'primary' : ''}`}>
                        + Create
                    </button>
                </Link>

                <div className="h-px bg-[#27272a] my-2" />

                <Link to="/gallery">
                    <button className={`w-full text-left ${isActive('/gallery') ? 'bg-[#27272a]' : ''}`}>
                        Gallery
                    </button>
                </Link>
                <button className="text-left w-full" disabled>
                    Settings
                </button>
            </nav>

            <div className="flex-1" />

            <div className="text-sm text-dim">
                v0.1.0-alpha
            </div>
        </aside>
    );
}
