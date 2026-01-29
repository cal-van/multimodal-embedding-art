import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';

interface LayoutProps {
    children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
    return (
        <div className="flex h-full">
            <Sidebar />
            <main className="flex-1 p-4 overflow-auto">
                {children}
            </main>
        </div>
    );
}
