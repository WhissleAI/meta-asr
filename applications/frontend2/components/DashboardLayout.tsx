'use client';

import { UserNav } from "@/components/UserNav";
import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

interface DashboardLayoutProps {
    children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
    return (
        <div className="flex flex-col min-h-screen">
            {/* Header */}
            <header className="flex h-16 items-center justify-between border-b bg-background px-4 lg:px-6 sticky top-0 z-10">
                {/* Left: Logo / App Name */}
                <Link href="/" className="text-lg md:text-xl font-semibold">
                    Whissle
                </Link>

                {/* Center: Model name */}
                <div className="text-base font-medium md:text-lg">
                    Meta ASR
                </div>

                {/* Right: User nav */}
                <div className="flex items-center gap-2">
                    <ThemeToggle />
                    <UserNav />
                </div>
            </header>

            {/* Main content */}
            <main className="flex-1 overflow-hidden">
                {children}
            </main>
        </div>
    );
}
