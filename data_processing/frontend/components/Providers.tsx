'use client';

import { SessionProvider } from 'next-auth/react';
import * as React from 'react';
import { ThemeProvider as NextThemesProvider } from 'next-themes';
// import { type ThemeProviderProps } from 'next-themes/dist/types';

interface ProvidersProps {
    children: React.ReactNode;
}

export default function Providers({ children, ...props }: React.ComponentProps<typeof NextThemesProvider> & ProvidersProps) {
    return (
        <NextThemesProvider {...props}>
            <SessionProvider>{children}</SessionProvider>
        </NextThemesProvider>
    );
}
