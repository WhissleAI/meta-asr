'use client';
import React, { useEffect, useState } from 'react';
import { loadPrompts } from '@/utils/loadPrompt';

export default function Home() {
    const [prompts, setPrompts] = useState<{ name: string; content: string; }[]>([]);

    useEffect(() => {
        loadPrompts().then(setPrompts);
    }, []);

    return (
        <div>
            <h1>Prompt Viewer</h1>
            {prompts.map(({ name, content }) => (
                <div key={name} style={{ marginBottom: '2rem' }}>
                    <h3>{name}</h3>
                    <textarea
                        value={content}
                        rows={15}
                        cols={100}
                        readOnly
                    />
                </div>
            ))}
        </div>
    );
}
