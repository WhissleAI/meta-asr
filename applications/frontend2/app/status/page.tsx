'use client';

import { useState, useEffect } from 'react';

interface StatusResponse {
    message: string;
    docs_url: string;
    html_interface: string;
    endpoints: {
        transcription_only: string;
        full_annotation: string;
    };
    gemini_configured: boolean;
    whissle_available: boolean;
    whissle_configured: boolean;
    age_gender_model_loaded: boolean;
    emotion_model_loaded: boolean;
    device: string;
}

export default function StatusPage() {
    const [status, setStatus] = useState<StatusResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchStatus = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/status`);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const data = await response.json();
                setStatus(data);
            } catch (err: any) {
                setError(err.message || 'Failed to fetch status');
            } finally {
                setIsLoading(false);
            }
        };
        fetchStatus();
    }, []);

    return (
        <div className="container p-4 max-w-7xl mx-auto space-y-6">
            <h2 className="text-xl font-semibold mb-4">API Status</h2>
            {isLoading && <p className="text-gray-600">Loading...</p>}
            {error && (
                <div className="p-4 bg-red-100 text-red-700 rounded">
                    <h3 className="font-semibold">Error</h3>
                    <p>{error}</p>
                </div>
            )}
            {status && (
                <div className="p-4 bg-gray-100 rounded space-y-4">
                    <div>
                        <h3 className="font-semibold">Message</h3>
                        <p>{status.message}</p>
                    </div>
                    <div>
                        <h3 className="font-semibold">Endpoints</h3>
                        <ul className="list-disc pl-5">
                            <li>Documentation: <a href={status.docs_url} className="text-blue-500 underline">{status.docs_url}</a></li>
                            <li>HTML Interface: <a href={status.html_interface} className="text-blue-500 underline">{status.html_interface}</a></li>
                            <li>Transcription Only: <span className="text-blue-500">{status.endpoints.transcription_only}</span></li>
                            <li>Full Annotation: <span className="text-blue-500">{status.endpoints.full_annotation}</span></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-semibold">Configuration</h3>
                        <ul className="list-disc pl-5">
                            <li>Gemini Configured: <span className={status.gemini_configured ? 'text-green-600' : 'text-red-600'}>{status.gemini_configured ? 'Yes' : 'No'}</span></li>
                            <li>Whissle Available: <span className={status.whissle_available ? 'text-green-600' : 'text-red-600'}>{status.whissle_available ? 'Yes' : 'No'}</span></li>
                            <li>Whissle Configured: <span className={status.whissle_configured ? 'text-green-600' : 'text-red-600'}>{status.whissle_configured ? 'Yes' : 'No'}</span></li>
                            <li>Age/Gender Model Loaded: <span className={status.age_gender_model_loaded ? 'text-green-600' : 'text-red-600'}>{status.age_gender_model_loaded ? 'Yes' : 'No'}</span></li>
                            <li>Emotion Model Loaded: <span className={status.emotion_model_loaded ? 'text-green-600' : 'text-red-600'}>{status.emotion_model_loaded ? 'Yes' : 'No'}</span></li>
                            <li>Device: <span className="text-blue-500">{status.device}</span></li>
                        </ul>
                    </div>
                </div>
            )}
        </div>
    );
}