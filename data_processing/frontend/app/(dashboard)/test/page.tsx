'use client';

import { useState } from 'react';

interface ProcessResponse {
    message: string;
    output_file: string;
    processed_files: number;
    saved_records: number;
    errors: number;
}

export default function Home() {
    const [directoryPath, setDirectoryPath] = useState('');
    const [modelChoice, setModelChoice] = useState<'gemini' | 'whissle' | 'deepgram'>('gemini');
    const [outputPath, setOutputPath] = useState('');
    const [annotations, setAnnotations] = useState({
        age: false,
        gender: false,
        emotion: false,
        entity: false,
        intent: false,
    });
    const [response, setResponse] = useState<ProcessResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleAnnotationChange = (key: keyof typeof annotations) => {
        setAnnotations((prev) => ({ ...prev, [key]: !prev[key] }));
    };

    const handleSubmit = async () => {
        setIsLoading(true);
        setError(null);
        setResponse(null);

        const selectedAnnotations = Object.keys(annotations).filter(
            (key) => annotations[key as keyof typeof annotations]
        );

        const endpoint = selectedAnnotations.length > 0
            ? '/create_annotated_manifest/'
            : '/create_transcription_manifest/';

        try {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    directory_path: directoryPath,
                    model_choice: modelChoice,
                    output_jsonl_path: outputPath,
                    ...(selectedAnnotations.length > 0 && { annotations: selectedAnnotations }),
                }),
            });
            if (!res.ok) {
                throw new Error(`HTTP error! Status: ${res.status}`);
            }
            const data = await res.json();
            setResponse(data);
        } catch (err: any) {
            setError(err.message || 'An error occurred');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-4 space-y-6">
            <h2 className="text-xl font-semibold">Process Audio Files</h2>
            <p className="text-sm text-gray-600">
                Enter a server-side directory path accessible to the backend (e.g., /home/dchauhan/workspace/test).
            </p>
            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium">Directory Path</label>
                    <input
                        type="text"
                        value={directoryPath}
                        onChange={(e) => setDirectoryPath(e.target.value)}
                        placeholder="/home/dchauhan/workspace/test"
                        className="mt-1 block w-full border rounded p-2"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Model Choice</label>
                    <select
                        value={modelChoice}
                        onChange={(e) => setModelChoice(e.target.value as 'gemini' | 'whissle' | 'deepgram')}
                        className="mt-1 block w-full border rounded p-2"
                    >
                        <option value="gemini">Gemini</option>
                        <option value="whissle">Whissle</option>
                        <option value="deepgram">Deepgram</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium">Output JSONL Path</label>
                    <input
                        type="text"
                        value={outputPath}
                        onChange={(e) => setOutputPath(e.target.value)}
                        placeholder="/home/dchauhan/workspace/test/test2.jsonl"
                        className="mt-1 block w-full border rounded p-2"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Annotations</label>
                    <div className="mt-2 space-y-2">
                        {Object.keys(annotations).map((key) => (
                            <label key={key} className="flex items-center">
                                <input
                                    type="checkbox"
                                    checked={annotations[key as keyof typeof annotations]}
                                    onChange={() => handleAnnotationChange(key as keyof typeof annotations)}
                                    className="mr-2"
                                    disabled={modelChoice === 'deepgram'} // Disable annotations for Deepgram
                                />
                                {key.charAt(0).toUpperCase() + key.slice(1)}
                            </label>
                        ))}
                    </div>
                </div>
                <button
                    onClick={handleSubmit}
                    disabled={isLoading}
                    className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
                >
                    {isLoading ? 'Processing...' : 'Process'}
                </button>
            </div>
            {response && (
                <div className="mt-4 p-4 bg-gray-100 rounded">
                    <h3 className="font-semibold">Result</h3>
                    <p>{response.message}</p>
                    <p>Output File: {response.output_file}</p>
                    <p>Processed: {response.processed_files}</p>
                    <p>Saved: {response.saved_records}</p>
                    <p>Errors: {response.errors}</p>
                    <p>Model Used: {modelChoice.charAt(0).toUpperCase() + modelChoice.slice(1)}</p>
                    {Object.values(annotations).some((v) => v) && (
                        <div>
                            <h4 className="font-semibold mt-2">Selected Annotations:</h4>
                            <ul className="list-disc pl-5">
                                {Object.keys(annotations)
                                    .filter((key) => annotations[key as keyof typeof annotations])
                                    .map((key) => (
                                        <li key={key}>{key.charAt(0).toUpperCase() + key.slice(1)}</li>
                                    ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
            {error && (
                <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
                    <h3 className="font-semibold">Error</h3>
                    <p>{error}</p>
                </div>
            )}
        </div>
    );
}