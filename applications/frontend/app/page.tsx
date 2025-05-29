'use client';

import { useState } from 'react';
import axios from 'axios';

interface ProcessResponse {
  message: string;
  output_file: string;
  processed_files: number;
  saved_records: number;
  errors: number;
}

export default function Home() {
  const [directoryPath, setDirectoryPath] = useState('');
  const [modelChoice, setModelChoice] = useState<'gemini' | 'whissle'>('gemini');
  const [outputPath, setOutputPath] = useState('');
  const [response, setResponse] = useState<ProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (endpoint: string) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await axios.post<ProcessResponse>(
        `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`,
        {
          directory_path: directoryPath,
          model_choice: modelChoice,
          output_jsonl_path: outputPath,
        }
      );
      setResponse(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto p-6">
      <h2 className="text-xl font-semibold">Process Audio Files</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium">Directory Path</label>
          <input
            type="text"
            value={directoryPath}
            onChange={(e) => setDirectoryPath(e.target.value)}
            placeholder="/path/to/audio"
            className="mt-1 block w-full border rounded p-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium">Model Choice</label>
          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value as 'gemini' | 'whissle')}
            className="mt-1 block w-full border rounded p-2"
          >
            <option value="gemini">Gemini</option>
            <option value="whissle">Whissle</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium">Output JSONL Path</label>
          <input
            type="text"
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            placeholder="/path/to/output.jsonl"
            className="mt-1 block w-full border rounded p-2"
          />
        </div>
        <div className="flex space-x-4">
          <button
            onClick={() => handleSubmit('/create_transcription_manifest/')}
            disabled={isLoading}
            className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
          >
            {isLoading ? 'Processing...' : 'Transcribe Only'}
          </button>
          <button
            onClick={() => handleSubmit('/create_annotated_manifest/')}
            disabled={isLoading}
            className="bg-green-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
          >
            {isLoading ? 'Processing...' : 'Transcribe & Annotate'}
          </button>
        </div>
      </div>
      {response && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h3 className="font-semibold">Result</h3>
          <p>{response.message}</p>
          <p>Output File: {response.output_file}</p>
          <p>Processed: {response.processed_files}</p>
          <p>Saved: {response.saved_records}</p>
          <p>Errors: {response.errors}</p>
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