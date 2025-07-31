import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const fastApiUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000'; // Ensure your FastAPI URL is correctly configured

    const response = await fetch(`${fastApiUrl}/process_gcs_file/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      // Ensure the error structure matches what your frontend expects or handle accordingly
      return NextResponse.json({ error: `FastAPI error: ${response.status} ${errorText}` }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    let errorMessage = 'An unknown error occurred';
    if (error instanceof Error) {
      errorMessage = error.message;
    }
    // Ensure the error structure matches what your frontend expects
    return NextResponse.json({ error: `Proxy error: ${errorMessage}` }, { status: 500 });
  }
}
