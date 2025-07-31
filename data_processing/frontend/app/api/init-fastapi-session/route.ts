import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import prisma from "@/lib/prisma";

const FASTAPI_URL = process.env.NEXT_PUBLIC_API_URL; // Your FastAPI backend URL

export async function POST() {
  const session = await getServerSession(authOptions);

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    // 1. Fetch user's API keys from Next.js app's database
    const userApiKeysFromDb = await prisma.apiKey.findMany({
      where: { userId: userId },
      select: { provider: true, key: true }, // Select only necessary fields
    });

    if (!userApiKeysFromDb || userApiKeysFromDb.length === 0) {
      // No keys stored, but still might want to inform FastAPI about the user session
      // Or decide if an empty init is useful. For now, let's proceed.
      console.log(
        `No API keys found in DB for user ${userId} to initialize FastAPI session.`
      );
      // Depending on requirements, you might return success or a specific message.
      // For now, we'll allow initializing with an empty list if no keys are found.
    }

    // 2. Prepare data for FastAPI
    const fastapiSessionData = {
      user_id: userId,
      api_keys: userApiKeysFromDb.map((ak) => ({
        provider: ak.provider,
        key: ak.key,
      })),
    };

    // 3. Call FastAPI's /init_session endpoint
    const fastapiResponse = await fetch(`${FASTAPI_URL}/init_session/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Add any other necessary headers, e.g., an internal auth token if FastAPI is protected
      },
      body: JSON.stringify(fastapiSessionData),
    });

    if (!fastapiResponse.ok) {
      const errorData = await fastapiResponse.json();
      console.error(
        `FastAPI session init failed for user ${userId}:`,
        errorData
      );
      return NextResponse.json(
        {
          error: "Failed to initialize session with backend service",
          details: errorData.detail || "Unknown error",
        },
        { status: fastapiResponse.status }
      );
    }

    const responseData = await fastapiResponse.json();
    return NextResponse.json({
      message: "FastAPI session initialized successfully",
      details: responseData,
    });
  } catch (error) {
    console.error(
      `Error initializing FastAPI session for user ${userId}:`,
      error
    );
    return NextResponse.json(
      { error: "Internal server error during FastAPI session initialization" },
      { status: 500 }
    );
  }
}
