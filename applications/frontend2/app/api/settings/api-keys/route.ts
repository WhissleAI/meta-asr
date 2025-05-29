import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/app/api/auth/[...nextauth]/route"; // Updated import path

// Get API keys for the logged-in user
export async function GET() {
  const session = await getServerSession(authOptions);

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const apiKeys = await prisma.apiKey.findMany({
      where: { userId: session.user.id },
      orderBy: { provider: "asc" },
    });
    return NextResponse.json(apiKeys);
  } catch (error) {
    console.error("Error fetching API keys:", error);
    return NextResponse.json(
      { error: "Failed to fetch API keys" },
      { status: 500 }
    );
  }
}

// Create or Update an API key
export async function POST(request: Request) {
  const session = await getServerSession(authOptions);

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const { provider, key } = await request.json();

    if (!provider || !key) {
      return NextResponse.json(
        { error: "Provider and key are required" },
        { status: 400 }
      );
    }

    // IMPORTANT: In a real application, the 'key' should be encrypted before saving to the database.
    // For simplicity in this example, it's stored as plain text.

    const upsertedApiKey = await prisma.apiKey.upsert({
      where: {
        userId_provider: {
          userId: session.user.id,
          provider: provider,
        },
      },
      update: {
        key: key,
      },
      create: {
        userId: session.user.id,
        provider: provider,
        key: key,
      },
    });

    return NextResponse.json(upsertedApiKey, { status: 201 });
  } catch (error) {
    console.error("Error saving API key:", error);
    // Check for unique constraint violation if needed, though upsert handles it.
    return NextResponse.json(
      { error: "Failed to save API key" },
      { status: 500 }
    );
  }
}
