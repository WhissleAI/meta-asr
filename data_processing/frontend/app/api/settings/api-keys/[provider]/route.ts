import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/app/api/auth/[...nextauth]/route"; // Updated import path

// Delete an API key
export async function DELETE(
  request: Request,
  { params }: { params: { provider: string } }
) {
  const session = await getServerSession(authOptions);

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const provider = params.provider;

  if (!provider) {
    return NextResponse.json(
      { error: "Provider is required" },
      { status: 400 }
    );
  }

  try {
    await prisma.apiKey.delete({
      where: {
        userId_provider: {
          userId: session.user.id,
          provider: provider,
        },
      },
    });
    return NextResponse.json(
      { message: "API key deleted successfully" },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error deleting API key:", error);
    // Handle cases where the key might not exist (e.g., Prisma.PrismaClientKnownRequestError with code P2025)
    if ((error as any).code === "P2025") {
      return NextResponse.json({ error: "API key not found" }, { status: 404 });
    }
    return NextResponse.json(
      { error: "Failed to delete API key" },
      { status: 500 }
    );
  }
}
