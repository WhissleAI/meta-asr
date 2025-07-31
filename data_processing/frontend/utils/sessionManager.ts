// utils/sessionManager.ts
import { toast } from "sonner";

export async function initFastApiUserSession() {
  try {
    const response = await fetch("/api/init-fastapi-session", {
      method: "POST",
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Failed to initialize FastAPI session:", errorData);
      toast.error("Failed to sync API keys with backend service.", {
        description:
          errorData.details ||
          errorData.error ||
          "Please try again or re-login.",
      });
      return false;
    }

    const responseData = await response.json();
    console.log(
      "FastAPI session initialized/updated successfully:",
      responseData.details?.message || responseData.message
    );
    // toast.success("API keys synced with backend service."); // Optional: can be too noisy
    return true;
  } catch (error) {
    console.error("Error calling /api/init-fastapi-session:", error);
    toast.error("Client-side error syncing API keys.", {
      description:
        (error as Error).message ||
        "Please check your connection and try again.",
    });
    return false;
  }
}
