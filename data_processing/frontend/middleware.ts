export { default } from "next-auth/middleware";

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api/auth (Auth.js API routes)
     * - auth (custom auth pages like signin, verify-request)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public assets (files in the public folder)
     */
    "/((?!api/auth|auth|_next/static|_next/image|favicon.ico|public/).*)",
  ],
};
