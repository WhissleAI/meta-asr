import NextAuth, { DefaultSession, DefaultUser } from "next-auth";
import { JWT } from "next-auth/jwt";

declare module "next-auth" {
  interface Session {
    user: {
      id?: string; // Add your custom property id
    } & DefaultSession["user"]; // Keep the default properties
  }

  interface User extends DefaultUser {
    // You can add other custom properties to the User object if needed
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    sub?: string; // The 'sub' property usually holds the user ID
    // You can add other custom properties to the JWT token if needed
  }
}
