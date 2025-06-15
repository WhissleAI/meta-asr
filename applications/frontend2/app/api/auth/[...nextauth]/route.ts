import NextAuth, {
  AuthOptions,
  SessionStrategy,
  User as NextAuthUser,
  Session,
} from "next-auth";
import { JWT } from "next-auth/jwt"; // Corrected JWT import
import { PrismaAdapter } from "@auth/prisma-adapter";
import EmailProvider from "next-auth/providers/email";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google"; // Add this line
import GitHubProvider from "next-auth/providers/github"; // Add this line
import prisma from "@/lib/prisma";
import nodemailer from "nodemailer";
import bcrypt from "bcrypt";

export const authOptions: AuthOptions = {
  adapter: PrismaAdapter(prisma),
  providers: [
    GoogleProvider({
      // Add GoogleProvider configuration
      clientId: process.env.GOOGLE_CLIENT_ID as string,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
    }),
    GitHubProvider({
      // Add GitHubProvider configuration
      clientId: process.env.GITHUB_CLIENT_ID as string,
      clientSecret: process.env.GITHUB_CLIENT_SECRET as string,
    }),
    EmailProvider({
      server: {
        host: process.env.EMAIL_SERVER_HOST,
        port: Number(process.env.EMAIL_SERVER_PORT),
        auth: {
          user: process.env.EMAIL_SERVER_USER,
          pass: process.env.EMAIL_SERVER_PASSWORD,
        },
      },
      from: process.env.EMAIL_FROM,
      async sendVerificationRequest({
        identifier: email,
        url,
        provider: { server, from },
      }) {
        const { host } = new URL(url);
        const transport = nodemailer.createTransport(server);
        await transport.sendMail({
          to: email,
          from,
          subject: `Sign in to ${host}`,
          text: `Sign in to ${host}\n\n${url}\n\n`,
          html: `<p>Sign in to <strong>${host}</strong></p><p><a href="${url}">Click here to sign in</a></p>`,
        });
      },
    }),
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials): Promise<NextAuthUser | null> {
        if (!credentials?.email || !credentials.password) {
          return null;
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        if (user && user.password) {
          const isValidPassword = await bcrypt.compare(
            credentials.password,
            user.password
          );
          if (isValidPassword) {
            return {
              id: user.id,
              name: user.name,
              email: user.email,
              image: user.image,
            };
          }
        }
        return null;
      },
    }),
  ],
  session: {
    strategy: "jwt" as SessionStrategy,
  },
  callbacks: {
    async session({ session, token }: { session: Session; token: JWT }) {
      // Used Session type
      if (token?.sub && session.user) {
        (session.user as { id?: string }).id = token.sub; // Type assertion for adding id
      }
      return session;
    },
    async jwt({ token, user }: { token: JWT; user?: NextAuthUser }) {
      if (user) {
        token.sub = user.id;
      }
      return token;
    },
  },
  pages: {
    signIn: "/auth/signin",
    verifyRequest: "/auth/verify-request",
  },
  secret: process.env.NEXTAUTH_SECRET,
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
