# Next.js Auth & Dashboard Starter

This is a [Next.js](https://nextjs.org/) project demonstrating a comprehensive authentication setup using [NextAuth.js](https://next-auth.js.org/) and a dashboard interface built with [Shadcn UI](https://ui.shadcn.com/).

## Features

- **Authentication:**
  - Credentials (Email & Password) sign-up and sign-in.
  - Email verification via magic link (using Nodemailer with a test service like Mailtrap).
  - OAuth sign-in/sign-up with Google.
  - OAuth sign-in/sign-up with GitHub.
  - Session management with database persistence (PostgreSQL via Prisma).
  - Protected routes using Next.js Middleware.
- **Database:**
  - [Prisma ORM](https://www.prisma.io/) for database interactions.
  - PostgreSQL database (configured via `DATABASE_URL` environment variable).
- **UI & UX:**
  - Modern UI components from [Shadcn UI](https://ui.shadcn.com/).
  - Dashboard layout with a sidebar and header.
  - User profile and settings accessible via dialog pop-ups from a user navigation menu.
  - Separate pages for Sign In and Sign Up.
- **Project Structure:**
  - Next.js App Router.
  - Clear separation of API routes, UI components, and utility functions.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (version 18.x or later recommended)
- [npm](https://www.npmjs.com/), [yarn](https://yarnpkg.com/), or [pnpm](https://pnpm.io/)
- A PostgreSQL database instance.
- Google Cloud Platform project for Google OAuth credentials.
- GitHub account for GitHub OAuth App credentials.
- An email testing service like [Mailtrap](https://mailtrap.io/) (or your own SMTP server) for email verification.

### 1. Clone the Repository (if applicable)

```bash
# If you have this project in a Git repository:
git clone <your-repository-url>
cd <project-directory>
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install
# or
pnpm install
```

### 3. Set Up Environment Variables

Create a `.env` file in the root of your project by copying the example below. **Do not commit your `.env` file to version control.**

```env
# Prisma / Database
DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require"

# NextAuth.js
# Generate a strong secret: openssl rand -base64 32
NEXTAUTH_SECRET="YOUR_NEXTAUTH_SECRET"
NEXTAUTH_URL="http://localhost:3000"

# Google OAuth Credentials
GOOGLE_CLIENT_ID="YOUR_GOOGLE_CLIENT_ID"
GOOGLE_CLIENT_SECRET="YOUR_GOOGLE_CLIENT_SECRET"

# GitHub OAuth Credentials
GITHUB_CLIENT_ID="YOUR_GITHUB_CLIENT_ID"
GITHUB_CLIENT_SECRET="YOUR_GITHUB_CLIENT_SECRET"

# Email (Nodemailer for Magic Links/Verification)
# Example using Mailtrap sandbox:
EMAIL_SERVER_USER="YOUR_MAILTRAP_USER"
EMAIL_SERVER_PASSWORD="YOUR_MAILTRAP_PASSWORD"
EMAIL_SERVER_HOST="sandbox.smtp.mailtrap.io"
EMAIL_SERVER_PORT="2525"
EMAIL_FROM="Auth App <noreply@example.com>"
```

**Fill in the placeholder values:**

- `DATABASE_URL`: Your PostgreSQL connection string.
- `NEXTAUTH_SECRET`: A strong, random string used to encrypt JWTs and session data. You can generate one using `openssl rand -base64 32` in your terminal.
- `GOOGLE_CLIENT_ID` & `GOOGLE_CLIENT_SECRET`: Obtain these from your [Google Cloud Console](https://console.cloud.google.com/).
  - **Authorized JavaScript origins**: `http://localhost:3000`
  - **Authorized redirect URIs**: `http://localhost:3000/api/auth/callback/google`
- `GITHUB_CLIENT_ID` & `GITHUB_CLIENT_SECRET`: Obtain these by creating an OAuth App on [GitHub](https://github.com/settings/developers).
  - **Homepage URL**: `http://localhost:3000`
  - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
- `EMAIL_SERVER_*` and `EMAIL_FROM`: Configure these for your email provider (e.g., Mailtrap for development, or a production SMTP service).

### 4. Set Up Prisma and Database

Apply database migrations and generate the Prisma client:

```bash
npx prisma migrate dev --name init # Or your desired migration name
npx prisma generate
```

This will create the necessary tables in your database based on the `prisma/schema.prisma` file.

### 5. Initialize Shadcn UI (if not already set up)

If you are setting up the project from scratch and Shadcn UI is not yet configured, run:

```bash
npx shadcn@latest init
```

Follow the prompts. The typical configuration used in this project is:

- Style: Default
- Base Color: Slate
- Global CSS: `app/globals.css`
- CSS Variables: Yes
- Tailwind Config: `tailwind.config.ts` (or `.js`)
- Import Alias for Components: `@/components`
- Import Alias for Utils: `@/lib/utils`
- React Server Components: Yes

Then, add the necessary components (if they are not already present in `components/ui`):

```bash
npx shadcn-ui@latest add button card input label separator sheet dropdown-menu avatar dialog
```

### 6. Run the Development Server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure Overview

- `app/`: Contains all routes, pages, and layouts (App Router).
  - `app/api/auth/[...nextauth]/route.ts`: NextAuth.js dynamic API route.
  - `app/api/auth/signup/route.ts`: Custom API route for user registration.
  - `app/auth/`: Contains sign-in, sign-up, and verification request pages.
  - `app/(dashboard)/`: Example of a route group for dashboard pages (if used).
  - `app/profile/`: User profile page (currently a dialog in UserNav).
  - `app/layout.tsx`: Root layout for the application.
  - `app/page.tsx`: Homepage/main dashboard content area.
- `components/`: Shared React components.
  - `components/ui/`: Shadcn UI components.
  - `components/DashboardLayout.tsx`: Main layout for the dashboard view.
  - `components/UserNav.tsx`: User avatar and dropdown menu in the header.
  - `components/Providers.tsx`: Wraps the app with `SessionProvider`.
- `lib/`: Utility functions and libraries.
  - `lib/prisma.ts`: Prisma client instantiation (singleton).
  - `lib/utils.ts`: Utility functions from Shadcn UI (e.g., `cn`).
- `prisma/`: Prisma schema and migrations.
  - `prisma/schema.prisma`: Defines database models.
- `middleware.ts`: Next.js middleware for protecting routes.
- `.env`: Environment variables (ignored by Git).

## Learn More (Next.js & Vercel)

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
