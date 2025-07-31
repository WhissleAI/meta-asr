'use client';

import { signIn, useSession } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link'; // Import Link
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

// Optional: Add icons for social logins
// import { Icons } from "@/components/icons"; // Assuming you might create an icons utility

function SignInForm() {
    const { status } = useSession();
    const router = useRouter();
    const searchParams = useSearchParams();
    const callbackUrl = searchParams.get('callbackUrl') || '/';

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (status === 'authenticated') {
            router.push(callbackUrl);
        }
    }, [status, router, callbackUrl]);

    const handleCredentialsSignIn = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);
        const result = await signIn('credentials', {
            redirect: false,
            email,
            password,
            callbackUrl
        });
        if (result?.error) {
            // Map common errors to more user-friendly messages
            if (result.error === 'CredentialsSignin') {
                setError('Invalid email or password. Please try again.');
            } else {
                setError(result.error);
            }
        } else if (result?.url) {
            router.push(result.url); // Should redirect to callbackUrl on success
        }
        setIsLoading(false);
    };

    const handleEmailSignIn = async () => {
        setError('');
        if (!email) {
            setError('Please enter your email address to sign in with a magic link.');
            return;
        }
        setIsLoading(true);
        const result = await signIn('email', {
            redirect: false,
            email,
            callbackUrl
        });
        if (result?.error) {
            setError(result.error);
        } else {
            router.push('/auth/verify-request');
        }
        setIsLoading(false);
    };

    const handleSocialSignIn = async (provider: string) => {
        setIsLoading(true);
        await signIn(provider, { callbackUrl });
        // setIsLoading(false); // NextAuth handles redirection or error display
    };

    if (status === 'loading') {
        return <div className="flex justify-center items-center min-h-screen">Loading...</div>;
    }
    if (status === 'authenticated') {
        return <div className="flex justify-center items-center min-h-screen">Redirecting...</div>;
    }

    return (
        <div className="flex justify-center items-center min-h-screen bg-background p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="text-center">
                    <CardTitle className="text-2xl font-bold">Sign In</CardTitle>
                    <CardDescription>Choose your preferred sign-in method</CardDescription>
                </CardHeader>
                <CardContent>
                    {error && <p className="text-sm font-medium text-destructive mb-4 text-center">{error}</p>}
                    <form onSubmit={handleCredentialsSignIn} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="email">Email</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                disabled={isLoading}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="password">Password</Label>
                            <Input
                                id="password"
                                type="password"
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                disabled={isLoading}
                            />
                        </div>
                        <Button type="submit" className="w-full" disabled={isLoading}>
                            {isLoading ? 'Signing In...' : 'Sign In with Email & Password'}
                        </Button>
                    </form>

                    <Separator className="my-6" />

                    <div className="space-y-3">
                        <Button variant="outline" className="w-full" onClick={() => handleSocialSignIn('google')} disabled={isLoading}>
                            {/* <Icons.google className="mr-2 h-4 w-4" /> */}
                            Sign In with Google
                        </Button>
                        <Button variant="outline" className="w-full" onClick={() => handleSocialSignIn('github')} disabled={isLoading}>
                            {/* <Icons.gitHub className="mr-2 h-4 w-4" /> */}
                            Sign In with GitHub
                        </Button>
                    </div>

                    <Separator className="my-6" />

                    <div className="space-y-3 text-center">
                        <p className="text-sm text-muted-foreground">Or sign in with a magic link (enter email above)</p>
                        <Button variant="link" onClick={handleEmailSignIn} disabled={isLoading || !email} className="w-full">
                            {isLoading ? 'Sending Link...' : 'Send Magic Link'}
                        </Button>
                    </div>
                </CardContent>
                <CardFooter className="text-center text-sm">
                    <p className="text-muted-foreground">
                        Don&apos;t have an account? <Link href="/auth/signup" className="font-semibold text-primary hover:underline">Sign Up</Link>
                    </p>
                </CardFooter>
            </Card>
        </div>
    );
}

export default function SignInPage() {
    return (
        <Suspense fallback={<div className="flex justify-center items-center min-h-screen">Loading page...</div>}>
            <SignInForm />
        </Suspense>
    );
}
