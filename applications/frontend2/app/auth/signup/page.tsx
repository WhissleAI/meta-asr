'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { signIn } from 'next-auth/react'; // Keep signIn for OAuth
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function SignUpPage() {
    const router = useRouter();
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isOauthLoading, setIsOauthLoading] = useState(false); // Separate loading state for OAuth

    const handleCredentialsSignUp = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const res = await fetch('/api/auth/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password }),
            });

            if (res.ok) {
                const signInResult = await signIn('email', { email, redirect: false, callbackUrl: '/' });
                if (signInResult?.error) {
                    setError(signInResult.error);
                } else {
                    router.push('/auth/verify-request');
                }
            } else {
                const data = await res.json();
                setError(data.message || 'Sign up failed. Please try again.');
            }
        } catch (err) {
            console.error('Sign up error:', err);
            setError('An unexpected error occurred. Please try again.');
        }
        setIsLoading(false);
    };

    const handleSocialSignUp = async (provider: string) => {
        setIsOauthLoading(true);
        // For OAuth sign-up, NextAuth.js handles user creation if they don't exist.
        // The callbackUrl will redirect after successful sign-up/sign-in.
        await signIn(provider, { callbackUrl: '/' });
        // No need to setIsOauthLoading(false) as page will redirect or error handled by NextAuth
    };

    return (
        <div className="flex justify-center items-center min-h-screen bg-background p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="text-center">
                    <CardTitle className="text-2xl font-bold">Create an Account</CardTitle>
                    <CardDescription>Enter your details or use a social provider.</CardDescription>
                </CardHeader>
                <CardContent>
                    {error && <p className="text-sm font-medium text-destructive mb-4 text-center">{error}</p>}

                    <div className="space-y-3 mb-6">
                        <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => handleSocialSignUp('google')}
                            disabled={isLoading || isOauthLoading}
                        >
                            {/* Optional: Add Google Icon */}
                            {isOauthLoading ? 'Processing...' : 'Sign Up with Google'}
                        </Button>
                        <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => handleSocialSignUp('github')}
                            disabled={isLoading || isOauthLoading}
                        >
                            {/* Optional: Add GitHub Icon */}
                            {isOauthLoading ? 'Processing...' : 'Sign Up with GitHub'}
                        </Button>
                    </div>

                    <div className="relative mb-6">
                        {/* <Separator /> */}
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t" />
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                            <span className="bg-background px-2 text-muted-foreground">Or continue with email</span>
                        </div>
                    </div>

                    <form onSubmit={handleCredentialsSignUp} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="name">Full Name</Label>
                            <Input
                                id="name"
                                type="text"
                                placeholder="John Doe"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                required
                                disabled={isLoading || isOauthLoading}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="email">Email</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                disabled={isLoading || isOauthLoading}
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
                                minLength={6}
                                disabled={isLoading || isOauthLoading}
                            />
                        </div>
                        <Button type="submit" className="w-full" disabled={isLoading || isOauthLoading}>
                            {isLoading ? 'Creating Account...' : 'Create Account & Verify Email'}
                        </Button>
                    </form>
                </CardContent>
                <CardFooter className="text-center text-sm">
                    <p className="text-muted-foreground">
                        Already have an account?{' '}
                        <Link href="/auth/signin" className="font-semibold text-primary hover:underline">
                            Sign In
                        </Link>
                    </p>
                </CardFooter>
            </Card>
        </div>
    );
}
