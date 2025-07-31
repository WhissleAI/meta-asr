'use client';

import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Image from 'next/image'; // For displaying user image

export default function ProfilePage() {
    const { data: session, status } = useSession();
    const router = useRouter();

    if (status === 'loading') {
        return <p style={{ textAlign: 'center', marginTop: '50px' }}>Loading profile...</p>;
    }

    if (status === 'unauthenticated') {
        // Should be handled by middleware, but as a fallback:
        router.push('/auth/signin');
        return <p style={{ textAlign: 'center', marginTop: '50px' }}>Redirecting to sign in...</p>;
    }

    return (
        <div style={{ maxWidth: '600px', margin: '50px auto', padding: '20px', border: '1px solid #ccc', borderRadius: '8px', textAlign: 'center' }}>
            <h1>User Profile</h1>
            {session?.user ? (
                <>
                    {session.user.image && (
                        <Image
                            src={session.user.image}
                            alt={`${session.user.name || 'User'}'s profile picture`}
                            width={100}
                            height={100}
                            style={{ borderRadius: '50%', margin: '20px auto' }}
                        />
                    )}
                    <p><strong>Name:</strong> {session.user.name || 'Not provided'}</p>
                    <p><strong>Email:</strong> {session.user.email || 'Not provided'}</p>
                    {/* You can add more details here if they are available in your session user object */}
                    <button
                        onClick={() => signOut({ callbackUrl: '/' })}
                        style={{
                            marginTop: '20px',
                            padding: '10px 20px',
                            backgroundColor: '#dc3545',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer'
                        }}
                    >
                        Sign Out
                    </button>
                </>
            ) : (
                <p>Could not load user profile.</p>
            )}
            <button
                onClick={() => router.push('/')}
                style={{
                    marginTop: '20px',
                    marginLeft: '10px',
                    padding: '10px 20px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                }}
            >
                Back to Home
            </button>
        </div>
    );
}
