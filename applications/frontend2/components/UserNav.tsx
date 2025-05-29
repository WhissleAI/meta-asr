'use client';

import { useState } from 'react';
import {
    Avatar,
    AvatarFallback,
    AvatarImage,
} from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuGroup,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogClose
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { signOut, useSession } from "next-auth/react";

export const UserNav = () => {
    const { data: session } = useSession();
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    if (!session?.user) {
        return null;
    }

    const getInitials = (name?: string | null) => {
        if (!name) return "U";
        const names = name.split(' ');
        if (names.length > 1) {
            return `${names[0][0]}${names[names.length - 1][0]}`.toUpperCase();
        }
        return name.substring(0, 2).toUpperCase();
    };

    return (
        <>
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                        <Avatar className="h-9 w-9">
                            <AvatarImage src={session.user.image ?? ""} alt={session.user.name ?? "User"} />
                            <AvatarFallback>{getInitials(session.user.name)}</AvatarFallback>
                        </Avatar>
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end" forceMount>
                    <DropdownMenuLabel className="font-normal">
                        <div className="flex flex-col space-y-1">
                            <p className="text-sm font-medium leading-none">{session.user.name}</p>
                            <p className="text-xs leading-none text-muted-foreground">
                                {session.user.email}
                            </p>
                        </div>
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuGroup>
                        <DropdownMenuItem onSelect={() => setIsProfileOpen(true)} className="cursor-pointer">
                            Profile
                        </DropdownMenuItem>
                        <DropdownMenuItem disabled className="cursor-not-allowed">
                            Billing (soon)
                        </DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => setIsSettingsOpen(true)} className="cursor-pointer">
                            Settings
                        </DropdownMenuItem>
                    </DropdownMenuGroup>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => signOut({ callbackUrl: '/' })} className="cursor-pointer">
                        Log out
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            {/* Profile Dialog */}
            <Dialog open={isProfileOpen} onOpenChange={setIsProfileOpen}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>Profile</DialogTitle>
                        <DialogDescription>
                            This is your user profile information.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <p><strong>Name:</strong> {session.user.name}</p>
                        <p><strong>Email:</strong> {session.user.email}</p>
                        {session.user.image && (
                            <div className="flex justify-center">
                                <Avatar className="h-24 w-24">
                                    <AvatarImage src={session.user.image} alt={session.user.name ?? "User"} />
                                    <AvatarFallback>{getInitials(session.user.name)}</AvatarFallback>
                                </Avatar>
                            </div>
                        )}
                        <p className="text-sm text-muted-foreground">More profile details will be shown here.</p>
                    </div>
                    <DialogFooter>
                        <DialogClose asChild>
                            <Button type="button" variant="secondary">
                                Close
                            </Button>
                        </DialogClose>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Settings Dialog */}
            <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>Settings</DialogTitle>
                        <DialogDescription>
                            Manage your application settings here.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <p className="text-sm text-muted-foreground">Dummy settings content. More options will be available soon.</p>
                        {/* Example Setting */}
                        <div className="flex items-center justify-between">
                            <Label htmlFor="dark-mode" className="flex flex-col space-y-1">
                                <span>Dark Mode</span>
                                <span className="font-normal leading-snug text-muted-foreground">
                                    Toggle dark mode for the application.
                                </span>
                            </Label>
                            {/* Switch component would go here if added */}
                            <Button variant="outline" size="sm" disabled>Toggle (soon)</Button>
                        </div>
                    </div>
                    <DialogFooter>
                        <DialogClose asChild>
                            <Button type="button" variant="secondary">
                                Close
                            </Button>
                        </DialogClose>
                        <Button type="submit" disabled>Save changes (soon)</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}
