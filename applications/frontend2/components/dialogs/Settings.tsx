import React from 'react'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogClose
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";


type SettingsProps = {
    isSettingsOpen: boolean;
    setIsSettingsOpen: (open: boolean) => void;
};

const SettingsDialog = ({ isSettingsOpen, setIsSettingsOpen }: SettingsProps) => {
    return (
        <>
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
            </Dialog></>
    )
}

export default SettingsDialog