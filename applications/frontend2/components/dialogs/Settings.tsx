import React, { useState, useEffect, FormEvent } from 'react';
import {
    Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter, DialogClose
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { Trash2, Edit3, PlusCircle } from 'lucide-react';

interface ApiKey {
    id: string;
    provider: string;
    key: string; // In a real app, you might not want to send the full key back to the client
    createdAt: string;
    // Add other relevant fields if necessary
}

const LLM_PROVIDERS = ["openai", "gemini", "deepgram", "whissle"]; // Add more as needed

type SettingsProps = {
    isSettingsOpen: boolean;
    setIsSettingsOpen: (open: boolean) => void;
};

const SettingsDialog = ({ isSettingsOpen, setIsSettingsOpen }: SettingsProps) => {
    const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isFetchingKeys, setIsFetchingKeys] = useState(false);
    const [currentProvider, setCurrentProvider] = useState("");
    const [currentKey, setCurrentKey] = useState("");
    const [isEditing, setIsEditing] = useState<string | null>(null); // Stores provider of key being edited

    const fetchApiKeys = async () => {
        setIsFetchingKeys(true);
        try {
            const response = await fetch('/api/settings/api-keys');
            if (!response.ok) {
                throw new Error('Failed to fetch API keys');
            }
            const data = await response.json();
            setApiKeys(data);
        } catch (error) {
            console.error("Error fetching API keys:", error);
            toast.error((error as Error).message || "Could not fetch API keys.");
        } finally {
            setIsFetchingKeys(false);
        }
    };

    useEffect(() => {
        if (isSettingsOpen) {
            fetchApiKeys();
            // Reset form when dialog opens
            setCurrentProvider("");
            setCurrentKey("");
            setIsEditing(null);
        }
    }, [isSettingsOpen]);

    const handleAddOrUpdateApiKey = async (e: FormEvent) => {
        e.preventDefault();
        if (!currentProvider || !currentKey) {
            toast.error("Provider and API Key cannot be empty.");
            return;
        }
        setIsLoading(true);
        try {
            const response = await fetch('/api/settings/api-keys', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: currentProvider, key: currentKey }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to save API key');
            }
            toast.success(`API key for ${currentProvider} ${isEditing ? 'updated' : 'added'} successfully.`);
            setCurrentProvider("");
            setCurrentKey("");
            setIsEditing(null);
            fetchApiKeys(); // Refresh the list
        } catch (error) {
            console.error("Error saving API key:", error);
            toast.error((error as Error).message || "Could not save API key.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleDeleteApiKey = async (provider: string) => {
        setIsLoading(true);
        try {
            const response = await fetch(`/api/settings/api-keys/${provider}`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to delete API key');
            }
            toast.success(`API key for ${provider} deleted successfully.`);
            fetchApiKeys(); // Refresh the list
            if (isEditing === provider) { // If deleting the key being edited, reset form
                setCurrentProvider("");
                setCurrentKey("");
                setIsEditing(null);
            }
        } catch (error) {
            console.error("Error deleting API key:", error);
            toast.error((error as Error).message || "Could not delete API key.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleEdit = (apiKey: ApiKey) => {
        setIsEditing(apiKey.provider);
        setCurrentProvider(apiKey.provider);
        setCurrentKey(""); // For security, don't display existing key. User must re-enter.
        toast.info(`Editing API key for ${apiKey.provider}. Please enter the new key.`);
    };

    return (
        <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
            <DialogContent className="sm:max-w-md md:max-w-lg">
                <DialogHeader>
                    <DialogTitle>Manage API Keys</DialogTitle>
                    <DialogDescription>
                        Add, edit, or delete your LLM provider API keys here.
                    </DialogDescription>
                </DialogHeader>

                <form onSubmit={handleAddOrUpdateApiKey} className="grid gap-6 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="provider" className="text-right col-span-1">
                            Provider
                        </Label>
                        <Select
                            value={currentProvider}
                            onValueChange={setCurrentProvider}
                            disabled={isEditing !== null} // Disable if editing, provider shouldn't change
                        >
                            <SelectTrigger id="provider" className="col-span-3">
                                <SelectValue placeholder="Select a provider" />
                            </SelectTrigger>
                            <SelectContent>
                                {LLM_PROVIDERS.map(p => (
                                    <SelectItem key={p} value={p} disabled={apiKeys.some(ak => ak.provider === p && p !== currentProvider)}>
                                        {p.charAt(0).toUpperCase() + p.slice(1)}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="api-key" className="text-right col-span-1">
                            API Key
                        </Label>
                        <Input
                            id="api-key"
                            type="password" // Use password type for sensitive input
                            value={currentKey}
                            onChange={(e) => setCurrentKey(e.target.value)}
                            className="col-span-3"
                            placeholder={isEditing ? "Enter new key to update" : "Enter your API key"}
                        />
                    </div>
                    <div className="flex justify-end">
                        <Button type="submit" disabled={isLoading || !currentProvider || !currentKey}>
                            {isLoading ? 'Saving...' : (isEditing ? 'Update Key' : 'Add Key')}
                            {isEditing ? <Edit3 className="ml-2 h-4 w-4" /> : <PlusCircle className="ml-2 h-4 w-4" />}
                        </Button>
                        {isEditing && (
                            <Button type="button" variant="outline" size="sm" onClick={() => { setIsEditing(null); setCurrentProvider(""); setCurrentKey(""); }} className="ml-2">
                                Cancel Edit
                            </Button>
                        )}
                    </div>
                </form>

                <div className="mt-6">
                    <h3 className="text-lg font-medium mb-3">Stored API Keys</h3>
                    {isFetchingKeys ? (
                        <p className="text-sm text-muted-foreground">Loading keys...</p>
                    ) : apiKeys.length > 0 ? (
                        <div className="space-y-3 max-h-60 overflow-y-auto pr-2">
                            {apiKeys.map((apiKey) => (
                                <div key={apiKey.provider} className="flex items-center justify-between p-3 bg-muted rounded-md">
                                    <div>
                                        <p className="font-medium">{apiKey.provider.charAt(0).toUpperCase() + apiKey.provider.slice(1)}</p>
                                        <p className="text-xs text-muted-foreground">
                                            Added: {new Date(apiKey.createdAt).toLocaleDateString()}
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Button variant="outline" size="icon" onClick={() => handleEdit(apiKey)} disabled={isLoading} title="Edit Key">
                                            <Edit3 className="h-4 w-4" />
                                        </Button>
                                        <Button variant="destructive" size="icon" onClick={() => handleDeleteApiKey(apiKey.provider)} disabled={isLoading} title="Delete Key">
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground">No API keys stored yet.</p>
                    )}
                </div>

                <DialogFooter className="mt-6">
                    <DialogClose asChild>
                        <Button type="button" variant="secondary" onClick={() => setIsSettingsOpen(false)}>
                            Close
                        </Button>
                    </DialogClose>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

export default SettingsDialog;