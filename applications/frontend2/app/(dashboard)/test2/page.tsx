"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Loader2, FileAudio, CheckCircle, AlertCircle } from "lucide-react"

interface ProcessResponse {
    message: string
    output_file: string
    processed_files: number
    saved_records: number
    errors: number
}

export default function Home() {
    const [directoryPath, setDirectoryPath] = useState("")
    const [modelChoice, setModelChoice] = useState<"gemini" | "whissle" | "deepgram">("gemini")
    const [outputPath, setOutputPath] = useState("")
    const [annotations, setAnnotations] = useState({
        age: false,
        gender: false,
        emotion: false,
        entity: false,
        intent: false,
    })
    const [response, setResponse] = useState<ProcessResponse | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)

    const handleAnnotationChange = (key: keyof typeof annotations) => {
        setAnnotations((prev) => ({ ...prev, [key]: !prev[key] }))
    }

    const handleSubmit = async () => {
        setIsLoading(true)
        setError(null)
        setResponse(null)

        const selectedAnnotations = Object.keys(annotations).filter((key) => annotations[key as keyof typeof annotations])

        const endpoint = selectedAnnotations.length > 0 ? "/create_annotated_manifest/" : "/create_transcription_manifest/"

        try {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    directory_path: directoryPath,
                    model_choice: modelChoice,
                    output_jsonl_path: outputPath,
                    ...(selectedAnnotations.length > 0 && { annotations: selectedAnnotations }),
                }),
            })
            if (!res.ok) {
                throw new Error(`HTTP error! Status: ${res.status}`)
            }
            const data = await res.json()
            setResponse(data)
        } catch (err: any) {
            setError(err.message || "An error occurred")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="container mx-auto py-8 px-4 max-w-4xl">
            <div className="flex items-center gap-3 mb-6">
                <FileAudio className="h-8 w-8 text-primary" />
                <h1 className="text-3xl font-bold">Audio Processor</h1>
            </div>

            <Card className="mb-6">
                <CardHeader>
                    <CardTitle>Process Audio Files</CardTitle>
                    <CardDescription>
                        Enter a server-side directory path accessible to the backend to process audio files.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="space-y-2">
                        <Label htmlFor="directory-path">Directory Path</Label>
                        <Input
                            id="directory-path"
                            value={directoryPath}
                            onChange={(e) => setDirectoryPath(e.target.value)}
                            placeholder="/home/dchauhan/workspace/test"
                        />
                        <p className="text-sm text-muted-foreground">Path must be accessible to the backend server</p>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="model-choice">Model Choice</Label>
                        <Select
                            value={modelChoice}
                            onValueChange={(value) => setModelChoice(value as "gemini" | "whissle" | "deepgram")}
                        >
                            <SelectTrigger id="model-choice">
                                <SelectValue placeholder="Select a model" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="gemini">Gemini</SelectItem>
                                <SelectItem value="whissle">Whissle</SelectItem>
                                <SelectItem value="deepgram">Deepgram</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="output-path">Output JSONL Path</Label>
                        <Input
                            id="output-path"
                            value={outputPath}
                            onChange={(e) => setOutputPath(e.target.value)}
                            placeholder="/home/dchauhan/workspace/test/output.jsonl"
                        />
                    </div>

                    <Separator className="my-4" />

                    <div className="space-y-3">
                        <Label className="text-base">Annotations</Label>
                        <div className="grid grid-cols-2 gap-4 mt-2">
                            {Object.keys(annotations).map((key) => (
                                <div key={key} className="flex items-center space-x-2">
                                    <Checkbox
                                        id={`annotation-${key}`}
                                        checked={annotations[key as keyof typeof annotations]}
                                        onCheckedChange={() => handleAnnotationChange(key as keyof typeof annotations)}
                                        disabled={modelChoice === "deepgram"}
                                    />
                                    <Label
                                        htmlFor={`annotation-${key}`}
                                        className={`${modelChoice === "deepgram" ? "text-muted-foreground" : ""}`}
                                    >
                                        {key.charAt(0).toUpperCase() + key.slice(1)}
                                    </Label>
                                </div>
                            ))}
                        </div>
                        {modelChoice === "deepgram" && (
                            <p className="text-sm text-amber-600">Annotations are not available with Deepgram model</p>
                        )}
                    </div>
                </CardContent>
                <CardFooter>
                    <Button onClick={handleSubmit} disabled={isLoading || !directoryPath || !outputPath} className="w-full">
                        {isLoading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            "Process Audio Files"
                        )}
                    </Button>
                </CardFooter>
            </Card>

            {error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {response && (
                <Card>
                    <CardHeader className="pb-2">
                        <div className="flex items-center gap-2">
                            <CheckCircle className="h-5 w-5 text-green-500" />
                            <CardTitle>Processing Complete</CardTitle>
                        </div>
                        <CardDescription>{response.message}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <p className="text-sm font-medium">Output File</p>
                                <p className="text-sm text-muted-foreground break-all">{response.output_file}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-sm font-medium">Model Used</p>
                                <p className="text-sm text-muted-foreground">
                                    {modelChoice.charAt(0).toUpperCase() + modelChoice.slice(1)}
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4">
                            <div className="bg-muted rounded-lg p-3 text-center">
                                <p className="text-2xl font-bold">{response.processed_files}</p>
                                <p className="text-xs text-muted-foreground">Files Processed</p>
                            </div>
                            <div className="bg-muted rounded-lg p-3 text-center">
                                <p className="text-2xl font-bold">{response.saved_records}</p>
                                <p className="text-xs text-muted-foreground">Records Saved</p>
                            </div>
                            <div className="bg-muted rounded-lg p-3 text-center">
                                <p className="text-2xl font-bold text-red-500">{response.errors}</p>
                                <p className="text-xs text-muted-foreground">Errors</p>
                            </div>
                        </div>

                        {Object.values(annotations).some((v) => v) && (
                            <div className="space-y-2">
                                <p className="text-sm font-medium">Selected Annotations</p>
                                <div className="flex flex-wrap gap-2">
                                    {Object.keys(annotations)
                                        .filter((key) => annotations[key as keyof typeof annotations])
                                        .map((key) => (
                                            <Badge key={key} variant="secondary">
                                                {key.charAt(0).toUpperCase() + key.slice(1)}
                                            </Badge>
                                        ))}
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}
        </div>
    )
}
