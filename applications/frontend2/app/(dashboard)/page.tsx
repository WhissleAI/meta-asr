"use client"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import { loadPrompts } from "@/utils/loadPrompt"
import { AlertCircle, CheckCircle, Loader2 } from "lucide-react"
import { useEffect, useState, useRef } from "react"
import { useSession } from "next-auth/react"
import { initFastApiUserSession } from "@/utils/sessionManager"
import { toast } from "sonner"
import { AnnotationInstructions } from "../../components/AnnotationInstructions" // Added import

interface ProcessResponse {
  message: string
  output_file: string
  processed_files: number
  saved_records: number
  errors: number
}

interface GcsStatusMessage {
  status: string
  detail?: string
  data?: any
  error?: string
}

interface GcsProcessingResult {
  original_gcs_path: string
  downloaded_local_path?: string
  status_message: string
  duration?: number
  transcription?: string
  age_group?: string
  gender?: string
  emotion?: string
  bio_annotation_gemini?: any
  gemini_intent?: string
  prompt_used?: string
  error_details?: string[]
  overall_error?: string
}

export default function Home() {
  const { data: session, status: sessionStatus } = useSession()
  const [sourceType, setSourceType] = useState<"directory" | "gcs">("directory")
  const [inputPath, setInputPath] = useState("")
  const [transcriptionType, setTranscriptionType] = useState<"simple" | "annotated">("simple")
  const [modelChoice, setModelChoice] = useState<"gemini" | "whissle" | "deepgram" | "openai">("gemini")
  const [outputPath, setOutputPath] = useState("") // For directory source
  const [gcsOutputJsonlPath, setGcsOutputJsonlPath] = useState("") // New: For GCS source output
  const [annotations, setAnnotations] = useState({
    age: false,
    gender: false,
    emotion: false,
    entity: false,
    intent: false,
  })
  const [prompts, setPrompts] = useState<{ file: string; displayName: string; content: string }[]>([])
  const [selectedPrompt, setSelectedPrompt] = useState("")
  const [customPrompt, setCustomPrompt] = useState("")
  const [response, setResponse] = useState<ProcessResponse | GcsProcessingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [gcsProcessingStatus, setGcsProcessingStatus] = useState<string[]>([])
  const websocketRef = useRef<WebSocket | null>(null)
  const [segmentLength, setSegmentLength] = useState<number | string>("") // New state for segment length

  // Initialize FastAPI session on authenticated load
  useEffect(() => {
    if (sessionStatus === "authenticated") {
      console.log("User authenticated, attempting to initialize FastAPI session...")
      initFastApiUserSession()
    }
  }, [sessionStatus])

  // Simulate loading prompts from \lib\prompt_library.txt
  useEffect(() => {
    loadPrompts()
      .then((loadedPrompts) => {
        setPrompts(loadedPrompts.map(({ displayName, name, content }) => ({
          displayName,
          file: name,
          content: content || "", // Ensure content is always a string
        })))
        setSelectedPrompt(loadedPrompts[0]?.displayName || "")
        setCustomPrompt(loadedPrompts[0]?.content || "")
      })
      .catch((err) => {
        console.error("Failed to load prompts:", err)
        setError("Failed to load prompts")
        toast.error("Failed to load prompts", { description: (err as Error).message })
      })
  }, [])

  // Cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close()
        websocketRef.current = null
        console.log("GCS status WebSocket closed on unmount.")
      }
    }
  }, [])

  const handleAnnotationChange = (key: keyof typeof annotations) => {
    setAnnotations((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const handleSubmit = async () => {
    if (sessionStatus !== "authenticated" || !session?.user?.id) {
      setError("User not authenticated. Please sign in.")
      toast.error("Authentication Error", { description: "User session not found." })
      setIsLoading(false)
      return
    }
    if (!inputPath) {
      toast.error("Missing Input Path", { description: "Please enter a valid path for the selected source type." })
      setIsLoading(false)
      return
    }
    if (sourceType === "gcs" && !inputPath.startsWith("gs://")) {
      toast.error("Invalid GCS Path", { description: "Please enter a valid GCS path starting with gs://" })
      setIsLoading(false)
      return
    }
    if (sourceType === "directory" && !outputPath) {
      toast.error("Missing Output Path", { description: "Output path is required for directory processing." })
      setIsLoading(false)
      return
    }
    if (sourceType === "gcs" && !gcsOutputJsonlPath) { // New: Check for GCS output path
      toast.error("Missing GCS Output JSONL Path", { description: "Output JSONL path is required for GCS processing." })
      setIsLoading(false)
      return
    }

    const userId = session.user.id
    setIsLoading(true)
    setError(null)
    setResponse(null)
    setGcsProcessingStatus([])

    const selectedAnnotations = transcriptionType === "annotated"
      ? Object.keys(annotations).filter((key) => annotations[key as keyof typeof annotations])
      : []

    // Log basic tracking
    console.log("Audio processing started for user:", userId, {
      sourceType,
      transcriptionType,
      modelChoice,
      inputPath,
      outputPath: sourceType === "directory" ? outputPath : gcsOutputJsonlPath, // Modified: use gcsOutputJsonlPath for GCS
      annotations: selectedAnnotations,
      prompt: transcriptionType === "annotated" ? customPrompt : null,
      segment_length_sec: sourceType === "directory" && segmentLength ? Number(segmentLength) : null, // Add segment length to log
      timestamp: new Date().toISOString(),
    })

    if (sourceType === "directory") {
      const endpoint = segmentLength // If segmentLength is set, use the new endpoint
        // ? "/trim_audio_and_transcribe/"
        ? "/trim_transcribe_annotate/"
        : transcriptionType === "annotated" && selectedAnnotations.length > 0
          ? "/create_annotated_manifest/"
          : "/create_transcription_manifest/"

      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            directory_path: inputPath,
            model_choice: modelChoice,
            output_jsonl_path: outputPath,
            ...(selectedAnnotations.length > 0 && { annotations: selectedAnnotations }),
            ...(transcriptionType === "annotated" && { prompt: customPrompt }),
            ...(segmentLength && { segment_length_sec: Number(segmentLength) }), // Add segment length to request
          }),
        })
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({ detail: `HTTP error! Status: ${res.status}` }))
          throw new Error(errorData.detail || `HTTP error! Status: ${res.status}`)
        }
        const data = await res.json()
        setResponse(data)
        toast.success("Directory processing complete!", { description: data.message })
      } catch (err: any) {
        setError(err.message || "An error occurred during directory processing.")
        toast.error("Directory Processing Error", { description: err.message || "An unknown error occurred." })
      } finally {
        setIsLoading(false)
      }
    } else if (sourceType === "gcs") {
      const fastapiBaseUrl = process.env.NEXT_PUBLIC_API_URL
      if (!fastapiBaseUrl) {
        setError("Configuration Error: FastAPI URL is not configured.")
        toast.error("Configuration Error", { description: "FastAPI URL is not configured." })
        setIsLoading(false)
        return
      }
      const cleanFastapiUrl = fastapiBaseUrl.endsWith('/') ? fastapiBaseUrl.slice(0, -1) : fastapiBaseUrl
      const wsProtocol = cleanFastapiUrl.startsWith('https://') ? 'wss://' : 'ws://'
      const wsUrl = `${wsProtocol}${cleanFastapiUrl.replace(/^https?:\/\//, '')}/ws/gcs_status/${userId}`

      console.log(`Connecting to WebSocket: ${wsUrl}`)
      setGcsProcessingStatus(prev => [...prev, `Connecting to WebSocket: ${wsUrl}`])

      if (websocketRef.current && websocketRef.current.readyState !== WebSocket.CLOSED) {
        websocketRef.current.close()
      }
      websocketRef.current = new WebSocket(wsUrl)

      const currentGcsPath = inputPath
      websocketRef.current.onopen = () => {
        setGcsProcessingStatus(prev => [...prev, "WebSocket connected. Triggering backend processing..."])
        triggerGcsProcessingApi(currentGcsPath)
      }

      websocketRef.current.onmessage = (event) => {
        try {
          const messageData: GcsStatusMessage = JSON.parse(event.data as string)
          console.log("WebSocket message received:", messageData)
          const statusLine = `[${new Date().toLocaleTimeString()}] ${messageData.status}: ${messageData.detail || ""}`
          setGcsProcessingStatus(prev => [...prev.slice(-10), statusLine]) // Keep last 10 messages
        } catch (e) {
          console.error("Error parsing WebSocket message or updating state:", e)
          setGcsProcessingStatus(prev => [...prev, "Error processing WebSocket message."])
        }
      }

      websocketRef.current.onerror = (errorEvent) => {
        let errorDetail = "An unknown WebSocket error occurred."
        if (errorEvent instanceof ErrorEvent && errorEvent.message) errorDetail = errorEvent.message
        setGcsProcessingStatus(prev => [...prev, `WebSocket error: ${errorDetail}`])
        toast.error("WebSocket Connection Error")
        setIsLoading(false)
        if (websocketRef.current) websocketRef.current.close()
      }

      websocketRef.current.onclose = (event) => {
        setGcsProcessingStatus(prev => [...prev, `WebSocket disconnected. Code: ${event.code}`])
        websocketRef.current = null
      }
    }
  }

  const triggerGcsProcessingApi = async (currentGcsPath: string) => {
    setIsLoading(true); // Ensure loading is true at the start
    setError(null);
    setResponse(null);
    // setGcsProcessingStatus([]); // Keep previous status messages or clear as preferred

    try {
      const selectedAnnotationsList = transcriptionType === "annotated"
        ? Object.keys(annotations).filter(key => annotations[key as keyof typeof annotations])
        : []
      const apiRequestBody = {
        user_id: session?.user?.id,
        gcs_path: currentGcsPath,
        model_choice: modelChoice,
        annotations: selectedAnnotationsList,
        prompt: transcriptionType === "annotated" && customPrompt ? customPrompt : null,
        output_jsonl_path: gcsOutputJsonlPath, // New: Add GCS output path to request
      }
      const fetchResponse = await fetch('/api/process-gcs-file-proxy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiRequestBody),
      })

      if (!fetchResponse.ok) {
        let errorMsg = `HTTP error! Status: ${fetchResponse.status}`;
        try {
          // Try to parse the error response as JSON
          const errorData = await fetchResponse.json();
          errorMsg = errorData.error || errorData.detail || JSON.stringify(errorData);
        } catch (e) {
          // If parsing as JSON fails, try to get the response as text
          try {
            const errorText = await fetchResponse.text();
            errorMsg = errorText || errorMsg; // Use text if available and not empty
            console.error("Received non-JSON error response:", errorText);
          } catch (textError) {
            console.error("Failed to get error response as text:", textError);
          }
        }
        toast.error("GCS File Processing Failed", { description: errorMsg });
        setResponse({
          original_gcs_path: currentGcsPath,
          status_message: `Error: ${errorMsg}`,
          overall_error: errorMsg
        } as GcsProcessingResult);
        return; // Exit after handling error
      }

      // If fetchResponse.ok is true, then try to parse the successful response as JSON
      const resultData = await fetchResponse.json(); // This is the line (around 264) that might throw

      toast.success("GCS File Processing Complete", { description: resultData.message || "Processing finished." });
      setResponse(resultData.data as GcsProcessingResult);

    } catch (error) { // This catch handles network errors or errors from fetchResponse.json() itself
      console.error("Error calling GCS processing API or parsing JSON:", error);
      const clientErrorMsg = (error instanceof Error ? error.message : String(error)) || "Could not send request or parse response.";
      toast.error("GCS Processing Request Error", { description: clientErrorMsg });
      setResponse({
        original_gcs_path: currentGcsPath,
        status_message: `Client-side error: ${clientErrorMsg}`,
        overall_error: "ClientRequestError",
      } as GcsProcessingResult);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-4xl">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Process Audio Files</CardTitle>
          <CardDescription>
            Enter a server-side directory or GCS path and choose transcription options.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="source-type">Audio Source Type</Label>
            <Select
              value={sourceType}
              onValueChange={(value) => {
                setSourceType(value as "directory" | "gcs")
                setInputPath("")
                setOutputPath("")
                setGcsOutputJsonlPath("") // New: Reset GCS output path on source change
                setResponse(null)
                setError(null)
                setGcsProcessingStatus([])
              }}
              disabled={isLoading}
            >
              <SelectTrigger id="source-type">
                <SelectValue placeholder="Select source type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="directory">Server Directory</SelectItem>
                <SelectItem value="gcs">Google Cloud Storage</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="input-path">Input Path</Label>
            <Input
              id="input-path"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder={sourceType === "directory" ? "/home/user/workspace/test" : "gs://your-bucket-name/path/to/audio.wav"}
              disabled={isLoading}
            />
            <p className="text-sm text-muted-foreground">
              {sourceType === "directory" ? "Path must be accessible to the backend server" : "Enter a valid GCS path starting with gs://"}
            </p>
          </div>

          {sourceType === "gcs" && ( // New: Input field for GCS output path
            <div className="space-y-2">
              <Label htmlFor="gcs-output-path">Output JSONL Path (GCS)</Label>
              <Input
                id="gcs-output-path"
                value={gcsOutputJsonlPath}
                onChange={(e) => setGcsOutputJsonlPath(e.target.value)}
                placeholder="/path/on/server/to/gcs_output.jsonl"
                disabled={isLoading}
              />
              <p className="text-sm text-muted-foreground">
                Server-side path where the GCS processing result (JSONL) will be saved.
              </p>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="transcription-type">Transcription Type</Label>
            <Select
              value={transcriptionType}
              onValueChange={(value) => {
                setTranscriptionType(value as "simple" | "annotated")
                setModelChoice("gemini")
                if (value === "simple") {
                  setAnnotations({ age: false, gender: false, emotion: false, entity: false, intent: false })
                  setSelectedPrompt(prompts[0]?.displayName || "")
                  setCustomPrompt(prompts[0]?.content || "")
                } else if (value === "annotated") {
                  setSelectedPrompt(prompts[0]?.displayName || "")
                  setCustomPrompt(prompts[0]?.content || "")
                }
              }}
              disabled={isLoading}
            >
              <SelectTrigger id="transcription-type">
                <SelectValue placeholder="Select transcription type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="simple">Transcription</SelectItem>
                <SelectItem value="annotated">Annotated Transcription</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="model-choice">Model Choice</Label>
            <Select
              value={modelChoice}
              onValueChange={(value) => setModelChoice(value as "gemini" | "whissle" | "deepgram" | "openai")}
              disabled={isLoading}
            >
              <SelectTrigger id="model-choice">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gemini">Gemini</SelectItem>
                <SelectItem value="whissle">Whissle</SelectItem>
                {transcriptionType === "simple" && <SelectItem value="deepgram">Deepgram</SelectItem>}
                {transcriptionType === "annotated" && <SelectItem value="openai">Open AI</SelectItem>}
              </SelectContent>
            </Select>
          </div>

          {transcriptionType === "annotated" && (
            <>
              <div className="space-y-3">
                <Label className="text-base">Annotations</Label>
                <div className="grid grid-cols-2 gap-4 mt-2">
                  {Object.keys(annotations).map((key) => (
                    <div key={key} className="flex items-center space-x-2">
                      <Checkbox
                        id={`annotation-${key}`}
                        checked={annotations[key as keyof typeof annotations]}
                        onCheckedChange={() => handleAnnotationChange(key as keyof typeof annotations)}
                        disabled={isLoading || modelChoice === "deepgram"}
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

              <div className="space-y-2">
                <Label htmlFor="prompt-choice">Annotation Prompt</Label>
                <Select
                  value={selectedPrompt}
                  onValueChange={(value) => {
                    setSelectedPrompt(value)
                    setCustomPrompt(prompts.find((p) => p.displayName === value)?.content || "")
                  }}
                  disabled={isLoading}
                >
                  <SelectTrigger id="prompt-choice">
                    <SelectValue placeholder="Select a prompt" />
                  </SelectTrigger>
                  <SelectContent>
                    {prompts.map((prompt) => (
                      <SelectItem key={prompt.file} value={prompt.displayName} className="truncate max-w-xs break-words">
                        {prompt.displayName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="custom-prompt">Modify Prompt</Label>
                <Textarea
                  id="custom-prompt"
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Modify the selected prompt here..."
                  rows={4}
                  disabled={isLoading}
                />
              </div>
            </>
          )}

          <Separator className="my-4" />

          {sourceType === "directory" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="output-path">Output JSONL Path (Directory)</Label>
                <Input
                  id="output-path"
                  value={outputPath}
                  onChange={(e) => setOutputPath(e.target.value)}
                  placeholder="/home/user/workspace/test/output.jsonl"
                  disabled={isLoading}
                />
              </div>
              <div className="space-y-2 mt-4"> {/* Added mt-4 for spacing */}
                <Label htmlFor="segment-length">Segment Length (seconds)</Label>
                <Input
                  id="segment-length"
                  type="number"
                  value={segmentLength}
                  onChange={(e) => setSegmentLength(e.target.value)}
                  placeholder="e.g., 10 (trims audio into 10-second segments)"
                  disabled={isLoading}
                />
                <p className="text-sm text-muted-foreground">
                  Optional. If provided, audio files will be trimmed into segments of this length before transcription.
                </p>
              </div>
            </>
          )}
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleSubmit}
            disabled={isLoading || !inputPath || (sourceType === "directory" && !outputPath) || (transcriptionType === "annotated" && !customPrompt)}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              `Process ${sourceType === "directory" ? "Directory" : "GCS"} Audio Files`
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

      {gcsProcessingStatus.length > 0 && sourceType === "gcs" && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>GCS Processing Status</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs bg-muted p-4 rounded-md max-h-60 overflow-y-auto whitespace-pre-wrap break-all">
              {gcsProcessingStatus.join("\n")}
            </pre>
          </CardContent>
        </Card>
      )}

      {response && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <CardTitle>Processing Complete</CardTitle>
            </div>
            <CardDescription>
              {sourceType === "directory"
                ? (response as ProcessResponse).message
                : (response as GcsProcessingResult).status_message}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {sourceType === "directory" && (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Output File</p>
                  <p className="text-sm text-muted-foreground break-all">
                    {(response as ProcessResponse).output_file}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Model Used</p>
                  <p className="text-sm text-muted-foreground">
                    {modelChoice.charAt(0).toUpperCase() + modelChoice.slice(1)}
                  </p>
                </div>
              </div>
            )}
            {sourceType === "gcs" && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm font-medium">GCS Path</p>
                    <p className="text-sm text-muted-foreground break-all">
                      {(response as GcsProcessingResult).original_gcs_path}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Model Used</p>
                    <p className="text-sm text-muted-foreground">
                      {modelChoice.charAt(0).toUpperCase() + modelChoice.slice(1)}
                    </p>
                  </div>
                </div>
                {(response as GcsProcessingResult).downloaded_local_path && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Downloaded to (temp)</p>
                    <p className="text-sm text-muted-foreground break-all">
                      {(response as GcsProcessingResult).downloaded_local_path}
                    </p>
                  </div>
                )}
                {(response as GcsProcessingResult).duration !== undefined &&
                  (response as GcsProcessingResult).duration !== null && (
                    <div className="space-y-1">
                      <p className="text-sm font-medium">Duration</p>
                      <p className="text-sm text-muted-foreground">
                        {(response as GcsProcessingResult).duration.toFixed(2)}s
                      </p>
                    </div>
                  )}
                {(response as GcsProcessingResult).transcription && (
                  <div>
                    <Label className="text-sm font-medium">Transcription</Label>
                    <Textarea
                      readOnly
                      value={(response as GcsProcessingResult).transcription}
                      rows={5}
                      className="text-sm mt-1"
                    />
                  </div>
                )}
              </>
            )}

            {sourceType === "directory" && (
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-muted rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold">{(response as ProcessResponse).processed_files}</p>
                  <p className="text-xs text-muted-foreground">Files Processed</p>
                </div>
                <div className="bg-muted rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold">{(response as ProcessResponse).saved_records}</p>
                  <p className="text-xs text-muted-foreground">Records Saved</p>
                </div>
                <div className="bg-muted rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-red-500">{(response as ProcessResponse).errors}</p>
                  <p className="text-xs text-muted-foreground">Errors</p>
                </div>
              </div>
            )}

            {transcriptionType === "annotated" && (
              <>
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
                <div className="space-y-2">
                  <p className="text-sm font-medium">Used Prompt</p>
                  <p className="text-sm text-muted-foreground break-all">{customPrompt}</p>
                </div>
                {sourceType === "gcs" && (response as GcsProcessingResult).age_group && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Age Group</p>
                    <p className="text-sm text-muted-foreground">
                      {(response as GcsProcessingResult).age_group}
                    </p>
                  </div>
                )}
                {sourceType === "gcs" && (response as GcsProcessingResult).gender && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Gender</p>
                    <p className="text-sm text-muted-foreground">
                      {(response as GcsProcessingResult).gender}
                    </p>
                  </div>
                )}
                {sourceType === "gcs" && (response as GcsProcessingResult).emotion && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Emotion</p>
                    <p className="text-sm text-muted-foreground">
                      {(response as GcsProcessingResult).emotion}
                    </p>
                  </div>
                )}
                {sourceType === "gcs" && (response as GcsProcessingResult).gemini_intent && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Gemini Intent</p>
                    <p className="text-sm text-muted-foreground">
                      {(response as GcsProcessingResult).gemini_intent}
                    </p>
                  </div>
                )}
                {sourceType === "gcs" && (response as GcsProcessingResult).bio_annotation_gemini && (
                  <div>
                    <p className="text-sm font-medium">Gemini BIO Tags</p>
                    <pre className="text-xs bg-muted p-2 rounded-md max-h-40 overflow-y-auto whitespace-pre-wrap break-all">
                      {JSON.stringify((response as GcsProcessingResult).bio_annotation_gemini, null, 2)}
                    </pre>
                  </div>
                )}
                {sourceType === "gcs" && (response as GcsProcessingResult).error_details &&
                  (response as GcsProcessingResult).error_details!.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm font-medium text-destructive">Error Details</p>
                      <ul className="list-disc list-inside text-xs text-destructive space-y-1">
                        {(response as GcsProcessingResult).error_details!.map((err, idx) => (
                          <li key={idx}>{err}</li>
                        ))}
                      </ul>
                    </div>
                  )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      <AnnotationInstructions /> {/* Added the new component here */}
    </div>
  )
}