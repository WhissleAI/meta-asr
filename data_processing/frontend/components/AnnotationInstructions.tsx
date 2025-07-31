// components/AnnotationInstructions.tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const AnnotationInstructions = () => {
    return (
        <Card className="mt-6">
            <CardHeader>
                <CardTitle>How to Use Annotation Features</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
                <p>To get annotated transcriptions, follow these steps:</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <strong>Select Source:</strong> Choose either "Server Directory" or "Google Cloud Storage" and provide the necessary path.
                    </li>
                    <li>
                        <strong>Transcription Type:</strong> Set to "Annotated Transcription".
                    </li>
                    <li>
                        <strong>Model Choice:</strong>
                        <ul className="list-circle pl-5 space-y-0.5 mt-1">
                            <li>For <strong>Entity Recognition</strong> and <strong>Intent Detection</strong>, ensure "Gemini" is selected as the model. These annotations are performed by the Gemini model.</li>
                            <li>Other models like "Whissle" or "Deepgram" primarily provide transcription and may support other annotations like age/gender if configured.</li>
                        </ul>
                    </li>
                    <li>
                        <strong>Annotations to Apply:</strong>
                        <ul className="list-circle pl-5 space-y-0.5 mt-1">
                            <li>Check the boxes for the specific annotations you need (e.g., Age, Gender, Emotion, Entity, Intent).</li>
                            <li>Age, Gender, and Emotion are typically processed by local models on the backend.</li>
                        </ul>
                    </li>
                    <li>
                        <strong>Annotation Prompt (for Gemini Entity/Intent):</strong>
                        <ul className="list-circle pl-5 space-y-0.5 mt-1">
                            <li>This section appears if "Entity" or "Intent" is selected.</li>
                            <li>You can choose a pre-defined base prompt from the dropdown.</li>
                            <li>You can modify the selected prompt or write a completely custom one in the "Modify Prompt" textarea. This prompt guides the Gemini model.</li>
                            <li>A prompt is <strong>required</strong> if you select "Entity" or "Intent" annotations.</li>
                        </ul>
                    </li>
                    <li>
                        <strong>Process:</strong> Click the "Process Directory" or "Process GCS File" button.
                    </li>
                </ul>
                <p className="font-semibold mt-2">Output Details:</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        The output will be a JSONL file. Each line corresponds to an audio file.
                    </li>
                    <li>
                        <strong>`text` / `original_transcription`</strong>: The transcribed text.
                    </li>
                    <li>
                        <strong>`bio_annotation_gemini`</strong>: If "Entity" annotation was selected, this field will contain:
                        <ul className="list-circle pl-5 space-y-0.5 mt-1">
                            <li>`tokens`: A list of words/punctuation from the transcription.</li>
                            <li>`tags`: A corresponding list of BIO (Beginning, Inside, Outside) tags for each token (e.g., B-PERSON_NAME, I-LOCATION, O).</li>
                        </ul>
                    </li>
                    <li>
                        <strong>`gemini_intent`</strong>: If "Intent" annotation was selected, this field will contain the detected overall intent of the utterance.
                    </li>
                    <li>
                        <strong>`age_group`, `gender`, `emotion`</strong>: Will be populated if these annotations were selected and successfully processed.
                    </li>
                    <li>
                        <strong>`prompt_used`</strong>: Shows the actual prompt submitted to Gemini for entity/intent tasks.
                    </li>
                    <li>
                        <strong>`error` / `error_details`</strong>: Will contain information if any part of the processing failed for a file.
                    </li>
                </ul>
            </CardContent>
        </Card>
    );
};
