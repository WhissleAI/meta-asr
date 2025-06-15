"""
Pydantic models and data structures for the API
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field as PydanticField

class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"
    deepgram = "deepgram"

class ProcessRequest(BaseModel):
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    model_choice: ModelChoice = PydanticField(..., description="The transcription model to use.")
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")
    annotations: Optional[List[str]] = PydanticField(
        None,
        description="List of annotations to include (age, gender, emotion, entity, intent).",
        example=["age", "gender", "emotion", "entity", "intent"] # Added entity and intent to example
    )

class TranscriptionJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

# New Pydantic model for BIO annotation structure
class BioAnnotation(BaseModel):
    tokens: List[str]
    tags: List[str]

# Updated AnnotatedJsonlRecord to match the new output format
class AnnotatedJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None  # Processed/cleaned transcription text
    original_transcription: Optional[str] = None  # Original transcription text
    duration: Optional[float] = None
    task_name: Optional[str] = None # e.g., "NER" or "Annotation"
    gender: Optional[str] = None
    age_group: Optional[str] = None
    emotion: Optional[str] = None
    gemini_intent: Optional[str] = None
    ollama_intent: Optional[str] = None # Placeholder for future Ollama integration
    bio_annotation_gemini: Optional[BioAnnotation] = None
    bio_annotation_ollama: Optional[BioAnnotation] = None # Placeholder for future Ollama integration
    error: Optional[str] = None

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int
