import os
import gc
import re
import json
import time
import uuid
import base64
import torch
import librosa
import asyncio
import logging
import numpy as np
import soundfile as sf
import sys
import resampy
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from deepgram import DeepgramClient, PrerecordedOptions 
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, Request,
    File, UploadFile, Form, Query
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydanticField
import uvicorn

from dotenv import load_dotenv


import torch.nn as nn
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

import google.generativeai as genai

try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
except ImportError:
    print("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass # Dummy class

# load_dotenv('/home/dchauhan/workspace/meta-asr/applications/.env')
load_dotenv('D:/z-whissle/meta-asr/.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# --- BIO Annotation Constants ---
# Moved ENTITY_TYPES and INTENT_TYPES to be global constants as they are used across functions
ENTITY_TYPES = [
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
    "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT",
    "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE",
    "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD",
    "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER",
    "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER",
    "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE",
    "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE",
    "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER",
    "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD",
    "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME",
    "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME",
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN"
]

INTENT_TYPES = [
    "INFORM", "QUESTION", "REQUEST", "COMMAND", "GREETING", "CONFIRMATION", "NEGATION",
    "ACKNOWLEDGEMENT", "INQUIRY", "FAREWELL", "APOLOGY", "THANKS", "COMPLAINT",
    "FEEDBACK", "SUGGESTION", "ASSISTANCE", "NAVIGATION", "TRANSACTION", "SCHEDULING",
    "UNKNOWN_INTENT" # Added for robust handling
]

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Initialize FastAPI ---
app = FastAPI(
    title="Audio Processing API",
    description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities, "
                "and saves results to a JSONL manifest file.",
    version="1.5.0" # Incremented version for new output format
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- Model Loading ---
age_gender_model = None
age_gender_processor = None
emotion_model = None
emotion_feature_extractor = None

current_file_path = Path(__file__).parent
static_dir = current_file_path / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_html_path = current_file_path / "static" / "index.html"
    if not index_html_path.is_file():
        logger.error(f"HTML file not found at: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html_path)

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3) # Assuming 3 genders (Male, Female, Other/Unknown)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1) # Pool over sequence length
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        return hidden_states, logits_age, logits_gender

try:
    logger.info("Loading Age/Gender model...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
    age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
    age_gender_model.eval()
    logger.info("Age/Gender model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Age/Gender model: {e}", exc_info=True)

try:
    logger.info("Loading Emotion model...")
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
    emotion_model.eval()
    logger.info("Emotion model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Emotion model: {e}", exc_info=True)

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API configured successfully.")
        GEMINI_CONFIGURED = True
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
else:
    logger.warning("Warning: GOOGLE_API_KEY environment variable not set. Gemini features will be unavailable.")

if WHISSLE_AVAILABLE and WHISSLE_AUTH_TOKEN:
    logger.info("Whissle Auth Token found.")
    WHISSLE_CONFIGURED = True
elif WHISSLE_AVAILABLE:
    logger.warning("Warning: WHISSLE_AUTH_TOKEN environment variable not set. Whissle model will be unavailable.")

class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"
    deepgram = "deepgram"


# Check Deepgram configuration
DEEPGRAM_CONFIGURED = bool(DEEPGRAM_API_KEY)
if DEEPGRAM_CONFIGURED:
    try:
        DEEPGRAM_CLIENT = DeepgramClient(DEEPGRAM_API_KEY)
        logger.info("Deepgram client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Deepgram client: {e}")
        DEEPGRAM_CONFIGURED = False
else:
    logger.warning("Deepgram API key not set. Deepgram transcription disabled.")

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

# --- Helper Functions ---
def validate_paths(dir_path_str: str, output_path_str: str) -> Tuple[Path, Path]:
    dir_path = Path(dir_path_str)
    output_jsonl_path = Path(output_path_str)

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {dir_path_str}")

    if not output_jsonl_path.parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Output directory does not exist: {output_jsonl_path.parent}")

    return dir_path, output_jsonl_path

def discover_audio_files(directory_path: Path) -> List[Path]:
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory_path.glob(f"*{ext}"))
        audio_files.extend(directory_path.glob(f"*{ext.upper()}")) # Also check for uppercase extensions
    audio_files.sort() # Ensure consistent order
    logger.info(f"Discovered {len(audio_files)} audio files in {directory_path}")
    return audio_files


def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Convert to mono
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=False)
        return None, None, f"Failed to load audio: {type(e).__name__}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as le:
             logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=False)
             return None

def predict_age_gender(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if age_gender_model is None or age_gender_processor is None:
        return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, None, "Empty audio data provided for Age/Gender."
    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = age_gender_model(input_values)
        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]
        return float(age_pred), int(gender_pred_idx), None
    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=False)
        return None, None, f"Age/Gender prediction failed: {type(e).__name__}"

def predict_emotion(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[str], Optional[str]]:
    if emotion_model is None or emotion_feature_extractor is None:
        return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, "Empty audio data provided for Emotion."
    min_length = int(sampling_rate * 0.1)
    if len(audio_data) < min_length:
        return "SHORT_AUDIO", None
    try:
        inputs = emotion_feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")
        return emotion_label, None
    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=False)
        return None, f"Emotion prediction failed: {type(e).__name__}"

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> tuple[Optional[str], Optional[str]]:
    if not WHISSLE_CONFIGURED:
        return None, "Whissle is not configured."
    try:
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        if isinstance(response, dict):
             text = response.get('text')
             if text is not None: return text.strip(), None
             else:
                 error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                 return None, f"Whissle API error: {error_detail}"
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str):
            return response.transcript.strip(), None
        else: return None, f"Unexpected Whissle response format: {type(response)}"
    except Exception as e:
        return None, f"Whissle SDK error: {type(e).__name__}: {e}"

def get_mime_type(audio_file_path: Path) -> str:
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_gemini_single(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    model_name = "models/gemini-1.5-flash" # Or "models/gemini-1.5-pro" for higher quality potential
    try: model = genai.GenerativeModel(model_name)
    except Exception as e: return None, f"Error initializing Gemini model: {e}"

    mime_type = get_mime_type(audio_path)
    uploaded_file = None
    try:
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini resource {uploaded_file.name}: {del_e}")
            return None, error_msg

        prompt = "Transcribe the audio accurately. Provide only the spoken text. If no speech is detected, return an empty string."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})
        
        # Ensure file is deleted after response
        try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name); uploaded_file = None
        except Exception as del_e: logger.warning(f"Could not delete Gemini resource {uploaded_file.name} after transcription: {del_e}")

        if response.candidates:
            try:
                if hasattr(response, 'text') and response.text is not None: transcription = response.text.strip()
                elif response.candidates[0].content.parts: transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else: return None, "Gemini response candidate found, but no text content."
                return transcription if transcription else "", None # Return empty string for no speech
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                return None, f"Error parsing Gemini transcription response: {resp_e}"
        else:
            error_message = f"No candidates from Gemini transcription for {audio_path.name}."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                 error_message += f" Feedback: {feedback}"
            return None, error_message
    except Exception as e:
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error during transcription: {type(e).__name__}: {e}"
    finally: # Redundant with above but good practice for ensuring cleanup
        if uploaded_file and hasattr(uploaded_file, 'name'):
             try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
             except Exception as del_e: logger.warning(f"Could not delete Gemini file {uploaded_file.name} in finally: {del_e}")


def get_annotation_prompt(texts_to_annotate: List[str]) -> str:
    """
    Generates the core prompt for BIO entity and utterance-level intent annotation.
    """
    all_entity_types_str = ", ".join(ENTITY_TYPES)
    all_intent_types_str = ", ".join(INTENT_TYPES)

    return f'''You are an expert linguistic annotator for English text.
You will receive a list of English sentences. Each sentence is a raw lowercase transcription.

Your task is crucial and requires precision. For each sentence, you must:
1.  **TOKENIZE:** Split the sentence into individual words (tokens).
2.  **ASSIGN BIO TAGS:** For each token, assign exactly one BIO tag according to the following rules:
    *   **ENTITY TAGS (Priority):** Identify entities using the provided `ENTITY_TYPES` list.
        *   `B-<ENTITY_TYPE>` for the *beginning* of an entity phrase (e.g., `B-PERSON_NAME`).
        *   `I-<ENTITY_TYPE>` for *inside* an entity phrase (e.g., `I-PERSON_NAME`).
    *   **UTTERANCE INTENT TAGS (Default/Fallback):** If a token is *not* part of any specific entity, it should be tagged to reflect the overall intent of the utterance.
        *   The first token of the sentence (if not an entity) should be `B-<UTTERANCE_INTENT>`.
        *   Subsequent non-entity tokens should be `I-<UTTERANCE_INTENT>`.
        *   The `<UTTERANCE_INTENT>` should be chosen from the `INTENT_TYPES` list.
    *   **IMPORTANT:** Ensure every token has a tag. If no specific entity or clear intent can be assigned, use `O` (Outside) for tokens.

3.  **EXTRACT INTENT:** In addition to tagging, determine and provide the single overall `intent` of the utterance as a separate field. This `intent` should be one of the `INTENT_TYPES`.

4.  **OUTPUT FORMAT (CRITICAL):** Return a JSON array of objects. Each object in the array must contain:
    *   `text`: The original lowercase input sentence (for verification purposes).
    *   `tokens`: A JSON array of the tokenized words.
    *   `tags`: A JSON array of the BIO tags, corresponding one-to-one with the `tokens` array.
    *   `intent`: A single string representing the overall utterance intent.

**ENTITY TYPES LIST (USE ONLY THESE FOR ENTITY TAGS):**
{json.dumps(ENTITY_TYPES, ensure_ascii=False, indent=2)}

**INTENT TYPES LIST (USE ONE FOR UTTERANCE INTENT AND FOR DEFAULT TAGS):**
{json.dumps(INTENT_TYPES, ensure_ascii=False, indent=2)}

**Example Input String 1 (with entities):**
"then if he becomes a champion, he's entitled to more money after that and champion end"

**CORRECT Example Output 1 (assuming intent is INFORM and "champion" is a PROJECT_NAME):**
```json
[
  {{
    "text": "then if he becomes a champion, he's entitled to more money after that and champion end",
    "tokens": ["then", "if", "he", "becomes", "a", "champion", ",", "he's", "entitled", "to", "more", "money", "after", "that", "and", "champion", "end"],
    "tags": ["B-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O"],
    "intent": "INFORM"
  }}
]
Sentences to Annotate Now:
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
async def annotate_text_structured_with_gemini(text_to_annotate: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
    """
    Annotates a single text string using Gemini to get structured BIO tags and intent.
    Returns (tokens, tags, intent, error_message).
    """
    if not GEMINI_CONFIGURED:
        return None, None, None, "Gemini API is not configured."
    if not text_to_annotate or text_to_annotate.isspace():
        return [], [], "NO_SPEECH_INPUT", None # Return empty lists and specific intent for empty input
    prompt = get_annotation_prompt([text_to_annotate.lower()])

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash") # Or your preferred Gemini model
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json", # Request JSON output
            ),
            safety_settings=safety_settings,
            request_options={'timeout': 120} # Increased timeout
        )

        if response.candidates:
            raw_json_output = response.text.strip()
            logger.debug(f"Gemini raw JSON output for BIO: {raw_json_output}")
            
            try:
                parsed_data_list = json.loads(raw_json_output)
                if not isinstance(parsed_data_list, list) or not parsed_data_list:
                    logger.error(f"Gemini BIO annotation did not return a list or returned an empty list: {raw_json_output}")
                    return None, None, None, "Gemini BIO: Invalid or empty list format"

                annotation_object = parsed_data_list[0] # Expecting one object for one input sentence

                tokens = annotation_object.get("tokens")
                tags = annotation_object.get("tags")
                intent = annotation_object.get("intent")

                if not (isinstance(tokens, list) and isinstance(tags, list) and isinstance(intent, str)):
                    logger.error(f"Gemini BIO: Invalid types for tokens, tags, or intent. Tokens: {type(tokens)}, Tags: {type(tags)}, Intent: {type(intent)}")
                    return None, None, None, "Gemini BIO: Type mismatch in parsed data"
                
                if len(tokens) != len(tags):
                    logger.error(f"Gemini BIO: Mismatch between token ({len(tokens)}) and tag ({len(tags)}) counts.")
                    return None, None, None, "Gemini BIO: Token/Tag count mismatch"
                    
                return tokens, tags, intent.upper(), None # Success
            except json.JSONDecodeError as json_e:
                logger.error(f"Gemini BIO annotation JSON decoding failed: {json_e}. Response: {raw_json_output}")
                return None, None, None, f"Gemini BIO: JSONDecodeError - {json_e}"
            except Exception as e:
                logger.error(f"Error parsing Gemini BIO annotation response: {e}. Response: {raw_json_output}")
                return None, None, None, f"Gemini BIO: Parsing error - {e}"
        else:
            error_message = f"No candidates from Gemini BIO annotation for text: {text_to_annotate[:100]}..."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                error_message += f" Feedback: {feedback}"
            logger.error(error_message)
            return None, None, None, error_message

    except Exception as e:
        logger.error(f"Gemini API/SDK error during BIO annotation: {type(e).__name__}: {e}", exc_info=True)
        return None, None, None, f"Gemini API/SDK error: {type(e).__name__}"


async def transcribe_with_deepgram_single(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    if not DEEPGRAM_CONFIGURED:
        return None, "Deepgram not configured."
    try:
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=False,
            language="en"
        )
        # Run synchronous Deepgram call in a thread to avoid blocking
        response = await asyncio.to_thread(
            DEEPGRAM_CLIENT.listen.prerecorded.v("1").transcribe_file,
            {"buffer": buffer_data, "mimetype": get_mime_type(audio_path)},
            options
        )
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        if not transcript:
            return "", None # Return empty string for no speech, no error
        return transcript, None
    except Exception as e:
        logger.error(f"Deepgram transcription error for {audio_path.name}: {e}")
        return None, f"Deepgram error: {str(e)}"


@app.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED: 
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    if model_choice == ModelChoice.gemini and not GEMINI_CONFIGURED: 
        raise HTTPException(status_code=400, detail="Gemini not configured.")
    if model_choice == ModelChoice.deepgram and not DEEPGRAM_CONFIGURED:
        raise HTTPException(status_code=400, detail="Deepgram not configured.")
    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e: 
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0; saved_records_count = 0; error_count = 0
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: Optional[str] = None; transcription_text: Optional[str] = None; duration: Optional[float] = None
                logger.info(f"--- Processing {audio_file.name} (Transcription Only) ---")
                processed_files_count += 1
                try:
                    duration = get_audio_duration(audio_file)
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                    else:  # Deepgram
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None: # Should only happen if a transcription function returns None, not ""
                        file_error = "Transcription returned None."
                    # transcription_text can be "" for no speech, which is fine
                except Exception as e:
                    file_error = f"Unexpected error: {type(e).__name__}: {e}"

                record = TranscriptionJsonlRecord(audio_filepath=str(audio_file.resolve()), text=transcription_text, duration=duration, model_used_for_transcription=model_choice.value, error=file_error)
                try: 
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    if not file_error: 
                        file_error = f"JSONL write error: {write_e}"
                    else: 
                        file_error += f"; JSONL write error: {write_e}"
                if file_error: 
                    error_count += 1
                else: 
                    saved_records_count += 1
                gc.collect()
    except IOError as e: 
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

    msg = f"Processed {processed_files_count}/{len(audio_files)}. Saved: {saved_records_count}. Errors: {error_count}."
    return ProcessResponse(message=msg, output_file=str(output_jsonl_path), processed_files=processed_files_count, saved_records=saved_records_count, errors=error_count)


@app.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    # Check if Gemini is required for entity/intent and configured
    requires_gemini_for_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    if requires_gemini_for_annotation and not GEMINI_CONFIGURED:
        raise HTTPException(status_code=400, detail="Gemini not configured (required for entity/intent annotation).")

    # Check if Age/Gender/Emotion models are needed and loaded
    needs_age_gender = process_request.annotations and any(a in ["age", "gender"] for a in process_request.annotations)
    needs_emotion = process_request.annotations and "emotion" in process_request.annotations

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _:
                pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(
            message="No audio files. Empty manifest created.",
            output_file=str(output_jsonl_path),
            processed_files=0,
            saved_records=0,
            errors=0
        )

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
         for audio_file in audio_files:
            file_error_details: List[str] = []
            
            # Initialize data for the record
            record_data: Dict[str, Any] = {
                "audio_filepath": str(audio_file.resolve()),
                "task_name": "NER", # Set task name to "NER" as requested
            }

            logger.info(f"--- Processing {audio_file.name} (Selective Annotation) ---")
            processed_files_count += 1

            # 1. Get Duration
            record_data["duration"] = get_audio_duration(audio_file)

            # 2. Load Audio for A/G/E if requested
            audio_data: Optional[np.ndarray] = None
            sample_rate: Optional[int] = None
            if needs_age_gender or needs_emotion:
                audio_data, sample_rate, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
                if load_err:
                    file_error_details.append(load_err)
                elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                    file_error_details.append("Audio load/SR mismatch for A/G/E.")
            
            # 3. Transcribe Audio
            transcription_text: Optional[str] = None
            transcription_error: Optional[str] = None
            if model_choice == ModelChoice.whissle:
                transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
            elif model_choice == ModelChoice.gemini:
                transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
            else: # Deepgram
                transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
            
            if transcription_error:
                file_error_details.append(f"Transcription: {transcription_error}")
                transcription_text = None # Mark as failed if error
            elif transcription_text is None:
                file_error_details.append("Transcription returned None.")
            
            # Store both original transcription and processed text
            record_data["original_transcription"] = transcription_text
            record_data["text"] = transcription_text  # For now, same as original; can be processed later

            # 4. Predict Age/Gender/Emotion if requested
            if (needs_age_gender or needs_emotion) and audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                tasks = []
                if needs_age_gender:
                    if age_gender_model is None: file_error_details.append("A/G_WARN: Age/Gender model not loaded.")
                    else: tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                if needs_emotion:
                    if emotion_model is None: file_error_details.append("EMO_WARN: Emotion model not loaded.")
                    else: tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))
                
                if tasks: # Only run gather if there are tasks
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, tuple) and len(result) == 3:  # Age/Gender result (age_pred, gender_idx, error)
                            age_pred, gender_idx, age_gender_err = result
                            if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
                            else:
                                if "age" in process_request.annotations:
                                    try:
                                        actual_age = round(age_pred, 1)
                                        age_brackets: List[Tuple[float, str]] = [
                                            (18, "0-17"), (25, "18-24"), (35, "25-34"),
                                            (45, "35-44"), (55, "45-54"), (65, "55-64"),
                                            (float('inf'), "65+")
                                        ]
                                        age_group = "Unknown"
                                        for threshold, bracket in age_brackets:
                                            if actual_age < threshold:
                                                age_group = bracket
                                                break
                                        record_data["age_group"] = age_group
                                    except Exception as age_e:
                                        logger.error(f"Error formatting age_group: {age_e}")
                                        record_data["age_group"] = "Error"

                                if "gender" in process_request.annotations:
                                    gender_str = "Unknown"
                                    if gender_idx == 1: gender_str = "Male"
                                    elif gender_idx == 0: gender_str = "Female"
                                    record_data["gender"] = gender_str
                        elif isinstance(result, tuple) and len(result) == 2:  # Emotion result (emotion_label, error)
                            emotion_label, emotion_err = result
                            if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
                            else:
                                if "emotion" in process_request.annotations:
                                    record_data["emotion"] = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"
                                
            # 5. Annotate Entities/Intent with Gemini if requested and transcription exists
            if requires_gemini_for_annotation and transcription_text and transcription_text.strip() != "":
                tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(transcription_text)
                if gemini_anno_err:
                    file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                    if "intent" in process_request.annotations: record_data["gemini_intent"] = "ANNOTATION_FAILED"
                else:
                    if "entity" in process_request.annotations: record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                    if "intent" in process_request.annotations: record_data["gemini_intent"] = intent # intent is already uppercase from the function
            elif requires_gemini_for_annotation: # If annotation requested but transcription is empty/failed
                if transcription_text == "":
                    if "intent" in process_request.annotations: record_data["gemini_intent"] = "NO_SPEECH"
                else: # transcription_text is None (transcription failed)
                    if "intent" in process_request.annotations: record_data["gemini_intent"] = "TRANSCRIPTION_FAILED"


            # Set final error message
            final_error_msg = "; ".join(file_error_details) if file_error_details else None
            record_data["error"] = final_error_msg

            # Create Pydantic record and write to file
            current_errors_before_write = error_count
            try:
                record = AnnotatedJsonlRecord(**record_data)
                outfile.write(record.model_dump_json(exclude_none=True) + "\n")
            except Exception as write_e:
                logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                if not final_error_msg: # If the error was *only* during write
                    final_error_msg = f"JSONL write error: {write_e}"
                if error_count == current_errors_before_write: # Ensure it's counted once
                     error_count += 1
            
            # Determine if record was truly successful (no processing errors, no write errors)
            if not final_error_msg:
                saved_records_count += 1
            elif error_count == current_errors_before_write and final_error_msg: # If processing errors caused error_count to increment already
                error_count += 1

            # Clean up GPU memory
            del audio_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        truly_successful_saves = saved_records_count # Renamed from previous calculation
        final_message = (
            f"Processed {processed_files_count}/{len(audio_files)} files for selective annotation. "
            f"{truly_successful_saves} records successfully saved (no internal errors). "
            f"{error_count} files encountered errors or warnings (check 'error' field in JSONL)."
        )
        return ProcessResponse(
            message=final_message,
            output_file=str(output_jsonl_path),
            processed_files=processed_files_count,
            saved_records=truly_successful_saves,
            errors=error_count
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")


@app.get("/status", summary="API Status")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API v1.5.0",
        "docs_url": "/docs", "html_interface": "/",
        "endpoints": { "transcription_only": "/create_transcription_manifest/", "full_annotation": "/create_annotated_manifest/" },
        "gemini_configured": GEMINI_CONFIGURED, "whissle_available": WHISSLE_AVAILABLE, "whissle_configured": WHISSLE_CONFIGURED,
        "age_gender_model_loaded": age_gender_model is not None, "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
    }


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    fastapi_script_name = Path(__file__).stem
    app_module_string = f"{fastapi_script_name}:app"
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = "info"
    logger.info(f"Starting FastAPI server for '{app_module_string}' on {host}:{port}...")
    logger.info(f"Log Level: {log_level.upper()}, Reload Mode: {'Enabled' if reload else 'Disabled'}")
    logger.info(f"Docs: http://{host}:{port}/docs, UI: http://{host}:{port}/")
    uvicorn.run(app_module_string, host=host, port=port, reload=reload, reload_dirs=[str(script_dir)] if reload else None, log_level=log_level)