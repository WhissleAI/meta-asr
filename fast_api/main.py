# -*- coding: utf-8 -*-
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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, Request,
    File, UploadFile, Form, Query
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field as PydanticField # Alias to avoid conflict
import uvicorn

from dotenv import load_dotenv

# --- Model Imports ---
import torch.nn as nn
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    # Wav2Vec2Model, # Base model not needed directly if using custom head model
    Wav2Vec2PreTrainedModel
)

import google.generativeai as genai

# Try importing WhissleClient - handle potential import error
try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
except ImportError:
    print("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass # Dummy class

# --- Configuration & Setup ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN") # Assuming token via env var now
GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Initialize FastAPI ---
app = FastAPI(
    title="Audio Processing API",
    description="Transcribes audio, predicts Age/Gender/Emotion, annotates Intent/Entities, "
                "and saves results to a JSONL file.",
    version="1.3.0" # Incremented version
)

# --- Model Loading ---
age_gender_model = None
age_gender_processor = None
emotion_model = None
emotion_feature_extractor = None

# Resolve the path relative to the current file
current_file_path = Path(__file__).parent
static_dir = current_file_path / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False) # Route to serve the HTML file
async def serve_index():
    index_html_path = current_file_path / "static" / "index.html"
    return FileResponse(index_html_path)

# --- Age/Gender Model Definition (Copied from provided code) ---
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

# Need Wav2Vec2Model for AgeGenderModel's definition
from transformers import Wav2Vec2Model

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1) # Predicts a single value for age
        self.gender = ModelHead(config, 3) # Predicts 3 classes for gender
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1) # Pool features
        logits_age = self.age(hidden_states)
        # Apply softmax AFTER getting logits if needed for probabilities elsewhere
        logits_gender = self.gender(hidden_states) # Return logits directly first
        # gender_probs = torch.softmax(logits_gender, dim=1) # Calculate probs if needed
        return hidden_states, logits_age, logits_gender # Return logits

try:
    logger.info("Loading Age/Gender model...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
    age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
    age_gender_model.eval() # Set to evaluation mode
    logger.info("Age/Gender model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Age/Gender model: {e}", exc_info=True)
    # Decide if the app should fail or continue without this feature
    # raise RuntimeError("Could not load essential Age/Gender model.") from e

try:
    logger.info("Loading Emotion model...")
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
    emotion_model.eval() # Set to evaluation mode
    logger.info("Emotion model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Emotion model: {e}", exc_info=True)
    # Decide if the app should fail or continue without this feature
    # raise RuntimeError("Could not load essential Emotion model.") from e

# --- Configure Gemini ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API configured successfully.")
        GEMINI_CONFIGURED = True
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
else:
    logger.warning("Warning: GOOGLE_API_KEY environment variable not set. Gemini features will be unavailable.")

# --- Configure Whissle ---
if WHISSLE_AVAILABLE and WHISSLE_AUTH_TOKEN:
    logger.info("Whissle Auth Token found.")
    WHISSLE_CONFIGURED = True
    # Test connection maybe?
    # try:
    #     _ = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN) # Test instantiation
    #     logger.info("Whissle client initialized successfully (basic check).")
    #     WHISSLE_CONFIGURED = True
    # except Exception as e:
    #      logger.error(f"Failed to initialize Whissle client with token: {e}. Whissle will be unavailable.")
elif WHISSLE_AVAILABLE:
    logger.warning("Warning: WHISSLE_AUTH_TOKEN environment variable not set. Whissle model will be unavailable.")
else:
    pass # Message already printed

# --- Pydantic Models ---
class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"

class ProcessRequest(BaseModel):
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    model_choice: ModelChoice = PydanticField(..., description="The transcription model to use.")
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")

class JsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None # Combined annotated text
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None # Store processing errors for this file

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int

# --- Helper Functions ---

def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Loads audio file, resamples if necessary. Returns (audio_array, sample_rate, error_message)."""
    try:
        # Use soundfile for broader format support and direct array loading
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Ensure mono
        if audio.ndim > 1:
            logger.warning(f"Audio file {audio_path.name} is not mono ({audio.ndim} channels). Converting to mono by averaging.")
            audio = np.mean(audio, axis=1)

        # Resample if necessary
        if sr != target_sr:
            logger.warning(f"Resampling {audio_path.name} from {sr}Hz to {target_sr}Hz.")
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr

        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=True)
        return None, None, f"Failed to load audio: {e}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Gets the duration of an audio file in seconds."""
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception as e:
        logger.warning(f"Could not get duration for {audio_path.name} using soundfile info: {e}. Trying librosa load.")
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as le:
             logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=True)
             return None

def predict_age_gender(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Predicts age and gender. Returns (predicted_age, predicted_gender_idx, error_message)."""
    if age_gender_model is None or age_gender_processor is None:
        return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, None, "Empty audio data provided for Age/Gender."

    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None

        with torch.no_grad():
            outputs = age_gender_model(input_values)

        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]

        return float(age_pred), int(gender_pred_idx), None

    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=True)
        return None, None, f"Age/Gender prediction failed: {e}"

def predict_emotion(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[str], Optional[str]]:
    """Predicts emotion. Returns (emotion_label, error_message)."""
    if emotion_model is None or emotion_feature_extractor is None:
        return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, "Empty audio data provided for Emotion."
    if len(audio_data) < sampling_rate * 0.1:
        logger.warning("Audio data too short for reliable emotion prediction.")
        return "SHORT_AUDIO", None

    try:
        inputs = emotion_feature_extractor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = emotion_model(**inputs)

        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")

        return emotion_label, None

    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=True)
        return None, f"Emotion prediction failed: {e}"

def format_age_gender_emotion_tags(age: Optional[float], gender_idx: Optional[int], emotion_label: Optional[str]) -> str:
    """Formats the predicted labels into a tag string."""
    tags = []
    if age is not None:
        try:
            actual_age = round(age, 1)
            age_brackets: List[Tuple[float, str]] = [
                (18, "0_18"), (30, "18_30"), (45, "30_45"),
                (60, "45_60"), (float('inf'), "60PLUS")
            ]
            age_tag = "UNKNOWN"
            for threshold, bracket in age_brackets:
                if actual_age < threshold:
                    age_tag = bracket
                    break
            tags.append(f"AGE_{age_tag}")
        except Exception as age_e:
             logger.error(f"Error formatting age tag for age value {age}: {age_e}")
             tags.append("AGE_ERROR")
    else:
        tags.append("AGE_UNKNOWN")

    if gender_idx is not None:
        if gender_idx == 1: tags.append("GENDER_MALE")
        elif gender_idx == 0: tags.append("GENDER_FEMALE")
        else: tags.append("GENDER_OTHER")
    else:
        tags.append("GENDER_UNKNOWN")

    if emotion_label and emotion_label != "SHORT_AUDIO":
        emotion_tag = emotion_label.upper().replace(" ", "_")
        tags.append(f"EMOTION_{emotion_tag}")
    elif emotion_label == "SHORT_AUDIO":
         tags.append("EMOTION_SHORT_AUDIO")
    else:
        tags.append("EMOTION_UNKNOWN")

    return " ".join(tags)

def combine_tags(gemini_annotated_text: str, age_gender_emotion_tags: str) -> str:
    """Inserts age/gender/emotion tags before the INTENT tag or at the end."""
    if not gemini_annotated_text:
        return age_gender_emotion_tags

    intent_match = re.search(r'\sINTENT_[A-Z_]+$', gemini_annotated_text)
    if intent_match:
        insertion_point = intent_match.start()
        return f"{gemini_annotated_text[:insertion_point]} {age_gender_emotion_tags}{gemini_annotated_text[insertion_point:]}".strip()
    else:
        return f"{gemini_annotated_text} {age_gender_emotion_tags}".strip()

# --- Transcription/Annotation Functions ---

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> tuple[Optional[str], Optional[str]]:
    """Transcribe using Whissle. Returns (text, error_message)."""
    if not WHISSLE_CONFIGURED:
        return None, "Whissle is not configured."
    try:
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        logger.info(f"Transcribing {audio_path.name} with Whissle...")
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        logger.debug(f"Whissle response for {audio_path.name}: {str(response)[:100]}...")

        if isinstance(response, dict):
             text = response.get('text')
             if text is not None: return text, None
             else:
                 error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                 logger.error(f"Whissle API error for {audio_path.name}: {error_detail} | Full response: {response}")
                 return None, f"Whissle API error: {error_detail}"
        elif hasattr(response, 'transcript'):
            logger.info(f"Received Whissle response object with transcript for {audio_path.name}.")
            return response.transcript, None
        else:
             logger.error(f"Unexpected Whissle response format for {audio_path.name}: {type(response)} | Content: {response}")
             return None, f"Unexpected Whissle response format: {type(response)}"

    except Exception as e:
        logger.error(f"Error transcribing {audio_path.name} with Whissle: {e}", exc_info=True)
        error_msg = f"Whissle SDK error: {type(e).__name__}: {e}"
        return None, error_msg

def get_mime_type(audio_file_path: Path) -> str:
    """Determines the MIME type of an audio file."""
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_gemini_single(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Transcribe using Gemini. Returns (text, error_message)."""
    if not GEMINI_CONFIGURED:
         return None, "Gemini API is not configured."
    model_name = "models/gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Error initializing Gemini model {model_name}: {e}", exc_info=True)
        return None, f"Error initializing Gemini model: {e}"

    mime_type = get_mime_type(audio_path)
    logger.info(f"Transcribing {audio_path.name} with Gemini (MIME: {mime_type})...")
    uploaded_file = None
    try:
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path))
        while uploaded_file.state.name == "PROCESSING":
            logger.info(f"Waiting for Gemini file processing: {audio_path.name}")
            await asyncio.sleep(2)
            uploaded_file = await asyncio.to_thread(genai.get_file, uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            logger.error(f"Gemini file processing failed for {audio_path.name}")
            # Attempt to delete the failed file
            try: await asyncio.to_thread(genai.delete_file, uploaded_file.name); logger.debug(f"Deleted failed Gemini file: {uploaded_file.name}")
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini file {uploaded_file.name}: {del_e}")
            return None, f"Gemini file processing failed: {uploaded_file.state}"

        prompt = "Transcribe the audio accurately. Return ONLY the spoken English words. If no speech, return empty string. Do NOT include sound descriptions."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})

        # Cleanup successful upload
        try: await asyncio.to_thread(genai.delete_file, uploaded_file.name); logger.debug(f"Deleted temp Gemini file: {uploaded_file.name}"); uploaded_file = None
        except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name}: {del_e}")

        # Process response
        transcription = ""
        if response.candidates:
            try:
                # Check multiple ways text might be present, prioritizing `response.text`
                if hasattr(response, 'text') and response.text:
                    transcription = response.text.strip()
                elif response.candidates[0].content.parts:
                    transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else: # Handle case where parts might be empty or text attribute missing
                     logger.warning(f"Gemini candidate exists but no text found for {audio_path.name}. Response: {response}")
                     return None, "Gemini response candidate found, but no text content."


                if transcription: logger.info(f"Gemini transcription for {audio_path.name}: {transcription[:100]}..."); return transcription, None
                else: logger.info(f"No speech detected by Gemini in {audio_path.name}."); return "", None # Return empty string for no speech
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e: # Added TypeError
                logger.warning(f"Could not extract text from Gemini response for {audio_path.name}: {resp_e}. Response: {response}"); return None, f"Error parsing Gemini response: {resp_e}"
        else: # Handle blocked responses or empty candidate list
            error_message = "No candidates returned from Gemini."
            try:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     error_message += f" Feedback: {response.prompt_feedback}"
                elif hasattr(response, 'error') and response.error: # Check for explicit error attribute
                     error_message += f" API Error: {response.error}"
            except Exception as fb_e: error_message += f" (Feedback/Error access error: {fb_e})"
            logger.error(f"Gemini transcription error for {audio_path.name}: {error_message}"); return None, error_message
    except Exception as e:
        logger.error(f"Error during Gemini transcription for {audio_path.name}: {e}", exc_info=True)
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try: await asyncio.to_thread(genai.delete_file, uploaded_file.name); logger.info(f"Deleted temp Gemini file after error: {uploaded_file.name}")
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API error: {type(e).__name__}: {e}"
    finally:
        # Ensure cleanup even if unexpected returns happen before deletion
        if uploaded_file and hasattr(uploaded_file, 'name'):
             try: await asyncio.to_thread(genai.delete_file, uploaded_file.name); logger.debug(f"Deleted temp Gemini file in finally: {uploaded_file.name}")
             except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} in finally: {del_e}")


def remove_existing_tags(text):
    """Removes only legacy NER_/ENTITY_ tags, preserving others."""
    if not isinstance(text, str): return text
    # Preserve specific known tags using placeholders
    preserve_patterns = {
        # Match specific tags exactly or with expected patterns
        r'\bAGE_[A-Z0-9_]+\b': ' __AGE_TAG__ ',
        r'\bGENDER_(MALE|FEMALE|OTHER)\b': ' __GENDER_TAG__ ', # More specific
        r'\bEMOTION_[A-Z_]+\b': ' __EMOTION_TAG__ ',
        r'\bSPEAKER_CHANGE\b': ' __SPEAKER_TAG__ ',
        # --- ADD ANY OTHER TAGS TO PRESERVE HERE ---
        r'\bINTENT_[A-Z_]+\b': ' __INTENT_TAG__ ', # Also preserve existing intent if needed
    }
    preserved_tags_map = {}
    placeholder_count = 0

    def replace_preserve(match):
        nonlocal placeholder_count
        tag = match.group(0)
        placeholder = f"__PRESERVED_{placeholder_count}__"
        preserved_tags_map[placeholder] = tag
        placeholder_count += 1
        # Add spaces around placeholder to prevent merging with words
        return f" {placeholder} "

    temp_text = text
    for pattern, _ in preserve_patterns.items(): # Don't need the replacement value here
        # Use the replace_preserve function to handle unique placeholders
        temp_text = re.sub(pattern, replace_preserve, temp_text, flags=re.IGNORECASE) # Ignore case for matching

    # Remove only legacy tags: NER_ followed by word characters, or ENTITY_ followed by word chars (IF THEY ARE UNWANTED)
    # If you want Gemini to ADD ENTITY_ tags, DO NOT remove them here unless they are definitely old/wrong.
    # Let's assume we only want to remove NER_ tags and maybe malformed/unwanted ENTITY_ tags.
    # Remove NER_ tags specifically
    cleaned_text = re.sub(r'\bNER_\w+\s?', '', temp_text, flags=re.IGNORECASE)
    # Optionally remove specific unwanted ENTITY tags if needed, e.g.:
    # cleaned_text = re.sub(r'\bENTITY_(OLDTAG1|OLDTAG2)\s?', '', cleaned_text, flags=re.IGNORECASE)

    # Remove END tags IF they are unassociated with a preserved/new entity tag (this might be too broad)
    # It's safer to handle END tag fixing in fix_end_tags based on context.
    # cleaned_text = re.sub(r'\s?END\b', '', cleaned_text, flags=re.IGNORECASE) # Commenting out for safety

    # Restore preserved tags
    for placeholder, original_tag in preserved_tags_map.items():
        # Replace placeholder surrounded by spaces with original tag surrounded by spaces
        # Ensure case is restored correctly from the map
        cleaned_text = cleaned_text.replace(f" {placeholder} ", f" {original_tag} ")

    # Clean up multiple spaces and strip leading/trailing whitespace
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text


def fix_end_tags(text):
    """Cleans up spacing around tags, including fixing split entity types."""
    if not isinstance(text, str): return text

    # --- ADDED: Fix split entity types like 'ENTITY_TYPE X END' -> 'ENTITY_TYPEX END' ---
    # Matches ENTITY_TYPE (group 1), a space, a single letter/digit (group 2), spaces, END
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)\s([A-Z0-9])\s+END\b', r'\1\2 END', text, flags=re.IGNORECASE)
    # ------------------------------------------------------------------------------------

    # Ensure space before END if attached to a word: wordEND -> word END
    text = re.sub(r'(\S)END\b', r'\1 END', text, flags=re.IGNORECASE) # Use ignorecase here too
    # Ensure space after ENTITY tag if attached to next word: ENTITY_TYPEword -> ENTITY_TYPE word
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text, flags=re.IGNORECASE)
    # Ensure space before INTENT tag if attached to a word: wordINTENT_ -> word INTENT_
    text = re.sub(r'(\S)(INTENT_[A-Z_]+)', r'\1 \2', text, flags=re.IGNORECASE)

    # Ensure space *after* END if followed immediately by a non-space char (e.g. ENDword -> END word)
    text = re.sub(r'\bEND(\S)', r'END \1', text, flags=re.IGNORECASE)

    # Clean up extra spaces that might have been introduced or were already there
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


async def annotate_text_with_gemini(text_to_annotate: str) -> tuple[Optional[str], Optional[str]]:
    """Annotates text for intent/entities using Gemini with the new prompt."""
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    if not text_to_annotate or text_to_annotate.isspace(): logger.warning("Skipping Gemini annotation: Input text is empty."); return "", None # Return empty string for empty input

    # Clean legacy tags before sending to Gemini.
    # Keep existing AGE/GENDER/EMOTION/SPEAKER tags as per the new prompt's instruction.
    cleaned_text_for_prompt = remove_existing_tags(text_to_annotate)
    if not cleaned_text_for_prompt or cleaned_text_for_prompt.isspace():
        logger.warning(f"Skipping Gemini annotation: Text empty after cleaning. Original: '{text_to_annotate}'")
        # Return the original text if cleaning removed everything, preserving original tags
        return text_to_annotate, None

    # --- THIS IS THE NEW PROMPT ---
    entity_list_json = json.dumps([
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
        "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN",
        "PRODUCT", "SERVICE", "CATEGORY", "BRAND", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD",
        "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE",
        "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "ORDER_NUMBER", "DELIVERY_ESTIMATE", "RECIPE", "INGREDIENT",
        "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD",
        "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
        "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
        "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
        "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
        "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
        "POLLUTION_LEVEL", "UV_INDEX",
        "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
        # Removed "NUMBER" as it wasn't in the example list and might be too general
    ])

    prompt = f'''You are given a sentence that may contain existing tags like GENDER_FEMALE, EMOTION_NEU, AGE_45_60, and SPEAKER_CHANGE.

Your task is to:
1. Remove any legacy entity tags such as NER_PERSON, NER_NORP, etc. (This step is already done on the input text).
2. Preserve the existing tags (like AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE) exactly as they appear in the input text.
3. **ONLY annotate entities from the following list.** Do NOT invent new entity types or annotate entities that are not explicitly listed here: {entity_list_json}.
4. Insert entity tags in the format `ENTITY_<TYPE>` before each recognized entity from the list and append ` END` (with a preceding space) immediately after the entity text. **Crucially, ensure the full entity type (e.g., PERSON_NAME, CURRENCY) is kept together and the space appears *only* before the `END` marker.**
5. Focus on identifying and tagging entities that are specifically mentioned in the provided list. Avoid tagging general concepts not on the list.
6. Classify the **intent** of the sentence and add exactly ONE `INTENT_<INTENT_TYPE>` tag at the VERY END of the sentence, after all other text and tags.
7. Return the full annotated text as a single string.

**IMPORTANT NOTES:**
- Do NOT modify or remove the pre-existing AGE, GENDER, EMOTION, or SPEAKER_CHANGE tags.
- Annotate using **only** the `ENTITY_<TYPE> ... END` format and **only** types from the provided list.
- **Correct Spacing Example:** `ENTITY_PERSON_NAME david END` is CORRECT. `ENTITY_PERSON_NAM E david END` is INCORRECT. `ENTITY_CURRENCY $5 END` is CORRECT. `ENTITY_CURRENC Y $5 END` is INCORRECT.
- Ensure there is a space between the entity text and the `END` tag. E.g., `ENTITY_PRICE 50 END` NOT `ENTITY_PRICE 50END`.
- Place the single `INTENT_<TYPE>` tag at the absolute end of the output string.

Example Input (after cleaning): "david needs 20 apples SPEAKER_CHANGE AGE_45_60 GENDER_MALE EMOTION_HAP"
Example Output: "ENTITY_PERSON_NAME david END needs 20 apples SPEAKER_CHANGE AGE_45_60 GENDER_MALE EMOTION_HAP INTENT_REQUEST"
(Note: '20' is not tagged in the example because 'NUMBER' was removed from the list for clarity, adjust if 'NUMBER' should be tagged)

Text to Annotate: "{cleaned_text_for_prompt}"'''
    # --- END OF NEW PROMPT ---

    logger.info(f"Annotating text with Gemini (first 100 chars after cleaning): {cleaned_text_for_prompt[:100]}...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # Or your preferred Gemini model
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            request_options={'timeout': 180},
            # Add safety settings if needed
            # safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none', 'SEXUAL':'block_none', 'DANGEROUS':'block_none'}
       )

        if response.candidates:
             annotated_text = ""
             try:
                 # Prioritize response.text if available
                 if hasattr(response, 'text') and response.text:
                     annotated_text = response.text.strip()
                 # Fallback to parts if response.text isn't populated (sometimes happens)
                 elif response.candidates[0].content.parts:
                     annotated_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
                 else:
                    logger.warning(f"Gemini annotation: No text found in response candidate for input '{cleaned_text_for_prompt[:50]}...'. Response: {response}")
                    return text_to_annotate, "Gemini response candidate found, but no text content." # Return original text + error

                 # Clean potential markdown code blocks
                 if annotated_text.startswith("```") and annotated_text.endswith("```"):
                     annotated_text = re.sub(r'^```[a-z]*\n?|\n?```$', '', annotated_text).strip()

                 # Apply tag fixing (including the split entity type fix)
                 final_text = fix_end_tags(annotated_text)

                 # Basic validation: Check if an INTENT tag exists at the end
                 if not re.search(r'INTENT_[A-Z_]+$', final_text):
                     logger.warning(f"Gemini annotation result missing INTENT tag at the end: '{final_text[-50:]}'")
                     # Optionally append a default intent or return with error? For now, just log.
                     # final_text += " INTENT_UNKNOWN" # Example fallback

                 logger.info(f"Gemini annotation successful (first 100 chars): {final_text[:100]}...")
                 return final_text, None

             except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                 logger.warning(f"Could not extract text from Gemini annotation response for input '{cleaned_text_for_prompt[:50]}...': {resp_e}. Response: {response}")
                 return text_to_annotate, f"Error parsing Gemini annotation response: {resp_e}" # Return original text + error

        else: # Handle blocked responses etc.
            error_message = "No candidates returned from Gemini annotation."
            try:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     error_message += f" Feedback: {response.prompt_feedback}"
                elif hasattr(response, 'error') and response.error:
                     error_message += f" API Error: {response.error}"
            except Exception as fb_e: error_message += f" (Feedback/Error access error: {fb_e})"
            logger.error(f"Gemini annotation error for input '{cleaned_text_for_prompt[:50]}...': {error_message}")
            return text_to_annotate, error_message # Return original text + error

    except Exception as e:
        logger.error(f"Error during Gemini annotation call for input '{cleaned_text_for_prompt[:50]}...': {e}", exc_info=True)
        # Check for specific Google API errors if possible
        # Example: if isinstance(e, google.api_core.exceptions.InvalidArgument): ...
        return text_to_annotate, f"Gemini API annotation error: {type(e).__name__}: {e}" # Return original text + error


# --- FastAPI Endpoint ---

@app.post("/process_audio_directory/",
          response_model=ProcessResponse,
          summary="Process Audio Files and Save to JSONL",
          description="Loads audio files from a directory, transcribes (Gemini/Whissle), "
                      "predicts Age/Gender/Emotion from audio, annotates Intent/Entities (Gemini), "
                      "combines results, and saves to a single JSONL file.")
async def process_audio_directory_endpoint(process_request: ProcessRequest):
    """
    Handles audio processing and saves results to JSONL.
    """
    results_list: List[JsonlRecord] = []
    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    # --- Input Validation ---
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
         raise HTTPException(status_code=400, detail="Whissle model selected, but not configured.")
    if model_choice == ModelChoice.gemini and not GEMINI_CONFIGURED:
        raise HTTPException(status_code=400, detail="Gemini model selected, but not configured.")
    # Check if annotation models are required/available
    annotation_required = True # Assume annotation is always required for now
    models_missing = []
    if age_gender_model is None: models_missing.append("Age/Gender")
    if emotion_model is None: models_missing.append("Emotion")
    if not GEMINI_CONFIGURED and annotation_required: models_missing.append("Gemini (for annotation)")
    if models_missing:
         # Decide behaviour: Warn or Fail
         warning_msg = f"Processing will continue, but annotations for {', '.join(models_missing)} will be unavailable."
         logger.warning(warning_msg)
         # If essential, raise error:
         # raise HTTPException(status_code=503, detail=f"Required models not loaded: {', '.join(models_missing)}")


    try:
        dir_path = Path(process_request.directory_path).resolve(strict=True)
        if not dir_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    except (FileNotFoundError, ValueError, TypeError) as path_e:
         raise HTTPException(status_code=400, detail=f"Invalid directory path: {process_request.directory_path}. Error: {path_e}")

    try:
        output_jsonl_path = Path(process_request.output_jsonl_path).resolve()
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if output_jsonl_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory, must be a file path: {output_jsonl_path}")
        # Overwrite existing file
        # if output_jsonl_path.exists():
        #      logger.warning(f"Output file {output_jsonl_path} exists. It will be overwritten.")
        #      # output_jsonl_path.unlink() # Uncomment to delete before writing
    except Exception as out_path_e:
        raise HTTPException(status_code=400, detail=f"Invalid output JSONL path: {process_request.output_jsonl_path}. Error: {out_path_e}")


    # --- File Discovery ---
    try:
        audio_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        logger.info(f"Found {len(audio_files)} audio files in {dir_path}")
    except Exception as e:
         logger.error(f"Error reading directory {dir_path}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error reading directory: {e}")

    if not audio_files:
        # Still create an empty file? Or just return? Let's return.
        return ProcessResponse(
            message=f"No supported audio files found in {dir_path}. Output file not created.",
            output_file=str(output_jsonl_path),
            processed_files=0, saved_records=0, errors=0
        )

    # --- Processing Loop ---
    # Open file *before* loop to append records incrementally
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: Optional[str] = None
                combined_text: Optional[str] = None
                duration: Optional[float] = None
                audio_data: Optional[np.ndarray] = None
                sample_rate: Optional[int] = None
                transcription_text: Optional[str] = None
                age_gender_emotion_tags: str = "" # Initialize

                logger.info(f"--- Processing {audio_file.name} ---")
                processed_files_count += 1

                try:
                    # 1. Get Duration & Load Audio
                    duration = get_audio_duration(audio_file)
                    if duration is None:
                        file_error = "Failed to get audio duration."
                    else:
                        audio_data, sample_rate, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
                        if load_err:
                            file_error = load_err
                        elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                            file_error = "Audio loading failed or SR mismatch after load."

                    # Only proceed if audio loaded successfully
                    if file_error is None and audio_data is not None:
                        # 2. Transcription (Async)
                        transcription_error = None
                        if model_choice == ModelChoice.whissle:
                            transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                        else: # Gemini
                            transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)

                        if transcription_error:
                            file_error = f"Transcription failed: {transcription_error}"
                            # Log the error but potentially still process AGE/GENDER/EMO? Yes.
                        elif transcription_text is None:
                            # Should not happen if error handling is correct, but safeguard
                            file_error = "Transcription returned None without error."
                        else:
                            transcription_text = transcription_text.strip() # Clean whitespace

                        # 3. Predict Age/Gender/Emotion (regardless of transcription success/failure, if audio loaded)
                        # Run these predictions using asyncio.to_thread for non-blocking IO
                        age_pred, gender_idx, age_gender_err = await asyncio.to_thread(
                            predict_age_gender, audio_data, TARGET_SAMPLE_RATE
                        )
                        if age_gender_err: logger.warning(f"Age/Gender prediction failed for {audio_file.name}: {age_gender_err}")

                        emotion_label, emotion_err = await asyncio.to_thread(
                            predict_emotion, audio_data, TARGET_SAMPLE_RATE
                        )
                        if emotion_err: logger.warning(f"Emotion prediction failed for {audio_file.name}: {emotion_err}")

                        # Format Age/Gender/Emotion Tags (always do this if audio was loaded)
                        age_gender_emotion_tags = format_age_gender_emotion_tags(age_pred, gender_idx, emotion_label)

                        # 4. Annotate with Gemini IF transcription was successful and text exists
                        if transcription_text is not None and transcription_text != "":
                             # We already have AGE/GENDER/EMO tags, pass them into the annotation function's input
                             # So Gemini can preserve them as per the prompt.
                             text_with_pre_tags = f"{transcription_text} {age_gender_emotion_tags}".strip()

                             gemini_annotated_text, gemini_err = await annotate_text_with_gemini(text_with_pre_tags)

                             if gemini_err:
                                 logger.warning(f"Gemini Intent/Entity annotation failed for {audio_file.name}: {gemini_err}. Using text + A/G/E tags.")
                                 # Fallback: Use the text + AGE/GENDER/EMOTION tags without further annotation
                                 combined_text = text_with_pre_tags
                                 # Append a generic intent if annotation failed?
                                 if not re.search(r'INTENT_[A-Z_]+$', combined_text): combined_text += " INTENT_UNKNOWN"
                             elif gemini_annotated_text is None or gemini_annotated_text.strip() == "":
                                  logger.warning(f"Gemini annotation returned empty/None for {audio_file.name}. Using text + A/G/E tags.")
                                  combined_text = text_with_pre_tags # Fallback
                                  if not re.search(r'INTENT_[A-Z_]+$', combined_text): combined_text += " INTENT_UNKNOWN"
                             else:
                                 # Success: Gemini handled transcription + A/G/E + added its own tags
                                 combined_text = gemini_annotated_text # Use the fully annotated text
                        elif transcription_text == "":
                             # Handle case of empty transcription (no speech detected)
                             logger.info(f"No speech detected in {audio_file.name}. Applying only A/G/E tags.")
                             combined_text = f"[No speech detected] {age_gender_emotion_tags}".strip()
                             # Add a specific intent for no speech?
                             combined_text += " INTENT_NO_SPEECH"
                        else:
                             # Handle transcription failure - use only A/G/E tags
                             logger.warning(f"Transcription failed for {audio_file.name}. Applying only A/G/E tags.")
                             combined_text = f"[Transcription Failed] {age_gender_emotion_tags}".strip()
                             combined_text += " INTENT_TRANSCRIPTION_FAILED"


                except Exception as e:
                    logger.error(f"Unexpected error processing {audio_file.name}: {e}", exc_info=True)
                    file_error = f"Unexpected processing error: {type(e).__name__}: {e}"
                    error_count += 1 # Count unexpected errors here
                    # Ensure A/G/E tags are added even if pipeline fails mid-way, if audio was loaded
                    if age_gender_emotion_tags and not combined_text:
                         combined_text = f"[Processing Error] {age_gender_emotion_tags}".strip()
                    elif not combined_text:
                         combined_text = "[Processing Error]"


                # Create JSONL record (even for errors)
                record = JsonlRecord(
                    audio_filepath=str(audio_file.resolve()),
                    text=combined_text, # Use the final combined text
                    duration=duration,
                    model_used_for_transcription=model_choice.value,
                    error=file_error # Log specific step errors
                )

                # Write record to file immediately
                outfile.write(record.model_dump_json(exclude_none=True) + "\n") # Pydantic v2+; exclude None fields
                # outfile.write(record.json(exclude_none=True) + "\n") # Pydantic v1; exclude None fields

                if file_error:
                    if error_count == 0: error_count = 1 # Ensure errors logged in loop are counted if unexpected didn't trigger
                    # Only count specific file errors if not already counted by unexpected exception
                    # error_count += 1 # This might double count if unexpected error also happened
                    pass # error_count is incremented by the unexpected exception handler if that runs
                else:
                    saved_records_count += 1 # Count records saved without specific file errors

                # Clean up
                del audio_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Finished processing. Wrote {processed_files_count} records attempt to {output_jsonl_path}")

    except IOError as e:
        logger.error(f"Failed to write output JSONL file {output_jsonl_path}: {e}", exc_info=True)
        # Return error even if some processing was done
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")


    # --- Final Response ---
    final_message = (
        f"Processed {processed_files_count}/{len(audio_files) if audio_files else 0} files from directory. "
        f"Attempted to save {processed_files_count} records to JSONL. "
        f"{saved_records_count} records saved without processing errors. "
        f"Encountered errors during processing for {error_count} files (check logs and 'error' field in JSONL)."
    )

    return ProcessResponse(
        message=final_message,
        output_file=str(output_jsonl_path),
        processed_files=processed_files_count,
        saved_records=saved_records_count, # Reflects records saved without errors
        errors=error_count # Reflects files with processing errors
    )


# --- Root Endpoint ---
@app.get("/", summary="API Root", description="Provides basic information about the API and model status.")
async def root():
    return {
        "message": "Welcome to the Audio Processing API v1.3.0",
        "docs_url": "/docs",
        "gemini_configured": GEMINI_CONFIGURED,
        "whissle_available": WHISSLE_AVAILABLE,
        "whissle_configured": WHISSLE_CONFIGURED,
        "age_gender_model_loaded": age_gender_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
        }

# --- Run the app ---
if __name__ == "__main__":
    # Determine the directory of the current script
    script_dir = Path(__file__).parent.resolve()

    # Define the default app path relative to the script directory
    # Assumes your script is named 'fastapi_script.py' or similar inside the project dir
    # If your script is named 'main.py', use "main:app"
    # Adjust 'fastapi_script_name' to the actual name of your Python file
    fastapi_script_name = Path(__file__).stem # Gets the filename without extension
    app_module_string = f"{fastapi_script_name}:app"

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_env = os.getenv("RELOAD", "false").lower() # Default reload to false for stability
    reload = reload_env == "true"

    logger.info(f"Starting FastAPI server for '{app_module_string}' on {host}:{port}...")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    if reload:
        logger.info(f"Reload mode enabled. Watching for changes in: {script_dir}")

    uvicorn.run(
        app_module_string,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(script_dir)] if reload else None
    ) 