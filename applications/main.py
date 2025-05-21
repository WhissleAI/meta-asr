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

load_dotenv('/home/dchauhan/workspace/meta-asr/applications/.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
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
    description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities, "
                "and saves results to a JSONL manifest file.",
    version="1.4.1" # Incremented version for fixes
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
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
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

class ProcessRequest(BaseModel):
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    model_choice: ModelChoice = PydanticField(..., description="The transcription model to use.")
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")
    annotations: Optional[List[str]] = PydanticField(
        None,
        description="List of annotations to include (age, gender, emotion, entity, intent).",
        example=["age", "gender", "emotion"]
    )

class TranscriptionJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

class AnnotatedJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int

def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=False) # Keep exc_info False for brevity in logs unless debugging
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

def format_age_gender_emotion_tags(age: Optional[float], gender_idx: Optional[int], emotion_label: Optional[str]) -> str:
    tags = []
    if age is not None:
        try:
            actual_age = round(age, 1)
            age_brackets: List[Tuple[float, str]] = [
                (18, "0_17"), (25, "18_24"), (35, "25_34"),
                (45, "35_44"), (55, "45_54"), (65, "55_64"),
                (float('inf'), "65PLUS")
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
    else: tags.append("AGE_UNKNOWN")

    if gender_idx is not None:
        if gender_idx == 1: tags.append("GENDER_MALE")
        elif gender_idx == 0: tags.append("GENDER_FEMALE")
        else: tags.append("GENDER_UNKNOWN")
    else: tags.append("GENDER_UNKNOWN")

    if emotion_label and emotion_label != "SHORT_AUDIO":
        emotion_tag = emotion_label.upper().replace(" ", "_")
        tags.append(f"EMOTION_{emotion_tag}")
    elif emotion_label == "SHORT_AUDIO":
         tags.append("EMOTION_SHORT_AUDIO")
    else: tags.append("EMOTION_UNKNOWN")
    return " ".join(tags)

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
    finally:
        if uploaded_file and hasattr(uploaded_file, 'name'):
             try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
             except Exception as del_e: logger.warning(f"Could not delete Gemini file {uploaded_file.name} in finally: {del_e}")

def remove_existing_tags(text: str) -> str:
    """
    Cleans text for Gemini annotation input.
    - PRESERVES AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE.
    - REMOVES pre-existing INTENT_* tags (Gemini will re-classify intent).
    - Decision on ENTITY_ tags: Currently PRESERVES correctly formatted ENTITY_...END tags
      because the Gemini annotation prompt includes examples of preserving them.
      If Gemini should re-tag ALL entities, uncomment the ENTITY_ removal line.
    """
    if not isinstance(text, str): return text

    preserve_patterns = [
        r'\bAGE_[A-Z0-9_]+\b',
        r'\bGENDER_(MALE|FEMALE|OTHER|UNKNOWN)\b',
        r'\bEMOTION_[A-Z_]+\b',
        r'\bSPEAKER_CHANGE\b',
        r'ENTITY_[A-Z0-9_]+\s+[\s\S]*?\s+END\b', # Preserve existing, well-formed entity tags
    ]

    preserved_tags_map = {}
    placeholder_count = 0

    def replace_with_placeholder(match):
        nonlocal placeholder_count
        tag_content = match.group(0)
        placeholder = f"__PRESERVED_TAG_CONTENT_{placeholder_count}__"
        preserved_tags_map[placeholder] = tag_content
        placeholder_count += 1
        return f" {placeholder} " # Add spaces for clean separation

    temp_text_with_placeholders = text
    for pattern in preserve_patterns:
        temp_text_with_placeholders = re.sub(pattern, replace_with_placeholder, temp_text_with_placeholders, flags=re.IGNORECASE)

    # Remove ANY INTENT_ tag from the text that now contains placeholders
    cleaned_text = re.sub(r'\s*INTENT_[A-Z_0-9]+\s*', ' ', temp_text_with_placeholders, flags=re.IGNORECASE)

    # Remove other legacy tags if necessary (NER_, old ENTITY_ formats not matching the preserve pattern)
    cleaned_text = re.sub(r'\bNER_\w+\s*', ' ', cleaned_text, flags=re.IGNORECASE)
    # If you had other specific old entity formats to remove:
    # cleaned_text = re.sub(r'\bOLD_ENTITY_FORMAT\s+.*?\s+END_OLD', ' ', cleaned_text, flags=re.IGNORECASE)

    # Restore preserved tags
    for placeholder, original_content in preserved_tags_map.items():
        cleaned_text = cleaned_text.replace(f" {placeholder} ", f" {original_content} ")

    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text

def fix_end_tags(text: str) -> str:
    if not isinstance(text, str): return text
    # Fix split entity types like 'ENTITY_TYPE X END' -> 'ENTITY_TYPEX END'
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)\s+([A-Z0-9])\s+END\b', r'\1\2 END', text, flags=re.IGNORECASE)
    # Ensure space before END if attached: wordEND -> word END
    text = re.sub(r'(\S)END\b', r'\1 END', text, flags=re.IGNORECASE)
    # Ensure space after ENTITY_TYPE if attached: ENTITY_TYPEword -> ENTITY_TYPE word
    text = re.sub(r'\b(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text, flags=re.IGNORECASE)
    # Ensure space before INTENT_ if attached: wordINTENT_ -> word INTENT_
    text = re.sub(r'(\S)(INTENT_[A-Z_0-9]+\b)', r'\1 \2', text, flags=re.IGNORECASE)
    # Ensure space *after* END if followed by non-space: ENDword -> END word
    text = re.sub(r'\bEND(\S)', r'END \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

async def annotate_text_with_gemini(text_to_annotate: str) -> tuple[Optional[str], Optional[str]]:
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    if not text_to_annotate or text_to_annotate.isspace():
        return "", None

    cleaned_text_for_prompt = remove_existing_tags(text_to_annotate)
    if not cleaned_text_for_prompt or cleaned_text_for_prompt.isspace():
        return text_to_annotate, None

    entity_list_json_str = """
    ["PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "NUMBER", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
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
    "PRODUCT", "SERVICE", "CATEGORY", "BRAND_NAME", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD",
    "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE",
    "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "RECIPE", "INGREDIENT",
    "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD",
    "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
    "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
    "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
    "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
    "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
    "POLLUTION_LEVEL", "UV_INDEX",
    "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND",
    "USER_HANDLE", "EMAIL_ADDRESS", "PHONE_NUMBER", "IPV4_ADDRESS", "IPV6_ADDRESS", "GPS_COORDINATE", "LATITUDE", "LONGITUDE", "GEOLOCATION",
    "STREET_NAME", "BUILDING_NUMBER", "FLOOR_NUMBER", "BUSINESS_NAME", "MODEL", "SERIAL_NUMBER", "IMEI", "IMSI", "DEVICE_ID",
    "OS_VERSION", "FILE_PATH", "FILE_NAME", "FILE_EXTENSION", "DOCUMENT_TITLE", "DOCUMENT_ID", "LEGAL_ENTITY", "TAX_DOCUMENT",
    "BILLING_ADDRESS", "SHIPPING_ADDRESS", "COUPON_CODE", "LOYALTY_CARD_NUMBER", "PRODUCT_ID", "SKU", "BARCODE", "QR_CODE",
    "TRANSACTION_CODE", "EVENT_ID", "SESSION_ID", "ACTION_ID", "CLICK_POSITION", "SCROLL_DEPTH",
    "VIDEO_ID", "AUDIO_TRACK", "SUBTITLE_LANGUAGE", "CHAPTER_TITLE", "CHAPTER_NUMBER", "EPISODE_NUMBER",
    "MOVIE_TITLE", "DIRECTOR_NAME", "ACTOR_NAME", "BOOK_TITLE", "AUTHOR_NAME", "PUBLISHER", "ISBN", "ISSN",
    "COURSE_NAME", "INSTRUCTOR_NAME", "STUDENT_ID", "GRADE", "CLASSROOM_NUMBER", "SCHOOL_NAME", "DEGREE_PROGRAM",
    "MAJOR", "MINOR", "CERTIFICATE_NAME", "EXAM_SCORE", "CERTIFICATION_ID", "TRAINING_PROGRAM",
    "PLATFORM", "APPLICATION", "SOFTWARE_PACKAGE", "API_ENDPOINT", "SERVICE_NAME", "SERVER_IP", "DATABASE_TABLE",
    "QUERY", "ERROR_CODE", "LOG_LEVEL", "SESSION_DURATION", "BROWSER_TYPE", "DEVICE_TYPE"
    ]
    """
    parsed_entity_list = json.loads(entity_list_json_str)
    entity_list_for_prompt_str = ", ".join(f'"{entity}"' for entity in parsed_entity_list)

    prompt = f'''Analyze the following sentence. It might already contain tags like AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE, or correctly formatted ENTITY_<TYPE> existing entity END tags.

Your tasks are:
1.  **Preserve Existing Tags:** Keep any AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE, or correctly pre-tagged `ENTITY_<TYPE> text END` tags exactly where they are. Do not modify or move them unless instructed below.
2.  **Annotate NEW Entities:** Identify and tag NEW entities in parts of the text NOT already tagged as `ENTITY_<TYPE> text END`. Tag these new entities **only** from this specific list: [{entity_list_for_prompt_str}]. Use the format `ENTITY_<TYPE> identified text END`. Ensure `<TYPE>` exactly matches an entry in the list. Do not tag anything not on the list. Place the start tag immediately before the entity and the ` END` tag (with a preceding space) immediately after it.
3.  **Classify Intent:** Determine the primary intent of the sentence. Choose **one** concise, descriptive intent label from this suggested list if appropriate, or a similar one if needed: `REQUEST_INFO`, `PROVIDE_INFO`, `BOOK_APPOINTMENT`, `CANCEL_APPOINTMENT`, `RESCHEDULE_APPOINTMENT`, `PROVIDE_FEEDBACK`, `MAKE_PURCHASE`, `RETURN_ITEM`, `TRACK_ORDER`, `SOCIAL_CHITCHAT`, `HEALTH_UPDATE`, `REQUEST_ASSISTANCE`, `TECHNICAL_SUPPORT`, `CONFIRMATION`, `DENIAL`, `GREETING`, `FAREWELL`, `AGREEMENT`, `DISAGREEMENT`, `EXPRESS_EMOTION`, `OTHER`. If no clear intent fits, use `OTHER`.
4.  **Add Intent Tag:** Append exactly ONE `INTENT_<ChosenIntentLabel>` tag at the VERY END of the entire processed sentence, after all text, entity tags, preserved tags, and ` END` markers.
5.  **Output:** Return the fully processed sentence as a single string. Do not wrap the entire output in quotes or markdown code blocks.

**CRITICAL FORMATTING RULES:**
*   Do NOT add extra spaces around the `ENTITY_<TYPE>` start tag or after the `END` tag (except for the final INTENT tag).
*   Ensure there is exactly one space between the identified entity text and its ` END` tag. Example: `ENTITY_CITY london END` (Correct).
*   The <TYPE> in `ENTITY_<TYPE>` must be a single, unbroken string from the provided entity list. Do NOT insert spaces within <TYPE>. Example: `ENTITY_PERSON_NAME david END` (Correct), `ENTITY_PERSON_NAM E david END` (Incorrect).
*   The final `INTENT_<TYPE>` tag must be the absolute last part of the output string, with no characters or spaces following it.

**Example Input (Tagging new entities, preserving AGE/GENDER):** "can you book a flight for alice to paris on next tuesday GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL"
**Example Output:** "can you book a flight for ENTITY_PERSON_NAME alice END to ENTITY_CITY paris END on ENTITY_DATE next tuesday END GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL INTENT_BOOK_APPOINTMENT"

**Example Input (Preserving existing correct ENTITY tags and AGE/GENDER, adding intent):** "The order ENTITY_ORDER_NUMBER 12345 END is for ENTITY_PRODUCT_NAME SuperWidget END. SPEAKER_CHANGE GENDER_MALE AGE_45_54"
**Example Output:** "The order ENTITY_ORDER_NUMBER 12345 END is for ENTITY_PRODUCT_NAME SuperWidget END. SPEAKER_CHANGE GENDER_MALE AGE_45_54 INTENT_PROVIDE_INFO"

**Example Input (No relevant entities from the list, preserving AGE/GENDER):** "Hello there! How are you doing today? AGE_30_40 EMOTION_HAPPY"
**Example Output:** "Hello there! How are you doing today? AGE_30_40 EMOTION_HAPPY INTENT_SOCIAL_CHITCHAT"

**Text to Annotate:** "{cleaned_text_for_prompt}"'''

    logger.info(f"Annotating with Gemini (model: gemini-1.5-pro-latest, temp: 0.1, input: '{cleaned_text_for_prompt[:100]}...')")
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = await asyncio.to_thread(
            model.generate_content, contents=[prompt],
            generation_config=generation_config, request_options={'timeout': 180}
        )

        if response.candidates:
            annotated_text = ""
            try:
                if response.candidates[0].content and response.candidates[0].content.parts:
                    annotated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
                elif hasattr(response, 'text') and response.text is not None:
                     annotated_text = response.text.strip()
                else:
                    return text_to_annotate, "Gemini response candidate found, but no text content."

                if not annotated_text:
                    return text_to_annotate, "Gemini response text extracted as empty."

                if annotated_text.startswith("```") and annotated_text.endswith("```"):
                    annotated_text = re.sub(r'^```[a-z]*\n?|\n?```$', '', annotated_text).strip()
                
                if len(annotated_text) > 1 and annotated_text.startswith('"') and annotated_text.endswith('"'):
                    content_inside_quotes = annotated_text[1:-1]
                    # Heuristic: only strip if intent is not already well-formed inside the quotes
                    if not re.search(r'INTENT_[A-Z_0-9]+$', content_inside_quotes.strip()):
                        annotated_text = content_inside_quotes
                    else: # Intent seems to be at end within quotes, strip and let validation handle
                        annotated_text = content_inside_quotes

                final_text = fix_end_tags(annotated_text)
                final_text_stripped = final_text.strip()

                if not re.search(r'INTENT_[A-Z_0-9]+$', final_text_stripped):
                    logger.warning(f"Gemini annotation result missing expected INTENT tag at the end: '{final_text_stripped[-70:]}' (Original from model: '{annotated_text[-70:]}')")
                    if re.search(r'INTENT_[A-Z_0-9]+', final_text_stripped):
                         logger.warning("An INTENT tag was found but not at the end. Review `remove_existing_tags` logic for annotation input.")
                    final_text = f"{final_text_stripped} INTENT_UNKNOWN".strip()
                    logger.info("Appended INTENT_UNKNOWN due to missing/misplaced intent tag.")
                else:
                    final_text = final_text_stripped
                
                logger.info(f"Gemini annotation successful (result: '{final_text[:100]}...')")
                return final_text, None
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                return text_to_annotate, f"Error parsing Gemini response: {resp_e}"
        else:
            error_message = f"No candidates from Gemini for '{cleaned_text_for_prompt[:50]}...'."
            if response.prompt_feedback:
                feedback_msg = f"Block reason: {response.prompt_feedback.block_reason}. Ratings: {response.prompt_feedback.safety_ratings}"
                error_message += f" Feedback: {feedback_msg}"
            return text_to_annotate, error_message
    except Exception as e:
        return text_to_annotate, f"Gemini API error: {type(e).__name__}: {e}"

def validate_paths(directory_path: str, output_jsonl_path: str) -> Tuple[Path, Path]:
    try:
        dir_path = Path(directory_path).resolve(strict=True)
        if not dir_path.is_dir(): raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    except Exception as path_e: raise HTTPException(status_code=400, detail=f"Invalid directory path: {directory_path}. Error: {path_e}")
    try:
        output_path = Path(output_jsonl_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir(): raise HTTPException(status_code=400, detail=f"Output path is a directory: {output_path}")
        if not os.access(output_path.parent, os.W_OK): raise HTTPException(status_code=403, detail=f"No write permissions: {output_path.parent}")
        if output_path.exists(): logger.warning(f"Output file {output_path} exists and will be overwritten.")
    except HTTPException: raise
    except Exception as out_path_e: raise HTTPException(status_code=400, detail=f"Invalid output path: {output_jsonl_path}. Error: {out_path_e}")
    return dir_path, output_path

def discover_audio_files(dir_path: Path) -> List[Path]:
    try:
        audio_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        logger.info(f"Found {len(audio_files)} supported audio files in {dir_path}")
        return audio_files
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error scanning directory: {e}")

@app.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED: raise HTTPException(status_code=400, detail="Whissle not configured.")
    if model_choice == ModelChoice.gemini and not GEMINI_CONFIGURED: raise HTTPException(status_code=400, detail="Gemini not configured.")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
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
                    if model_choice == ModelChoice.whissle: transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    else: transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                    if transcription_error: file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None: file_error = "Transcription returned None."
                    else: transcription_text = transcription_text.strip()
                except Exception as e: file_error = f"Unexpected error: {type(e).__name__}: {e}"

                record = TranscriptionJsonlRecord(audio_filepath=str(audio_file.resolve()), text=transcription_text, duration=duration, model_used_for_transcription=model_choice.value, error=file_error)
                try: outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    if not file_error: file_error = f"JSONL write error: {write_e}"
                    else: file_error += f"; JSONL write error: {write_e}"
                if file_error: error_count += 1
                else: saved_records_count += 1
                gc.collect()
    except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")
    
    msg = f"Processed {processed_files_count}/{len(audio_files)}. Saved: {saved_records_count}. Errors: {error_count}."
    return ProcessResponse(message=msg, output_file=str(output_jsonl_path), processed_files=processed_files_count, saved_records=saved_records_count, errors=error_count)

# @app.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
# async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
#     model_choice = process_request.model_choice
#     if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED: raise HTTPException(status_code=400, detail="Whissle not configured.")
#     if not GEMINI_CONFIGURED: raise HTTPException(status_code=400, detail="Gemini not configured (required for annotation).")
    
#     models_missing = []
#     if age_gender_model is None: models_missing.append("Age/Gender")
#     if emotion_model is None: models_missing.append("Emotion")
#     if models_missing: logger.warning(f"Annotation models not loaded: {', '.join(models_missing)}. Proceeding without these.")

#     dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
#     audio_files = discover_audio_files(dir_path)

#     if not audio_files:
#         try:
#             with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
#         except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
#         return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

#     processed_files_count = 0; saved_records_count = 0; error_count = 0
#     try:
#         with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
#             for audio_file in audio_files:
#                 file_error_details: List[str] = []; combined_text: Optional[str] = None; duration: Optional[float] = None
#                 audio_data: Optional[np.ndarray] = None; sample_rate: Optional[int] = None
#                 transcription_text: Optional[str] = None; age_gender_emotion_tags: str = ""
#                 logger.info(f"--- Processing {audio_file.name} (Full Annotation) ---")
#                 processed_files_count += 1
#                 try:
#                     duration = get_audio_duration(audio_file)
#                     audio_data, sample_rate, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
#                     if load_err: file_error_details.append(load_err)
#                     elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE: file_error_details.append("Audio load/SR mismatch.")

#                     if model_choice == ModelChoice.whissle: transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
#                     else: transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
#                     if transcription_error: file_error_details.append(f"Transcription: {transcription_error}")
#                     elif transcription_text is None: file_error_details.append("Transcription returned None.")
#                     else: transcription_text = transcription_text.strip()

#                     age_pred, gender_idx, emotion_label = None, None, None
#                     if audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
#                         try:
#                             age_gender_task = asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE)
#                             emotion_task = asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE)
#                             age_gender_result, emotion_result = await asyncio.gather(age_gender_task, emotion_task)
#                             age_pred, gender_idx, age_gender_err = age_gender_result
#                             if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
#                             emotion_label, emotion_err = emotion_result
#                             if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
#                         except Exception as pred_e: file_error_details.append(f"A/G/E Error: {pred_e}")
#                     elif not ("Audio load/SR mismatch" in " ".join(file_error_details) or load_err):
#                         file_error_details.append("Skipped A/G/E (audio load issue).")
                    
#                     age_gender_emotion_tags = format_age_gender_emotion_tags(age_pred, gender_idx, emotion_label)

#                     if transcription_text is not None and transcription_text != "":
#                         text_with_pre_tags = f"{transcription_text} {age_gender_emotion_tags}".strip()
#                         gemini_annotated_text, gemini_err = await annotate_text_with_gemini(text_with_pre_tags)
#                         if gemini_err:
#                             file_error_details.append(f"ANNOTATION_FAIL: {gemini_err}")
#                             combined_text = text_with_pre_tags
#                             if not re.search(r'\sINTENT_[A-Z_0-9]+$', combined_text): combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_FAILED".strip()
#                         elif gemini_annotated_text is None or gemini_annotated_text.strip() == "":
#                             file_error_details.append("ANNOTATION_EMPTY: Gemini returned empty.")
#                             combined_text = text_with_pre_tags
#                             if not re.search(r'\sINTENT_[A-Z_0-9]+$', combined_text): combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_EMPTY".strip()
#                         else: combined_text = gemini_annotated_text
#                     elif transcription_text == "":
#                         combined_text = f"[NO_SPEECH] {age_gender_emotion_tags}".strip()
#                         combined_text = f"{combined_text.strip()} INTENT_NO_SPEECH".strip()
#                     else: # Transcription failed
#                         combined_text = f"[TRANSCRIPTION_FAILED] {age_gender_emotion_tags}".strip()
#                         combined_text = f"{combined_text.strip()} INTENT_TRANSCRIPTION_FAILED".strip()
#                 except Exception as e:
#                     file_error_details.append(f"Unexpected error: {type(e).__name__}: {e}")
#                     if not combined_text: combined_text = f"[PROCESSING_ERROR] {age_gender_emotion_tags}".strip(); combined_text = f"{combined_text.strip()} INTENT_PROCESSING_ERROR".strip()

#                 final_error_msg = "; ".join(file_error_details) if file_error_details else None
#                 record = AnnotatedJsonlRecord(audio_filepath=str(audio_file.resolve()), text=combined_text, duration=duration, model_used_for_transcription=model_choice.value, error=final_error_msg)
                
#                 current_errors_before_write = error_count
#                 try: outfile.write(record.model_dump_json(exclude_none=True) + "\n")
#                 except Exception as write_e:
#                     logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
#                     if not final_error_msg: final_error_msg = f"JSONL write error: {write_e}" # This might not be captured on record if record is already formed
#                     # Ensure error_count is incremented if write fails, even if processing was 'ok'
#                     if error_count == current_errors_before_write and not file_error_details: # Only inc if no other proc error already counted it
#                          error_count += 1
                
#                 # Increment error count IF there were processing errors OR if final_error_msg now includes a write error
#                 # and it wasn't counted by the write exception block
#                 if file_error_details: # If any processing error occurred
#                     if error_count == current_errors_before_write: # Ensure it's counted once
#                         error_count +=1
#                 elif not file_error_details and not final_error_msg : # No processing errors, no write errors
#                      saved_records_count += 1
#                 # If final_error_msg has something but file_error_details was empty, it implies write error was the only issue
#                 # This is tricky to count perfectly without more state; the current logic prioritizes counting if any error string is present.

#                 del audio_data; gc.collect()
#                 if torch.cuda.is_available(): torch.cuda.empty_cache()
#         logger.info(f"Finished full annotation. Attempted {processed_files_count} records.")
#     except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

# selective annotation 
@app.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    if process_request.annotations and "entity" in process_request.annotations or "intent" in process_request.annotations:
        if not GEMINI_CONFIGURED:
            raise HTTPException(status_code=400, detail="Gemini not configured (required for entity/intent annotation).")

    # Validate annotations
    valid_annotations = {"age", "gender", "emotion", "entity", "intent"}
    if process_request.annotations:
        invalid_annotations = set(process_request.annotations) - valid_annotations
        if invalid_annotations:
            raise HTTPException(status_code=400, detail=f"Invalid annotations: {invalid_annotations}")

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
                combined_text: Optional[str] = None
                duration: Optional[float] = None
                audio_data: Optional[np.ndarray] = None
                sample_rate: Optional[int] = None
                transcription_text: Optional[str] = None
                age_gender_emotion_tags: str = ""
                logger.info(f"--- Processing {audio_file.name} (Selective Annotation) ---")
                processed_files_count += 1

                try:
                    duration = get_audio_duration(audio_file)
                    if process_request.annotations and any(a in ["age", "gender", "emotion"] for a in process_request.annotations):
                        audio_data, sample_rate, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
                        if load_err:
                            file_error_details.append(load_err)
                        elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                            file_error_details.append("Audio load/SR mismatch.")

                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    else:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                    if transcription_error:
                        file_error_details.append(f"Transcription: {transcription_error}")
                    elif transcription_text is None:
                        file_error_details.append("Transcription returned None.")
                    else:
                        transcription_text = transcription_text.strip()

                    age_pred, gender_idx, emotion_label = None, None, None
                    if process_request.annotations and audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                        try:
                            tasks = []
                            if "age" in process_request.annotations or "gender" in process_request.annotations:
                                tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                            if "emotion" in process_request.annotations:
                                tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))
                            results = await asyncio.gather(*tasks, return_exceptions=True)

                            for result in results:
                                if isinstance(result, tuple) and len(result) == 3:  # Age/Gender
                                    age_pred, gender_idx, age_gender_err = result
                                    if age_gender_err:
                                        file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                elif isinstance(result, tuple) and len(result) == 2:  # Emotion
                                    emotion_label, emotion_err = result
                                    if emotion_err:
                                        file_error_details.append(f"EMO_WARN: {emotion_err}")

                        except Exception as pred_e:
                            file_error_details.append(f"A/G/E Error: {pred_e}")

                    # Format only requested tags
                    if process_request.annotations:
                        tags = []
                        if "age" in process_request.annotations and age_pred is not None:
                            try:
                                actual_age = round(age_pred, 1)
                                age_brackets = [
                                    (18, "0_17"), (25, "18_24"), (35, "25_34"),
                                    (45, "35_44"), (55, "45_54"), (65, "55_64"),
                                    (float('inf'), "65PLUS")
                                ]
                                age_tag = "UNKNOWN"
                                for threshold, bracket in age_brackets:
                                    if actual_age < threshold:
                                        age_tag = bracket
                                        break
                                tags.append(f"AGE_{age_tag}")
                            except Exception as age_e:
                                logger.error(f"Error formatting age tag: {age_e}")
                                tags.append("AGE_ERROR")
                        elif "age" in process_request.annotations:
                            tags.append("AGE_UNKNOWN")

                        if "gender" in process_request.annotations and gender_idx is not None:
                            if gender_idx == 1:
                                tags.append("GENDER_MALE")
                            elif gender_idx == 0:
                                tags.append("GENDER_FEMALE")
                            else:
                                tags.append("GENDER_UNKNOWN")
                        elif "gender" in process_request.annotations:
                            tags.append("GENDER_UNKNOWN")

                        if "emotion" in process_request.annotations and emotion_label and emotion_label != "SHORT_AUDIO":
                            emotion_tag = emotion_label.upper().replace(" ", "_")
                            tags.append(f"EMOTION_{emotion_tag}")
                        elif "emotion" in process_request.annotations and emotion_label == "SHORT_AUDIO":
                            tags.append("EMOTION_SHORT_AUDIO")
                        elif "emotion" in process_request.annotations:
                            tags.append("EMOTION_UNKNOWN")

                        age_gender_emotion_tags = " ".join(tags)

                    if transcription_text is not None and transcription_text != "":
                        text_with_pre_tags = f"{transcription_text} {age_gender_emotion_tags}".strip() if age_gender_emotion_tags else transcription_text
                        if process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations):
                            gemini_annotated_text, gemini_err = await annotate_text_with_gemini(text_with_pre_tags)
                            if gemini_err:
                                file_error_details.append(f"ANNOTATION_FAIL: {gemini_err}")
                                combined_text = text_with_pre_tags
                                if "intent" in process_request.annotations and not re.search(r'\sINTENT_[A-Z_0-9]+$', combined_text):
                                    combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_FAILED".strip()
                            elif gemini_annotated_text is None or gemini_annotated_text.strip() == "":
                                file_error_details.append("ANNOTATION_EMPTY: Gemini returned empty.")
                                combined_text = text_with_pre_tags
                                if "intent" in process_request.annotations and not re.search(r'\sINTENT_[A-Z_0-9]+$', combined_text):
                                    combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_EMPTY".strip()
                            else:
                                combined_text = gemini_annotated_text
                        else:
                            combined_text = text_with_pre_tags
                    elif transcription_text == "":
                        combined_text = f"[NO_SPEECH] {age_gender_emotion_tags}".strip() if age_gender_emotion_tags else "[NO_SPEECH]"
                        if process_request.annotations and "intent" in process_request.annotations:
                            combined_text = f"{combined_text.strip()} INTENT_NO_SPEECH".strip()
                    else:
                        combined_text = f"[TRANSCRIPTION_FAILED] {age_gender_emotion_tags}".strip() if age_gender_emotion_tags else "[TRANSCRIPTION_FAILED]"
                        if process_request.annotations and "intent" in process_request.annotations:
                            combined_text = f"{combined_text.strip()} INTENT_TRANSCRIPTION_FAILED".strip()

                except Exception as e:
                    file_error_details.append(f"Unexpected error: {type(e).__name__}: {e}")
                    if not combined_text:
                        combined_text = f"[PROCESSING_ERROR] {age_gender_emotion_tags}".strip() if age_gender_emotion_tags else "[PROCESSING_ERROR]"
                        if process_request.annotations and "intent" in process_request.annotations:
                            combined_text = f"{combined_text.strip()} INTENT_PROCESSING_ERROR".strip()

                final_error_msg = "; ".join(file_error_details) if file_error_details else None
                record = AnnotatedJsonlRecord(
                    audio_filepath=str(audio_file.resolve()),
                    text=combined_text,
                    duration=duration,
                    model_used_for_transcription=model_choice.value,
                    error=final_error_msg
                )

                current_errors_before_write = error_count
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                    if not final_error_msg:
                        final_error_msg = f"JSONL write error: {write_e}"
                    if error_count == current_errors_before_write and not file_error_details:
                        error_count += 1

                if file_error_details:
                    if error_count == current_errors_before_write:
                        error_count += 1
                elif not file_error_details and not final_error_msg:
                    saved_records_count += 1

                del audio_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        truly_successful_saves = processed_files_count - error_count
        final_message = (
            f"Processed {processed_files_count}/{len(audio_files)} files for selective annotation. "
            f"Attempted to save {processed_files_count} records. "
            f"{truly_successful_saves} records are considered fully successful (no errors reported). "
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

    # Adjust final saved_records_count based on errors. If total processed is X and errors is Y, saved is X-Y.
    # This provides a clearer picture than trying to perfectly sync saved_records_count increments within the loop.
    # However, this assumes an error means the record wasn't useful. The current saved_records_count is more optimistic.
    # Let's refine the message to be more precise:
    truly_successful_saves = processed_files_count - error_count # A stricter definition of success

    final_message = (
        f"Processed {processed_files_count}/{len(audio_files)} files for full annotation. "
        f"Attempted to save {processed_files_count} records. "
        f"{truly_successful_saves} records are considered fully successful (no errors reported). "
        f"{error_count} files encountered errors or warnings (check 'error' field in JSONL)."
    )
    return ProcessResponse(message=final_message, output_file=str(output_jsonl_path), processed_files=processed_files_count, saved_records=truly_successful_saves, errors=error_count)

@app.get("/status", summary="API Status")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API v1.4.1",
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