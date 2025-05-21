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
from pydantic import BaseModel, Field, HttpUrl
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

try:
    import ollama
    from ollama import chat as ollama_chat_func 
    OLLAMA_SDK_AVAILABLE = True
except ImportError:
    print("Warning: Ollama SDK not found or failed to import. Ollama model will be unavailable.")
    OLLAMA_SDK_AVAILABLE = False
    class ollama: 
        class Client:
            def __init__(self, host=None): pass
            def chat(self, model, messages, stream=False): pass
        class APIError(Exception): pass
        class ResponseError(APIError): pass 
    def ollama_chat_func(model, messages, stream=False): pass 


load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
OLLAMA_API_BASE_URL_ENV = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME_ENV = os.getenv("OLLAMA_MODEL_NAME")


GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
OLLAMA_CONFIGURED = False
OLLAMA_MODEL_TO_USE: Optional[str] = None
OLLAMA_CLIENT: Optional[ollama.Client] = None 


AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Initialize FastAPI ---
app = FastAPI(
    title="Audio Processing API",
    description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities "
                "(using Gemini or Ollama), and saves results to a JSONL manifest file with custom format.",
    version="1.6.0" # Incremented version for new output format
)

# --- Model Loading (Age/Gender, Emotion) ---
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

# --- API Configurations (Gemini, Whissle, Ollama) ---
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

if OLLAMA_SDK_AVAILABLE:
    if OLLAMA_MODEL_NAME_ENV:
        OLLAMA_MODEL_TO_USE = OLLAMA_MODEL_NAME_ENV
        try:
            temp_client = ollama.Client(host=OLLAMA_API_BASE_URL_ENV)
            temp_client.show(OLLAMA_MODEL_TO_USE) 
            OLLAMA_CLIENT = temp_client 
            logger.info(f"Ollama configured and verified with model '{OLLAMA_MODEL_TO_USE}' at '{OLLAMA_API_BASE_URL_ENV}'.")
            OLLAMA_CONFIGURED = True
        except ollama.ResponseError as e:
            if hasattr(e, 'status_code') and e.status_code == 404:
                 logger.error(f"Ollama model '{OLLAMA_MODEL_TO_USE}' not found at '{OLLAMA_API_BASE_URL_ENV}'. Please pull the model. Error: {e}")
            else:
                 logger.error(f"Ollama API error during client initialization/model check for '{OLLAMA_MODEL_TO_USE}' at '{OLLAMA_API_BASE_URL_ENV}'. Status: {e.status_code if hasattr(e, 'status_code') else 'N/A'}. Error: {e}")
            OLLAMA_CONFIGURED = False
        except Exception as e: 
            logger.error(f"Failed to initialize/verify Ollama client with model '{OLLAMA_MODEL_TO_USE}' at '{OLLAMA_API_BASE_URL_ENV}': {e}")
            OLLAMA_CONFIGURED = False
    else:
        logger.warning("Warning: OLLAMA_MODEL_NAME environment variable not set. Ollama features will be unavailable.")
        OLLAMA_CONFIGURED = False 
else:
    logger.warning("Warning: Ollama SDK not installed. Ollama features will be unavailable.")
    OLLAMA_CONFIGURED = False 

# --- Pydantic Models ---
class TranscriptionModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"

class AnnotationModelProvider(str, Enum):
    gemini = "gemini"
    ollama = "ollama"

class ProcessRequest(BaseModel):
    directory_path: str = Field(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    output_jsonl_path: str = Field(..., description="Absolute path for the output JSONL manifest file.", example="/path/to/output/results.jsonl")
    transcription_model_choice: TranscriptionModelChoice = Field(default=TranscriptionModelChoice.gemini, description="Choice of transcription model.")
    annotation_provider: AnnotationModelProvider = Field(default=AnnotationModelProvider.gemini, description="Choice of annotation provider.")

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int

class TranscriptionJsonlRecord(BaseModel): # For transcription-only endpoint
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

class CustomAnnotatedJsonlRecord(BaseModel): # New model for the desired output format
    audio_filepath: str
    text: Optional[str] = Field(None, description="Transcription with ENTITY tags only")
    duration: Optional[float] = None
    task_name: Optional[str] = Field(None, description="Extracted from INTENT_ tag (e.g., INFORM, REQUEST_INFO)")
    gender: Optional[str] = Field(None, description="Predicted gender (e.g., Male, Female, Unknown)")
    age_group: Optional[str] = Field(None, description="Predicted age group (e.g., 18-24, 65+)")
    emotion: Optional[str] = Field(None, description="Predicted emotion (e.g., Happy, Neutral, Unknown)")

# --- Utility Functions ---
def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        if sr != target_sr: audio = resampy.resample(audio, sr, target_sr); sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=False)
        return None, None, f"Failed to load audio: {type(e).__name__}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    try: return sf.info(str(audio_path)).duration
    except Exception:
        try: return librosa.get_duration(path=str(audio_path))
        except Exception as le: logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=False); return None

def predict_age_gender(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if age_gender_model is None or age_gender_processor is None: return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0: return None, None, "Empty audio data for Age/Gender."
    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad(): outputs = age_gender_model(input_values)
        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]
        return float(age_pred), int(gender_pred_idx), None
    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=False)
        return None, None, f"Age/Gender prediction failed: {type(e).__name__}"

def predict_emotion(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[str], Optional[str]]:
    if emotion_model is None or emotion_feature_extractor is None: return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0: return None, "Empty audio data for Emotion."
    if len(audio_data) < int(sampling_rate * 0.1): return "SHORT_AUDIO", None
    try:
        inputs = emotion_feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad(): outputs = emotion_model(**inputs)
        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")
        return emotion_label, None
    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=False)
        return None, f"Emotion prediction failed: {type(e).__name__}"

def format_age_gender_emotion_tags_for_llm(age: Optional[float], gender_idx: Optional[int], emotion_label: Optional[str]) -> str:
    # This function prepares tags to be *sent* to the LLM for preservation
    tags = []
    if age is not None:
        try:
            actual_age = round(age, 1)
            age_brackets = [(18, "0_17"), (25, "18_24"), (35, "25_34"), (45, "35_44"), (55, "45_54"), (65, "55_64"), (float('inf'), "65PLUS")]
            age_tag = next((bracket for threshold, bracket in age_brackets if actual_age < threshold), "UNKNOWN")
            tags.append(f"AGE_{age_tag}")
        except Exception: tags.append("AGE_ERROR")
    else: tags.append("AGE_UNKNOWN")

    if gender_idx == 1: tags.append("GENDER_MALE")
    elif gender_idx == 0: tags.append("GENDER_FEMALE")
    else: tags.append("GENDER_UNKNOWN")

    if emotion_label and emotion_label != "SHORT_AUDIO": tags.append(f"EMOTION_{emotion_label.upper().replace(' ', '_')}")
    elif emotion_label == "SHORT_AUDIO": tags.append("EMOTION_SHORT_AUDIO")
    else: tags.append("EMOTION_UNKNOWN")
    return " ".join(tags)

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> tuple[Optional[str], Optional[str]]:
    if not WHISSLE_CONFIGURED: return None, "Whissle is not configured."
    try:
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        if isinstance(response, dict): text = response.get('text'); return (text.strip(), None) if text else (None, response.get('error') or response.get('message', 'Unknown Whissle API error'))
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str): return response.transcript.strip(), None
        else: return None, f"Unexpected Whissle response format: {type(response)}"
    except Exception as e: return None, f"Whissle SDK error: {type(e).__name__}: {e}"

def get_mime_type(audio_file_path: Path) -> str:
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_gemini_single(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    try: model = genai.GenerativeModel("models/gemini-1.5-flash")
    except Exception as e: return None, f"Error initializing Gemini model: {e}"
    uploaded_file = None
    try:
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=get_mime_type(audio_path))
        while uploaded_file.state.name == "PROCESSING": await asyncio.sleep(2); uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
        if uploaded_file.state.name != "ACTIVE": err_msg = f"Gemini file processing failed: {uploaded_file.state.name}"; return None, err_msg
        prompt = "Transcribe audio. Provide only spoken text. If no speech, return empty string."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})
        if response.candidates:
            transcription = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            return transcription if transcription else "", None
        else: return None, f"No candidates from Gemini. Feedback: {getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback)) if hasattr(response, 'prompt_feedback') else 'N/A'}"
    except Exception as e: return None, f"Gemini transcription error: {type(e).__name__}: {e}"
    finally:
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete Gemini file {uploaded_file.name}: {del_e}")

def remove_existing_tags(text: str) -> str: # For preparing LLM input
    if not isinstance(text, str): return text
    preserve_patterns = [ r'\bAGE_[A-Z0-9_]+\b', r'\bGENDER_(MALE|FEMALE|OTHER|UNKNOWN)\b', r'\bEMOTION_[A-Z_]+\b', r'\bSPEAKER_CHANGE\b', r'ENTITY_[A-Z0-9_]+\s+[\s\S]*?\s+END\b' ]
    preserved_tags_map, placeholder_count = {}, 0
    def replace_with_placeholder(match): nonlocal placeholder_count; tag_content, placeholder = match.group(0), f"__P_{placeholder_count}__"; preserved_tags_map[placeholder] = tag_content; placeholder_count += 1; return f" {placeholder} "
    temp_text = text
    for pattern in preserve_patterns: temp_text = re.sub(pattern, replace_with_placeholder, temp_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*INTENT_[A-Z_0-9]+\s*', ' ', temp_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bNER_\w+\s*', ' ', cleaned, flags=re.IGNORECASE)
    for placeholder, original in preserved_tags_map.items(): cleaned = cleaned.replace(f" {placeholder} ", f" {original} ")
    return re.sub(r'\s{2,}', ' ', cleaned).strip()

def fix_end_tags(text: str) -> str: # For LLM output cleanup
    if not isinstance(text, str): return text
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)\s+([A-Z0-9])\s+END\b', r'\1\2 END', text, flags=re.IGNORECASE)
    text = re.sub(r'(\S)END\b', r'\1 END', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\S)(INTENT_[A-Z_0-9]+\b)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\bEND(\S)', r'END \1', text, flags=re.IGNORECASE)
    return re.sub(r'\s{2,}', ' ', text).strip()

_entity_list_json_str = """["PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "NUMBER", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS", "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN", "PRODUCT", "SERVICE", "CATEGORY", "BRAND_NAME", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD", "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE", "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "RECIPE", "INGREDIENT", "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD", "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE", "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION", "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA", "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE", "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY", "POLLUTION_LEVEL", "UV_INDEX", "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND", "USER_HANDLE", "EMAIL_ADDRESS", "PHONE_NUMBER", "IPV4_ADDRESS", "IPV6_ADDRESS", "GPS_COORDINATE", "LATITUDE", "LONGITUDE", "GEOLOCATION", "STREET_NAME", "BUILDING_NUMBER", "FLOOR_NUMBER", "BUSINESS_NAME", "MODEL", "SERIAL_NUMBER", "IMEI", "IMSI", "DEVICE_ID", "OS_VERSION", "FILE_PATH", "FILE_NAME", "FILE_EXTENSION", "DOCUMENT_TITLE", "DOCUMENT_ID", "LEGAL_ENTITY", "TAX_DOCUMENT", "BILLING_ADDRESS", "SHIPPING_ADDRESS", "COUPON_CODE", "LOYALTY_CARD_NUMBER", "PRODUCT_ID", "SKU", "BARCODE", "QR_CODE", "TRANSACTION_CODE", "EVENT_ID", "SESSION_ID", "ACTION_ID", "CLICK_POSITION", "SCROLL_DEPTH", "VIDEO_ID", "AUDIO_TRACK", "SUBTITLE_LANGUAGE", "CHAPTER_TITLE", "CHAPTER_NUMBER", "EPISODE_NUMBER", "MOVIE_TITLE", "DIRECTOR_NAME", "ACTOR_NAME", "BOOK_TITLE", "AUTHOR_NAME", "PUBLISHER", "ISBN", "ISSN", "COURSE_NAME", "INSTRUCTOR_NAME", "STUDENT_ID", "GRADE", "CLASSROOM_NUMBER", "SCHOOL_NAME", "DEGREE_PROGRAM", "MAJOR", "MINOR", "CERTIFICATE_NAME", "EXAM_SCORE", "CERTIFICATION_ID", "TRAINING_PROGRAM", "PLATFORM", "APPLICATION", "SOFTWARE_PACKAGE", "API_ENDPOINT", "SERVICE_NAME", "SERVER_IP", "DATABASE_TABLE", "QUERY", "ERROR_CODE", "LOG_LEVEL", "SESSION_DURATION", "BROWSER_TYPE", "DEVICE_TYPE"]"""
_parsed_entity_list = json.loads(_entity_list_json_str)
_entity_list_for_prompt_str = ", ".join(f'"{entity}"' for entity in _parsed_entity_list)

def _build_annotation_prompt(cleaned_text_for_prompt: str) -> str:
    return f'''Analyze the following sentence. It might already contain tags like AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE, or correctly formatted ENTITY_<TYPE> existing entity END tags.

Your tasks are:
1.  **Preserve Existing Tags:** Keep any AGE_*, GENDER_*, EMOTION_*, SPEAKER_CHANGE, or correctly pre-tagged `ENTITY_<TYPE> text END` tags exactly where they are.
2.  **Annotate NEW Entities:** Identify and tag NEW entities in parts of the text NOT already tagged as `ENTITY_<TYPE> text END`. Tag these new entities **only** from this specific list: [{_entity_list_for_prompt_str}]. Use the format `ENTITY_<TYPE> identified text END`.
3.  **Classify Intent:** Determine the primary intent. Choose **one** label from: `REQUEST_INFO`, `PROVIDE_INFO`, `BOOK_APPOINTMENT`, `CANCEL_APPOINTMENT`, `RESCHEDULE_APPOINTMENT`, `PROVIDE_FEEDBACK`, `MAKE_PURCHASE`, `RETURN_ITEM`, `TRACK_ORDER`, `SOCIAL_CHITCHAT`, `HEALTH_UPDATE`, `REQUEST_ASSISTANCE`, `TECHNICAL_SUPPORT`, `CONFIRMATION`, `DENIAL`, `GREETING`, `FAREWELL`, `AGREEMENT`, `DISAGREEMENT`, `EXPRESS_EMOTION`, `OTHER`. If no clear intent fits, use `OTHER`.
4.  **Add Intent Tag:** Append exactly ONE `INTENT_<ChosenIntentLabel>` tag at the VERY END of the entire processed sentence.
5.  **Output:** Return the fully processed sentence as a single string. Do not wrap in quotes or markdown.

**CRITICAL FORMATTING RULES:**
*   `ENTITY_<TYPE> text END` (Correct). `<TYPE>` must be from the list, no spaces within `<TYPE>`.
*   Final `INTENT_<TYPE>` tag must be the absolute last part of the output string.

**Example Input (Tagging new entities, preserving AGE/GENDER):** "can you book a flight for alice to paris on next tuesday GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL"
**Example Output:** "can you book a flight for ENTITY_PERSON_NAME alice END to ENTITY_CITY paris END on ENTITY_DATE next tuesday END GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL INTENT_BOOK_APPOINTMENT"

**Text to Annotate:** "{cleaned_text_for_prompt}"'''

async def _annotate_text_llm_common(text_to_annotate: str, llm_provider: AnnotationModelProvider) -> tuple[Optional[str], Optional[str]]:
    if not text_to_annotate or text_to_annotate.isspace(): return "", None
    cleaned_text_for_prompt = remove_existing_tags(text_to_annotate)
    if not cleaned_text_for_prompt or cleaned_text_for_prompt.isspace(): return text_to_annotate, None
    
    prompt = _build_annotation_prompt(cleaned_text_for_prompt)
    annotated_text: Optional[str] = None
    error_detail: Optional[str] = None

    if llm_provider == AnnotationModelProvider.gemini:
        if not GEMINI_CONFIGURED: return text_to_annotate, "Gemini API not configured."
        logger.info(f"Annotating with Gemini (input: '{cleaned_text_for_prompt[:100]}...')")
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            response = await asyncio.to_thread(model.generate_content, contents=[prompt], generation_config=generation_config, request_options={'timeout': 180})
            if response.candidates:
                annotated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            else: error_detail = f"No candidates from Gemini. Feedback: {getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback)) if hasattr(response, 'prompt_feedback') else 'N/A'}"
        except Exception as e: error_detail = f"Gemini API error: {type(e).__name__}: {e}"

    elif llm_provider == AnnotationModelProvider.ollama:
        if not OLLAMA_CONFIGURED or not OLLAMA_MODEL_TO_USE: return text_to_annotate, f"Ollama not configured or model not set (current: {OLLAMA_MODEL_TO_USE or 'Not Set'})."
        logger.info(f"Annotating with Ollama (model: '{OLLAMA_MODEL_TO_USE}', input: '{cleaned_text_for_prompt[:100]}...')")
        try:
            response_data = await asyncio.to_thread(ollama_chat_func, model=OLLAMA_MODEL_TO_USE, messages=[{'role': 'user', 'content': prompt}])
            if response_data and isinstance(response_data, dict) and response_data.get('message', {}).get('content'):
                annotated_text = response_data['message']['content'].strip()
            else: error_detail = f"Unexpected Ollama response: {str(response_data)[:200]}"
        except ollama.ResponseError as e: error_detail = f"Ollama API ResponseError: Status {e.status_code if hasattr(e, 'status_code') else 'N/A'}. {str(e)[:200]}"
        except ollama.APIError as e: error_detail = f"Ollama APIError: {str(e)[:200]}"
        except Exception as e: error_detail = f"Ollama annotation error: {type(e).__name__}: {e}"

    if error_detail: return text_to_annotate, error_detail
    if not annotated_text: return text_to_annotate, f"{llm_provider.value} response text extracted as empty."

    # Common cleanup for LLM outputs
    if annotated_text.startswith("```") and annotated_text.endswith("```"):
        annotated_text = re.sub(r'^```[a-z]*\n?|\n?```$', '', annotated_text).strip()
    if len(annotated_text) > 1 and annotated_text.startswith('"') and annotated_text.endswith('"'):
        content_inside_quotes = annotated_text[1:-1]
        if not re.search(r'INTENT_[A-Z_0-9]+$', content_inside_quotes.strip()): annotated_text = content_inside_quotes
        else: annotated_text = content_inside_quotes
    
    final_text = fix_end_tags(annotated_text).strip()
    if not re.search(r'INTENT_[A-Z_0-9]+$', final_text):
        logger.warning(f"{llm_provider.value} annotation missing INTENT tag: '{final_text[-70:]}'")
        final_text = f"{final_text} INTENT_UNKNOWN".strip()
    
    logger.info(f"{llm_provider.value} annotation successful (result: '{final_text[:100]}...')")
    return final_text, None

# --- New parsing function for custom output format ---
def parse_custom_output_fields(llm_annotated_text: Optional[str]) -> Dict[str, Optional[str]]:
    parsed_fields = {
        "text_with_entities": None,
        "task_name": "UNKNOWN",
        "gender": "Unknown",
        "age_group": "Unknown",
        "emotion": "Unknown",
    }
    if not llm_annotated_text:
        return parsed_fields

    text_for_entities = llm_annotated_text

    # 1. Extract Intent (Task Name)
    intent_match = re.search(r'\sINTENT_([A-Z_0-9]+)$', text_for_entities)
    if intent_match:
        parsed_fields["task_name"] = intent_match.group(1).upper()
        text_for_entities = text_for_entities[:intent_match.start()].strip()

    # 2. Extract Gender
    gender_pattern = r'\s?\bGENDER_(MALE|FEMALE|OTHER|UNKNOWN)\b\s?'
    gender_match = re.search(gender_pattern, text_for_entities, re.IGNORECASE)
    if gender_match:
        gender_tag = gender_match.group(1).upper()
        if gender_tag == "MALE": parsed_fields["gender"] = "Male"
        elif gender_tag == "FEMALE": parsed_fields["gender"] = "Female"
        else: parsed_fields["gender"] = "Unknown"
        text_for_entities = re.sub(gender_pattern, ' ', text_for_entities, count=1, flags=re.IGNORECASE).strip()

    # 3. Extract Age Group
    age_pattern = r'\s?\bAGE_([0-9]{1,2}_[0-9]{1,2}|[0-9]{1,2}PLUS|UNKNOWN|ERROR)\b\s?'
    age_match = re.search(age_pattern, text_for_entities, re.IGNORECASE)
    if age_match:
        age_tag_raw = age_match.group(1).upper()
        if age_tag_raw == "UNKNOWN" or age_tag_raw == "ERROR": parsed_fields["age_group"] = "Unknown"
        elif "PLUS" in age_tag_raw: parsed_fields["age_group"] = age_tag_raw.replace("PLUS", "+")
        elif "_" in age_tag_raw: parsed_fields["age_group"] = age_tag_raw.replace("_", "-")
        else: parsed_fields["age_group"] = age_tag_raw 
        text_for_entities = re.sub(age_pattern, ' ', text_for_entities, count=1, flags=re.IGNORECASE).strip()
    
    # 4. Extract Emotion
    emotion_pattern = r'\s?\bEMOTION_([A-Z_]+)\b\s?'
    emotion_match = re.search(emotion_pattern, text_for_entities, re.IGNORECASE)
    if emotion_match:
        emotion_tag_raw = emotion_match.group(1).upper()
        if emotion_tag_raw in ["SHORT_AUDIO", "UNKNOWN"]: parsed_fields["emotion"] = "Unknown"
        else: parsed_fields["emotion"] = emotion_tag_raw.replace("_", " ").capitalize()
        text_for_entities = re.sub(emotion_pattern, ' ', text_for_entities, count=1, flags=re.IGNORECASE).strip()

    parsed_fields["text_with_entities"] = re.sub(r'\s{2,}', ' ', text_for_entities).strip()
    if parsed_fields["text_with_entities"] in {"[NO_SPEECH]", "[TRANSCRIPTION_FAILED]", "[PROCESSING_ERROR]"}:
        # If the remaining text is a placeholder, it means original transcription was that.
        # The `task_name` would reflect NO_SPEECH or TRANSCRIPTION_FAILED if set correctly by the earlier logic.
        pass # Keep the placeholder in text_with_entities for these cases.
        
    return parsed_fields


def validate_paths(directory_path: str, output_jsonl_path: str) -> Tuple[Path, Path]:
    try: dir_path = Path(directory_path).resolve(strict=True)
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid dir path: {e}")
    if not dir_path.is_dir(): raise HTTPException(status_code=404, detail=f"Dir not found: {dir_path}")
    try: output_path = Path(output_jsonl_path).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid output path: {e}")
    if output_path.is_dir(): raise HTTPException(status_code=400, detail=f"Output path is dir: {output_path}")
    if not os.access(output_path.parent, os.W_OK): raise HTTPException(status_code=403, detail=f"No write perm: {output_path.parent}")
    if output_path.exists(): logger.warning(f"Output file {output_path} exists and will be overwritten.")
    return dir_path, output_path

def discover_audio_files(dir_path: Path) -> List[Path]:
    try: files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]; logger.info(f"Found {len(files)} audio files in {dir_path}"); return files
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error scanning dir: {e}")

@app.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    trans_model = process_request.transcription_model_choice
    if trans_model == TranscriptionModelChoice.whissle and not WHISSLE_CONFIGURED: raise HTTPException(status_code=400, detail="Whissle not configured.")
    if trans_model == TranscriptionModelChoice.gemini and not GEMINI_CONFIGURED: raise HTTPException(status_code=400, detail="Gemini for transcription not configured.")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)
    if not audio_files:
        try: open(output_jsonl_path, "w").close(); return ProcessResponse(message="No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)
        except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to create empty output: {e}")

    processed, saved, errors = 0, 0, 0
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                processed += 1; file_err, text, duration_val = None, None, None
                logger.info(f"--- Processing {audio_file.name} (Transcription Only: {trans_model.value}) ---")
                try:
                    duration_val = get_audio_duration(audio_file)
                    if trans_model == TranscriptionModelChoice.whissle: text, trans_err = await transcribe_with_whissle_single(audio_file)
                    else: text, trans_err = await transcribe_with_gemini_single(audio_file)
                    if trans_err: file_err = f"Transcription failed: {trans_err}"
                    elif text is None: file_err = "Transcription returned None."
                    else: text = text.strip()
                except Exception as e: file_err = f"Unexpected error: {type(e).__name__}: {e}"
                
                record = TranscriptionJsonlRecord(audio_filepath=str(audio_file.resolve()), text=text, duration=duration_val, model_used_for_transcription=trans_model.value, error=file_err)
                try: outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e: file_err = f"{file_err or ''}; JSONL write error: {write_e}".strip('; ')
                if file_err: errors += 1
                else: saved += 1
                gc.collect()
    except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to write output: {e}")
    return ProcessResponse(message=f"Processed {processed}/{len(audio_files)}. Saved: {saved}. Errors: {errors}.", output_file=str(output_jsonl_path), processed_files=processed, saved_records=saved, errors=errors)

@app.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest (Custom Format)")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    trans_model = process_request.transcription_model_choice
    annot_provider = process_request.annotation_provider

    if trans_model == TranscriptionModelChoice.whissle and not WHISSLE_CONFIGURED: raise HTTPException(status_code=400, detail="Whissle (transcription) not configured.")
    if trans_model == TranscriptionModelChoice.gemini and not GEMINI_CONFIGURED: raise HTTPException(status_code=400, detail="Gemini (transcription) not configured.")
    if annot_provider == AnnotationModelProvider.gemini and not GEMINI_CONFIGURED: raise HTTPException(status_code=400, detail="Gemini (annotation) not configured.")
    if annot_provider == AnnotationModelProvider.ollama and not OLLAMA_CONFIGURED:
        raise HTTPException(status_code=400, detail=f"Ollama (annotation) not configured. Model: '{OLLAMA_MODEL_TO_USE or 'Not Set'}' at {OLLAMA_API_BASE_URL_ENV}.")

    missing_local_models = [m for m, loaded in [("Age/Gender", age_gender_model), ("Emotion", emotion_model)] if not loaded]
    if missing_local_models: logger.warning(f"Local models not loaded: {', '.join(missing_local_models)}. Predictions will be 'Unknown'.")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)
    if not audio_files:
        try: open(output_jsonl_path, "w").close(); return ProcessResponse(message="No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)
        except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to create empty output: {e}")

    processed, saved, errors_count = 0, 0, 0
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                processed += 1
                file_error_details: List[str] = []
                duration_val, audio_data, sr = None, None, None
                transcription_text, llm_annotated_text = None, None
                
                logger.info(f"--- Processing {audio_file.name} (Trans: {trans_model.value}, Annot: {annot_provider.value}) ---")
                
                try: # Main processing block for a single file
                    duration_val = get_audio_duration(audio_file)
                    audio_data, sr, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
                    if load_err: file_error_details.append(load_err)
                    elif audio_data is None or sr != TARGET_SAMPLE_RATE: file_error_details.append("Audio load/SR mismatch.")

                    # 1. Transcription
                    if trans_model == TranscriptionModelChoice.whissle: transcription_text, trans_err = await transcribe_with_whissle_single(audio_file)
                    else: transcription_text, trans_err = await transcribe_with_gemini_single(audio_file)
                    if trans_err: file_error_details.append(f"Transcription: {trans_err}")
                    elif transcription_text is None: file_error_details.append("Transcription returned None.")
                    else: transcription_text = transcription_text.strip()

                    # 2. Local A/G/E predictions
                    age_pred_val, gender_idx_val, emotion_label_val = None, None, None
                    if audio_data is not None and sr == TARGET_SAMPLE_RATE:
                        if age_gender_model and emotion_model:
                            try:
                                age_pred_val, gender_idx_val, age_gender_err = await asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE)
                                if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                emotion_label_val, emotion_err = await asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE)
                                if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
                            except Exception as pred_e: file_error_details.append(f"A/G/E Error: {pred_e}")
                        else: file_error_details.append("Skipped A/G/E (local models not loaded).")
                    elif not load_err: file_error_details.append("Skipped A/G/E (audio load issue).")
                    
                    age_gender_emotion_tags_for_llm = format_age_gender_emotion_tags_for_llm(age_pred_val, gender_idx_val, emotion_label_val)

                    # 3. LLM Annotation
                    if transcription_text is not None and transcription_text != "":
                        text_to_llm = f"{transcription_text} {age_gender_emotion_tags_for_llm}".strip()
                        llm_annotated_text, annot_err = await _annotate_text_llm_common(text_to_llm, annot_provider)
                        if annot_err: file_error_details.append(f"ANNOTATION_FAIL ({annot_provider.value}): {annot_err}")
                        # llm_annotated_text will contain the original text_to_llm if LLM failed, or the annotated version
                    elif transcription_text == "":
                        llm_annotated_text = f"[NO_SPEECH] {age_gender_emotion_tags_for_llm} INTENT_NO_SPEECH".strip()
                    else: # Transcription failed
                        llm_annotated_text = f"[TRANSCRIPTION_FAILED] {age_gender_emotion_tags_for_llm} INTENT_TRANSCRIPTION_FAILED".strip()
                
                except Exception as e: # Catch-all for unexpected errors during a file's processing
                    logger.error(f"Unexpected critical error processing {audio_file.name}: {e}", exc_info=True)
                    file_error_details.append(f"CRITICAL_ERROR: {type(e).__name__}: {e}")
                    # Ensure llm_annotated_text is a string for parsing, even if it's just a placeholder
                    if llm_annotated_text is None:
                        age_gender_emotion_tags_for_llm_fallback = format_age_gender_emotion_tags_for_llm(None, None, None) # Default tags
                        llm_annotated_text = f"[PROCESSING_ERROR] {age_gender_emotion_tags_for_llm_fallback} INTENT_PROCESSING_ERROR".strip()

                # 4. Parse final fields for CustomAnnotatedJsonlRecord
                parsed_output_fields = parse_custom_output_fields(llm_annotated_text)

                custom_record = CustomAnnotatedJsonlRecord(
                    audio_filepath=str(audio_file.resolve()),
                    text=parsed_output_fields["text_with_entities"],
                    duration=round(duration_val, 2) if duration_val is not None else None,
                    task_name=parsed_output_fields["task_name"],
                    gender=parsed_output_fields["gender"],
                    age_group=parsed_output_fields["age_group"],
                    emotion=parsed_output_fields["emotion"]
                )
                
                try:
                    outfile.write(custom_record.model_dump_json(exclude_none=True) + "\n")
                    if not file_error_details: # Only count as saved if no processing errors occurred for this file
                        saved += 1
                    else: # If there were processing errors, count it as an error, even if written
                        errors_count +=1
                        logger.warning(f"File {audio_file.name} processed with errors: {'; '.join(file_error_details)}")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    errors_count +=1 # Count write error
                
                del audio_data; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        logger.info(f"Finished full annotation. Processed {processed} files.")
    except IOError as e: raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

    final_message = (
        f"Processed {processed}/{len(audio_files)} files. "
        f"Successfully saved records (no processing errors): {saved}. "
        f"Files with processing/write errors: {errors_count}."
    )
    return ProcessResponse(message=final_message, output_file=str(output_jsonl_path), processed_files=processed, saved_records=saved, errors=errors_count)

@app.get("/status", summary="API Status")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API v1.6.0",
        "docs_url": "/docs", "html_interface": "/",
        "endpoints": { "transcription_only": "/create_transcription_manifest/", "full_annotation": "/create_annotated_manifest/" },
        "transcription_models": { "gemini_configured": GEMINI_CONFIGURED, "whissle_available": WHISSLE_AVAILABLE, "whissle_configured": WHISSLE_CONFIGURED, },
        "annotation_providers": { "gemini_configured": GEMINI_CONFIGURED, "ollama_sdk_available": OLLAMA_SDK_AVAILABLE, "ollama_configured": OLLAMA_CONFIGURED, "ollama_model_used": OLLAMA_MODEL_TO_USE if OLLAMA_CONFIGURED else "N/A", "ollama_api_base_url": OLLAMA_API_BASE_URL_ENV, },
        "local_prediction_models": { "age_gender_model_loaded": age_gender_model is not None, "emotion_model_loaded": emotion_model is not None, },
        "compute_device": str(device)
    }

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    fastapi_script_name = Path(__file__).stem
    app_module_string = f"{fastapi_script_name}:app"
    host, port, reload_flag = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000")), os.getenv("RELOAD", "true").lower() == "true"
    log_level = "info"
    logger.info(f"Starting FastAPI server for '{app_module_string}' on {host}:{port} (Reload: {reload_flag}, Log: {log_level.upper()})")
    logger.info(f"Docs: http://{host}:{port}/docs, UI: http://{host}:{port}/")
    uvicorn.run(app_module_string, host=host, port=port, reload=reload_flag, reload_dirs=[str(script_dir)] if reload_flag else None, log_level=log_level)