# applications/config.py
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import torch
from enum import Enum
from pydantic import BaseModel, Field as PydanticField
from typing import Optional, List

# Load environment variables
load_dotenv('D:/z-whissle/meta-asr/.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model choice enum
class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"
    deepgram = "deepgram"

# Pydantic models for requests and responses
class ProcessRequest(BaseModel):
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    model_choice: ModelChoice = PydanticField(..., description="The transcription model to use.")
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")
    annotations: Optional[List[str]] = PydanticField(
        None, description="List of annotations to include (age, gender, emotion, entity, intent).",
        example=["age", "gender", "emotion", "entity", "intent"]
    )

class TranscriptionJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

class BioAnnotation(BaseModel):
    tokens: List[str]
    tags: List[str]

class AnnotatedJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    original_transcription: Optional[str] = None
    duration: Optional[float] = None
    task_name: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    emotion: Optional[str] = None
    gemini_intent: Optional[str] = None
    ollama_intent: Optional[str] = None
    bio_annotation_gemini: Optional[BioAnnotation] = None
    bio_annotation_ollama: Optional[BioAnnotation] = None
    error: Optional[str] = None

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int