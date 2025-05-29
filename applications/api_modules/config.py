"""
Configuration and constants for the API
"""
import os
import torch
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/dchauhan/workspace/meta-asr/applications/.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
DEEPGRAM_CONFIGURED = False
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# --- BIO Annotation Constants ---
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

# Configuration setup
def setup_api_configurations():
    """Setup API configurations for external services"""
    global GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED
    
    # Gemini configuration
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Gemini API configured successfully.")
            GEMINI_CONFIGURED = True
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
    else:
        logger.warning("Warning: GOOGLE_API_KEY environment variable not set. Gemini features will be unavailable.")

    # Whissle configuration
    try:
        from whissle import WhissleClient
        WHISSLE_AVAILABLE = True
        if WHISSLE_AUTH_TOKEN:
            logger.info("Whissle Auth Token found.")
            WHISSLE_CONFIGURED = True
        else:
            logger.warning("Warning: WHISSLE_AUTH_TOKEN environment variable not set. Whissle model will be unavailable.")
    except ImportError:
        logger.warning("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
        WHISSLE_AVAILABLE = False

    # Deepgram configuration
    DEEPGRAM_CONFIGURED = bool(DEEPGRAM_API_KEY)
    if DEEPGRAM_CONFIGURED:
        try:
            from deepgram import DeepgramClient
            DEEPGRAM_CLIENT = DeepgramClient(DEEPGRAM_API_KEY)
            logger.info("Deepgram client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {e}")
            DEEPGRAM_CONFIGURED = False
    else:
        logger.warning("Deepgram API key not set. Deepgram transcription disabled.")
