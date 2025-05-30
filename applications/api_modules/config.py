"""
Configuration and constants for the API
"""
import os
import torch
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
# These will be set based on environment variables, assumed to be loaded by main.py
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
    """Setup API configurations for external services. Relies on .env being loaded by main.py."""
    global GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED
    
    # Environment variables are now expected to be pre-loaded by main.py
    # The print statements below will show what os.getenv() retrieves at this point.
    # print(f"[SETUP_API_CONFIG_PRINT] In setup_api_configurations. config.py does not load .env anymore.")
    
    # Retrieve keys directly from environment
    google_api_key_local = os.getenv("GOOGLE_API_KEY")
    whissle_auth_token_local = os.getenv("WHISSLE_AUTH_TOKEN")
    deepgram_api_key_local = os.getenv("DEEPGRAM_API_KEY")

    # Debug prints for values retrieved by os.getenv within this function
    print(f"[SETUP_API_CONFIG_PRINT] Values from os.getenv inside setup_api_configurations:")
    print(f"[SETUP_API_CONFIG_PRINT] google_api_key_local is {'SET' if google_api_key_local else 'NOT SET'}")
    print(f"[SETUP_API_CONFIG_PRINT] whissle_auth_token_local is {'SET' if whissle_auth_token_local else 'NOT SET'}")
    print(f"[SETUP_API_CONFIG_PRINT] deepgram_api_key_local is {'SET' if deepgram_api_key_local else 'NOT SET'}")
    
    # Original debug logging using logger
    logger.info(f"DEBUG: setup_api_configurations called")
    logger.info(f"DEBUG: GOOGLE_API_KEY (from local var in func) found: {bool(google_api_key_local)}")
    logger.info(f"DEBUG: WHISSLE_AUTH_TOKEN (from local var in func) found: {bool(whissle_auth_token_local)}")
    logger.info(f"DEBUG: DEEPGRAM_API_KEY (from local var in func) found: {bool(deepgram_api_key_local)}")
    
    # Gemini configuration
    if google_api_key_local:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key_local)
            logger.info("Gemini API configured successfully.")
            GEMINI_CONFIGURED = True
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
            GEMINI_CONFIGURED = False # Ensure it's false on error
    else:
        logger.warning("Warning: GOOGLE_API_KEY environment variable not set. Gemini features will be unavailable.")
        GEMINI_CONFIGURED = False

    # Whissle configuration
    # WHISSLE_AVAILABLE = False # Initialize
    try:
        from whissle import WhissleClient # Keep import to check availability
        # WHISSLE_AVAILABLE = True # SDK is present
        if whissle_auth_token_local:
            logger.info("Whissle Auth Token found.")
            WHISSLE_CONFIGURED = True
        else:
            logger.warning("Warning: WHISSLE_AUTH_TOKEN environment variable not set. Whissle model will be unavailable.")
            WHISSLE_CONFIGURED = False
    except ImportError:
        logger.warning("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
        # WHISSLE_AVAILABLE = False
        WHISSLE_CONFIGURED = False

    # Deepgram configuration
    if deepgram_api_key_local:
        try:
            from deepgram import DeepgramClient
            # DEEPGRAM_CLIENT = DeepgramClient(deepgram_api_key_local) # Initialize if needed globally, or locally in transcription module
            logger.info("Deepgram API key found. Client can be initialized.")
            DEEPGRAM_CONFIGURED = True
        except Exception as e: # Should be ImportError for DeepgramClient if SDK missing, or other errors for bad key
            logger.error(f"Failed to prepare for Deepgram client initialization (or SDK missing): {e}")
            DEEPGRAM_CONFIGURED = False
    else:
        logger.warning("Deepgram API key not set. Deepgram transcription disabled.")
        DEEPGRAM_CONFIGURED = False
    
    # Final status log
    logger.info(f"DEBUG: Configuration complete - Gemini: {GEMINI_CONFIGURED}, Whissle: {WHISSLE_CONFIGURED}, Deepgram: {DEEPGRAM_CONFIGURED}")

# Getter functions for configuration status
def is_gemini_configured():
    """Check if Gemini is configured"""
    return GEMINI_CONFIGURED

def is_whissle_configured():
    """Check if Whissle is configured"""
    return WHISSLE_CONFIGURED

def is_deepgram_configured():
    """Check if Deepgram is configured"""
    return DEEPGRAM_CONFIGURED

def get_api_configurations():
    """Get all API configuration statuses"""
    return {
        "gemini_configured": GEMINI_CONFIGURED,
        "whissle_configured": WHISSLE_CONFIGURED,
        "deepgram_configured": DEEPGRAM_CONFIGURED
    }
