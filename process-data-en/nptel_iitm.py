import os
import gc
import json
import re
import torch
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
import torch.nn as nn
import google.generativeai as genai
import ollama # Import the ollama client
import time

# --- Initial Setup ---
load_dotenv()
torch.cuda.empty_cache()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device("cuda:0")
else:
    print("CUDA not available, using CPU.")
    device = torch.device("cpu")

# Configure Google Generative AI
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}. Gemini annotation will be skipped if chosen.")
    genai = None

# Configure Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b") # Changed to gemma:7b for broader availability
ollama_client = None
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    # Test connection
    ollama_client.list() 
    print(f"Ollama client configured for model '{OLLAMA_MODEL}' at '{OLLAMA_HOST}'.")
except Exception as e:
    print(f"Error configuring Ollama client or connecting: {e}. Ollama annotation will be skipped if chosen.")
    ollama_client = None


# --- Age/Gender Model Definition (unchanged) ---
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
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"
    if len(audio_data) < sampling_rate * 0.1: 
        pass

    try:
        inputs = model_info['feature_extractor'](
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        emotion_label = model_info['model'].config.id2label.get(predicted_class_idx, "Unknown")
        return emotion_label.upper()
    except Exception as e:
        print(f"Error during emotion extraction: {e}")
        return "Extraction_Error"


# --- Data Structures (unchanged) ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float 
    gender: str 
    transcription: str 
    emotion: str 
    chunk_filename: str
    duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    @staticmethod
    def get_age_bucket(age: float) -> str:
        """
        Maps a raw age prediction (0-1 float, scaled to 0-100) to an age group string.
        """
        actual_age = round(age * 100) 
        if actual_age <= 17:
            return "0-17"
        elif actual_age <= 29: 
            return "18-29"
        elif actual_age <= 44: 
            return "30-44"
        elif actual_age <= 59: 
            return "45-59"
        else: 
            return "60+"

# --- File Handling (unchanged) ---
def get_nptel_file_data(dataset_base_dir: str) -> List[Tuple[str, str]]:
    audio_root_dir = os.path.join(dataset_base_dir, "Audio")
    transcription_root_dir = os.path.join(dataset_base_dir, "transcription")

    if not os.path.isdir(audio_root_dir):
        print(f"Error: Audio directory not found: {audio_root_dir}")
        return []
    if not os.path.isdir(transcription_root_dir):
        print(f"Error: Transcription directory not found: {transcription_root_dir}")
        return []

    print(f"Scanning for audio files in: {audio_root_dir}")
    print(f"Scanning for transcriptions in: {transcription_root_dir}")

    transcriptions_map = {}
    for trans_subdir_name in os.listdir(transcription_root_dir):
        trans_subdir_path = os.path.join(transcription_root_dir, trans_subdir_name)
        if os.path.isdir(trans_subdir_path):
            text_file_path = os.path.join(trans_subdir_path, "text")
            if os.path.isfile(text_file_path):
                print(f"  Reading transcriptions from: {text_file_path}")
                try:
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_basename, text_content = parts
                                transcriptions_map[audio_basename] = text_content
                            else:
                                if line.strip(): 
                                    print(f"    Warning: Malformed line {line_num+1} in {text_file_path}: '{line.strip()}'")
                except Exception as e:
                    print(f"    Error reading transcription file {text_file_path}: {e}")

    if not transcriptions_map:
        print("Warning: No transcriptions loaded. Check transcription file paths and content in subdirectories of 'transcription'.")

    file_data_list = []
    wav_files_found = 0
    wav_files_matched = 0

    for audio_file_name in os.listdir(audio_root_dir):
        if audio_file_name.lower().endswith('.wav'):
            wav_files_found += 1
            audio_basename = os.path.splitext(audio_file_name)[0]
            audio_full_path = os.path.join(audio_root_dir, audio_file_name) 

            if audio_basename in transcriptions_map:
                file_data_list.append((audio_full_path, transcriptions_map[audio_basename]))
                wav_files_matched +=1
            else:
                print(f"  Warning: No transcription found for audio file: {audio_file_name} (basename: {audio_basename})")

    print(f"Found {wav_files_found} .wav files in {audio_root_dir}.")
    print(f"Loaded {len(transcriptions_map)} unique transcription entries from 'text' files.")
    print(f"Successfully matched {wav_files_matched} audio files with transcriptions.")
    if not file_data_list:
         print(f"No matching audio-transcription pairs found for dataset: {dataset_base_dir}")

    return file_data_list


# --- Core Annotation Prompt and Post-Processing Logic ---

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
def get_annotation_prompt(texts_to_annotate: List[str]) -> str:
    """
    Generates the core prompt for BIO entity and utterance-level intent annotation.
    """
    # Define common intent types, these should also appear in the tags if no entity is found
    INTENT_TYPES = ["INFORM", "QUESTION", "REQUEST", "COMMAND", "GREETING", "CONFIRMATION", "NEGATION", 
                    "ACKNOWLEDGEMENT", "INQUIRY", "FAREWELL", "APOLOGY", "THANKS", "COMPLAINT", 
                    "FEEDBACK", "SUGGESTION", "ASSISTANCE", "NAVIGATION", "TRANSACTION", "SCHEDULING",
                    "UNKNOWN_INTENT"] # Added UNKNOWN_INTENT for robust handling

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
def parse_llm_bio_output(llm_response_json_string: str, original_texts: List[str]) -> List[Dict[str, Any]]:
    """
    Parses the LLM's raw JSON output string into a list of structured BIO annotations.
    Each item in the list will contain 'tokens', 'tags', and 'intent'.
    Handles markdown blocks and potential malformed JSON.
    """
    processed_results = []
    # Clean markdown code blocks if present
    cleaned_response = llm_response_json_string.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    elif cleaned_response.startswith("```"):
         cleaned_response = cleaned_response[len("```"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-len("```")].strip()

    try:
        data = json.loads(cleaned_response)
        
        # Ensure 'data' is a list. Some LLMs might return a single object or wrap in { "sentences": [...] }
        if isinstance(data, dict) and "sentences" in data and isinstance(data["sentences"], list):
            data = data["sentences"]
        elif not isinstance(data, list):
            # If it's a single object (not a list of objects), wrap it for consistent processing
            if isinstance(data, dict) and all(k in data for k in ["tokens", "tags", "intent"]):
                data = [data]
            else:
                raise ValueError("LLM response is not a list of objects or a single valid object.")

        if len(data) != len(original_texts):
            print(f"Warning: Mismatch in LLM response length. Expected {len(original_texts)}, got {len(data)}. Padding/truncating.")
            # Pad or truncate to match expected length
            if len(data) < len(original_texts):
                data.extend([{"tokens": [], "tags": [], "intent": "PARSE_ERROR", "text": orig_text} for orig_text in original_texts[len(data):]])
            else: # If LLM returned more than expected
                data = data[:len(original_texts)]
        
        for i, item in enumerate(data):
            tokens = item.get("tokens", [])
            tags = item.get("tags", [])
            intent = item.get("intent", "UNKNOWN_INTENT").upper() 

            # Basic validation and normalization
            if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                print(f"Warning: Invalid tokens in LLM response for index {i} (original text: '{original_texts[i]}'). Tokens: {tokens}")
                tokens = original_texts[i].lower().split() # Fallback to simple split
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                print(f"Warning: Invalid tags in LLM response for index {i} (original text: '{original_texts[i]}'). Tags: {tags}")
                tags = ["O"] * len(tokens) # Default to 'O'
            
            # Ensure lengths match, if not, try to recover or default
            if len(tokens) != len(tags):
                print(f"Warning: Token-tag length mismatch for index {i} (original text: '{original_texts[i]}'). Tokens: {len(tokens)}, Tags: {len(tags)}. Attempting recovery.")
                if len(tokens) > 0:
                    tags = tags[:len(tokens)] + ["O"] * (len(tokens) - len(tags)) # Pad tags if too short
                else:
                    tokens = original_texts[i].lower().split() # Re-tokenize if tokens was empty
                    tags = ["O"] * len(tokens) # Default all to O

            # Normalize tags: uppercase types, handle potential spaces
            cleaned_tags = []
            for tag in tags:
                if tag.upper() == "O":
                    cleaned_tags.append("O")
                elif tag.startswith("B-") or tag.startswith("I-"):
                    parts = tag.split('-', 1)
                    if len(parts) == 2:
                        prefix, type_part = parts
                        cleaned_type_part = type_part.replace(' ', '').upper()
                        cleaned_tags.append(f"{prefix.upper()}-{cleaned_type_part}")
                    else: # Malformed B- or I- tag, default to O
                        cleaned_tags.append("O")
                else: # Any other unexpected tag, default to O
                    cleaned_tags.append("O")
            
            # Ensure final tokens and tags match length after all normalization
            if len(tokens) != len(cleaned_tags):
                print(f"Final warning: Token-tag length mismatch after normalization for index {i} (original text: '{original_texts[i]}'). Forcing consistency.")
                cleaned_tags = cleaned_tags[:len(tokens)] + ["O"] * (len(tokens) - len(cleaned_tags))

            processed_results.append({
                "tokens": tokens,
                "tags": cleaned_tags,
                "intent": intent
            })
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed for LLM response: {e}. Raw response snippet: {cleaned_response[:500]}...")
        # Return error entries for all original texts if parsing fails for the whole batch
        processed_results = [{"tokens": original_texts[i].lower().split(), "tags": ["PARSE_ERROR"] * len(original_texts[i].lower().split()), "intent": "JSON_PARSE_ERROR"} for i in range(len(original_texts))]
    except Exception as e:
        print(f"Unexpected error in parse_llm_bio_output: {e}. Raw response snippet: {cleaned_response[:500]}...")
        processed_results = [{"tokens": original_texts[i].lower().split(), "tags": ["ERROR"] * len(original_texts[i].lower().split()), "intent": "GENERAL_PARSE_ERROR"} for i in range(len(original_texts))]

    return processed_results
def annotate_batch_texts_ollama(texts_to_annotate: List[str]) -> List[Dict[str, Any]]:
    """Annotates a batch of texts using Ollama and parses into BIO format."""
    if not ollama_client:
        print("Error: Ollama client not configured. Skipping Ollama annotation.")
        return [{"tokens": text.lower().split(), "tags": ["NOT_CONFIGURED"] * len(text.lower().split()), "intent": "OLLAMA_NOT_CONFIGURED"} for text in texts_to_annotate]
    if not texts_to_annotate:
        return []
    prompt_content = get_annotation_prompt(texts_to_annotate)
    max_retries = 3
    retry_delay = 5

    messages = [{'role': 'user', 'content': prompt_content}]

    for attempt in range(max_retries):
        try:
            # Ollama's chat endpoint can accept `format='json'` which helps, but doesn't guarantee structure.
            response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, format='json') 
            assistant_reply = response['message']['content'].strip()

            # ADDED DEBUGGING
            print(f"--- Ollama Annotation Debug ---")
            print(f"Input texts_to_annotate count: {len(texts_to_annotate)}")
            print(f"Ollama raw response length: {len(assistant_reply)}")
            print(f"Ollama raw response (first 1000 chars): >>>\n{assistant_reply[:1000]}\n<<<")
            # END ADDED DEBUGGING

            parsed_results = parse_llm_bio_output(assistant_reply, texts_to_annotate)

            # ADDED DEBUGGING
            print(f"Parsed results count: {len(parsed_results)}")
            if parsed_results:
                print(f"First parsed result example: {parsed_results[0] if parsed_results else 'N/A'}")
                if len(texts_to_annotate) > 0 and len(parsed_results) == len(texts_to_annotate):
                    match_ok = True
                    for idx, (orig, parsed) in enumerate(zip(texts_to_annotate, parsed_results)):
                        if not ("tokens" in parsed and "tags" in parsed and "intent" in parsed):
                            print(f"  Problem with parsed result at index {idx}: {parsed}")
                            match_ok = False
                            break
                        if not parsed.get("tokens") and orig: # Original text was there, but tokens are empty or missing
                            print(f"  Empty/missing tokens for non-empty original text at index {idx}. Original: '{orig}', Parsed: {parsed}")
                    if match_ok:
                         print(f"All parsed results seem to have tokens, tags, and intent fields.")
                elif len(texts_to_annotate) > 0 and len(parsed_results) != len(texts_to_annotate):
                    print(f"  Length mismatch: Expected {len(texts_to_annotate)} input texts, Got {len(parsed_results)} parsed results.")
            elif len(texts_to_annotate) > 0:
                 print(f"  Parsed results list is empty, but expected {len(texts_to_annotate)} results.")
            print(f"--- End Ollama Annotation Debug ---")
            # END ADDED DEBUGGING

            if len(parsed_results) == len(texts_to_annotate) and all("tokens" in p and "tags" in p for p in parsed_results):
                return parsed_results
            else:
                print(f"Warning: Ollama parsing returned inconsistent or incomplete results. Attempt {attempt+1}/{max_retries}.")
                if attempt == max_retries - 1:
                    return [{"tokens": text.lower().split(), "tags": ["PARSE_FAIL"] * len(text.lower().split()), "intent": "OLLAMA_PARSE_FAIL"} for text in texts_to_annotate]
                time.sleep(retry_delay * (attempt + 1))

        except Exception as e:
            print(f"Error calling/processing Ollama AI (Attempt {attempt+1}/{max_retries}): {e}")
            if "connection refused" in str(e).lower() or "timeout" in str(e).lower():
                print("Ollama server might not be running or accessible.")
            time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1:
                return [{"tokens": text.lower().split(), "tags": ["API_ERROR"] * len(text.lower().split()), "intent": "OLLAMA_API_ERROR"} for text in texts_to_annotate]

    print("Error: Max retries reached for Ollama annotation.")
    return [{"tokens": text.lower().split(), "tags": ["MAX_RETRIES"] * len(text.lower().split()), "intent": "OLLAMA_MAX_RETRIES"} for text in texts_to_annotate]

def annotate_batch_texts_gemini(texts_to_annotate: List[str]) -> List[Dict[str, Any]]:
    """Annotates a batch of texts using Google Gemini and parses into BIO format."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping Gemini annotation.")
        return [{"tokens": text.lower().split(), "tags": ["NOT_CONFIGURED"] * len(text.lower().split()), "intent": "GEMINI_NOT_CONFIGURED"} for text in texts_to_annotate]
    if not texts_to_annotate:
        return []
    
    prompt_content = get_annotation_prompt(texts_to_annotate)
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt_content)
            assistant_reply = response.text.strip()

            parsed_results = parse_llm_bio_output(assistant_reply, texts_to_annotate)

            if len(parsed_results) == len(texts_to_annotate) and all("tokens" in p and "tags" in p for p in parsed_results):
                return parsed_results
            else:
                print(f"Warning: Gemini parsing returned inconsistent or incomplete results. Attempt {attempt+1}/{max_retries}.")
                if attempt == max_retries - 1:
                    return [{"tokens": text.lower().split(), "tags": ["PARSE_FAIL"] * len(text.lower().split()), "intent": "GEMINI_PARSE_FAIL"} for text in texts_to_annotate]
                time.sleep(retry_delay * (attempt + 1))

        except Exception as e:
            print(f"Error calling/processing Gemini AI (Attempt {attempt+1}/{max_retries}): {e}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                print("Gemini API quota/rate limit reached.")
            time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1:
                return [{"tokens": text.lower().split(), "tags": ["API_ERROR"] * len(text.lower().split()), "intent": "GEMINI_API_ERROR"} for text in texts_to_annotate]

    print("Error: Max retries reached for Gemini annotation.")
    return [{"tokens": text.lower().split(), "tags": ["MAX_RETRIES"] * len(text.lower().split()), "intent": "GEMINI_MAX_RETRIES"} for text in texts_to_annotate]
def format_gender(gender_str: str) -> str:
    """Formats raw gender string to desired casing."""
    if gender_str.upper() == "FEMALE": return "Female"
    if gender_str.upper() == "MALE": return "Male"
    if gender_str.upper() == "OTHER": return "Other"
    return "Unknown" # For "UNKNOWN" or "ERROR"

def format_emotion(emotion_str: str) -> str:
    """Maps raw emotion labels to desired abbreviations."""
    emotion_map = {
        "ANGER": "Ang",
        "NEUTRAL": "Neu",
        "SADNESS": "Sad",
        "HAPPINESS": "Hap",
        "FEAR": "Fea",
        "DISGUST": "Dis",
        "SURPRISE": "Sur",
        "ERRORLOADINGMODEL": "Err",
        "NO_AUDIO": "NA",
        "EXTRACTION_ERROR": "Err",
        "UNKNOWN": "Unk" # For "Unknown" from model
    }
    return emotion_map.get(emotion_str.upper(), "Unk")

# --- Main Processing Function (MODIFIED) ---
def process_audio_and_annotate(
    dataset_input_dir: str,
    output_json_path: str, # Renamed for clarity
    batch_size: int = 10,
    annotation_model_choice: str = "gemini" # 'gemini', 'ollama', 'both', 'none'
) -> None:
    output_dir_for_file = os.path.dirname(output_json_path)
    os.makedirs(output_dir_for_file, exist_ok=True)
    try:
        print(f"Attempting to clear/create output file: {output_json_path}")
        with open(output_json_path, 'w') as f_clear:
            f_clear.write("[]") # Initialize with an empty JSON array for safety, or just clear
        print(f"Output file {output_json_path} is ready.")
    except IOError as e:
        print(f"Error preparing output file {output_json_path}: {e}. Please check permissions.")
        return

    print("Loading audio analysis models...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model: {e}. Exiting.")
        return

    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"Error loading Emotion model: {e}. Emotion extraction will use 'ErrorLoadingModel'.")

    print("-" * 30)
    print(f"Processing NPTEL-style dataset from: {dataset_input_dir}")
    print(f"Output will be saved to: {output_json_path}")
    print(f"Annotation model choice: {annotation_model_choice.upper()}")
    print("-" * 30)

    file_data_pairs = get_nptel_file_data(dataset_input_dir)
    if not file_data_pairs:
        print("No matching audio files and transcriptions found. Exiting.")
        return

    total_files = len(file_data_pairs)
    print(f"Found {total_files} audio files with transcriptions to process.")

    all_processed_records = [] # Initialize list to store all records
    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    for i, (audio_path, original_transcription_text) in enumerate(file_data_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            transcription = original_transcription_text.lower() # Raw lowercase for LLM input
            if not transcription:
                print(f"  Skipping: Empty transcription for {audio_path}")
                continue

            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1:
                     print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                     continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading audio {audio_path}: {load_err}")
                 continue

            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values)
                age_raw = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                gender_raw = gender_map.get(gender_idx, "UNKNOWN")
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}")
                age_raw = -1.0
                gender_raw = "ERROR"

            emotion_raw = extract_emotion(signal, sr, emotion_model_info)

            record = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": duration,
                "original_transcription": transcription, 
                "predicted_gender_raw": gender_raw,
                "predicted_age_raw": age_raw,
                "predicted_emotion_raw": emotion_raw,
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_for_llm_annotation = [rec["original_transcription"] for rec in processed_records_buffer]
                
                default_bio_data = {"tokens": [], "tags": [], "intent": "NOT_ANNOTATED"}
                gemini_annotated_data = [dict(default_bio_data, tokens=text.lower().split()) for text in texts_for_llm_annotation]
                ollama_annotated_data = [dict(default_bio_data, tokens=text.lower().split()) for text in texts_for_llm_annotation]

                if annotation_model_choice in ["gemini", "both"]:
                    print("  Calling Gemini for annotation...")
                    gemini_annotated_data = annotate_batch_texts_gemini(texts_for_llm_annotation)
                
                if annotation_model_choice in ["ollama", "both"]:
                    print("  Calling Ollama for annotation...")
                    ollama_annotated_data = annotate_batch_texts_ollama(texts_for_llm_annotation)

                if len(gemini_annotated_data) != current_batch_size or len(ollama_annotated_data) != current_batch_size:
                    print(f"  CRITICAL ERROR: Final annotation data count mismatch after LLM calls! Skipping save for this batch.")
                    processed_records_buffer = [] # Clear buffer even on error to avoid reprocessing
                    continue

                for j, record_data in enumerate(processed_records_buffer):
                    current_gemini_data = gemini_annotated_data[j]
                    current_ollama_data = ollama_annotated_data[j]

                    final_record = {
                        "audio_filepath": record_data["audio_filepath"],
                        "text": record_data["original_transcription"],
                        "original_transcription": record_data["original_transcription"],
                        "duration": record_data["duration"],
                        "task_name": "OTHER",
                        "gender": format_gender(record_data["predicted_gender_raw"]),
                        "age_group": ChunkData.get_age_bucket(record_data["predicted_age_raw"]),
                        "emotion": format_emotion(record_data["predicted_emotion_raw"]),
                    }

                    if annotation_model_choice in ["gemini", "both"]:
                        final_record["gemini_intent"] = current_gemini_data["intent"]
                        final_record["bio_annotation_gemini"] = {
                            "tokens": current_gemini_data["tokens"],
                            "tags": current_gemini_data["tags"]
                        }
                    else:
                        final_record["gemini_intent"] = "NOT_USED"
                        final_record["bio_annotation_gemini"] = {"tokens": [], "tags": []}

                    if annotation_model_choice in ["ollama", "both"]:
                        final_record["ollama_intent"] = current_ollama_data["intent"]
                        final_record["bio_annotation_ollama"] = {
                            "tokens": current_ollama_data["tokens"],
                            "tags": current_ollama_data["tags"]
                        }
                    else:
                        final_record["ollama_intent"] = "NOT_USED"
                        final_record["bio_annotation_ollama"] = {"tokens": [], "tags": []}
                    
                    all_processed_records.append(final_record) # Append to the main list
                
                print(f"--- Batch {batch_num} processed and added to memory ({len(processed_records_buffer)} records). Total in memory: {len(all_processed_records)} ---")
                processed_records_buffer = [] 
                torch.cuda.empty_cache()
                gc.collect()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing file {audio_path}: {e}")
            traceback.print_exc()
            continue

    # After the loop, write all accumulated records to the JSON file
    try:
        print(f"\nWriting all {len(all_processed_records)} processed records to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_processed_records, f_out, ensure_ascii=False, indent=4)
        total_records_saved = len(all_processed_records)
        print(f"Successfully wrote {total_records_saved} records to {output_json_path}.")
    except IOError as io_err:
        print(f"  Error writing final JSON to {output_json_path}: {io_err}")
    except Exception as write_err:
        print(f"  Unexpected error writing final JSON: {write_err}")

    print("\n" + "="*30)
    print(f"Processing Finished.")
    print(f"Total files processed attempt: {files_processed_count}/{total_files}")
    print(f"Total records saved to {output_json_path}: {total_records_saved}")
    print("="*30)
if __name__ == "__main__":
    INPUT_DATASET_DIR = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/english_190"
    output_filename = f"{os.path.basename(INPUT_DATASET_DIR)}_structured_bio_annotated.json" # Changed extension to .json
    output_dir_for_jsonl = INPUT_DATASET_DIR 

    FINAL_OUTPUT_JSON = os.path.join(output_dir_for_jsonl, output_filename) # Renamed variable for clarity

    PROCESSING_BATCH_SIZE = 5 

    # --- CHOOSE YOUR ANNOTATION MODEL(S) HERE ---
    # Options: 'gemini', 'ollama', 'both', 'none'
    # SELECTED_ANNOTATION_MODEL = "gemini" 
    # SELECTED_ANNOTATION_MODEL = "ollama"
    SELECTED_ANNOTATION_MODEL = "both" # Recommended for testing both outputs
    # SELECTED_ANNOTATION_MODEL = "none" # Use only transcription without LLM annotation

    print(f"Starting NPTEL Audio Processing and Annotation Workflow...")
    print(f"Input Dataset Directory: {INPUT_DATASET_DIR}")
    print(f"Final Output File: {FINAL_OUTPUT_JSON}") # Updated variable name
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print(f"Selected Annotation Model(s): {SELECTED_ANNOTATION_MODEL.upper()}")
    print("-" * 30)

    if not os.path.isdir(INPUT_DATASET_DIR):
        print(f"ERROR: Input dataset directory not found: {INPUT_DATASET_DIR}")
        exit(1)

    # Pre-check for chosen models
    if SELECTED_ANNOTATION_MODEL in ["gemini", "both"] and not genai:
         print("ERROR: Gemini selected but not configured. Please check GOOGLE_API_KEY. Exiting.")
         exit(1)
    if SELECTED_ANNOTATION_MODEL in ["ollama", "both"] and not ollama_client:
         print(f"ERROR: Ollama selected but not configured/connected. Please ensure Ollama server is running and model '{OLLAMA_MODEL}' is pulled. Exiting.")
         exit(1)
    if SELECTED_ANNOTATION_MODEL not in ['gemini', 'ollama', 'both', 'none']:
        print(f"ERROR: Invalid annotation model choice '{SELECTED_ANNOTATION_MODEL}'. Please choose from 'gemini', 'ollama', 'both', 'none'.")
        exit(1)


    process_audio_and_annotate(
        dataset_input_dir=INPUT_DATASET_DIR,
        output_json_path=FINAL_OUTPUT_JSON, # Updated variable name
        batch_size=PROCESSING_BATCH_SIZE,
        annotation_model_choice=SELECTED_ANNOTATION_MODEL
    )

    print("Workflow complete.")