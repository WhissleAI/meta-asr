# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Devanagari script
import os
import gc
import json
import re
import torch
import librosa
import numpy as np
import pandas as pd # Added for easier TSV reading
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
import time
import soundfile as sf

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
    # Explicitly load API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not found or is empty.")
        print("Attempting to load from .env file if present...")
        load_dotenv() # Try loading again if not set initially
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY environment variable is still not found or empty after loading .env.")

    genai.configure(api_key=api_key)
    # Optional: Test configuration with a simple list_models call
    # models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    # if not models:
    #     print("Warning: No models found that support generateContent. Check API key permissions.")
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None # Set genai to None to prevent further errors if configuration failed


# --- Age/Gender Model Definition (Unchanged - Assumed Language Agnostic) ---
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

# --- Emotion Extraction Function (Unchanged - Assumed Mostly Language Agnostic) ---
# Note: Emotion model performance might vary across languages like Bhojpuri.
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: Optional[dict] = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"

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

# --- Data Structures (Unchanged - Structure is Language Agnostic) ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: str
    transcription: str # This will hold Bhojpuri text
    emotion: str
    chunk_filename: str
    duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    def get_formatted_text(self) -> str:
        if not self.segments: return ""
        segment = self.segments[0]
        age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender
        emotion_text = segment.emotion.upper()
        transcription = segment.transcription.strip() # Keep original case for AI
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        # Ensure single space between transcription and metadata
        return f"{transcription.strip()} {metadata.strip()}"

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100) # Model outputs 0-1 range
        age_brackets = [
            (18, "0_18"), (30, "18_30"), (45, "30_45"),
            (60, "45_60"), (float('inf'), "60PLUS")
        ]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS" # Default

# --- *** NEW Function for loading from TSV *** ---
def load_from_tsv(tsv_file_path: str, audio_base_dir: str) -> List[Tuple[str, str]]:
    """
    Loads data from a TSV file specific to the Bhojpuri dataset structure.

    Args:
        tsv_file_path: Path to the .tsv file (e.g., train.tsv, dev.tsv).
        audio_base_dir: Path to the directory containing ALL audio files (e.g., 'audio').

    Returns:
        A list of tuples: [(full_audio_path, transcription_text), ...].
        Returns empty list on error.
    """
    pairs = []
    missing_audio_files = 0
    malformed_lines = 0
    print(f"Reading data from TSV: {tsv_file_path}")
    print(f"Looking for audio files in: {audio_base_dir}")

    try:
        # Use pandas for robust TSV reading, explicitly using UTF-8
        df = pd.read_csv(tsv_file_path, sep='\t', header=None, names=['id', 'filename', 'transcription'], encoding='utf-8', on_bad_lines='warn')

        for index, row in df.iterrows():
            try:
                # Check for NaN or empty strings which might indicate parsing issues or empty cells
                if pd.isna(row['filename']) or pd.isna(row['transcription']) or not row['filename'] or not row['transcription']:
                     print(f"Warning: Skipping malformed/empty line {index + 1} in TSV: {row.to_list()}")
                     malformed_lines += 1
                     continue

                filename = str(row['filename']).strip()
                transcription = str(row['transcription']).strip()

                # Construct the full path to the audio file
                audio_path = os.path.join(audio_base_dir, filename)

                # Check if the audio file actually exists
                if os.path.isfile(audio_path):
                    pairs.append((audio_path, transcription))
                else:
                    print(f"Warning: Audio file listed in TSV not found: {audio_path} (from line {index + 1})")
                    missing_audio_files += 1

            except Exception as line_err:
                print(f"Error processing line {index + 1} in {tsv_file_path}: {line_err} - Row Data: {row.to_list()}")
                malformed_lines += 1

        print(f"Read {len(df)} lines from TSV.")
        print(f"Successfully created {len(pairs)} audio-transcription pairs.")
        if malformed_lines > 0:
            print(f"Found {malformed_lines} malformed or skipped lines in TSV.")
        if missing_audio_files > 0:
            print(f"Warning: {missing_audio_files} audio files listed in TSV were not found in {audio_base_dir}.")

        if not pairs:
             print(f"Warning: No valid audio-transcription pairs were created from {tsv_file_path}. Check paths and TSV format.")

    except FileNotFoundError:
        print(f"Error: TSV file not found: {tsv_file_path}")
        return []
    except pd.errors.ParserError as pe:
        print(f"Error parsing TSV file {tsv_file_path}: {pe}. Check delimiter and file integrity.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {tsv_file_path}: {e}")
        return []

    return pairs


# --- AI Annotation Functions ---

def correct_entity_tag_spaces(text: str) -> str:
    """Removes spaces within ENTITY_TYPE names like 'CIT Y' -> 'CITY'."""
    # (Unchanged - Logic is based on tag format, not language)
    if not isinstance(text, str): return text

    def replace_spaces(match):
        tag_part = match.group(1)
        type_part = tag_part[len("ENTITY_"):]
        corrected_type = type_part.replace(' ', '')
        return f"ENTITY_{corrected_type}"

    pattern = r'\b(ENTITY_[A-Z0-9_ ]*?[A-Z0-9_])(?=\s+\S)'
    corrected_text = re.sub(pattern, replace_spaces, text)
    return corrected_text


def fix_end_tags_and_spacing(text: str) -> str:
    """
    Cleans up AI-generated annotations, focusing on END tag placement,
    spacing around tags, internal spaces in tag types, and general punctuation.
    Handles Devanagari script characters via \w in regex. (Unchanged - Language Agnostic)
    """
    if not isinstance(text, str):
        return text

    # --- Pre-processing ---
    # 1. Normalize whitespace to single spaces and strip ends
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Tag Structure Correction ---
    # 2. Fix spaces *within* ENTITY_TYPE names (e.g., "ENTITY_PERSON_ NAM E" -> "ENTITY_PERSON_NAME")
    def remove_internal_spaces(match):
        tag_prefix = match.group(1) # e.g., "ENTITY_"
        tag_body = match.group(2)   # e.g., "PERSON_ NAM E"
        corrected_body = re.sub(r'\s+', '', tag_body) # Remove all spaces
        return f"{tag_prefix}{corrected_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\b', remove_internal_spaces, text)

    # 3. Ensure space *after* ENTITY_TAG (if followed by non-space)
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)

    # 4. Ensure space *before* END tag (if preceded by non-space)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)

    # 5. Ensure space *after* END tag (if followed by non-space)
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text)

    # --- Redundant/Misplaced END Tag Removal ---
    # 6. Remove duplicate END tags (run twice for overlapping cases like END END END)
    text = re.sub(r'\bEND\s+END\b', 'END', text)
    text = re.sub(r'\bEND\s+END\b', 'END', text)

    # 7. Remove END tag if it immediately precedes metadata or end-of-string
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text) # Run again

    # --- Final Spacing and Punctuation ---
    # 8. General/Common punctuation spacing rules
    text = re.sub(r'\s+([?!:;,.])', r'\1', text) # Space BEFORE
    text = re.sub(r'([?!:;,.])(\w)', r'\1 \2', text) # Space AFTER
    #    Handle Devanagari danda (।)
    text = re.sub(r'\s+(\।)', r'\1', text)
    text = re.sub(r'(\।)(\w)', r'\1 \2', text)

    # 9. Final whitespace normalization and stripping
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- *** MODIFIED PROMPT FOR BHOJPURI *** ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for Bhojpuri)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate # Return original texts
    if not texts_to_annotate:
        return []

    # Input texts already have metadata like: "bhojpuri text AGE_X GENDER_Y EMOTION_Z"

    # *** REFINED BHOJPURI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Bhojpuri** text written in the **Devanagari** script. Your task is to process a list of Bhojpuri sentences, each already containing `AGE_*`, `GENDER_*`, and `EMOTION_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, and `EMOTION_*` tags at the end of each sentence **must** remain exactly as they are.
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Bhojpuri Text Only):**
    *   Identify entities **only** within the main **Bhojpuri** transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Bhojpuri entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_LOCATION`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Bhojpuri entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation).
    *   **Example:** For the entity "पटना", the correct annotation is `ENTITY_CITY पटना END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the **Bhojpuri** sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Bhojpuri Language Specifics:**
    *   Handle Devanagari script correctly.
    *   Ensure correct spacing around common punctuation (like `.`, `?`, `!`, `।`). Remove space before punctuation, ensure space after punctuation if followed by a word.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Using Placeholder Bhojpuri):**

*   **Input:** `"जौनपुर के जिलाधिकारी अगला दुइ बुध अउर बियफे के छुट्टी कइले बाड़े AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_LOCATION जौनपुर END के जिलाधिकारी अगला ENTITY_DATE दुइ बुध END अउर ENTITY_DATE बियफे END के छुट्टी कइले बाड़े AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

*   **Input:** `"हमनी के देस खेती किसानी पर निर्भर बा AGE_45_60 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"हमनी के देस खेती किसानी पर निर्भर बा AGE_45_60 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`
    *(Note: No entities identified in this example sentence)*

*   **Input:** `"का हाल बा रउआ? AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"का हाल बा रउआ? AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL INTENT_QUESTION"`

**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_LOCA TION जौनपुर END ...` (Space in TYPE)
*   `... ENTITY_LOCATION जौनपुरEND ...` (Missing space before END)
*   `... ENTITY_LOCATION जौनपुर ENDके ...` (Missing space after END)
*   `... कइले बाड़े END AGE_...` (Unnecessary END before metadata)
*   `... अगला END दुइ END बुध END ...` (Incorrectly adding END after non-entity words)

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED BHOJPURI PROMPT ENDS HERE ***

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Use Flash for speed/cost
            # Configure safety settings to be less restrictive if needed, but be cautious
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
            # response = model.generate_content(prompt, safety_settings=safety_settings)
            response = model.generate_content(prompt)


            # Check for safety blocks before accessing .text
            if not response.candidates:
                 print(f"Warning: Gemini response blocked due to safety settings (Attempt {attempt+1}). Response parts: {response.parts}")
                 # Return original text with error marker - OR you could retry/raise
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_SAFETY_BLOCK" for t in texts_to_annotate]

            # If not blocked, proceed to extract text
            assistant_reply = response.text.strip()


            # --- Robust JSON Extraction ---
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Handle potential markdown code fences
                if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()

                if assistant_reply.startswith('[') and assistant_reply.endswith(']'):
                    json_str = assistant_reply
                else:
                    print(f"Error: Could not extract valid JSON list from response (Attempt {attempt+1}). Response snippet:\n---\n{assistant_reply[:500]}\n---")
                    # If it looks like a list *without* brackets, try adding them
                    if assistant_reply.strip().startswith('{') and assistant_reply.strip().endswith('}'):
                         print("Trying to wrap response in list brackets []...")
                         json_str = f"[{assistant_reply}]" # Example: if only one item was returned as a plain object
                    else:
                         raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            # --- JSON Parsing and Post-Processing ---
            try:
                annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed specifically on extracted string (Attempt {attempt+1}): {json_e}")
                print("Extracted string snippet:", json_str[:500])
                # Provide original text with error marker on final attempt
                if attempt == max_retries - 1:
                     return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE_FINAL" for t in texts_to_annotate]
                raise json_e # Raise to trigger retry

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str):
                          final_sentence = fix_end_tags_and_spacing(sentence)
                          processed_sentences.append(final_sentence)
                     else:
                          print(f"Warning: Non-string item received in annotation list at index {idx}: {sentence}")
                          try: original_text = texts_to_annotate[idx]
                          except IndexError: original_text = "ORIGINAL_TEXT_INDEX_ERROR"
                          processed_sentences.append(fix_end_tags_and_spacing(original_text) + " ANNOTATION_ERROR_NON_STRING")

                if len(processed_sentences) == len(texts_to_annotate):
                    print(f"Debug: Batch annotation successful (Attempt {attempt+1}). Lengths match.")
                    return processed_sentences
                else:
                    # This case should ideally not happen if the non-string handling is correct
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1:
                        return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                    else: raise ValueError("Processing error lead to length mismatch.")

            else:
                print(f"Error: API returned invalid list or mismatched length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1:
                     return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation...")
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors
            if "rate limit" in str(e).lower():
                 print("Rate limit likely hit. Increasing delay and retrying...")
                 time.sleep(retry_delay * (attempt + 1) * 5) # Exponential backoff for rate limits
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                 print("FATAL: Invalid Google API Key or permission issue.")
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            # Handle potential content blocking explicitly if generate_content raises it (though the candidate check above is preferred)
            elif "response was blocked" in str(e).lower():
                 print(f"Warning: Gemini response blocked (Attempt {attempt+1}).")
                 if attempt == max_retries - 1:
                     return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_SAFETY_BLOCK_FINAL" for t in texts_to_annotate]
                 time.sleep(retry_delay * (attempt+1)) # Simple retry for blocks
            else:
                 time.sleep(retry_delay * (attempt + 1)) # Standard delay for other errors

            if attempt == max_retries - 1:
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]


# --- *** MODIFIED Main Processing Function for Bhojpuri *** ---
def process_audio_and_annotate(
    tsv_file_path: str,
    audio_base_dir: str,
    output_jsonl_path: str,
    age_gender_model: AgeGenderModel, # Pass loaded models
    processor: Wav2Vec2Processor,     # Pass loaded models
    emotion_model_info: Optional[dict], # Pass loaded models
    batch_size: int = 10
) -> None:
    """
    Processes Bhojpuri audio files and text from a specific TSV split file,
    extracts metadata, gets transcriptions, formats text, annotates with AI
    (using Bhojpuri prompt), and saves final JSONL output for that split.

    Args:
        tsv_file_path (str): Path to the input TSV file (e.g., train.tsv).
        audio_base_dir (str): Path to the base directory containing audio files.
        output_jsonl_path (str): Path to save the output JSONL file for this split.
        age_gender_model: Loaded Age/Gender model.
        processor: Loaded Wav2Vec2Processor.
        emotion_model_info: Loaded Emotion model info dict (or None).
        batch_size (int): Number of records to process before AI annotation/saving.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Clear output file at the very beginning for this specific split ---
    try:
        print(f"Attempting to clear/create output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear:
            f_clear.write("") # Write empty string to clear or create the file
        print(f"Output file {output_jsonl_path} is ready.")
    except IOError as e:
        print(f"Error preparing output file {output_jsonl_path}: {e}. Please check permissions. Skipping this split.")
        return # Stop processing for this split if file can't be prepared

    print("-" * 30)
    print(f"Starting processing for split: {os.path.basename(tsv_file_path)}")
    print(f"  Input TSV: {tsv_file_path}")
    print(f"  Audio Dir: {audio_base_dir}")
    print(f"  Output JSONL: {output_jsonl_path}")
    print("-" * 30)

    # Get audio paths and corresponding transcriptions using the new TSV function
    file_pairs = load_from_tsv(tsv_file_path, audio_base_dir)
    if not file_pairs:
        print(f"No matching audio-transcription data found for {os.path.basename(tsv_file_path)}. Skipping this split.")
        return

    total_files = len(file_pairs)
    print(f"Found {total_files} audio files with matching transcriptions to process for this split.")

    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Files for this split ---
    for i, (audio_path, transcription) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # 1. Transcription is already loaded
            if not transcription:
                print(f"  Skipping: Empty transcription provided for {audio_path}")
                continue

            # 2. Load Audio
            signal, sr, duration = None, 16000, 0
            try:
                try:
                    signal_raw, sr_orig = sf.read(audio_path, dtype='float32')
                except Exception as sf_err:
                    print(f"  Soundfile failed ({sf_err}), trying librosa...")
                    signal_raw, sr_orig = librosa.load(audio_path, sr=None, mono=True)

                target_sr = 16000
                if sr_orig != target_sr:
                    signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    if signal_float.ndim > 1: signal_mono = np.mean(signal_float, axis=1)
                    else: signal_mono = signal_float
                    signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr)
                    sr = target_sr
                else:
                    signal = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    if signal.ndim > 1: signal = np.mean(signal, axis=1)
                    sr = sr_orig

                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1:
                    print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                    continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading/processing audio {audio_path}: {load_err}")
                 continue

            # 3. Extract Age/Gender
            age, gender = -1.0, "ERROR"
            max_len_inference = 30 * sr # Limit input length for stability/memory
            inputs, input_values, input_values_truncated = None, None, None
            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values
                if input_values.shape[1] > max_len_inference:
                    # print(f"  Truncating input for inference ({input_values.shape[1]} > {max_len_inference})") # Optional debug
                    input_values_truncated = input_values[:, :max_len_inference].to(device)
                else:
                    input_values_truncated = input_values.to(device)

                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values_truncated)
                age = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                gender = gender_map.get(gender_idx, "UNKNOWN")
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     print(f"  CUDA OOM Error during Age/Gender extraction. Skipping file.")
                     torch.cuda.empty_cache(); gc.collect(); continue
                 else: print(f"  Runtime Error during Age/Gender extraction: {e}"); age = -1.0; gender = "ERROR"
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}"); age = -1.0; gender = "ERROR"

            # 4. Extract Emotion
            emotion = "ERROR"
            emotion_signal = None
            try:
                 if input_values is not None and input_values.shape[1] > max_len_inference:
                     emotion_signal = signal[:max_len_inference]
                 else:
                      emotion_signal = signal
                 emotion = extract_emotion(emotion_signal, sr, emotion_model_info)
            except Exception as emotion_err:
                 print(f"  Error during Emotion extraction: {emotion_err}"); emotion = "ERROR"

            # 5. Create Initial Record
            try:
                # Attempt to parse speaker ID from filename (adjust if needed)
                # Assuming format like Regional-Gorakhpur-Bhojpuri-1730-20191217182853_23.wav
                # We might use the first part or a combination as speaker ID
                # Using the full basename without extension as a unique ID proxy for now
                speaker = os.path.splitext(os.path.basename(audio_path))[0]
            except Exception:
                speaker = "UNKNOWN_SPEAKER" # Fallback
            segment_data = AudioSegment(start_time=0, end_time=duration, speaker=speaker, age=age, gender=gender, transcription=transcription, emotion=emotion, chunk_filename=os.path.basename(audio_path), duration=duration)
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text()
            # Include more raw info in the buffer if useful for debugging later
            record = {
                 "audio_filepath": chunk.filepath,
                 "duration": duration,
                 "initial_text": initial_formatted_text, # Text sent to AI
                 "raw_transcription": transcription,   # Original transcription
                 "raw_age_output": age,
                 "raw_gender_prediction": gender,
                 "raw_emotion_prediction": emotion,
                 "speaker_id": speaker
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 6. Annotate and Save in Batches
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) for {os.path.basename(output_jsonl_path)} ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                # Use the Bhojpuri-specific annotation function
                annotated_texts = annotate_batch_texts(texts_to_annotate)

                if len(annotated_texts) != current_batch_size:
                     print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                     # Decide how to handle: skip batch? try again? stop? For now, just skip saving this batch.
                else:
                    try:
                        lines_written_in_batch = 0
                        # Ensure UTF-8 for writing Devanagari text
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                # Final record structure for JSONL
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text, # This contains the final Bhojpuri text with tags
                                }
                                if "ANNOTATION_ERROR" in annotated_text:
                                     print(f"  Warning: Saving record for {os.path.basename(record_data['audio_filepath'])} with annotation error flag: {annotated_text.split()[-1]}")

                                # Ensure JSON dumps handles Unicode correctly
                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1
                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total saved for this split: {total_records_saved} ---")
                    except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                # Clear buffer and memory regardless of save success to proceed
                processed_records_buffer = []
                del texts_to_annotate, annotated_texts
                if inputs: del inputs
                if input_values: del input_values
                if input_values_truncated: del input_values_truncated
                if signal: del signal
                if 'signal_raw' in locals(): del signal_raw
                if 'signal_float' in locals(): del signal_float
                if 'signal_mono' in locals(): del signal_mono
                if 'logits_age' in locals(): del logits_age, logits_gender
                if emotion_signal: del emotion_signal
                if 'outputs' in locals() and outputs: del outputs # Check if outputs exists
                torch.cuda.empty_cache(); gc.collect()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            # Need to decide if we want to save the current buffer partially
            # For now, just stop.
            return # Stop processing this split

        except Exception as e:
            print(f"  FATAL ERROR processing file {os.path.basename(audio_path)}: {e}")
            import traceback; traceback.print_exc()
            # Clean up GPU memory in case of OOM or other GPU related errors before continuing
            torch.cuda.empty_cache(); gc.collect()
            continue # Try the next file in the split

    print("\n" + "="*30 + f"\nProcessing Finished for split: {os.path.basename(tsv_file_path)}.\nTotal files processing attempted: {files_processed_count}/{total_files}\nTotal records saved to {output_jsonl_path}: {total_records_saved}\n" + "="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Bhojpuri Data *** ---
    # >>> IMPORTANT: SET THIS TO YOUR BHOJPURI DATA LOCATION <<<
    BHOJPURI_BASE_DIR = "/external4/datasets/madasr/bhojpuri/data/hi_in"
    # <<< END OF PATH TO CHANGE <<<

    PROCESSING_BATCH_SIZE = 10 # Adjust based on API limits and system memory

    # --- API Key and Setup ---
    load_dotenv() # Load .env file if present
    if not os.getenv("GOOGLE_API_KEY") and genai is None: # Check again if initial config failed
        print("ERROR: GOOGLE_API_KEY environment variable not found and GenAI not configured.")
        print("Please set it in a .env file or your environment.")
        exit(1)
    elif genai is None:
        # Attempt re-configuration if GOOGLE_API_KEY might have been loaded by load_dotenv() just now
        print("Attempting Google GenAI configuration again...")
        try:
             api_key_recheck = os.getenv("GOOGLE_API_KEY")
             if not api_key_recheck: raise ValueError("GOOGLE_API_KEY still not found.")
             genai.configure(api_key=api_key_recheck)
             print("Google Generative AI re-configured successfully.")
        except Exception as e:
             print(f"ERROR: Failed to configure Google Generative AI on retry: {e}")
             exit(1)

    # --- Load Models ONCE before processing splits ---
    print("Loading models...")
    loaded_processor = None
    loaded_age_gender_model = None
    loaded_emotion_model_info = {}

    # Age/Gender Model
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        loaded_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        loaded_age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        loaded_age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model ({age_gender_model_name}): {e}. Exiting.")
        exit(1) # Exit if essential model fails

    # Emotion Model
    emotion_model_name = "superb/hubert-large-superb-er"
    try:
        loaded_emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        loaded_emotion_model_info['model'].eval()
        loaded_emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"Warning: Error loading Emotion model ({emotion_model_name}): {e}. Emotion extraction will use 'ErrorLoadingModel'.")
        loaded_emotion_model_info = None # Indicate model load failure but continue

    print("-" * 30)
    print("Starting Bhojpuri Audio Processing and Annotation Workflow...")
    print(f"Input Base Directory: {BHOJPURI_BASE_DIR}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    # Check base directory exists
    if not os.path.isdir(BHOJPURI_BASE_DIR):
        print(f"ERROR: Base directory not found: {BHOJPURI_BASE_DIR}")
        exit(1)

    # Define the splits to process
    splits = ["train", "dev", "test"]
    audio_dir = os.path.join(BHOJPURI_BASE_DIR, "audio") # Common audio directory

    # Check if common audio directory exists
    if not os.path.isdir(audio_dir):
        print(f"ERROR: Common audio directory not found: {audio_dir}")
        exit(1)

    # --- Loop through each split (train, dev, test) ---
    for split in splits:
        tsv_path = os.path.join(BHOJPURI_BASE_DIR, f"{split}.tsv")
        output_jsonl_path = os.path.join(BHOJPURI_BASE_DIR, f"{split}_annotated.jsonl") # Output in the same directory

        # *** MODIFIED: Construct split-specific audio directory ***
        split_audio_dir = os.path.join(audio_dir, split) # e.g., audio/train, audio/dev

        # Check if the TSV file for the split exists
        if not os.path.isfile(tsv_path):
            print(f"Warning: TSV file for split '{split}' not found at {tsv_path}. Skipping this split.")
            continue

        # *** NEW: Check if the specific audio directory for the split exists ***
        if not os.path.isdir(split_audio_dir):
             print(f"Warning: Audio directory for split '{split}' not found at {split_audio_dir}. Skipping this split.")
             continue

        # Call the processing function for the current split
        # *** MODIFIED: Pass the split-specific audio directory ***
        process_audio_and_annotate(
            tsv_file_path=tsv_path,
            audio_base_dir=split_audio_dir, # Use the specific directory
            output_jsonl_path=output_jsonl_path,
            age_gender_model=loaded_age_gender_model, # Pass loaded models
            processor=loaded_processor,             # Pass loaded models
            emotion_model_info=loaded_emotion_model_info, # Pass loaded models
            batch_size=PROCESSING_BATCH_SIZE
        )
        # Optional: Add extra cleanup between splits if memory is very tight
        # torch.cuda.empty_cache(); gc.collect()

    print("\nWorkflow complete for all specified splits.")
