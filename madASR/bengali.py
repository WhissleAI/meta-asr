# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Bengali script
import os
import gc
import json
import re
import torch
import librosa
import numpy as np
import pandas as pd # For reading TSV
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not found or is empty.")
        print("Attempting to load from .env file if present...")
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY environment variable is still not found or empty after loading .env.")

    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None


# --- Age/Gender Model Definition (Unchanged) ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features; x = self.dropout(x); x = self.dense(x)
        x = torch.tanh(x); x = self.dropout(x); x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config; self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1); self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]; hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: Optional[dict] = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0: return "No_Audio"
    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        return model_info['model'].config.id2label.get(predicted_class_idx, "Unknown").upper()
    except Exception as e: print(f"Error during emotion extraction: {e}"); return "Extraction_Error"

# --- Data Structures (Unchanged) ---
@dataclass
class AudioSegment:
    start_time: float; end_time: float; speaker: str; age: float; gender: str
    transcription: str # This will hold Bengali text
    emotion: str; chunk_filename: str; duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list); filepath: str = ""
    def get_formatted_text(self) -> str:
        if not self.segments: return ""
        segment = self.segments[0]; age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender; emotion_text = segment.emotion.upper()
        transcription = segment.transcription.strip()
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        return f"{transcription.strip()} {metadata.strip()}"
    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100)
        brackets = [(18, "0_18"), (30, "18_30"), (45, "30_45"), (60, "45_60"), (float('inf'), "60PLUS")]
        for threshold, bracket in brackets:
            if actual_age < threshold: return bracket
        return "60PLUS"

# --- *** NEW Function for loading Bengali TSV *** ---
def load_bengali_tsv(tsv_file_path: str, audio_files_dir: str) -> List[Tuple[str, str, str]]:
    """
    Loads data from the specific Bengali utt_spk_text.tsv format.

    Args:
        tsv_file_path: Path to the utt_spk_text.tsv file.
        audio_files_dir: Path to the directory containing the .flac audio files (e.g., 'data').

    Returns:
        A list of tuples: [(full_audio_path, transcription_text, speaker_id), ...].
        Returns empty list on error.
    """
    pairs = []
    missing_audio_files = 0
    malformed_lines = 0
    print(f"Reading data from Bengali TSV: {tsv_file_path}")
    print(f"Looking for audio files in: {audio_files_dir}")

    try:
        # Read the TSV with 3 columns, no header, UTF-8 encoding
        df = pd.read_csv(
            tsv_file_path,
            sep='\t',
            header=None,
            names=['utterance_id', 'speaker_id', 'transcription'],
            encoding='utf-8',
            on_bad_lines='warn' # Log problematic lines
        )

        for index, row in df.iterrows():
            try:
                # Check for NaN or empty strings
                if pd.isna(row['utterance_id']) or pd.isna(row['speaker_id']) or pd.isna(row['transcription']) \
                   or not str(row['utterance_id']).strip() or not str(row['transcription']).strip():
                    print(f"Warning: Skipping malformed/empty line {index + 1} in TSV: {row.to_list()}")
                    malformed_lines += 1
                    continue

                utterance_id = str(row['utterance_id']).strip()
                speaker_id = str(row['speaker_id']).strip() # Keep speaker ID
                transcription = str(row['transcription']).strip()

                # Construct the expected audio filename (utterance_id + .flac)
                audio_filename = f"{utterance_id}.flac"
                audio_path = os.path.join(audio_files_dir, audio_filename)

                # Check if the audio file actually exists
                if os.path.isfile(audio_path):
                    pairs.append((audio_path, transcription, speaker_id)) # Store path, text, and speaker
                else:
                    # Be less verbose, maybe print only first few missing ones?
                    if missing_audio_files < 5:
                        print(f"Warning: Audio file listed in TSV not found: {audio_path} (from line {index + 1})")
                    elif missing_audio_files == 5:
                         print("Warning: Further missing audio file warnings will be suppressed.")
                    missing_audio_files += 1

            except Exception as line_err:
                print(f"Error processing line {index + 1} in {tsv_file_path}: {line_err} - Row Data: {row.to_list()}")
                malformed_lines += 1

        print(f"Read {len(df)} lines from TSV.")
        print(f"Successfully created {len(pairs)} audio-transcription-speaker tuples.")
        if malformed_lines > 0:
            print(f"Found {malformed_lines} malformed or skipped lines in TSV.")
        if missing_audio_files > 0:
            print(f"Warning: {missing_audio_files} audio files listed in TSV were not found in {audio_files_dir}.")

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

def correct_entity_tag_spaces(text: str) -> str: # Unchanged
    if not isinstance(text, str): return text
    def replace_spaces(match):
        tag_part = match.group(1); type_part = tag_part[len("ENTITY_"):]
        corrected_type = type_part.replace(' ', '')
        return f"ENTITY_{corrected_type}"
    pattern = r'\b(ENTITY_[A-Z0-9_ ]*?[A-Z0-9_])(?=\s+\S)'
    return re.sub(pattern, replace_spaces, text)

def fix_end_tags_and_spacing(text: str) -> str: # Unchanged (Handles Devanagari/Bengali via \w)
    if not isinstance(text, str): return text
    text = re.sub(r'\s+', ' ', text).strip()
    def remove_internal_spaces(match):
        prefix, body = match.group(1), match.group(2)
        cleaned_body = re.sub(r'\\s+', '', body)
        return f"{prefix}{cleaned_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\\b', remove_internal_spaces, text)
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text) # Space after ENTITY_TAG
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text) # Space before END
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text) # Space after END
    text = re.sub(r'\bEND\s+END\b', 'END', text); text = re.sub(r'\bEND\s+END\b', 'END', text) # Dup ENDs
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text) # END before metadata/EOF
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text) # Run again
    text = re.sub(r'\s+([?!:;,।.])', r'\1', text) # Space before punct (added Bengali dā̃ṛi ।)
    text = re.sub(r'([?!:;,।.])(\w)', r'\1 \2', text) # Space after punct
    text = re.sub(r'\s+', ' ', text).strip() # Final whitespace
    return text

# --- *** MODIFIED PROMPT FOR BENGALI *** ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for Bengali)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate
    if not texts_to_annotate: return []

    # Input texts already have metadata like: "bengali text AGE_X GENDER_Y EMOTION_Z"

    # *** REFINED BENGALI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Bengali** text written in the **Bengali script**. Your task is to process a list of Bengali sentences, each already containing `AGE_*`, `GENDER_*`, and `EMOTION_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, and `EMOTION_*` tags at the end of each sentence **must** remain exactly as they are.
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Bengali Text Only):**
    *   Identify entities **only** within the main **Bengali** transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Bengali entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_LOCATION`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Bengali entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation like the Bengali dā̃ṛi '।').
    *   **Example:** For the entity "কলকাতা", the correct annotation is `ENTITY_CITY কলকাতা END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the **Bengali** sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Bengali Language Specifics:**
    *   Handle Bengali script correctly.
    *   Ensure correct spacing around common punctuation (like `?`, `!`, `।`). Remove space before punctuation, ensure space after punctuation if followed by a word.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Bengali):**

*   **Input:** `"বীরভূম জেলা হাসপাতালে AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_LOCATION বীরভূম জেলা END হাসপাতালে AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

*   **Input:** `"আপনার গল্প ছাড়া AGE_45_60 GENDER_FEMALE EMOTION_SAD"`
*   **Correct Output:** `"আপনার গল্প ছাড়া AGE_45_60 GENDER_FEMALE EMOTION_SAD INTENT_INFORM"`
    *(Note: No entities identified)*

*   **Input:** `"মান্নান সৈয়দের মৃত্যুর পর বেরিয়েছে AGE_18_30 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_PERSON_NAME মান্নান সৈয়দের END মৃত্যুর পর বেরিয়েছে AGE_18_30 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_LOCA TION বীরভূম জেলা END ...` (Space in TYPE)
*   `... ENTITY_LOCATION বীরভূম জেলাEND ...` (Missing space before END)
*   `... ENTITY_LOCATION বীরভূম জেলা ENDহাসপাতালে ...` (Missing space after END)
*   `... পর বেরিয়েছে END AGE_...` (Unnecessary END before metadata)
*   `... বীরভূম END জেলা END ...` (Incorrectly adding END after non-entity words)

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED BENGALI PROMPT ENDS HERE ***

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            # Optional: Adjust safety settings if overly restrictive (use with caution)
            # safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            # response = model.generate_content(prompt, safety_settings=safety_settings)
            response = model.generate_content(prompt)

            # --- Handle potential safety blocks ---
            if not response.candidates:
                 print(f"Warning: Gemini response blocked due to safety settings (Attempt {attempt+1}). Response parts: {response.parts}")
                 # Return original text with error marker
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_SAFETY_BLOCK" for t in texts_to_annotate]

            assistant_reply = response.text.strip()

            # --- Robust JSON Extraction ---
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match: json_str = json_match.group(0)
            else:
                if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()
                if assistant_reply.startswith('[') and assistant_reply.endswith(']'): json_str = assistant_reply
                else:
                    print(f"Error: Could not extract valid JSON list from response (Attempt {attempt+1}). Response snippet:\n---\n{assistant_reply[:500]}\n---")
                    raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            # --- JSON Parsing and Post-Processing ---
            try: annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed on extracted string (Attempt {attempt+1}): {json_e}"); print("Extracted string snippet:", json_str[:500])
                if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE_FINAL" for t in texts_to_annotate]
                raise json_e

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str): processed_sentences.append(fix_end_tags_and_spacing(sentence))
                     else:
                          print(f"Warning: Non-string item received at index {idx}: {sentence}")
                          original_text = texts_to_annotate[idx] if idx < len(texts_to_annotate) else "ORIGINAL_TEXT_OOB"
                          processed_sentences.append(fix_end_tags_and_spacing(original_text) + " ANNOTATION_ERROR_NON_STRING")
                if len(processed_sentences) == len(texts_to_annotate):
                    print(f"Debug: Batch annotation successful (Attempt {attempt+1})."); return processed_sentences
                else:
                    print(f"Error: Mismatch after processing non-strings. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                    else: raise ValueError("Processing error led to length mismatch.")
            else:
                print(f"Error: API returned invalid list or mismatched length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else: print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})..."); time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation..."); time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}"); import traceback; traceback.print_exc()
            if "rate limit" in str(e).lower(): time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                 print("FATAL: Invalid Google API Key or permission issue."); return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            elif "response was blocked" in str(e).lower(): # Handle explicit block exception
                 print(f"Warning: Gemini response blocked via exception (Attempt {attempt+1}).")
                 if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_SAFETY_BLOCK_FINAL" for t in texts_to_annotate]
                 time.sleep(retry_delay * (attempt+1))
            else: time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]


# --- *** MODIFIED Main Processing Function for Bengali *** ---
def process_audio_and_annotate_bengali(
    manifest_tsv_path: str,
    audio_files_dir: str,
    output_jsonl_path: str,
    age_gender_model: AgeGenderModel, # Pass loaded models
    processor: Wav2Vec2Processor,
    emotion_model_info: Optional[dict],
    batch_size: int = 10
) -> None:
    """
    Processes Bengali audio files and text from the specific utt_spk_text.tsv,
    extracts metadata, formats text, annotates with AI (Bengali prompt),
    and saves the final JSONL output.

    Args:
        manifest_tsv_path (str): Path to the input utt_spk_text.tsv file.
        audio_files_dir (str): Path to the directory containing .flac audio files.
        output_jsonl_path (str): Path to save the output JSONL file.
        age_gender_model: Loaded Age/Gender model.
        processor: Loaded Wav2Vec2Processor.
        emotion_model_info: Loaded Emotion model info dict (or None).
        batch_size (int): Number of records to process before AI annotation/saving.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Clear/Create output file ---
    try:
        print(f"Attempting to clear/create output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear: f_clear.write("")
        print(f"Output file {output_jsonl_path} is ready.")
    except IOError as e:
        print(f"Error preparing output file {output_jsonl_path}: {e}. Processing stopped.")
        return

    print("-" * 30)
    print(f"Starting Bengali processing:")
    print(f"  Manifest TSV: {manifest_tsv_path}")
    print(f"  Audio Dir:    {audio_files_dir}")
    print(f"  Output JSONL: {output_jsonl_path}")
    print("-" * 30)

    # Get audio paths, transcriptions, and speaker IDs using the Bengali TSV function
    file_tuples = load_bengali_tsv(manifest_tsv_path, audio_files_dir)
    if not file_tuples:
        print(f"No valid audio-transcription data found from {manifest_tsv_path}. Exiting.")
        return

    total_files = len(file_tuples)
    print(f"Found {total_files} audio files with matching transcriptions and speakers to process.")

    processed_records_buffer = []
    batch_num = 0; files_processed_count = 0; total_records_saved = 0

    # --- Process Files ---
    # Now loop through (audio_path, transcription, speaker_id) tuples
    for i, (audio_path, transcription, speaker_id) in enumerate(file_tuples):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)} (Speaker: {speaker_id})")
        try:
            # 1. Transcription & Speaker ID are already loaded
            if not transcription: print(f"  Skipping: Empty transcription for {audio_path}"); continue

            # 2. Load Audio
            signal, sr, duration = None, 16000, 0
            try:
                # Use soundfile primarily for flac
                signal_raw, sr_orig = sf.read(audio_path, dtype='float32')
                target_sr = 16000
                if sr_orig != target_sr:
                    if signal_raw.ndim > 1: signal_mono = np.mean(signal_raw, axis=1)
                    else: signal_mono = signal_raw
                    signal = librosa.resample(y=signal_mono.astype(np.float32), orig_sr=sr_orig, target_sr=target_sr)
                    sr = target_sr
                else:
                    signal = signal_raw if signal_raw.ndim == 1 else np.mean(signal_raw, axis=1)
                    sr = sr_orig
                if signal is None or len(signal) == 0: print(f"  Skipping: Failed load/empty audio {audio_path}"); continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1: print(f"  Skipping: Audio too short ({duration:.2f}s) {audio_path}"); continue
            except Exception as load_err: print(f"  Skipping: Error loading/processing audio {audio_path}: {load_err}"); continue

            # 3. Extract Age/Gender
            age, gender = -1.0, "ERROR"
            max_len_inference = 30 * sr # Limit input length
            inputs, input_values, input_values_truncated = None, None, None
            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values
                input_values_truncated = input_values[:, :max_len_inference].to(device) if input_values.shape[1] > max_len_inference else input_values.to(device)
                with torch.no_grad(): _, logits_age, logits_gender = age_gender_model(input_values_truncated)
                age = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender = {0: "FEMALE", 1: "MALE", 2: "OTHER"}.get(gender_idx, "UNKNOWN")
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"  CUDA OOM Error during Age/Gender. Skipping."); torch.cuda.empty_cache(); gc.collect(); continue
                 else: print(f"  Runtime Error Age/Gender: {e}"); age = -1.0; gender = "ERROR"
            except Exception as age_gender_err: print(f"  Error Age/Gender: {age_gender_err}"); age = -1.0; gender = "ERROR"

            # 4. Extract Emotion
            emotion = "ERROR"
            try:
                 emotion_signal = signal[:max_len_inference] if input_values is not None and input_values.shape[1] > max_len_inference else signal
                 emotion = extract_emotion(emotion_signal, sr, emotion_model_info)
            except Exception as emotion_err: print(f"  Error Emotion: {emotion_err}"); emotion = "ERROR"

            # 5. Create Initial Record (Use speaker_id from TSV)
            segment_data = AudioSegment(start_time=0, end_time=duration, speaker=speaker_id, age=age, gender=gender, transcription=transcription, emotion=emotion, chunk_filename=os.path.basename(audio_path), duration=duration)
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text()
            record = { "audio_filepath": chunk.filepath, "duration": duration, "initial_text": initial_formatted_text, "raw_transcription": transcription, "raw_age_output": age, "raw_gender_prediction": gender, "raw_emotion_prediction": emotion, "speaker_id": speaker_id } # Store speaker_id
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 6. Annotate and Save in Batches
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1; current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")
                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                annotated_texts = annotate_batch_texts(texts_to_annotate) # Use Bengali prompt

                if len(annotated_texts) != current_batch_size:
                     print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                else:
                    try:
                        lines_written_in_batch = 0
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out: # Append mode
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                final_record = {"audio_filepath": record_data["audio_filepath"], "duration": record_data["duration"], "text": annotated_text} # Final JSONL structure
                                if "ANNOTATION_ERROR" in annotated_text: print(f"  Warning: Saving record for {os.path.basename(record_data['audio_filepath'])} with annotation error: {annotated_text.split()[-1]}")
                                json_str = json.dumps(final_record, ensure_ascii=False) # Handle Bengali script
                                f_out.write(json_str + '\n'); lines_written_in_batch += 1
                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")
                    except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                processed_records_buffer = [] # Clear buffer
                # Explicit memory cleanup
                del texts_to_annotate, annotated_texts
                if inputs: del inputs;
                if input_values: del input_values;
                if input_values_truncated: del input_values_truncated;
                if signal: del signal;
                if 'signal_raw' in locals(): del signal_raw;
                if 'signal_mono' in locals(): del signal_mono;
                if 'logits_age' in locals(): del logits_age, logits_gender;
                if 'emotion_signal' in locals(): del emotion_signal
                if 'outputs' in locals() and outputs: del outputs
                torch.cuda.empty_cache(); gc.collect()

        except KeyboardInterrupt: print("\nProcessing interrupted by user."); return
        except Exception as e:
            print(f"  FATAL ERROR processing file {os.path.basename(audio_path)}: {e}")
            import traceback; traceback.print_exc(); torch.cuda.empty_cache(); gc.collect(); continue # Try next file

    print("\n" + "="*30 + f"\nBengali Processing Finished.\nTotal files processing attempted: {files_processed_count}/{total_files}\nTotal records saved to {output_jsonl_path}: {total_records_saved}\n" + "="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Bengali Data *** ---
    # >>> SET THIS TO YOUR BENGALI DATASET LOCATION <<<
    BENGALI_BASE_DIR = "/external4/datasets/madasr/asr_bengali"
    # >>> SET DESIRED OUTPUT FILENAME <<<
    FINAL_OUTPUT_JSONL = os.path.join(BENGALI_BASE_DIR, "bengali_annotated_output.jsonl")
    # <<< END OF PATHS TO CHANGE <<<

    PROCESSING_BATCH_SIZE = 10 # Adjust based on API limits/memory

    # --- API Key Setup ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY") and genai is None:
        print("ERROR: GOOGLE_API_KEY not found and GenAI not configured. Set in .env or environment."); exit(1)
    elif genai is None: # Attempt re-config if key might be available now
        print("Attempting Google GenAI configuration again...")
        try:
             api_key_recheck = os.getenv("GOOGLE_API_KEY")
             if not api_key_recheck: raise ValueError("GOOGLE_API_KEY still not found.")
             genai.configure(api_key=api_key_recheck)
             print("Google Generative AI re-configured successfully.")
        except Exception as e: print(f"ERROR: Failed to configure Google GenAI on retry: {e}"); exit(1)

    # --- Load Models ONCE ---
    print("Loading models...")
    loaded_processor = None; loaded_age_gender_model = None; loaded_emotion_model_info = {}
    try: # Age/Gender
        age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
        loaded_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        loaded_age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        loaded_age_gender_model.eval(); print("Age/Gender model loaded.")
    except Exception as e: print(f"FATAL Error loading Age/Gender model: {e}"); exit(1)
    try: # Emotion
        emotion_model_name = "superb/hubert-large-superb-er"
        loaded_emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        loaded_emotion_model_info['model'].eval()
        loaded_emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e: print(f"Warning: Error loading Emotion model: {e}. Using 'ErrorLoadingModel'."); loaded_emotion_model_info = None

    print("-" * 30)
    print("Starting Bengali Audio Processing and Annotation Workflow...")
    print(f"Input Base Directory: {BENGALI_BASE_DIR}")
    print(f"Final Output File:    {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    # --- Define Paths for Bengali Data ---
    manifest_path = os.path.join(BENGALI_BASE_DIR, "utt_spk_text.tsv")
    audio_directory = os.path.join(BENGALI_BASE_DIR, "data")

    # --- Check Paths ---
    if not os.path.isdir(BENGALI_BASE_DIR): print(f"ERROR: Base directory not found: {BENGALI_BASE_DIR}"); exit(1)
    if not os.path.isfile(manifest_path): print(f"ERROR: Manifest TSV not found: {manifest_path}"); exit(1)
    if not os.path.isdir(audio_directory): print(f"ERROR: Audio directory not found: {audio_directory}"); exit(1)

    # --- Run Processing ---
    process_audio_and_annotate_bengali(
        manifest_tsv_path=manifest_path,
        audio_files_dir=audio_directory,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        age_gender_model=loaded_age_gender_model,
        processor=loaded_processor,
        emotion_model_info=loaded_emotion_model_info,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("\nWorkflow complete for Bengali dataset.")