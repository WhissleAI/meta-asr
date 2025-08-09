import os
import gc
import json
import re
import torch
import librosa
import numpy as np
from pathlib import Path
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
import ollama
import time

# --- Ollama Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
ollama_client = None

# --- Initial Setup ---
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

try:
    print(f"Attempting to connect to Ollama host: {OLLAMA_HOST} with model: {OLLAMA_MODEL}")
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    ollama_client.list()
    print(f"Ollama client configured and server reached successfully. Using model: {OLLAMA_MODEL}")
except Exception as e:
    print(f"Error configuring or testing Ollama: {e}. Annotation step will likely fail.")
    ollama_client = None

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
        outputs = self.wav2vec2(input_values); hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0: return "No_Audio"
    if len(audio_data) < sampling_rate * 0.1: pass

    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        return model_info['model'].config.id2label.get(predicted_class_idx, "Unknown").upper()
    except Exception as e: print(f"Error emotion extraction: {e}"); return "Extraction_Error"

# --- Data Structures ---
@dataclass
class AudioSegment:
    start_time: float; end_time: float; speaker: str; age: float
    gender: str; transcription: str; emotion: str
    chunk_filename: str; duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    def get_text_for_llm(self) -> str: # MODIFIED: Only transcription
        if not self.segments: return ""
        segment = self.segments[0]
        return segment.transcription.strip().lower() # Just the plain transcription

    @staticmethod
    def get_age_bucket(age_float: float) -> str:
        if age_float < 0: return "UNKNOWN_AGE"
        actual_age = round(age_float * 100)
        age_brackets = [(18, "0_18"), (30, "18_30"), (45, "30_45"), (60, "45_60"), (float('inf'), "60PLUS")]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS"

# --- File Handling (Unchanged) ---
def get_audio_transcription_pairs(audio_dir: str, transcription_manifest_file: str) -> List[Tuple[str, str]]:
    transcriptions: Dict[str, str] = {};
    try:
        with open(transcription_manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip();
                if not line: continue
                parts = line.split(' ', 1)
                if len(parts) == 2: transcriptions[parts[0]] = parts[1]
                else: print(f"Warning: Skipping malformed line: {line}")
        print(f"Loaded {len(transcriptions)} transcriptions from {transcription_manifest_file}")
    except FileNotFoundError: print(f"Error: Manifest not found: {transcription_manifest_file}"); return []
    except Exception as e: print(f"Error reading manifest {transcription_manifest_file}: {e}"); return []
    if not transcriptions: return []
    pairs = []
    try:
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.flac', '.wav', '.mp3'))]
        for audio_filename in audio_files_list:
            audio_file_stem = os.path.splitext(audio_filename)[0]
            if audio_file_stem in transcriptions:
                pairs.append((os.path.join(audio_dir, audio_filename), transcriptions[audio_file_stem]))
        print(f"Found {len(pairs)} matching audio-transcription pairs.")
        if not pairs: print(f"Audio files: {len(audio_files_list)}, Transcriptions: {len(transcriptions)}")
        return pairs
    except FileNotFoundError: print(f"Error: Audio directory not found: {audio_dir}"); return []
    except Exception as e: print(f"Unexpected error in get_audio_transcription_pairs: {e}"); return []

# --- AI Annotation Helper Functions (IMPROVED) ---
def correct_entity_tag_spaces(text: str) -> str:
    if not isinstance(text, str): return text
    
    # Fix spaces inside entity type names (e.g., ENTITY_PERSON_NAM E → ENTITY_PERSON_NAME)
    def replace_spaces_in_entity(match):
        full_entity_tag = match.group(1)
        type_part = full_entity_tag[len("ENTITY_"):].replace(' ', '')
        return f"ENTITY_{type_part}"
    
    # Catch entity tags with spaces anywhere in the tag
    entity_space_pattern = r'(ENTITY_[A-Z0-9_ ]+?)(?=\s)'
    text = re.sub(entity_space_pattern, replace_spaces_in_entity, text)
    
    # Also catch entity tags at the end of the string
    text = re.sub(r'(ENTITY_[A-Z0-9_ ]+?)$', replace_spaces_in_entity, text)
    
    return text

def fix_end_tags_and_spacing(text: str) -> str: # Operates on LLM output (text_with_entities INTENT_TAG)
    if not isinstance(text, str): return text
    
    # First remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix entity tag formatting issues
    text = correct_entity_tag_spaces(text)
    
    # Remove redundant END tags (after sentence-ending punctuation)
    text = re.sub(r'([।?!:;,.])(\s+END)(?=\s+INTENT_|\s*$)', r'\1', text)
    
    # Remove redundant multiple END tags
    text = re.sub(r'END(\s+END)+', 'END', text)
    
    # Remove END tags that appear right before INTENT
    text = re.sub(r'\sEND\s+(INTENT_)', r' \1', text)
    
    # Ensure entity tags have proper spacing
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    
    # Ensure END tags have proper spacing
    text = re.sub(r'(\S)(END\b)', r'\1 END', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([।?!:;,.])', r'\1', text)
    text = re.sub(r'([।?!:;,.])(\w)', r'\1 \2', text)
    
    # Ensure each ENTITY tag has one corresponding END tag
    # Find all entity tags without END tags
    entities_without_end = re.finditer(r'(ENTITY_[A-Z0-9_]+\s+\S+(?:\s+\S+)*)(?!\s+END)(?=\s+INTENT_|\s+ENTITY_|\s*$)', text)
    
    # For each match, add END tag
    offset = 0
    for match in entities_without_end:
        start, end = match.span(1)
        # Adjust for previous insertions
        start += offset
        end += offset
        # Insert END tag
        text = text[:end] + ' END' + text[end:]
        # Update offset for next insertion
        offset += 4
    
    # Final cleanup of any doubled spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Ollama Annotation Function (MODIFIED PROMPT) ---
def annotate_batch_texts_ollama(plain_texts_to_annotate: List[str]) -> List[str]:
    if not ollama_client:
        print("Error: Ollama client not configured. Skipping.")
        return plain_texts_to_annotate
    if not plain_texts_to_annotate:
        return []

    # MODIFIED PROMPT: LLM only sees plain text.
    prompt_instructions = f'''You are an expert linguistic annotator for Hindi text.
You will receive a list of PLAIN HINDI SENTENCES.

Your task is crucial and requires precision:
1.  **ENTITY ANNOTATION:** Identify entities ONLY within the Hindi transcription. Use ONLY the entity types from the provided list.
2.  **ENTITY TAG FORMAT (VERY IMPORTANT):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES** are allowed within the `<TYPE>` part (e.g., use `PERSON_NAME`, NOT `PERSON_ NAM E`).
    *   Immediately **AFTER** the Hindi entity text, add a single space followed by `END`.
    *   Example: `... ENTITY_CITY दिल्ली END ...`
3.  **INTENT TAG:** Determine the single primary intent of the Hindi transcription (e.g., INFORM, QUESTION, REQUEST, COMMAND, GREETING, etc.). Add ONE `INTENT_<INTENT_TYPE>` tag at the absolute end of the entire string, AFTER all entity annotations.
4.  **OUTPUT FORMAT:** Respond ONLY with a valid JSON array of strings. Each string in the array must be the annotated sentence (original text + entity tags + intent tag). The array must have the same number of elements as the input list.
    Example output format for an input list of 2 sentences:
    ["annotated sentence 1 INTENT_TYPE1", "annotated sentence 2 INTENT_TYPE2"]
    DO NOT include any other text, explanations, or markdown formatting like ```json ... ``` around the JSON array. Just the raw JSON array.
5.  **HINDI SPECIFICS:** Handle Hindi script, punctuation (like ।), and spacing correctly according to standard Hindi rules. Ensure proper spacing around the inserted tags.

**ENTITY TYPES LIST (USE ONLY THESE):**
[
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

**EXAMPLES:**

**Example 1 - Basic Entity Recognition:**
Input: "मैंने कल मारिया को दिल्ली में देखा।"
Correct Output: "मैंने कल ENTITY_PERSON_NAME मारिया END को ENTITY_CITY दिल्ली END में देखा। INTENT_INFORM"

**Example 2 - Person & Location:**
Input: "राम ने कहा कि वह कल मुंबई जाएगा।"
Correct Output: "ENTITY_PERSON_NAME राम END ने कहा कि वह कल ENTITY_CITY मुंबई END जाएगा। INTENT_INFORM"

**Example 3 - Organization & City:**
Input: "टाटा समूह का मुख्यालय मुंबई में स्थित है।"
Correct Output: "ENTITY_ORGANIZATION टाटा समूह END का मुख्यालय ENTITY_CITY मुंबई END में स्थित है। INTENT_INFORM"

**Example 4 - Question Intent with Date:**
Input: "क्या आप मंगलवार को मिल सकते हैं?"
Correct Output: "क्या आप ENTITY_DATE मंगलवार END को मिल सकते हैं? INTENT_QUESTION"

**Example 5 - Multiple Entity Types Same Sentence:**
Input: "विवेकानंद ने शिकागो में अपना प्रसिद्ध भाषण दिया था।"
Correct Output: "ENTITY_PERSON_NAME विवेकानंद END ने ENTITY_CITY शिकागो END में अपना प्रसिद्ध भाषण दिया था। INTENT_INFORM"

**Example 6 - Nested Entities (Handle Correctly):**
Input: "भारत सरकार ने नई दिल्ली में एक बैठक बुलाई।"
Correct Output: "ENTITY_ORGANIZATION भारत सरकार END ने ENTITY_CITY नई दिल्ली END में एक बैठक बुलाई। INTENT_INFORM"

**Example 7 - Command Intent:**
Input: "यहां से मुंबई स्टेशन तक का रास्ता बताइए।"
Correct Output: "यहां से ENTITY_LOCATION मुंबई स्टेशन END तक का रास्ता बताइए। INTENT_COMMAND"

**Now, process the following list of sentences. Return only the JSON array of annotated strings:**
{json.dumps(plain_texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    max_retries = 3; retry_delay = 10; assistant_reply = ""

    for attempt in range(max_retries):
        try:
            print(f"  Sending batch to Ollama model {OLLAMA_MODEL} (Attempt {attempt+1}/{max_retries})...")
            response = ollama_client.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt_instructions}])
            assistant_reply = response['message']['content'].strip()

            if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
            elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
            if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()

            if not (assistant_reply.startswith('[') and assistant_reply.endswith(']')):
                 match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
                 if match: print("  Warning: Extracted JSON list from verbose output."); assistant_reply = match.group(0)
                 else: raise json.JSONDecodeError("Response not a JSON list.", assistant_reply, 0)

            annotated_sentences_raw = json.loads(assistant_reply)

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(plain_texts_to_annotate):
                processed_sentences = []
                for sentence_from_llm in annotated_sentences_raw: # This is "text_with_entities INTENT_TAG"
                     if isinstance(sentence_from_llm, str):
                          # Basic cleaning and fixing for LLM output
                          final_llm_output_sentence = fix_end_tags_and_spacing(sentence_from_llm)
                          processed_sentences.append(final_llm_output_sentence)
                     else:
                          print(f"Warning: Non-string item in annotation list: {sentence_from_llm}")
                          processed_sentences.append(str(sentence_from_llm))
                print("  Ollama annotation successful for batch.")
                return processed_sentences
            else:
                print(f"  Error: Ollama API mismatched length. Expected {len(plain_texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else 'Invalid Type'}")
                if attempt == max_retries - 1: return plain_texts_to_annotate
                time.sleep(retry_delay * (attempt + 1))
        except json.JSONDecodeError as e:
            print(f"  JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}\n  Response: {assistant_reply[:500]}")
            if attempt == max_retries - 1: return plain_texts_to_annotate
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"  Error calling/processing Ollama (Attempt {attempt+1}/{max_retries}): {e}")
            if "connection refused" in str(e).lower(): return plain_texts_to_annotate
            if attempt == max_retries - 1: return plain_texts_to_annotate
            time.sleep(retry_delay * (attempt + 1))
    print("Error: Max retries for Ollama annotation."); return plain_texts_to_annotate

# --- Main Processing Function (MODIFIED for new LLM input/output) ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    output_dir = os.path.dirname(output_jsonl_path); os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_jsonl_path, 'w') as f_clear: f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared.")
    except IOError as e: print(f"Error clearing output file {output_jsonl_path}: {e}."); return

    print("Loading models...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    loaded_age_gender_model: Optional[AgeGenderModel] = None
    loaded_processor: Optional[Wav2Vec2Processor] = None
    try:
        loaded_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        loaded_age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        loaded_age_gender_model.eval(); print("Age/Gender model loaded.")
    except Exception as e: print(f"FATAL Error loading Age/Gender model: {e}. Exiting."); return

    emotion_model_name = "superb/hubert-large-superb-er"; emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e: print(f"Warning: Error loading Emotion model: {e}.")

    audio_dir = os.path.join(base_dir, "audio")
    transcription_manifest_file = os.path.join(base_dir, "transcription.txt")
    if not (os.path.exists(audio_dir) and os.path.exists(transcription_manifest_file)):
        print(f"Error: Missing audio dir or manifest."); return

    file_pairs = get_audio_transcription_pairs(audio_dir, transcription_manifest_file)
    if not file_pairs: print("No matching audio-transcription pairs. Exiting."); return
    total_files = len(file_pairs); print(f"Found {total_files} pairs. Output: {output_jsonl_path}")

    processed_records_buffer = []; batch_num = 0; files_processed_count = 0; total_records_saved = 0

    for i, (audio_path, transcription_text) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            if not transcription_text: print(f"  Skipping: Empty transcription for {audio_path}"); continue
            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                if signal is None or len(signal) == 0: print(f"  Skipping: Empty audio {audio_path}"); continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1: print(f"  Skipping: Audio too short {audio_path}"); continue
            except Exception as load_err: print(f"  Skipping: Error loading audio {audio_path}: {load_err}"); continue

            age_raw = -1.0; gender_raw = "ERROR"
            if loaded_age_gender_model and loaded_processor:
                try:
                    inputs = loaded_processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                    input_values = inputs.input_values.to(device)
                    with torch.no_grad(): _, logits_age, logits_gender = loaded_age_gender_model(input_values)
                    age_raw = logits_age.cpu().numpy().item()
                    gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                    gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                    gender_raw = gender_map.get(gender_idx, "UNKNOWN_GENDER")
                except Exception as age_gender_err: print(f"  Error Age/Gender: {age_gender_err}")
            emotion_raw = extract_emotion(signal, sr, emotion_model_info)
            speaker_id = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            
            segment_data = AudioSegment(0, duration, speaker_id, age_raw, gender_raw, transcription_text, emotion_raw, os.path.basename(audio_path), duration)
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            
            # MODIFIED: text_for_llm is now just the plain transcription
            plain_text_for_llm = chunk.get_text_for_llm() 

            record_for_batch = {
                "audio_filepath": chunk.filepath,
                "duration": duration,
                "plain_text_for_llm": plain_text_for_llm, # This goes to LLM
                "raw_age_output": age_raw,
                "raw_gender_prediction": gender_raw,
                "raw_emotion_prediction": emotion_raw,
            }
            processed_records_buffer.append(record_for_batch); files_processed_count += 1

            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1; current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                # Input to Ollama is now a list of plain texts
                plain_texts_for_ollama = [rec["plain_text_for_llm"] for rec in processed_records_buffer]
                
                # Ollama output: list of "text_with_entities INTENT_TAG" strings
                annotated_texts_from_llm = annotate_batch_texts_ollama(plain_texts_for_ollama)

                if len(annotated_texts_from_llm) != current_batch_size:
                    print(f"  FATAL BATCH ERROR: Ollama annotation count mismatch! Skipping save.")
                else:
                    lines_written_in_batch = 0
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                        for record_data, llm_output_str in zip(processed_records_buffer, annotated_texts_from_llm):
                            
                            task_name = "UNKNOWN"; final_text_field = llm_output_str # Default
                            intent_match = re.search(r'\sINTENT_([A-Z0-9_]+)\s*$', llm_output_str)
                            if intent_match:
                                task_name = intent_match.group(1).upper()
                                final_text_field = llm_output_str[:intent_match.start()].strip() # Text before INTENT tag
                            else:
                                print(f"  Warning: INTENT tag not found or malformed in: ...{llm_output_str[-70:]}")
                                # If intent not found, final_text_field remains the full llm_output_str
                                # Consider if you want to strip if it looks like a failed intent: e.g. ends with " INTENT_"
                                if llm_output_str.endswith(" INTENT_"):
                                    final_text_field = llm_output_str[:-len(" INTENT_")].strip()


                            formatted_gender = "Unknown"
                            if isinstance(record_data["raw_gender_prediction"], str):
                                if record_data["raw_gender_prediction"] in ["MALE", "FEMALE", "OTHER"]:
                                     formatted_gender = record_data["raw_gender_prediction"].title()
                            
                            formatted_age_group = "Unknown"
                            if record_data["raw_age_output"] >= 0:
                                age_bucket_internal = ChunkData.get_age_bucket(record_data["raw_age_output"])
                                if age_bucket_internal != "UNKNOWN_AGE":
                                    formatted_age_group = age_bucket_internal.replace('_', '-')
                            
                            formatted_emotion = "Unknown"
                            raw_emo = record_data["raw_emotion_prediction"]
                            if isinstance(raw_emo, str) and \
                               not any(err_str in raw_emo for err_str in ["Error", "No_Audio", "Unknown"]):
                                formatted_emotion = raw_emo.title()

                            final_record = {
                                "audio_filepath": record_data["audio_filepath"],
                                "text": final_text_field, # This is now "transcription_with_ENTITY_tags"
                                "duration": record_data["duration"],
                                "task_name": task_name,
                                "gender": formatted_gender,
                                "age_group": formatted_age_group,
                                "emotion": formatted_emotion
                            }
                            json_str = json.dumps(final_record, ensure_ascii=False)
                            f_out.write(json_str + '\n'); lines_written_in_batch += 1
                    total_records_saved += lines_written_in_batch
                    print(f"--- Batch {batch_num} saved ({lines_written_in_batch} records). Total: {total_records_saved} ---")
                
                processed_records_buffer = []; torch.cuda.empty_cache(); gc.collect()
        except KeyboardInterrupt: print("\nProcessing interrupted."); break
        except Exception as e: print(f"  FATAL ERROR processing {audio_path}: {e}"); import traceback; traceback.print_exc(); continue
    print(f"\nProcessing Finished. Total processed: {files_processed_count}, Total saved: {total_records_saved}")

# --- Main Execution ---
if __name__ == "__main__":
    BASE_DATA_DIR = "/external3/databases/hindi_openslr_103"
    FINAL_OUTPUT_JSONL = os.path.join(BASE_DATA_DIR, "processed_output", "hn_annotated_ollama_simplified_llm_v3.jsonl") # New name
    PROCESSING_BATCH_SIZE = 5

    print("Starting Audio Processing (Ollama - Simplified LLM Input)...")
    print(f"Input Base: {BASE_DATA_DIR}, Output: {FINAL_OUTPUT_JSONL}, Model: {OLLAMA_MODEL}")

    if not ollama_client: print("ERROR: Ollama client failed. Exiting."); exit(1)
    os.makedirs(os.path.dirname(FINAL_OUTPUT_JSONL), exist_ok=True)
    process_audio_and_annotate(base_dir=BASE_DATA_DIR, output_jsonl_path=FINAL_OUTPUT_JSONL, batch_size=PROCESSING_BATCH_SIZE)
    print("Workflow complete.")