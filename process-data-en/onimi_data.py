import gc
import json
import torch
import librosa
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pyarrow.parquet as pq
from glob import glob
from huggingface_hub import snapshot_download
import logging
import io


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file.")
except ImportError:
    print("python-dotenv not found. Please install it with: pip install python-dotenv")
    print("Falling back to system environment variables.")

# --- Transformer and AI Model Imports ---
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch.nn as nn
import google.generativeai as genai
import time

# Removed Whisper dependency (dataset already contains transcriptions)
print("Skipping Whisper import – dataset already provides text.")

# --- Initial System and GPU Setup ---
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

# --- Google Generative AI Configuration ---
# Load Google API key from .env file or environment variable
is_gemini_configured = False
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
    # Test configuration by listing models
    next(genai.list_models())
    is_gemini_configured = True
    print("✅ Google Generative AI configured successfully.")
    print(f"   API Key loaded: {GOOGLE_API_KEY[:10]}...{GOOGLE_API_KEY[-4:]}")  # Show partial key for verification
    
except (ValueError, Exception) as e:
    print(f"❌ Warning: Google Generative AI could not be configured: {e}")
    print("   The annotation step will be skipped.")
    print("   Please add your GOOGLE_API_KEY to the .env file or set as environment variable.")

# --- Model Definitions for Local Inference (Age, Gender, Emotion) ---
class ModelHead(nn.Module):
    """A standard classification head for the Wav2Vec2 model."""
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
    """A multi-head Wav2Vec2 model for predicting age and gender."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)  # female, male, other
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Audio Processing and AI Helper Functions ---

def get_audio_files(audio_dir: str) -> List[str]:
    """Finds all audio files (e.g., .wav, .mp3) in the specified directory."""
    supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
    try:
        audio_files_list = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if os.path.splitext(f)[1].lower() in supported_formats
        ]
        print(f"Found {len(audio_files_list)} audio files in {audio_dir}")
        return audio_files_list
    except FileNotFoundError:
        print(f"Error: Directory not found: {audio_dir}.")
        return []
    except Exception as e:
        print(f"An error occurred in get_audio_files: {e}")
        return []

## Whisper transcription removed

def extract_emotion(audio_data: np.ndarray, sampling_rate: int, model_info: dict) -> str:
    """Extracts emotion from audio data using a pre-loaded model."""
    if not all(k in model_info for k in ['model', 'feature_extractor']):
        return "Model_Not_Loaded"
    try:
        inputs = model_info['feature_extractor'](
            audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model_info['model'](**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()
        return model_info['model'].config.id2label.get(predicted_class_idx, "Unk")
    except Exception as e:
        print(f"Error during emotion extraction: {e}")
        return "Extraction_Error"

def get_age_bucket(age: float) -> str:
    """Converts a continuous age value (0-1) to a string bucket."""
    actual_age = round(age * 100)
    if actual_age < 18: return "0-18"
    if actual_age < 30: return "18-30"
    if actual_age < 45: return "30-45"
    if actual_age < 60: return "45-60"
    return "60+"

def annotate_batch_qa(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate a batch of QA pairs returning entity-tagged question/answer plus intents only.

    We produce:
      question_annotated: question with ENTITY_<TYPE> <span> END tags (NO demographic or intent tokens appended)
      answer_annotated: answer with ENTITY_<TYPE> <span> END tags (NO intent token appended)
      question_intent: single INTENT_* tag
      answer_intent: single INTENT_* tag

    Demographic (AGE_/GENDER_/EMOTION_) tokens are appended later by our code using locally inferred values.
    """
    if not is_gemini_configured:
        # Passthrough fallback
        return [
            {
                **r,
                "question_annotated": r["question_text"],
                "answer_annotated": r["answer_text"],
                "question_intent": "INTENT_OTHER",
                "answer_intent": "INTENT_OTHER",
            } for r in records
        ]

    payload = []
    for r in records:
        payload.append({
            "question": r["question_text"],
            "answer": r["answer_text"],
        })

    prompt = f"""You are an expert conversational data annotator.
You will receive a JSON array named INPUT of objects with fields: question, answer.
For EACH object:
1. Insert entity annotations into the question and answer using this wrapping syntax (exactly): ENTITY_<TYPE> <original span text> END
   - Do not alter internal punctuation of the span. Punctuation immediately following the entity span should remain after END (e.g., ENTITY_PRODUCT Omni END,).
2. Allowed entity types (UPPERCASE, no spaces): PRODUCT, ACTIVITY, BODY_PART, DIET_FOOD, HEALTH_METRIC, EQUIPMENT, DURATION, REPETITION, PERSON_NAME, LOCATION, ORGANIZATION, DATE_TIME, TECHNOLOGY, CAPABILITY, TASK, FEATURE, FUNCTIONALITY, SERVICE.
3. Choose exactly ONE intent for the question and ONE for the answer from:
   INTENT_INTRODUCTION, INTENT_QUESTION, INTENT_INFORMATIONAL, INTENT_INSTRUCTION, INTENT_PERSONAL_EXPERIENCE, INTENT_MOTIVATION, INTENT_OTHER.
4. Do NOT add demographic tokens (AGE_*, GENDER_*, EMOTION_*) anywhere. We'll append those later.
5. Do NOT append the intent tokens inside the annotated question/answer strings; provide them separately.
6. Return a JSON array (no markdown fences) with objects each having keys exactly:
   question_annotated, answer_annotated, question_intent, answer_intent.
7. Maintain original sentence order and wording except for inserting the entity tags.

INPUT JSON Array:\n{json.dumps(payload, ensure_ascii=False)}\nReturn ONLY the JSON array."""

    max_retries = 3
    retry_delay_seconds = 10
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            cleaned = response.text.strip().removeprefix("```json").removeprefix("```").strip().removesuffix("```")
            data = json.loads(cleaned)
            if not isinstance(data, list) or len(data) != len(records):
                raise ValueError("Length mismatch or wrong format")
            output_records = []
            for base, ann in zip(records, data):
                output_records.append({
                    **base,
                    "question_annotated": ann.get("question_annotated", base["question_text"]),
                    "answer_annotated": ann.get("answer_annotated", base["answer_text"]),
                    "question_intent": ann.get("question_intent", "INTENT_OTHER"),
                    "answer_intent": ann.get("answer_intent", "INTENT_OTHER"),
                })
            return output_records
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gemini parse/format error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(retry_delay_seconds * (attempt + 1))
        except Exception as e:
            print(f"Gemini unexpected error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(retry_delay_seconds * (attempt + 1))

    # Fallback passthrough
    return [
        {
            **r,
            "question_annotated": r["question_text"],
            "answer_annotated": r["answer_text"],
            "question_intent": "INTENT_OTHER",
            "answer_intent": "INTENT_OTHER",
        } for r in records
    ]


def extract_intent_from_annotated_text(annotated_text: str) -> str:
    """
    Extracts the intent from an annotated text string.
    Looks for INTENT_ tags at the end of the string.
    """
    # Split by spaces and look for intent tags
    parts = annotated_text.split()
    for part in reversed(parts):  # Start from the end
        if part.startswith("INTENT_"):
            return part
    return "INTENT_OTHER"  # Default if no intent found


def clean_annotated_text_for_output(annotated_text: str, age_group: str, gender: str, emotion: str) -> str:
    """
    Replaces placeholder metadata with actual values and ensures proper formatting.
    """
    # Replace placeholders with actual values
    text = annotated_text.replace("AGE_PLACEHOLDER", f"AGE_{age_group.replace('-', '_')}")
    text = text.replace("GENDER_PLACEHOLDER", f"GENDER_{gender.upper()}")
    text = text.replace("EMOTION_PLACEHOLDER", f"EMOTION_{emotion.upper()}")
    
    return text


def load_audio_duration(audio_path: str) -> float:
    """Robust audio duration helper. Returns -1.0 on failure.
    Attempts librosa first then soundfile."""
    if not audio_path or not os.path.exists(audio_path):
        return -1.0
    try:
        signal, sr = librosa.load(audio_path, sr=None, mono=True)
        return round(len(signal) / sr, 2)
    except Exception as e_lib:
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return round(len(data) / sr, 2)
        except Exception as e_sf:
            logger.warning(f"Duration load failed for {audio_path}: {e_lib} | {e_sf}")
            return -1.0


def resolve_local_model(model_name: str, local_dir_env: str | None = None) -> str:
    """Ensure model is available locally (offline friendly).
    If HF hub cache already has it, transformers will reuse.
    If a LOCAL_<MODEL>_DIR environment variable is provided, prefer it.
    Otherwise snapshot_download into standard cache and return path."""
    if local_dir_env and os.path.isdir(local_dir_env):
        logger.info(f"Using provided local directory for {model_name}: {local_dir_env}")
        return local_dir_env
    try:
        # Allow forcing offline mode via env HF_OFFLINE=1
        local_files_only = os.environ.get("HF_OFFLINE", "0") == "1"
        local_path = snapshot_download(repo_id=model_name, revision=None, local_files_only=local_files_only)
        logger.info(f"Snapshot downloaded/available for {model_name}: {local_path}")
        return local_path
    except Exception as e:
        logger.warning(f"snapshot_download failed for {model_name}: {e}. Will rely on transformers internal caching.")
        return model_name  # transformers will attempt normal resolution


def safe_age_gender_predict(signal: np.ndarray, sr: int, processor, model) -> Tuple[str, str]:
    """Predict age bucket and gender with truncation + exception safety."""
    try:
        if signal is None or len(signal) == 0:
            return "UNK", "UNK"
        # Truncate very long audio to 30s for inference speed/memory
        max_seconds = 30
        if len(signal) > sr * max_seconds:
            signal = signal[: sr * max_seconds]
        inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            _, logits_age, logits_gender = model(input_values)
        age_val = logits_age.cpu().numpy().item()
        age_group = get_age_bucket(age_val)
        gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
        gender_map = {0: "Female", 1: "Male", 2: "Other"}
        gender = gender_map.get(gender_idx, "Unk")
        return age_group, gender
    except Exception as e:
        logger.warning(f"Age/Gender inference failed: {e}")
        return "UNK", "UNK"


def safe_emotion_predict(signal: np.ndarray, sr: int, emotion_model_info: dict) -> str:
    try:
        return extract_emotion(signal, sr, emotion_model_info)
    except Exception as e:
        logger.warning(f"Emotion inference failed: {e}")
        return "UNK"


def process_parquet_dataset(
    dataset_dir: str,
    output_jsonl_path: str,
    audio_subdir: str = None,
    batch_size: int = 25,
    max_rows: int | None = None
):
    """Iterate over VoiceAssistant-400K parquet shards and annotate QA.
    Input parquet expected columns: 'question', 'question_audio', 'answer'.
    We compute age/gender/emotion locally from the question audio (if available).
    Final JSONL schema per line:
        {
          "question_audio_path": <absolute path or null>,
          "question_text": <entity annotated question + two spaces + AGE_* GENDER_* EMOTION_* INTENT_*>,
          "answer_text": <entity annotated answer + space + INTENT_*>,
          "audio_duration_s": <float seconds or -1.0>
        }
    No separate columns for demographics or intents; embedded inside the text fields as requested.
    """
    # Ensure output directory
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    out_f = open(output_jsonl_path, 'w', encoding='utf-8')
    print(f"Writing JSONL to {output_jsonl_path}")

    # Load local models once (age/gender/emotion)
    print("Loading local audio attribute models (age/gender/emotion)...")
    age_gender_model_name = os.environ.get("AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-6-ft-age-gender")
    emotion_model_name = os.environ.get("EMOTION_MODEL", "superb/hubert-large-superb-er")

    # Resolve local snapshot paths (optional offline support)
    resolved_age_gender = resolve_local_model(age_gender_model_name, os.environ.get("AGE_GENDER_MODEL_DIR"))
    resolved_emotion = resolve_local_model(emotion_model_name, os.environ.get("EMOTION_MODEL_DIR"))

    try:
        processor = Wav2Vec2Processor.from_pretrained(resolved_age_gender)
        age_gender_model = AgeGenderModel.from_pretrained(resolved_age_gender).to(device)
        age_gender_model.eval()
        logger.info("Age/Gender model ready.")
    except Exception as e:
        logger.error(f"Failed to load Age/Gender model ({age_gender_model_name}): {e}")
        processor = None
        age_gender_model = None

    emotion_model_info = {}
    try:
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(resolved_emotion)
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(resolved_emotion).to(device)
        emotion_model_info['model'].eval()
        logger.info("Emotion model ready.")
    except Exception as e:
        logger.error(f"Failed to load Emotion model ({emotion_model_name}): {e}")
        emotion_model_info = {}

    parquet_paths = sorted(glob(os.path.join(dataset_dir, 'data', 'train-*.parquet')))
    print(f"Found {len(parquet_paths)} parquet shards.")

    buffer: List[Dict[str, Any]] = []
    processed = 0

    # Derive a probable audio root under dataset if not explicitly provided
    inferred_audio_root = audio_subdir or os.path.join(dataset_dir, 'data', 'audio')
    if not os.path.isdir(inferred_audio_root):
        # Some dumps store wavs directly under dataset_dir or dataset_dir/data
        alt1 = os.path.join(dataset_dir, 'data')
        if os.path.isdir(alt1):
            inferred_audio_root = alt1
        else:
            inferred_audio_root = dataset_dir

    for shard_idx, parquet_path in enumerate(parquet_paths):
        print(f"Reading shard {shard_idx+1}/{len(parquet_paths)}: {os.path.basename(parquet_path)}")
        table = pq.read_table(parquet_path)
        pandas_df = table.to_pandas()
        for _, row in pandas_df.iterrows():
            if max_rows is not None and processed >= max_rows:
                break
            question = str(row.get('question', '')).strip()
            answer = str(row.get('answer', '')).strip()
            qa_audio_struct = row.get('question_audio')
            audio_path = None
            signal = None
            sr = 16000
            duration = -1.0
            age_group = 'UNK'
            gender = 'UNK'
            emotion = 'UNK'
            if isinstance(qa_audio_struct, dict):
                # huggingface datasets typical struct {"path": ..., "bytes": ...}
                audio_path = qa_audio_struct.get('path')
                # Normalize and attempt resolution strategy:
                # 1) If relative and provided audio_subdir exists, join.
                # 2) Else try inferred_audio_root.
                # 3) Else if only a bare filename, join dataset_dir.
                if audio_path and not os.path.isabs(audio_path):
                    candidate_paths = []
                    if audio_subdir:
                        candidate_paths.append(os.path.join(audio_subdir, audio_path))
                    candidate_paths.append(os.path.join(inferred_audio_root, audio_path))
                    candidate_paths.append(os.path.join(dataset_dir, audio_path))
                    # Deduplicate while preserving order
                    seen = set(); ordered = []
                    for cp in candidate_paths:
                        if cp not in seen:
                            ordered.append(cp); seen.add(cp)
                    resolved = None
                    for cp in ordered:
                        if os.path.exists(cp):
                            resolved = cp
                            break
                    if resolved:
                        audio_path = resolved
                # If the referenced path doesn't exist but we have bytes, decode directly
                audio_bytes = qa_audio_struct.get('bytes') if isinstance(qa_audio_struct, dict) else None
                if audio_path and os.path.exists(audio_path):
                    duration = load_audio_duration(audio_path)
                    try:
                        signal, sr = librosa.load(audio_path, sr=16000, mono=True)
                    except Exception as e_load:
                        logger.warning(f"Primary audio load failed ({e_load}); trying soundfile")
                        try:
                            import soundfile as sf
                            signal, sr = sf.read(audio_path)
                            if sr != 16000:
                                signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
                                sr = 16000
                            if signal.ndim > 1:
                                signal = np.mean(signal, axis=1)
                        except Exception as e_sf:
                            logger.error(f"All audio load methods failed for {audio_path}: {e_sf}")
                            signal = None
                            sr = 16000
                elif audio_bytes:
                    # Decode bytes using soundfile->librosa fallback
                    try:
                        import soundfile as sf
                        with io.BytesIO(audio_bytes) as bio:
                            data, sr_read = sf.read(bio)
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        # Resample if needed
                        if sr_read != 16000:
                            data = librosa.resample(data, orig_sr=sr_read, target_sr=16000)
                            sr = 16000
                        else:
                            sr = sr_read
                        signal = data.astype(np.float32)
                        duration = round(len(signal) / sr, 2)
                        # Optionally persist decoded bytes for reproducibility if we have a reference name
                        if audio_path and not os.path.isabs(audio_path):
                            # Save under inferred_audio_root
                            os.makedirs(inferred_audio_root, exist_ok=True)
                            target_file = os.path.join(inferred_audio_root, os.path.basename(audio_path))
                            try:
                                import soundfile as sf
                                sf.write(target_file, signal, sr)
                                audio_path = target_file
                            except Exception as e_save:
                                logger.warning(f"Failed to save decoded audio bytes to {target_file}: {e_save}")
                    except Exception as e_bytes:
                        logger.error(f"Failed to decode inline audio bytes: {e_bytes}")
                        signal = None
                else:
                    # No path and no bytes
                    duration = -1.0

                # Run inferences if we obtained a signal
                if signal is not None and processor is not None and age_gender_model is not None:
                    age_group, gender = safe_age_gender_predict(signal, sr, processor, age_gender_model)
                if signal is not None and emotion_model_info:
                    emotion = safe_emotion_predict(signal, sr, emotion_model_info)

            record = {
                "question_audio_path": os.path.abspath(audio_path) if audio_path else None,
                "question_text": question,
                "answer_text": answer,
                "audio_duration_s": duration,
                "age_group": age_group,
                "gender": gender,
                "emotion": emotion,
            }
            buffer.append(record)
            processed += 1

            if len(buffer) >= batch_size:
                annotated = annotate_batch_qa(buffer)
                for rec in annotated:
                    # Compose final annotated texts
                    q_full = f"{rec['question_annotated']}  AGE_{rec['age_group']} GENDER_{rec['gender'].upper()} EMOTION_{rec['emotion'].upper()} {rec['question_intent']}".strip()
                    a_full = f"{rec['answer_annotated']} {rec['answer_intent']}".strip()
                    final_obj = {
                        "question_audio_path": rec['question_audio_path'],
                        "question_text": q_full,
                        "answer_text": a_full,
                        "audio_duration_s": rec['audio_duration_s']
                    }
                    out_f.write(json.dumps(final_obj, ensure_ascii=False) + '\n')
                out_f.flush()
                print(f"Wrote {processed} records so far.")
                buffer = []

        if max_rows is not None and processed >= max_rows:
            break

    # Flush remainder
    if buffer:
        annotated = annotate_batch_qa(buffer)
        for rec in annotated:
            q_full = f"{rec['question_annotated']}  AGE_{rec['age_group']} GENDER_{rec['gender'].upper()} EMOTION_{rec['emotion'].upper()} {rec['question_intent']}".strip()
            a_full = f"{rec['answer_annotated']} {rec['answer_intent']}".strip()
            final_obj = {
                "question_audio_path": rec['question_audio_path'],
                "question_text": q_full,
                "answer_text": a_full,
                "audio_duration_s": rec['audio_duration_s']
            }
            out_f.write(json.dumps(final_obj, ensure_ascii=False) + '\n')
        out_f.flush()
        print(f"Wrote final batch. Total {processed} records.")

    out_f.close()
    print(f"✅ Completed. Total records written: {processed}")


# --- Configuration and Execution ---
# !!! IMPORTANT: Update these paths to your specific directories !!!
VOICE_ASSISTANT_DIR = "/external3/databases/hf-omini-data/VoiceAssistant-400K"
OUTPUT_JSONL = "/external3/databases/hf-omini-data/voiceassistant_annotated_2.jsonl"
PROCESSING_BATCH_SIZE = 25
MAX_ROWS_DEBUG = None  # Set to None for full dataset

if __name__ == "__main__":
    print("Starting VoiceAssistant QA Annotation Workflow...")
    if not is_gemini_configured:
        print("⚠️ Gemini not configured: annotations will be passthrough.")
    try:
        process_parquet_dataset(
            dataset_dir=VOICE_ASSISTANT_DIR,
            output_jsonl_path=OUTPUT_JSONL,
            audio_subdir=os.path.join(VOICE_ASSISTANT_DIR, 'data', 'audio'),
            batch_size=PROCESSING_BATCH_SIZE,
            max_rows=MAX_ROWS_DEBUG
        )
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback; traceback.print_exc()
    print("Workflow complete.")