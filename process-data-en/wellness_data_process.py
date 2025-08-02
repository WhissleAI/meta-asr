import gc
import json
import torch
import librosa
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

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

# Attempt to import whisper
try:
    import whisper
    print("OpenAI Whisper library found.")
except ImportError:
    print("OpenAI Whisper library not found. Please install it with: pip install openai-whisper")
    whisper = None

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
        self.gender = ModelHead(config, 3) # Assuming 3 classes: female, male, other
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

def transcribe_audio_with_whisper(audio_path: str, whisper_model: Any) -> str | None:
    """
    Transcribes a single audio file using the loaded Whisper model.
    Returns the transcription text or None if it fails.
    """
    if whisper_model is None:
        print("Error: Whisper model not loaded.")
        return None
    print(f"  - Transcribing {os.path.basename(audio_path)} with Whisper...")
    try:
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"].strip()
        print(f"  - Transcription successful.")
        return transcription
    except Exception as e:
        print(f"  - ERROR: Whisper transcription failed for {os.path.basename(audio_path)}: {e}")
        return None

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

def annotate_batch_wellness_texts(batch_transcriptions: List[str]) -> List[str]:
    """
    Annotates a batch of wellness/fitness transcriptions using a specialized Gemini prompt.
    Returns a list of annotated strings with entity tags and intent classifications.
    """
    if not is_gemini_configured:
        print("Error: Gemini API not configured. Skipping annotation.")
        return batch_transcriptions  # Return original transcriptions

    # Add metadata tags to transcriptions (age, gender, emotion info will be added later)
    transcriptions_with_metadata = []
    for transcription in batch_transcriptions:
        # For now, we'll add placeholder metadata that will be replaced later in the processing
        transcriptions_with_metadata.append(f"{transcription} AGE_PLACEHOLDER GENDER_PLACEHOLDER EMOTION_PLACEHOLDER")

    prompt = f'''You are an expert AI assistant specializing in annotating English-language transcripts from the wellness and fitness domain.
Your task is to analyze transcribed sentences and tag specific entities and the speaker's intent according to a strict set of rules.

You will receive a JSON array of strings. Each string is a sentence that may already have metadata tags like AGE_*, GENDER_*, and EMOTION_* at the end.

Your Task and Rules:
1.  **INPUT/OUTPUT:** Your input is a JSON array of strings. Your output MUST be a JSON array containing the same number of strings.
2.  **PRESERVE EXISTING TAGS:** Keep the `AGE_`, `GENDER_`, and `EMOTION_` tags at the end of each sentence. Do not modify or move them.
3.  **ENTITY ANNOTATION:** Identify entities from the **Entity Types List** below. This includes both wellness-specific and general entities.
4.  **ENTITY TAG FORMAT (CRITICAL):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_ACTIVITY`).
    *   **NO SPACES** in the `<TYPE>` (e.g., use `BODY_PART`, not `BODY PART`).
    *   Immediately **AFTER** the entity text, add a single space and the word `END`.
5.  **SPEAKER INTENT TAG:** Based on the text, determine the single primary intent. Add **ONE** intent tag from the **Speaker Intent List** at the absolute end of the entire string, after all other tags.
6.  **OUTPUT FORMAT (CRITICAL):** Return ONLY a valid JSON array of strings. Do not include any other text or markdown formatting outside of the JSON array.

---
**Entity Types List (USE ONLY THESE):**

**Wellness & Fitness Entities:**
*   `ACTIVITY`: A specific exercise, workout type, or physical action (e.g., "running", "yoga", "deep squat", "meditation").
*   `BODY_PART`: Any part of the human body mentioned (e.g., "shoulders", "knees", "core", "hamstrings").
*   `DIET_FOOD`: A specific food, drink, supplement, or dietary style (e.g., "protein shake", "kale", "water", "vegan diet").
*   `HEALTH_METRIC`: A measurable health or performance indicator (e.g., "heart rate", "calories burned", "blood pressure").
*   `EQUIPMENT`: A piece of fitness equipment (e.g., "dumbbell", "yoga mat", "resistance band").
*   `DURATION`: A specific length of time for an activity (e.g., "30 seconds", "one minute", "for an hour").
*   `REPETITION`: A count of an exercise or action (e.g., "ten reps", "3 sets of 12").

**General Entities:**
*   `PERSON_NAME`: The name of a person (e.g., "Dr. Huberman", "Adriene").
*   `LOCATION`: A city, state, country, or specific place (e.g., "California", "Central Park").
*   `ORGANIZATION`: A company, institution, or brand (e.g., "Peloton", "World Health Organization").
*   `DATE_TIME`: A specific date or time of day (e.g., "tomorrow morning", "last week", "on Monday").

---
**Speaker Intent List (Choose ONE per sentence):**
*   `INTENT_INSTRUCTION`: Guiding the user through an action, exercise, or process.
*   `INTENT_MOTIVATION`: Encouraging the user or providing positive reinforcement.
*   `INTENT_INFORMATIONAL`: Providing factual information, tips, or explanations about health or fitness.
*   `INTENT_QUESTION`: Asking the user a direct question.
*   `INTENT_PERSONAL_EXPERIENCE`: Sharing a personal story, feeling, or feedback.
*   `INTENT_OTHER`: For any text that does not fit the categories above.

---
**Example:**

**Input JSON:**
["Next, we are going to do ten reps of a deep squat for 30 seconds. AGE_30_45 GENDER_FEMALE EMOTION_CALM"]

**CORRECT Output JSON:**
["Next, we are going to do ENTITY_REPETITION ten reps END of a ENTITY_ACTIVITY deep squat END for ENTITY_DURATION 30 seconds END. AGE_30_45 GENDER_FEMALE EMOTION_CALM INTENT_INSTRUCTION"]

---
Annotate the following sentences provided in the JSON array. Ensure your output is ONLY a JSON array of strings:
{json.dumps(transcriptions_with_metadata, ensure_ascii=False, indent=2)}
'''

    max_retries = 3
    retry_delay_seconds = 10
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
            # Clean up potential markdown code blocks from the response
            cleaned_response = response.text.strip().removeprefix("```json").removesuffix("```").strip()

            annotated_data = json.loads(cleaned_response)

            if isinstance(annotated_data, list) and len(annotated_data) == len(batch_transcriptions):
                # Validate that all items are strings
                for item in annotated_data:
                    if not isinstance(item, str):
                        raise ValueError("API returned non-string items in the array.")
                return annotated_data
            else:
                raise ValueError(f"API returned data of mismatched length. Expected {len(batch_transcriptions)}, got {len(annotated_data)}.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error decoding or validating Gemini response (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return batch_transcriptions  # Return original transcriptions if all attempts fail
            time.sleep(retry_delay_seconds * (attempt + 1))
        except Exception as e:
            print(f"An unexpected error occurred with Gemini API (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return batch_transcriptions  # Return original transcriptions if all attempts fail
            time.sleep(retry_delay_seconds * (attempt + 1))

    return batch_transcriptions  # Return original transcriptions if all attempts fail


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


def process_audio_and_annotate(audio_dir: str, output_json_path: str, batch_size: int = 10):
    """
    Main function to process audio files, transcribe, analyze, and save structured annotations
to a single JSON file.
    """
    # Ensure output directory exists and is writable
    output_dir = os.path.dirname(output_json_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✅ Output directory is writable: {output_dir}")
    except Exception as e:
        print(f"❌ ERROR: Cannot write to output directory {output_dir}: {e}")
        return

    # --- Load All Models ---
    print("Loading local models...")
    try:
        # Age/Gender model
        age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")

        # Emotion model
        emotion_model_name = "superb/hubert-large-superb-er"
        emotion_model_info = {
            'feature_extractor': AutoFeatureExtractor.from_pretrained(emotion_model_name),
            'model': AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        }
        emotion_model_info['model'].eval()
        print("Emotion model loaded.")

        # Whisper model for transcription
        if whisper:
            # Using the "base" model for a good balance of speed and accuracy.
            # Other options: "tiny", "small", "medium", "large"
            whisper_model = whisper.load_model("base")
            print("Whisper model loaded.")
        else:
            raise SystemExit("Whisper library is not installed. Cannot proceed.")

    except Exception as e:
        print(f"Fatal error loading a required model: {e}")
        return

    # --- Start Processing ---
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        print("❌ No audio files found. Exiting.")
        print(f"Checked directory: {audio_dir}")
        print("Supported formats: .wav, .mp3, .flac, .m4a")
        return

    print(f"📁 Found {len(audio_files)} audio files to process")
    print(f"📝 Output will be saved to: {output_json_path}")
    print(f"⚙️ Batch size: {batch_size}")

    total_files = len(audio_files)
    records_buffer = []
    all_processed_records = [] # List to hold all final records

    for i, audio_path in enumerate(audio_files):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # STEP 1: Transcribe with Whisper
            original_transcription = transcribe_audio_with_whisper(audio_path, whisper_model)
            if not original_transcription:
                print(f"  Skipping file due to empty transcription.")
                continue

            # STEP 2: Load audio for local models
            signal, sr = librosa.load(audio_path, sr=16000)
            duration = round(len(signal) / sr, 2)

            # STEP 3: Run local Age/Gender/Emotion models
            inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                _, logits_age, logits_gender = age_gender_model(input_values)

            age_val = logits_age.cpu().numpy().item()
            age_group = get_age_bucket(age_val)
            gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
            gender = {"0": "Female", "1": "Male", "2": "Other"}.get(str(gender_idx), "Unk")
            emotion = extract_emotion(signal, sr, emotion_model_info)

            # STEP 4: Buffer the pre-processed record
            record = {
                "audio_filepath": os.path.abspath(audio_path),
                "original_transcription": original_transcription,
                "duration": duration,
                "task_name": "WELLNESS_FITNESS", # Domain-specific task name
                "gender": gender,
                "age_group": age_group,
                "emotion": emotion,
            }
            records_buffer.append(record)

            # STEP 5: Annotate in batches when buffer is full or at the end
            if len(records_buffer) >= batch_size or (i + 1) == total_files:
                print(f"\n--- Annotating batch of {len(records_buffer)} records ---")
                
                transcriptions_to_annotate = [rec["original_transcription"] for rec in records_buffer]
                annotated_texts = annotate_batch_wellness_texts(transcriptions_to_annotate)

                # Combine buffer data with Gemini annotations and add to the main list
                for record_data, annotated_text in zip(records_buffer, annotated_texts):
                    # Extract intent from the annotated text
                    gemini_intent = extract_intent_from_annotated_text(annotated_text)
                    
                    # Clean and format the annotated text with actual metadata
                    final_annotated_text = clean_annotated_text_for_output(
                        annotated_text, 
                        record_data["age_group"], 
                        record_data["gender"], 
                        record_data["emotion"]
                    )

                    final_record = {
                        "audio_filepath": record_data["audio_filepath"],
                        "text": final_annotated_text,
                        "original_transcription": record_data["original_transcription"],
                        "duration": record_data["duration"],
                        "task_name": record_data["task_name"],
                        "gender": record_data["gender"],
                        "age_group": record_data["age_group"],
                        "emotion": record_data["emotion"],
                        "gemini_intent": gemini_intent.replace("INTENT_", "") if gemini_intent.startswith("INTENT_") else "OTHER"
                    }
                    all_processed_records.append(final_record)

                print(f"--- Batch processed. Total records collected: {len(all_processed_records)} ---")

                # Incremental save after each batch to prevent data loss
                try:
                    temp_output_path = output_json_path.replace('.json', '_temp.json')
                    with open(temp_output_path, 'w', encoding='utf-8') as f_temp:
                        json.dump(all_processed_records, f_temp, ensure_ascii=False, indent=2)
                    print(f"  Incremental save completed: {temp_output_path}")
                except Exception as e:
                    print(f"  Warning: Could not save incremental backup: {e}")

                # Clear buffer and release memory
                records_buffer = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"  !! FATAL ERROR processing file {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- Final Step: Write all collected records to a single JSON file ---
    print("\n" + "="*50)
    print(f"Processing Finished. Writing {len(all_processed_records)} records to JSON file...")
    
    if len(all_processed_records) == 0:
        print("No records to save. Check if audio files exist and processing completed successfully.")
        return
        
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # Write the final output
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_processed_records, f_out, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved {len(all_processed_records)} records to {output_json_path}")
        
        # Remove temporary file if it exists
        temp_output_path = output_json_path.replace('.json', '_temp.json')
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"Cleaned up temporary file: {temp_output_path}")
            
        # Print file size for verification
        file_size = os.path.getsize(output_json_path) / (1024 * 1024)  # MB
        print(f"Output file size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not write to output file {output_json_path}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save to a backup location
        try:
            backup_path = output_json_path.replace('.json', '_backup.json')
            with open(backup_path, 'w', encoding='utf-8') as f_backup:
                json.dump(all_processed_records, f_backup, ensure_ascii=False, indent=2)
            print(f"💾 Data saved to backup location: {backup_path}")
        except Exception as backup_error:
            print(f"Could not save backup either: {backup_error}")
    
    print("="*50)


# --- Configuration and Execution ---
# !!! IMPORTANT: Update these paths to your specific directories !!!
AUDIO_FILES_DIR = "/external4/datasets/bucket_data/wellness/overlap"
FINAL_OUTPUT_JSON = "/hydra2-prev/home/compute/workspace_himanshu/wellness_fitness_annotations.json"

# Batch size for processing and annotating. Lower if you have memory constraints.
PROCESSING_BATCH_SIZE = 5

if __name__ == "__main__":
    print("Starting Wellness & Fitness Audio Annotation Workflow...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(FINAL_OUTPUT_JSON), exist_ok=True)
    print(f"Output directory ensured: {os.path.dirname(FINAL_OUTPUT_JSON)}")
    
    # Check if audio directory exists
    if not os.path.exists(AUDIO_FILES_DIR):
        print(f"Warning: Audio directory '{AUDIO_FILES_DIR}' not found. Please update the path.")
        print("Creating the directory structure...")
        try:
            os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
            print(f"Created directory: {AUDIO_FILES_DIR}")
            print("Please add your audio files to this directory and run the script again.")
        except Exception as e:
            print(f"Could not create directory: {e}")
    
    # Check if Gemini is configured
    if not is_gemini_configured:
         print("\n❌ WARNING: Gemini API is not configured. The script will run but skip annotation.")
         print("   To enable annotation, please:")
         print("   1. Add your GOOGLE_API_KEY to the .env file, OR")
         print("   2. Set the GOOGLE_API_KEY environment variable")
         print("   Processing will continue with transcription and local model analysis only...")
    
    # Always run the processing function
    try:
        process_audio_and_annotate(
            audio_dir=AUDIO_FILES_DIR,
            output_json_path=FINAL_OUTPUT_JSON,
            batch_size=PROCESSING_BATCH_SIZE
        )
    except Exception as e:
        print(f"FATAL ERROR during processing: {e}")
        import traceback
        traceback.print_exc()

    print("Workflow complete.")