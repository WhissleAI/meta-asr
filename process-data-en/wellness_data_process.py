import gc
import json
import torch
import librosa
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

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
# Note: This part loads the Google API key from an environment variable.
# Make sure to set `export GOOGLE_API_KEY='your_api_key'` in your terminal.
is_gemini_configured = False
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
    # Test configuration by listing models
    next(genai.list_models())
    is_gemini_configured = True
    print("Google Generative AI configured successfully.")
except (ValueError, Exception) as e:
    print(f"Warning: Google Generative AI could not be configured: {e}")
    print("The annotation step will be skipped. Please set the GOOGLE_API_KEY environment variable.")

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

def annotate_batch_wellness_texts(batch_transcriptions: List[str]) -> List[Dict[str, Any]]:
    """
    Annotates a batch of wellness/fitness transcriptions using a specialized Gemini prompt.
    Returns a list of structured dictionaries as specified in the prompt.
    """
    if not is_gemini_configured:
        print("Error: Gemini API not configured. Skipping annotation.")
        return [{} for _ in batch_transcriptions] # Return empty dicts for each item

    prompt = f'''
You are an expert AI specializing in analyzing transcripts from wellness and fitness audio.
Your task is to process a list of transcripts and for each one, return a structured JSON object containing an intent classification and detailed BIO (Beginning, Inside, Outside) entity tagging.

**Follow these instructions carefully for EACH transcript in the input list:**

1.  **Analyze the Intent:** Determine the single primary intent of the speaker from the list below.
2.  **Perform Entity Recognition:** Identify and tag specific entities within the transcript using the BIO scheme.
3.  **Tokenize Correctly:** The 'tokens' list in your output must be the words from the original transcript.
4.  **Format Output as JSON:** Your final output must be a single JSON array `[...]`, where each element is a JSON object `{...}` corresponding to one of the input transcripts.

---
**Intent Categories (Choose ONE per transcript):**
*   `INSTRUCTION`: Guiding the user through an action, exercise, or process (e.g., "Now, lift your legs slowly").
*   `MOTIVATION`: Encouraging the user or providing positive reinforcement (e.g., "You're doing great, keep it up!").
*   `INFORM`: Providing factual information, tips, or explanations about health or fitness (e.g., "This exercise targets the core muscles").
*   `QUESTION`: Asking the user a direct question (e.g., "How does that feel?").
*   `FEEDBACK`: Commenting on form, progress, or sensation (e.g., "I can feel a deep stretch in my hamstrings").
*   `OTHER`: For any text that does not fit the categories above.

**Entity Types for BIO Tagging:**
*   `ACTIVITY`: A specific exercise, workout type, or physical action (e.g., "push-up", "running", "yoga", "meditation").
*   `BODY_PART`: Any part of the human body mentioned (e.g., "shoulders", "knees", "core", "back").
*   `DURATION`: A length of time (e.g., "30 seconds", "one minute", "an hour").
*   `REPETITION`: A count of an exercise or action (e.g., "ten reps", "3 sets").
*   `EQUIPMENT`: A piece of fitness equipment (e.g., "dumbbell", "mat", "resistance band").
*   `HEALTH_METRIC`: A measurable health or performance indicator (e.g., "heart rate", "calories").
*   `DIET_FOOD`: A specific food, drink, or dietary style (e.g., "protein", "water", "vegan").

---
**CRITICAL OUTPUT FORMAT:**
For each input transcript, create a JSON object with these exact keys:
{{
  "gemini_intent": "...",
  "bio_annotation_gemini": {{
    "tokens": ["word1", "word2", ...],
    "tags": ["B-TAG", "I-TAG", "O", ...]
  }}
}}

**Example:**
*Input Transcript:* "For the next 30 seconds, we will do a deep squat."
*Correct JSON Object Output for this transcript:*
{{
  "gemini_intent": "INSTRUCTION",
  "bio_annotation_gemini": {{
    "tokens": ["For", "the", "next", "30", "seconds", ",", "we", "will", "do", "a", "deep", "squat", "."],
    "tags":   ["O", "O", "O", "B-DURATION", "I-DURATION", "O", "O", "O", "O", "O", "B-ACTIVITY", "I-ACTIVITY", "O"]
  }}
}}

---
**Transcripts to Process (return a JSON array with one object for each):**
{json.dumps(batch_transcriptions, ensure_ascii=False, indent=2)}
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
                # Basic validation of the returned structure
                for item in annotated_data:
                    if not ("gemini_intent" in item and "bio_annotation_gemini" in item):
                        raise ValueError("Returned JSON object is missing required keys.")
                return annotated_data
            else:
                raise ValueError(f"API returned data of mismatched length. Expected {len(batch_transcriptions)}, got {len(annotated_data)}.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error decoding or validating Gemini response (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return [{} for _ in batch_transcriptions]
            time.sleep(retry_delay_seconds * (attempt + 1))
        except Exception as e:
            print(f"An unexpected error occurred with Gemini API (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return [{} for _ in batch_transcriptions]
            time.sleep(retry_delay_seconds * (attempt + 1))

    return [{} for _ in batch_transcriptions]


def process_audio_and_annotate(audio_dir: str, output_jsonl_path: str, batch_size: int = 10):
    """
    Main function to process audio files, transcribe, analyze, and save structured annotations.
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    # Clear the output file at the start
    with open(output_jsonl_path, 'w') as f_clear:
        pass

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
        print("No audio files found. Exiting.")
        return

    total_files = len(audio_files)
    records_buffer = []
    total_records_saved = 0

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
                gemini_annotations = annotate_batch_wellness_texts(transcriptions_to_annotate)

                # Combine buffer data with Gemini annotations and save
                lines_written = 0
                with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                    for record_data, gemini_data in zip(records_buffer, gemini_annotations):
                        # Final record combines local model results and Gemini results
                        final_record = {
                            "audio_filepath": record_data["audio_filepath"],
                            "text": record_data["original_transcription"], # 'text' and 'original_transcription' are the same here
                            "original_transcription": record_data["original_transcription"],
                            "duration": record_data["duration"],
                            "task_name": record_data["task_name"],
                            "gender": record_data["gender"],
                            "age_group": record_data["age_group"],
                            "emotion": record_data["emotion"],
                            "gemini_intent": gemini_data.get("gemini_intent", "FAILED_ANNOTATION"),
                            "bio_annotation_gemini": gemini_data.get("bio_annotation_gemini", {"tokens": [], "tags": []})
                        }
                        f_out.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                        lines_written += 1

                total_records_saved += lines_written
                print(f"--- Batch saved. {lines_written} records written to {output_jsonl_path}. Total saved: {total_records_saved} ---")

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

    print("\n" + "="*30)
    print("Processing Finished.")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)


# --- Configuration and Execution ---
# !!! IMPORTANT: Update these paths to your specific directories !!!
# Example for Google Drive mounted in Colab:
AUDIO_FILES_DIR = "/external4/datasets/bucket_data/wellness/overlap"
FINAL_OUTPUT_JSONL = "/hydra2-prev/home/compute/workspace_himanshu/wellness_fitness_annotations.jsonl"

# Batch size for processing and annotating. Lower if you have memory constraints.
PROCESSING_BATCH_SIZE = 5

if __name__ == "__main__":
    print("Starting Wellness & Fitness Audio Annotation Workflow...")
    # Create a dummy directory and file for demonstration if it doesn't exist
    if not os.path.exists(AUDIO_FILES_DIR):
        print(f"Warning: Demo directory '{AUDIO_FILES_DIR}' not found. Please update the path.")
        # You might want to create a dummy file for testing purposes
        # os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
        # print("A dummy directory has been created. Please add your audio files to it.")
    
    # Check if Gemini is configured before starting
    if not is_gemini_configured:
         print("\nERROR: Gemini API is not configured. The script cannot proceed with annotation.")
         print("Please ensure your GOOGLE_API_KEY is set up correctly in your environment.")
    else:
        process_audio_and_annotate(
            audio_dir=AUDIO_FILES_DIR,
            output_jsonl_path=FINAL_OUTPUT_JSONL,
            batch_size=PROCESSING_BATCH_SIZE
        )

    print("Workflow complete.")