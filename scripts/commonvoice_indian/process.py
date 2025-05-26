import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import shutil
from transformers import pipeline
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor
from functools import partial

load_dotenv()

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

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"

class AudioPredictor:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_age_and_gender(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                hidden_states, logits_age, logits_gender = self.model(**inputs)
            age = logits_age.squeeze().item() * 100
            gender_probs = logits_gender.squeeze()
            gender_idx = torch.argmax(gender_probs).item()
            gender_map = {0: "MALE", 1: "FEMALE", 2: "OTHER"}
            return round(age), gender_map.get(gender_idx, "OTHER")
        except Exception as e:
            print(f"Error processing audio: {audio_path}\nError: {str(e)}")
            return 25, "OTHER"

def get_age_bucket(age_text: str) -> str:
    age_to_number = {
        'twenties': 25,
        'thirties': 35,
        'fourties': 45,
        'fifties': 55,
        'sixties': 65,
        'seventies': 75,
        'eighties': 85,
        'nineties': 95
    }
    age = age_to_number.get(age_text.lower(), 20)
    age_brackets = [
        (18, "0_18"),
        (30, "18_30"),
        (45, "30_45"),
        (60, "45_60"),
        (float('inf'), "60PLUS")
    ]
    for threshold, bracket in age_brackets:
        if age < threshold:
            return bracket
    return "60PLUS"

def map_gender(gender: str) -> str:
    gender_mapping = {
        'male': 'MALE',
        'female': 'FEMALE',
        'other': 'OTHER'
    }
    return gender_mapping.get(gender.lower(), 'OTHER')

def predict_emotion(text: str, emotion_classifier) -> str:
    try:
        result = emotion_classifier(text)[0]
        emotion = result['label'].upper()
        return f"EMOTION_{emotion}"
    except Exception as e:
        print(f"Error predicting emotion for text: {text}\nError: {str(e)}")
        return "EMOTION_NEU"

def get_audio_duration(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    extension = Path(file_path).suffix.lower()
    try:
        if extension == '.opus':
            cmd = [
                'ffprobe',
                '-i', file_path,
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            try:
                duration = float(output)
                return round(duration, 2)
            except ValueError:
                print(f"Could not parse duration from ffprobe output: {output}")
                return None
        elif extension in ['.mp3', '.wav', '.ogg', '.flac']:
            if extension == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif extension == '.wav':
                audio = AudioSegment.from_wav(file_path)
            elif extension == '.ogg':
                audio = AudioSegment.from_ogg(file_path)
            elif extension == '.flac':
                audio = AudioSegment.from_file(file_path, format="flac")
        elif extension in ['.m4a', '.mp4', '.webm']:
            audio = AudioSegment.from_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        duration = len(audio) / 1000.0
        return round(duration, 2)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        if extension == '.opus':
            try:
                print(f"Trying alternative approach for {file_path}...")
                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                    cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-y',
                        temp_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    audio = AudioSegment.from_wav(temp_path)
                    duration = len(audio) / 1000.0
                    return round(duration, 2)
            except Exception as e2:
                print(f"Alternative approach also failed for {file_path}: {str(e2)}")
        try:
            print(f"Trying librosa to get duration for {file_path}...")
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return round(duration, 2)
        except Exception as e3:
            print(f"Librosa approach also failed for {file_path}: {str(e3)}")
            return None

def convert_single_file(file_name, original_clips_dir, new_clips_dir):
    """
    Converts a single audio file to WAV format.
    If the file is already in WAV format, it is copied directly.
    If the corresponding WAV file already exists in new_clips_dir, the conversion is skipped.
    """
    original_file_path = os.path.join(original_clips_dir, file_name)
    if not os.path.isfile(original_file_path):
        return  # Skip if the original file does not exist

    ext = Path(file_name).suffix.lower()
    new_file_name = Path(file_name).stem + ".wav"
    new_file_path = os.path.join(new_clips_dir, new_file_name)

    # Skip conversion if the target WAV file already exists.
    if os.path.exists(new_file_path):
        print(f"Skipping {file_name} as WAV file already exists.")
        return

    try:
        if ext == ".wav":
            shutil.copy2(original_file_path, new_file_path)
        else:
            audio = AudioSegment.from_file(original_file_path)
            audio.export(new_file_path, format="wav")
    except Exception as e:
        print(f"Error converting {original_file_path}: {e}")

def convert_audio_files_to_wav(language_code: str, original_clips_dir: str, output_base_dir: str) -> str:
    """
    Converts all audio files in original_clips_dir to WAV format using parallel processing and saves them under:
    /external4/datasets/cv_scripts_data/<language_code>/clips/
    Returns the path to the new clips directory.
    """
    new_clips_dir = os.path.join(output_base_dir, language_code, "clips")
    os.makedirs(new_clips_dir, exist_ok=True)
    
    file_list = os.listdir(original_clips_dir)
    process_file = partial(convert_single_file, original_clips_dir=original_clips_dir, new_clips_dir=new_clips_dir)
    
    # Set the ProcessPoolExecutor to use 4 CPU cores.
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(process_file, file_list),
            total=len(file_list),
            desc=f"Converting audio files for {language_code}"
        ))
    
    return new_clips_dir

def process_data(tsv_path: str, output_path: str, clips_dir: str, emotion_model_name: str,
                 use_converted: bool = False, converted_clips_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading emotion classifier...")
    emotion_classifier = pipeline(
        "text-classification",
        model=emotion_model_name,
        device=2 if torch.cuda.is_available() else -1
    )
    print("Loading age-gender model...")
    model = AgeGenderModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Use a more tolerant CSV reader to skip bad lines.
    print(f"Reading TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t', engine="python", on_bad_lines="skip")
    print(f"Found {len(df)} entries in TSV file")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        processed_files = 0
        skipped_files = 0
        audio_predictor = AudioPredictor()
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx}/{len(df)}")
            try:
                # If using converted files, replace the original extension with .wav
                if use_converted and converted_clips_dir is not None:
                    audio_filename = Path(row['path']).stem + ".wav"
                    audio_path = os.path.join(converted_clips_dir, audio_filename)
                else:
                    audio_path = os.path.join(clips_dir, row['path'])
                if not os.path.exists(audio_path):
                    print(f"Skipping missing file: {audio_path}")
                    skipped_files += 1
                    continue
                duration = get_audio_duration(audio_path)
                if duration is None:
                    print(f"Could not determine duration for {audio_path}, skipping...")
                    skipped_files += 1
                    continue
                if pd.notna(row.get('age')) and pd.notna(row.get('gender')):
                    age_bracket = get_age_bucket(row['age'])
                    gender = map_gender(row['gender'])
                else:
                    try:
                        predicted_age, predicted_gender = audio_predictor.predict_age_and_gender(audio_path)
                        age_bracket = get_age_bucket(str(round(predicted_age)))
                        gender = predicted_gender
                    except Exception as e:
                        print(f"Error predicting age/gender for {audio_path}: {str(e)}")
                        age_bracket = "30_45"
                        gender = "OTHER"
                sentence_text = row['sentence']
                emotion = predict_emotion(sentence_text, emotion_classifier)
                text = f"{sentence_text} AGE_{age_bracket} GENDER_{gender} {emotion}"
                entry = {
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": duration
                }
                out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                processed_files += 1
                if processed_files % 1000 == 0:
                    print(f"Processed {processed_files} files...")
                    print(f"Current stats - Processed: {processed_files}, Skipped: {skipped_files}")
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped_files += 1
        print(f"\nProcessing complete!")
        print(f"Total entries processed: {processed_files}")
        print(f"Total entries skipped: {skipped_files}")

if __name__ == "__main__":
    # List of language codes to process.
    language_codes = ["bn"]  #  "ur", "ml", "as", "hi", "mr", "ne-NP", "or", "pa-IN", "ta"

    # Use the same emotion classifier model for all languages.
    emotion_model = "j-hartmann/emotion-english-distilroberta-base"

    base_dataset_dir = "/external4/datasets/cv/cv-corpus-15.0-2023-09-08"
    output_base_dir = "/external4/datasets/cv_scripts_data"

    for lang in language_codes:
        print(f"\n=== Processing language: {lang} ===")
        base_dir = f"{base_dataset_dir}/{lang}"
        original_clips_dir = f"{base_dir}/clips"
        
        # Convert all audio files to WAV and store them in a language-specific folder using parallel processing (4 cores)
        print(f"Converting audio files for language: {lang}")
        converted_clips_dir = convert_audio_files_to_wav(lang, original_clips_dir, output_base_dir)
        
        # Process both "validated.tsv" and "test.tsv"
        tsv_files = ["validated.tsv", "test.tsv"]
        for tsv_file in tsv_files:
            tsv_path = f"{base_dir}/{tsv_file}"
            # Build an output file name based on language and TSV file name, e.g., ml_validated_manifest.jsonl.
            output_file_name = f"{lang}_{Path(tsv_file).stem}_manifest.jsonl"
            output_path = os.path.join(output_base_dir, output_file_name)
            print(f"\nProcessing TSV file: {tsv_path}")
            process_data(tsv_path, output_path, clips_dir=original_clips_dir,
                         emotion_model_name=emotion_model,
                         use_converted=True,
                         converted_clips_dir=converted_clips_dir)
