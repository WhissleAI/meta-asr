import os
import torch
import librosa
import numpy as np
import pandas as pd
import moviepy as mp
import torch.nn as nn
import soundfile as sf
from dotenv import load_dotenv
from whisper import load_model
from pyannote.audio import Pipeline
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
from pathlib import Path
from convert_mp4_mp3 import MP4AudioChunkConverter
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from datetime import datetime
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

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.getenv("HF_TOKEN"))

whisper_model = load_model("base").to(device)


def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0].detach().cpu().numpy()
        else:
            age_logits = y[1].detach().cpu().numpy()
            gender_logits = y[2].detach().cpu().numpy()
            y = [age_logits, gender_logits]

    return y


def get_speaker_changes(audio_path: str):
    diarization = pipeline({'audio': audio_path})
    speaker_changes = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_changes.append((turn.start, turn.end, speaker))
    return speaker_changes


def transcribe_audio_segment(audio_path: str):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def split_audio_by_speaker(audio_path: str, speaker_changes: list, output_dir: str = "spk_dir", max_duration: float = 20.0):
    os.makedirs(output_dir, exist_ok=True)

    signal, sr = librosa.load(audio_path, sr=16000)
    audio_segments = []

    for idx, (start, end, speaker) in enumerate(speaker_changes):
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        while start_sample < end_sample:
            segment_end_sample = min(start_sample + int(max_duration * sr), end_sample)
            segment = signal[start_sample:segment_end_sample]

            segment_filename = os.path.join(output_dir, f"speaker_{speaker}_segment_{idx}.wav")
            sf.write(segment_filename, segment, sr)
            audio_segments.append(segment_filename)

            start_sample = segment_end_sample

    return audio_segments
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    model_name = "superb/hubert-large-superb-er"
        
    if not hasattr(extract_emotion, 'model'):
        extract_emotion.model = AutoModelForAudioClassification.from_pretrained(model_name)
        extract_emotion.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
    if audio_data is None or len(audio_data) == 0:
        return "No Audio"
        
    if len(audio_data) < sampling_rate:
        return "Audio Too Short"

    inputs = extract_emotion.feature_extractor(
            audio_data, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
    )

    outputs = extract_emotion.model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
        
    return extract_emotion.model.config.id2label.get(predicted_class_idx, "Unknown")


def convert_mp4_to_wav(mp4_path: str) -> str:

    file_ext = os.path.splitext(mp4_path)[1].lower()
    
    if file_ext == '.mp4':
     
        output_dir = "./converted_audio"
        os.makedirs(output_dir, exist_ok=True)
        
     
        wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mp4_path))[0] + ".wav")
        
        video = mp.VideoFileClip(mp4_path)
        video.audio.write_audiofile(wav_path)
        video.close()
        
        return wav_path
    
    return mp4_path
@dataclass
class AudioSegment:
    """Data class to store processed audio segment information."""
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: int
    transcription: str
    emotion: str
    chunk_filename: str

@dataclass
class ChunkData:
    """Data class to store chunk-level aggregated data."""
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""
    
    def get_formatted_text(self) -> str:
        """Combine all segment transcriptions with their metadata."""
        texts = []
        current_speaker = None
        
        for segment in self.segments:
            age_bucket = self.get_age_bucket(segment.age)
            gender_text = "MALE" if segment.gender == 1 else "FEMALE"
            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{segment.emotion.upper()}"
            
            if current_speaker != segment.speaker:
                metadata += " SPEAKER_CHANGE"
                current_speaker = segment.speaker
                
            texts.append(f"{segment.transcription.lower()} {metadata}")
        
        return " ".join(texts)

    @staticmethod
    def get_age_bucket(age: float) -> str:
        """Map age to predefined buckets."""
        if age < 18: return "0_18"
        elif age < 30: return "18_30"
        elif age < 45: return "30_45"
        elif age < 60: return "45_60"
        else: return "60plus"

def create_output_directories(base_path: str) -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunks_dir = os.path.join(base_path, "audio_chunks", timestamp)
    results_dir = os.path.join(base_path, "results", timestamp)
    
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return chunks_dir, results_dir

def process_large_audio(
    audio_path: str, 
    chunk_duration: float = 20.0,
    output_base_dir: str = "output"
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    chunks_dir, results_dir = create_output_directories(output_base_dir)
    
    processed_audio_path = convert_mp4_to_wav(audio_path)
    signal, sr = librosa.load(processed_audio_path, sr=16000)

    all_data = []
    chunk_data: Dict[str, ChunkData] = defaultdict(ChunkData)
    chunk_size = int(chunk_duration * sr)

    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    for chunk_idx in range(0, len(signal), chunk_size):
        # Memory management for GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        end = min(chunk_idx + chunk_size, len(signal))
        chunk = signal[chunk_idx:end]

        if len(chunk) / sr < 1.0:
            continue
            
        chunk_filename = f"{base_filename}_chunk_{chunk_idx//chunk_size}.wav"
        chunk_path = os.path.join(chunks_dir, chunk_filename)

        if not os.path.exists(chunk_path):
            sf.write(chunk_path, chunk, sr)
        
        try:
            diarization = pipeline({'audio': chunk_path})
            speaker_changes = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            
            for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
    
                speaker_segment = chunk[start_sample:end_sample]
                
                if len(speaker_segment) / sr < 1.5:
                    continue
                    
                # Process audio segment
                y = processor(speaker_segment, sampling_rate=sr)
                y = y['input_values'][0]
                y = y.reshape(1, -1)
                y = torch.from_numpy(y).to(device)
  
                with torch.no_grad():
                    model_output = model(y)
                    age = float(model_output[1].detach().cpu().numpy()[0][0])
                    gender = np.argmax(model_output[2].detach().cpu().numpy())
    
                # Create temporary segment file
                temp_segment_path = os.path.join(
                    chunks_dir, 
                    f"temp_segment_{chunk_idx//chunk_size}_{speaker_idx}.wav"
                )
                sf.write(temp_segment_path, speaker_segment, sr)
                
                # Extract features
                transcription = whisper_model.transcribe(temp_segment_path)["text"]
                speaker_segment_audio, _ = librosa.load(temp_segment_path, sr=16000)
                emotion = extract_emotion(speaker_segment_audio)
                
                # Create segment object
                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker,
                    age=float(age),
                    gender=int(gender),
                    transcription=transcription,
                    emotion=emotion,
                    chunk_filename=chunk_filename
                )
                
                # Add to all_data for CSV
                all_data.append({
                    'Start Time': start_time,
                    'End Time': end_time,
                    'Speaker': speaker,
                    'Age': float(age),
                    'Gender': int(gender),
                    'Transcription': transcription,
                    'emotion': emotion,
                    'Audio File Path': chunk_filename
                })
                
                # Add to chunk_data for JSON
                relative_path = os.path.join("audio_chunks", base_filename, chunk_filename)
                chunk_data[chunk_filename].segments.append(segment)
                chunk_data[chunk_filename].filepath = relative_path
        
                # Clean up temporary files
                os.remove(temp_segment_path)
        
        except Exception as e:
            print(f"Error processing chunk {chunk_filename}: {e}")
            continue
        
        finally:
            # Clean up memory
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Create DataFrame for CSV
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(results_dir, f"{base_filename}_processed_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Processed audio data saved to {csv_path}")
    
    # Create chunk-level text data
    chunk_texts = [
        {
            "audio_filepath": data.filepath,
            "text": data.get_formatted_text()
        }
        for data in chunk_data.values()
    ]
    
    # Save JSON output with unique filename
    json_path = os.path.join(results_dir, f"{base_filename}_audio_text_pairs.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            chunk_texts,
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"Created JSON file with {len(chunk_texts)} entries: {json_path}")
    
    return df, chunk_texts


if __name__ == "__main__":
    download_dir = "downloads"
    
   
    audio_extensions = ['.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg']
    
    for filename in os.listdir(download_dir):
        audio_path = os.path.join(download_dir, filename)
        
        if os.path.isfile(audio_path):
            try:
                print(f"Processing audio file: {filename}")
                df, chunk_texts = process_large_audio(audio_path) 
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
