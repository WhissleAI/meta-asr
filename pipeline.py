import torch
import torch.nn as nn
import numpy as np
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from pyannote.audio import Pipeline
import soundfile as sf
from whisper import load_model
import os

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

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

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


def split_audio_by_speaker(audio_path: str, speaker_changes: list, output_dir: str = "speaker_segments", max_duration: float = 20.0):
    os.makedirs(output_dir, exist_ok=True)

    signal, sr = librosa.load(audio_path, sr=16000)
    audio_segments = []

    for idx, (start, end, speaker) in enumerate(speaker_changes):
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Check if segment duration exceeds max_duration
        while start_sample < end_sample:
            segment_end_sample = min(start_sample + int(max_duration * sr), end_sample)
            segment = signal[start_sample:segment_end_sample]

            segment_filename = os.path.join(output_dir, f"speaker_{speaker}_segment_{idx}.wav")
            sf.write(segment_filename, segment, sr)
            audio_segments.append(segment_filename)

            start_sample = segment_end_sample

    return audio_segments


def process_large_audio(audio_path: str, chunk_duration: float = 20.0):
    """
    Process large audio files in smaller chunks to manage GPU memory
    
    Args:
    - audio_path: Path to the audio file
    - chunk_duration: Duration of each chunk in seconds (default 20 seconds)
    
    Returns:
    - Pandas DataFrame with processed audio segments
    """

    signal, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(signal) / sr
    
    output_dir = "./output/youtube_segment"
    speaker_segments_dir = "./output/speaker_segments"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(speaker_segments_dir, exist_ok=True)

    all_data = []
    
    # Process audio in chunks
    chunk_size = int(chunk_duration * sr)
    for start in range(0, len(signal), chunk_size):
    
        torch.cuda.empty_cache()
        
        end = min(start + chunk_size, len(signal))
        chunk = signal[start:end]
 
        if len(chunk) / sr < 1.0:
            continue
        
        chunk_filename = f"chunk_{start//chunk_size}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)

        if not os.path.exists(chunk_path):
            sf.write(chunk_path, chunk, sr)
        
        try:
         
            diarization = pipeline({'audio': chunk_path})
            speaker_changes = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_changes.append((turn.start, turn.end, speaker))
            
         
            for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                # Calculate sample indices relative to the chunk
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract speaker segment
                speaker_segment = chunk[start_sample:end_sample]
                
                # Skip very short segments
                if len(speaker_segment) / sr < 1.0:
                    continue
                
           
                segment_filename = f"speaker_{speaker}_{speaker_idx:02d}_segment.wav"
                segment_path = os.path.join(speaker_segments_dir, segment_filename)
                
                # Save speaker segment
                sf.write(segment_path, speaker_segment, sr)
                
                # Prepare input for model
                y = processor(speaker_segment, sampling_rate=sr)
                y = y['input_values'][0]
                y = y.reshape(1, -1)
                y = torch.from_numpy(y).to(device)
                
                # Get model predictions
                with torch.no_grad():
                    model_output = model(y)
                    age = model_output[1].detach().cpu().numpy()[0]
                    gender = np.argmax(model_output[2].detach().cpu().numpy())
                
                # Transcribe segment
                transcription = whisper_model.transcribe(segment_path)["text"]
                
                # Store results
                all_data.append({
                    'Start Time': start_time,
                    'End Time': end_time,
                    'Speaker': speaker,
                    'Age': float(age),
                    'Gender': int(gender),
                    'Transcription': transcription,
                    'Audio File Path': segment_path
                })
        
        except Exception as e:
            print(f"Error processing chunk {chunk_filename}: {e}")
        
        # Optional: free up memory
        del chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save results
    output_csv = "output_large_audio.csv"
    df.to_csv(output_csv, index=False)
    print(f"Processed audio saved to {output_csv}")
    
    return df

# Usage
audio_path = "temp_audio.wav"
process_large_audio(audio_path)
