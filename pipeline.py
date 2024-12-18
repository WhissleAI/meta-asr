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
from convert_mp4_mp3 import MP4AudioChunkConverter
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
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

def process_large_audio(audio_path: str, chunk_duration: float = 20.0):
    # Convert MP4 to WAV if needed
    processed_audio_path = convert_mp4_to_wav(audio_path)
    
    # Load audio signal
    signal, sr = librosa.load(processed_audio_path, sr=16000)
    total_duration = len(signal) / sr
 
    # Create output directory
    output_dir = "chunks"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    chunk_size = int(chunk_duration * sr)
    
    # Generate base filename from the input audio path
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    for chunk_idx in range(0, len(signal), chunk_size):
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Extract chunk
        end = min(chunk_idx + chunk_size, len(signal))
        chunk = signal[chunk_idx:end]
        
        # Skip very short chunks
        if len(chunk) / sr < 1.0:
            continue
        
        # Create chunk filename using base filename and chunk index
        chunk_filename = f"{base_filename}_chunk_{chunk_idx//chunk_size}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
  
        # Save chunk if it doesn't exist
        if not os.path.exists(chunk_path):
            sf.write(chunk_path, chunk, sr)
        
        try:
            # Rest of the processing remains the same as in the original function
            diarization = pipeline({'audio': chunk_path})
            speaker_changes = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_changes.append((turn.start, turn.end, speaker))
            
            for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
    
                speaker_segment = chunk[start_sample:end_sample]
                
                if len(speaker_segment) / sr < 1.5:
                    continue

                # Process the speaker segment
                y = processor(speaker_segment, sampling_rate=sr)
                y = y['input_values'][0]
                y = y.reshape(1, -1)
                y = torch.from_numpy(y).to(device)
  
                with torch.no_grad():
                    model_output = model(y)
                    age = float(model_output[1].detach().cpu().numpy()[0][0])
                    gender = np.argmax(model_output[2].detach().cpu().numpy())
          
                # Temporarily save the speaker segment to transcribe
                temp_segment_path = os.path.join(output_dir, f"temp_segment_{chunk_idx//chunk_size}_{speaker_idx}.wav")
                sf.write(temp_segment_path, speaker_segment, sr)
                
                transcription = whisper_model.transcribe(temp_segment_path)["text"]
                speaker_segment_audio, _ = librosa.load(temp_segment_path, sr=16000)
    
                emotion = extract_emotion(speaker_segment_audio)
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
                
                # Remove temporary segment file
                os.remove(temp_segment_path)
        
        except Exception as e:
            print(f"Error processing chunk {chunk_filename}: {e}")
        
        del chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    output_csv = f"{base_filename}_processed_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"Processed audio data saved to {output_csv}")
    return df

if __name__ == "__main__":
    download_dir = "downloads"
    
    # List of supported audio file extensions
    # audio_extensions = ['.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg']
    
    for filename in os.listdir(download_dir):
        audio_path = os.path.join(download_dir, filename)
        
        if os.path.isfile(audio_path):
            try:
                print(f"Processing audio file: {filename}")
                process_large_audio(audio_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
