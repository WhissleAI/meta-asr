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


def split_audio_by_speaker(audio_path: str, speaker_changes: list, output_dir: str = "speaker_segments"):
    os.makedirs(output_dir, exist_ok=True)
    
    signal, sr = librosa.load(audio_path, sr=16000)
    audio_segments = []

    for idx, (start, end, speaker) in enumerate(speaker_changes):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = signal[start_sample:end_sample]

        segment_filename = os.path.join(output_dir, f"speaker_{speaker}_segment_{idx}.wav")
        sf.write(segment_filename, segment, sr)
        audio_segments.append(segment_filename)

    return audio_segments


def process_audio(audio_path: str):
    signal, sr = librosa.load(audio_path, sr=16000)
    sampling_rate = 16000

    output = process_func(signal, sampling_rate)

    speaker_changes = get_speaker_changes(audio_path)
    audio_segments = split_audio_by_speaker(audio_path, speaker_changes, output_dir="speaker_segments1")

    data = []
    for idx, (change, segment_path) in enumerate(zip(speaker_changes, audio_segments)):
        start_time, end_time, speaker = change
        
        segment_transcription = transcribe_audio_segment(segment_path)
        
        age = output[0][0]
        gender = np.argmax(output[1])

        data.append({
            'Start Time': start_time,
            'End Time': end_time,
            'Speaker': speaker,
            'Age': age,
            'Gender': gender,
            'Transcription': segment_transcription,
            'Audio File Path': segment_path
        })

    df = pd.DataFrame(data)
    df.to_csv("output1.csv", index=False)


audio_path = "test_audio/test2.mp3"
process_audio(audio_path)
