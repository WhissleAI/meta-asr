import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
from whisper import load_model
from flair.data import Sentence
from pyannote.audio import Pipeline
from collections import defaultdict
from flair.models import SequenceTagger
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model

load_dotenv()

# Flags
ENABLE_SPEAKER_CHANGE = True
ENABLE_TRANSCRIPTION = True

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
ner_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.getenv("HF_TOKEN"))
whisper_model = load_model("base").to(device)

def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    model_name = "superb/hubert-large-superb-er"
    if not hasattr(extract_emotion, 'model'):
        extract_emotion.model = AutoModelForAudioClassification.from_pretrained(model_name)
        extract_emotion.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = extract_emotion.feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    outputs = extract_emotion.model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    return extract_emotion.model.config.id2label.get(predicted_class_idx, "Unknown")

def process_ner(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    sentence = Sentence(text)
    ner_tagger.predict(sentence)
    entities = [{
        "text": entity.text,
        "type": entity.tag,
        "start_position": entity.start_position,
        "end_position": entity.end_position
    } for entity in sentence.get_spans('ner')]
    return text, entities

def process_audio_file(audio_path: str, input_dir: str, output_dir: str = "output") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    signal, sr = librosa.load(audio_path, sr=16000)
    diarization = pipeline({'audio': audio_path})
    speaker_changes = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    segments = []
    current_speaker = None
    for start_time, end_time, speaker in speaker_changes:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        speaker_segment = signal[start_sample:end_sample]
        if len(speaker_segment) / sr < 1.5:
            continue
        age, gender = 0, 0
        emotion = extract_emotion(speaker_segment)
        transcription = ""
        if ENABLE_TRANSCRIPTION:
            temp_path = os.path.join(output_dir, "temp_segment.wav")
            sf.write(temp_path, speaker_segment, sr)
            transcription = whisper_model.transcribe(temp_path)["text"]
            os.remove(temp_path)
        ner_text, ner_entities = process_ner(transcription.lower())
        rel_path = os.path.relpath(audio_path, input_dir)
        metadata = f"EMOTION_{emotion.upper()}"
        if ENABLE_SPEAKER_CHANGE and current_speaker != speaker:
            metadata += " SPEAKER_CHANGE"
            current_speaker = speaker
        segments.append({
            "audio_filepath": f"audio_chunks/{rel_path}",
            "text": f"{ner_text} {metadata}" if ENABLE_TRANSCRIPTION else metadata
        })
    return {"segments": segments}

def process_directory(input_dir: str, output_dir: str = "output"):
    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp3', '.wav', '.flac')):
            audio_path = os.path.join(input_dir, filename)
            try:
                result = process_audio_file(audio_path, input_dir, output_dir)
                results.extend(result['segments'])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    output_path = os.path.join(output_dir, "processed_audio.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return output_path

if __name__ == "__main__":
    input_dir = os.path.abspath("../meta-asr/output/test")
    output_dir = os.path.abspath("output")
    output_path = process_directory(input_dir, output_dir)
    print(f"Results saved to: {output_path}")
