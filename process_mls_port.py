import os
import json
import torch
import spacy
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from transformers import (
    Wav2Vec2ForSequenceClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    AutoModelForAudioClassification
)
import torch.nn as nn

load_dotenv()

# Load spaCy model - you can change this based on your target language
nlp = spacy.load('fr_core_news_lg')

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

class MultilangAudioModel:
    def __init__(self, model_path):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
    def forward(self, input_values):
        # Remove attention_mask from forward pass
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            return transcription

def process_audio(audio_path, model):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        input_values = model.processor(waveform, sampling_rate=sample_rate, 
                                     return_tensors="pt").input_values
        # Remove attention_mask from processing
        transcription = model.forward(input_values)
        return transcription
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using XLSR-53 based model for better multilingual support
model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = MultilangAudioModel.from_pretrained(model_name).to(device)

# Initialize emotion recognition model
emotion_model_name = "facebook/wav2vec2-large-xlsr-53"
emotion_processor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_name).to(device)

def load_transcriptions(trans_file: str) -> Dict[str, str]:
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                file_id = parts[0].replace('/', '_')
                transcriptions[file_id] = parts[1]
    return transcriptions

def process_ner(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    doc = nlp(text)
    
    entities = []
    formatted_text = text
    
    for ent in doc.ents:
        entity_text = ent.text
        entity_type = ent.label_
        replacement = f"NER_{entity_type} {entity_text} END"
        formatted_text = formatted_text.replace(entity_text, replacement, 1)
        
        entities.append({
            "text": entity_text,
            "type": entity_type,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    
    return formatted_text, entities

def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    if audio_data is None or len(audio_data) == 0:
        return "No Audio"
        
    if len(audio_data) < sampling_rate:
        return "Audio Too Short"

    inputs = emotion_processor(
        audio_data,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        
    return emotion_model.config.id2label.get(predicted_class_idx, "Unknown")

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: int
    transcription: str
    emotion: str
    chunk_filename: str
    ner_tags: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""
    
    def get_formatted_text(self) -> str:
        texts = []
        
        for segment in self.segments:
            age_bucket = self.get_age_bucket(segment.age)
            gender_text = "MALE" if segment.gender == 1 else "FEMALE"
            
            ner_text, entities = process_ner(segment.transcription.lower())
            segment.ner_tags = entities  
            
            # Include confidence scores in metadata if available
            confidence_info = ""
            if segment.confidence_scores:
                confidence_info = " ".join([
                    f"{key.upper()}_CONF_{value:.2f}"
                    for key, value in segment.confidence_scores.items()
                ])
                
            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{segment.emotion.upper()} {confidence_info}"
            
            texts.append(f"{ner_text} {metadata}")
        
        return " ".join(texts)

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100, 2)
        age_brackets = [
            (18, "0_18"),
            (30, "18_30"),
            (45, "30_45"),
            (60, "45_60"),
            (float('inf'), "60PLUS")
        ]
        
        for threshold, bracket in age_brackets:
            if actual_age < threshold:
                return bracket
        return "60PLUS"

def process_audio_directory(base_dir: str, transcriptions: Dict[str, str], output_dir: str) -> List[Dict[str, str]]:
    results = []
    audio_dir = os.path.join(base_dir, "audio")
    
    for speaker in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
            
        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            if not os.path.isdir(session_path):
                continue
                
            for audio_file in os.listdir(session_path):
                if not audio_file.endswith('.flac'):
                    continue
                    
                audio_path = os.path.join(session_path, audio_file)
                file_id = os.path.splitext(audio_file)[0]
                
                if file_id not in transcriptions:
                    print(f"No transcription found for {audio_file}")
                    continue
                
                transcription = transcriptions[file_id]
                print(f"Processing {audio_file}: {transcription}")
                
                try:
                    signal, sr = librosa.load(audio_path, sr=16000)
                    
                    # Update this part:
                    inputs = feature_extractor(
                        signal, 
                        sampling_rate=sr, 
                        return_tensors="pt",
                        padding=True
                    ).to(device)


                    with torch.no_grad():
                        model_output = model(**inputs)
                        age = float(model_output[1].detach().cpu().numpy()[0][0])
                        gender_logits = model_output[2].detach().cpu().numpy()[0]
                        gender = np.argmax(gender_logits)
                        gender_confidence = float(np.max(gender_logits))

                    emotion = extract_emotion(signal)
                    ner_text, ner_entities = process_ner(transcription)
                    
                    # Store confidence scores
                    confidence_scores = {
                        "gender": gender_confidence,
                    }
                    
                    segment_data = AudioSegment(
                        start_time=0,
                        end_time=len(signal) / sr,
                        speaker=speaker,
                        age=age,
                        gender=gender,
                        transcription=transcription,
                        emotion=emotion,
                        chunk_filename=audio_file,
                        ner_tags=ner_entities,
                        confidence_scores=confidence_scores
                    )
                    
                    relative_path = os.path.relpath(audio_path, base_dir)
                    
                    chunk = ChunkData()
                    chunk.segments.append(segment_data)
                    chunk.filepath = relative_path
                    
                    results.append({
                        "audio_filepath": chunk.filepath,
                        "text": chunk.get_formatted_text()
                    })
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
                    continue
    
    return results

def process_dataset(base_dir: str, output_dir: str = "output", batch_size: int = 10) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    trans_file = os.path.join(base_dir, "transcripts.txt")
    if not os.path.exists(trans_file):
        print(f"Transcription file not found: {trans_file}")
        return
        
    print("Loading transcriptions...")
    transcriptions = load_transcriptions(trans_file)
    print(f"Loaded {len(transcriptions)} transcriptions")
    
    print("Processing audio files...")
    all_results = []
    processed_count = 0
    batch_number = 1
    
    def save_batch(results, batch_num):
        if results:
            json_path = os.path.join(output_dir, f"audio_text_pairs_batch_{batch_num}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved batch {batch_num} with {len(results)} entries to: {json_path}")
    
    results = process_audio_directory(base_dir, transcriptions, output_dir)
    all_results.extend(results)
    processed_count += len(results)
    
    if processed_count >= batch_size:
        save_batch(all_results, batch_number)
        batch_number += 1
        all_results = []
        processed_count = 0
    
    if all_results:
        save_batch(all_results, batch_number)
        print(f"\nSaved final batch with {len(all_results)} entries")
    else:
        print("\nNo remaining results to save")

if __name__ == "__main__":
    base_dir = "/external2/datasets/librespeech/mls_portuguese/train"
    output_dir = "/external2/datasets/librespeech/mls_port_output"
    
    print(f"Processing files from base directory: {base_dir}")
    print(f"Saving output to: {output_dir}")
    
    process_dataset(base_dir, output_dir)