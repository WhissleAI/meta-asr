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
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn as nn

load_dotenv()

# Load spaCy model
nlp = spacy.load('de_core_news_lg')

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

def load_transcriptions(trans_file: str) -> Dict[str, str]:
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Split on tab or multiple spaces
            parts = line.strip().split('\t')
            if len(parts) != 2:
                parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                # Convert file path format from / to _ for matching
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
            
            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{segment.emotion.upper()}"
            
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
    
    # Iterate through speaker directories
    for speaker in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
            
        # Iterate through session directories
        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            if not os.path.isdir(session_path):
                continue
                
            # Process each audio file
            for audio_file in os.listdir(session_path):
                if not audio_file.endswith('.flac'):
                    continue
                    
                audio_path = os.path.join(session_path, audio_file)
                file_id = os.path.splitext(audio_file)[0]  # Remove .flac extension
                
                if file_id not in transcriptions:
                    print(f"No transcription found for {audio_file}")
                    continue
                
                transcription = transcriptions[file_id]
                print(f"Processing {audio_file}: {transcription}")
                
                try:
                    signal, sr = librosa.load(audio_path, sr=16000)
                    
                    y = processor(signal, sampling_rate=sr)
                    y = y['input_values'][0]
                    y = y.reshape(1, -1)
                    y = torch.from_numpy(y).to(device)

                    with torch.no_grad():
                        model_output = model(y)
                        age = float(model_output[1].detach().cpu().numpy()[0][0])
                        gender = np.argmax(model_output[2].detach().cpu().numpy())

                    emotion = extract_emotion(signal)
                    ner_text, ner_entities = process_ner(transcription)
                    
                    segment_data = AudioSegment(
                        start_time=0,
                        end_time=len(signal) / sr,
                        speaker=speaker,
                        age=age,
                        gender=gender,
                        transcription=transcription,
                        emotion=emotion,
                        chunk_filename=audio_file,
                        ner_tags=ner_entities
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

def process_dataset(base_dir: str, output_dir: str = "output", batch_size: int = 1000) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transcriptions
    trans_file = os.path.join(base_dir, "transcripts.txt")
    if not os.path.exists(trans_file):
        print(f"Transcription file not found: {trans_file}")
        return
        
    print("Loading transcriptions...")
    transcriptions = load_transcriptions(trans_file)
    print(f"Loaded {len(transcriptions)} transcriptions")
    
    # Process audio files
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
    
    # Save any remaining results
    if all_results:
        save_batch(all_results, batch_number)
        print(f"\nSaved final batch with {len(all_results)} entries")
    else:
        print("\nNo remaining results to save")

if __name__ == "__main__":
    base_dir = "/external2/datasets/librespeech/mls_german/train"
    output_dir = "/external2/datasets/librespeech/mls_german_output"
    
    print(f"Processing files from base directory: {base_dir}")
    print(f"Saving output to: {output_dir}")
    
    process_dataset(base_dir, output_dir)