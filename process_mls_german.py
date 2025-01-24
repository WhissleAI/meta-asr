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
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # No need to replace '/' with '_' as the IDs already use underscore format
                file_id = parts[0]
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
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(audio_dir):
        for audio_file in files:
            if not audio_file.endswith('.flac'):
                continue
                
            audio_path = os.path.join(root, audio_file)
            file_id = os.path.splitext(audio_file)[0]
            
            if file_id not in transcriptions:
                print(f"No transcription found for {file_id}")
                continue
            
            transcription = transcriptions[file_id]
            print(f"Processing {audio_file}: {transcription}")
            
            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                
                if signal is None or len(signal) == 0:
                    print(f"Failed to load audio file: {audio_path}")
                    continue

                # Process audio file
                y = processor(signal, sampling_rate=sr)
                y = y['input_values'][0]
                y = y.reshape(1, -1)
                y = torch.from_numpy(y).to(device)

                with torch.no_grad():
                    model_output = model(y)
                    age = float(model_output[1].detach().cpu().numpy()[0][0])
                    gender = np.argmax(model_output[2].detach().cpu().numpy())

                # Get emotion
                emotion = extract_emotion(signal)
                emotion_label = "NEU"  # Default neutral emotion
                if emotion:
                    # Map emotion to abbreviated format if needed
                    emotion_label = emotion.upper()[:3]

                # Process NER
                ner_text, ner_entities = process_ner(transcription)
                
                # Create relative path that matches the desired format
                relative_path = os.path.relpath(audio_path, base_dir)
                
                # Map age to ranges
                age_actual = round(age * 100, 2)
                if age_actual < 18:
                    age_range = "0_18"
                elif age_actual < 30:
                    age_range = "18_30"
                elif age_actual < 45:
                    age_range = "30_45"
                elif age_actual < 60:
                    age_range = "45_60"
                else:
                    age_range = "60PLUS"

                # Map gender (assuming 1 is male, others are female)
                gender_label = "MALE" if gender == 1 else "FEMALE"
                
                # Format the text field
                formatted_text = f"{ner_text} AGE_{age_range} GENDER_{gender_label} EMOTION_{emotion_label}"
                
                results.append({
                    "audio_filepath": relative_path,
                    "text": formatted_text
                })
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue

    return results

def process_dataset(base_dir: str, output_dir: str = "output", batch_size: int = 10) -> None:
    print(f"\nStarting process_dataset")
    print(f"Current time: 2025-01-23 14:10:23")
    print(f"User: zenitsu0509")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory with full path
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created/verified output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return

    # Check write permissions
    if not os.access(output_dir, os.W_OK):
        print(f"❌ No write permission for output directory: {output_dir}")
        return
    
    # Verify transcription file
    trans_file = os.path.join(base_dir, "transcripts.txt")
    if not os.path.exists(trans_file):
        print(f"❌ Transcription file not found: {trans_file}")
        return
        
    print("\nLoading transcriptions...")
    transcriptions = load_transcriptions(trans_file)
    print(f"✓ Loaded {len(transcriptions)} transcriptions")
    
    if not transcriptions:
        print("❌ No transcriptions loaded. Exiting.")
        return
    
    print("\nProcessing audio files...")
    results = process_audio_directory(base_dir, transcriptions, output_dir)
    
    if not results:
        print("❌ No results generated. Check if audio files were processed correctly.")
        return
    
    print(f"\nPreparing to save {len(results)} results...")
    
    # Save in batches of 10
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        # Create batch filename
        json_path = os.path.join(output_dir, f"audio_text_pairs_batch_{batch_num}.json")
        temp_path = json_path + '.tmp'
        
        print(f"\nSaving batch {batch_num}/{(len(results) + batch_size - 1) // batch_size}")
        print(f"Batch size: {len(batch)} entries")
        print(f"Output file: {json_path}")
        
        try:
            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, indent=2, ensure_ascii=False)
            print(f"✓ Written to temporary file: {temp_path}")
            
            # Rename to final filename
            os.replace(temp_path, json_path)
            print(f"✓ Successfully saved batch {batch_num} to: {json_path}")
            
        except Exception as e:
            print(f"❌ Error saving batch {batch_num}: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"✓ Cleaned up temporary file: {temp_path}")
                except:
                    print(f"❌ Failed to clean up temporary file: {temp_path}")
            continue
    
    # Save summary file
    try:
        summary_path = os.path.join(output_dir, "summary.json")
        summary = {
            "total_files": len(results),
            "total_batches": (len(results) + batch_size - 1) // batch_size,
            "batch_size": batch_size,
            "processing_date": "2025-01-23 14:10:23",
            "user": "zenitsu0509",
            "base_dir": base_dir,
            "output_dir": output_dir
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved summary to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error saving summary: {str(e)}")

if __name__ == "__main__":
    # Define base directories with absolute paths
    base_dir = "/external2/datasets/librespeech/mls_german/train"
    output_dir = "/external2/datasets/librespeech/mls_german_output"
    
    print("=" * 50)
    print("Starting MLS German Processing")
    print("=" * 50)
    print(f"Start time: 2025-01-23 14:10:23")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        process_dataset(base_dir, output_dir, batch_size=10)
        print("\n✓ Processing completed successfully.")
    except Exception as e:
        print(f"\n❌ Fatal error in process_dataset: {str(e)}")
    
    print("=" * 50)