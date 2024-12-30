import os
import json
import torch
import librosa
import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from typing import Dict, List, Any, Tuple

# Set device
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

def load_models():
    """Initialize and load all required models with proper error handling"""
    try:
        models = {
            'ner_tagger': SequenceTagger.load("flair/ner-english-ontonotes-large").to(device),
            'emotion_model': AutoModelForAudioClassification.from_pretrained(
                "superb/hubert-large-superb-er"
            ).to(device),
            'emotion_extractor': AutoFeatureExtractor.from_pretrained(
                "superb/hubert-large-superb-er"
            ),
            'age_gender_model': Wav2Vec2ForSequenceClassification.from_pretrained(
                "audeering/wav2vec2-large-robust-6-ft-age-gender"
            ).to(device),
            'age_gender_processor': Wav2Vec2Processor.from_pretrained(
                "audeering/wav2vec2-large-robust-6-ft-age-gender"
            )
        }
        return models
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def load_transcriptions(trans_file: str) -> Dict[str, str]:
    """Load transcriptions from the .trans.txt file"""
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, text = parts
                transcriptions[f"{file_id}.flac"] = text
    return transcriptions


def extract_emotion(audio_data: np.ndarray, models: Dict, sampling_rate: int = 16000) -> str:
    """Extract emotion from audio segment with improved error handling"""
    try:
        # Ensure audio data is the correct shape and type
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
        
        inputs = models['emotion_extractor'](
            audio_data, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = models['emotion_model'](**inputs)
        
        predicted_class_idx = outputs.logits[0].argmax(-1).item()
        emotion = models['emotion_model'].config.id2label.get(predicted_class_idx, "Unknown")
        
        if emotion == "Unknown":
            print("Warning: Emotion detection returned unknown class")
            
        return emotion
    except Exception as e:
        print(f"Error in emotion extraction: {str(e)}")
        return "NEUTRAL"  # Fallback emotion



def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> Tuple[float, float]:
    """Extract age and gender logits"""
    y = age_gender_processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        outputs = age_gender_model(y)
        age_logits = outputs.logits[0][0].item()  # Age prediction
        gender_logits = outputs.logits[0][1].item()  # Gender prediction

    return age_logits, gender_logits



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



def process_ner(text: str, models: Dict) -> Tuple[str, List[Dict[str, Any]]]:
    """Process text for named entities with improved error handling"""
    try:
        # Clean and prepare text
        text = text.strip()
        if not text:
            return "", []
            
        sentence = Sentence(text)
        models['ner_tagger'].predict(sentence)
        
        entities = []
        for entity in sentence.get_spans('ner'):
            # Validate entity boundaries
            if entity.start_position < 0 or entity.end_position > len(text):
                continue
                
            entities.append({
                "text": entity.text,
                "type": entity.tag,
                "start_position": entity.start_position,
                "end_position": entity.end_position,
                "confidence": entity.score  # Add confidence score
            })
            
        # Debug information
        if not entities:
            print(f"Warning: No entities found in text: {text[:100]}...")
            
        return text, entities
    except Exception as e:
        print(f"Error in NER processing: {str(e)}")
        return text, []


def process_directory(input_dir: str, output_dir: str = "output") -> str:
    """Process all audio files in the directory with improved error handling"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    try:
        # Load all models at once
        models = load_models()
        
        # Find transcription file
        trans_files = [f for f in os.listdir(input_dir) if f.endswith('.trans.txt')]
        if not trans_files:
            raise FileNotFoundError("No transcription file found")
            
        # Load transcriptions
        transcriptions = load_transcriptions(os.path.join(input_dir, trans_files[0]))
        
        # Process audio files
        for filename in sorted(os.listdir(input_dir)):
            if not filename.endswith('.flac'):
                continue
                
            audio_path = os.path.join(input_dir, filename)
            try:
                # Load and preprocess audio
                signal, sr = librosa.load(audio_path, sr=16000)
                
                # Extract features
                age_logits, gender_logits = process_func(signal, sr)
                age_bucket = get_age_bucket(age_logits)
                emotion = extract_emotion(signal, models)
                
                # Process transcription
                transcription = transcriptions.get(filename, "")
                ner_text, ner_entities = process_ner(transcription.lower(), models)
                
                # Create comprehensive result
                result = {
                    "audio_filepath": audio_path,
                    "text": ner_text,
                    "metadata": {
                        "emotion": emotion,
                        "age_bucket": age_bucket,
                        "gender_logit": float(gender_logits),
                        "named_entities": ner_entities
                    },
                    "processed_text": f"{ner_text} EMOTION_{emotion.upper()} AGE_{age_bucket}"
                }
                
                results.append(result)
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Save results
        output_path = os.path.join(output_dir, "processed_audio.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Error in directory processing: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_dir = "84/121123"
    output_dir = "output1"
    output_path = process_directory(input_dir, output_dir)
    print(f"Results saved to: {output_path}")
