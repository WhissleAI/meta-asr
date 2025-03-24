import os
import json
import torch
import librosa
import numpy as np
import torch.nn as nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from typing import Dict, List, Any

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



def get_age_bucket(age: float) -> str:
    actual_age: float = round(age*100, 2)

    age_brackets: List[tuple[float, str]] = [
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


def process_txt_data(
    txt_path: str,
    output_base_dir: str = "output"
) -> List[Dict[str, Any]]:
    results_dir = os.path.abspath(os.path.join(output_base_dir, "results"))
    os.makedirs(results_dir, exist_ok=True)

    all_data = []

    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue

                parts = line.split('|', 1)
                if len(parts) != 2:
                    print(f"Invalid line format: {line}")
                    continue

                audio_path = parts[0].strip()
                provided_text = parts[1].strip()

                if not os.path.exists(audio_path):
                    print(f"Audio file not found: {audio_path}")
                    continue

                try:
                    import soundfile as sf
                    signal, sr = sf.read(audio_path)
                    if sr != 16000:
                        import resampy
                        signal = resampy.resample(signal, sr, 16000)
                        sr = 16000
                except Exception as e:
                    print(f"Error loading audio file {audio_path}: {str(e)}")
                    continue

                try:
                    y = processor(signal, sampling_rate=sr)
                    y = y['input_values'][0]
                    y = y.reshape(1, -1)
                    y = torch.from_numpy(y).to(device)

                    with torch.no_grad():
                        model_output = model(y)
                        age = float(model_output[1].detach().cpu().numpy()[0][0])
                        gender = np.argmax(model_output[2].detach().cpu().numpy())

                    try:
                        emotion = extract_emotion(signal)
                    except Exception as e:
                        print(f"Error extracting emotion: {str(e)}")
                        emotion = "Unknown"

                    age_bucket = get_age_bucket(age)
                    gender_text = "MALE" if gender == 1 else "FEMALE"

                    emotion_text = emotion.upper().replace(" ", "_")
                    metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"

                    segment_data = {
                        'audio_file_path': os.path.abspath(audio_path),
                        'transcription': provided_text,
                        'age': float(age),
                        'age_bucket': age_bucket,
                        'gender': int(gender),
                        'gender_text': gender_text,
                        'emotion': emotion,
                        'formatted_text': f"{provided_text} {metadata}"
                    }

                    all_data.append(segment_data)

                except Exception as e:
                    print(f"Error processing audio file {audio_path}: {str(e)}")
                    continue

        if all_data:
            jsonl_path_output = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(txt_path))[0]}_processed_data.jsonl")
            with open(jsonl_path_output, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved JSONL to: {jsonl_path_output}")

            audio_text_pairs = [
                {
                    "audio_filepath": item['audio_file_path'],
                    "text": item['formatted_text']
                }
                for item in all_data
            ]

            audio_text_jsonl_path = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(txt_path))[0]}_audio_text_pairs.jsonl")
            with open(audio_text_jsonl_path, 'w', encoding='utf-8') as f:
                for item in audio_text_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved audio-text pairs JSONL to: {audio_text_jsonl_path}")

        return all_data

    except Exception as e:
        print(f"Error processing TXT file {txt_path}: {str(e)}")
        return []


if __name__ == "__main__":
    input_txt_path = "/hydra2-prev/home/compute/workspace_himanshu/transcriptions_using_gem.txt"  # Specify the txt file path here
    output_dir = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing TXT file: {input_txt_path}")
    print(f"Saving output to: {output_dir}")

    if os.path.isfile(input_txt_path) and input_txt_path.lower().endswith('.txt'):
        print(f"\nProcessing txt file: {input_txt_path}")
        data = process_txt_data(input_txt_path, output_base_dir=output_dir)
        print(f"Successfully processed {input_txt_path}")
        print(f"Processed {len(data)} audio segments")
    else:
        print(f"Error: {input_txt_path} is not a valid TXT file path.")