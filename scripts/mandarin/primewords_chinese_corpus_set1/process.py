#!/usr/bin/env python3
import json
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoTokenizer,
    AutoModelForTokenClassification
)
import torch.nn as nn
from tqdm import tqdm

# === Hard-coded configuration ===
INPUT_JSONL = "/external4/datasets/Mandarin/Primewords_Chinese_Corpus_Set_1/manifest.jsonl"
OUTPUT_JSONL = "/external4/datasets/Mandarin/Primewords_Chinese_Corpus_Set_1/intermediate.jsonl"
BATCH_SIZE = 32
NUM_WORKERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 100 # or an int to limit samples

# === Age brackets & mapping ===
age_brackets = [
    (18, "0_18"),
    (30, "18_30"),
    (45, "30_45"),
    (60, "45_60"),
    (float('inf'), "60PLUS")
]
def map_age_to_bracket(age):
    for max_age, label in age_brackets:
        if age <= max_age:
            return label
    return "60PLUS"

# === Emotion mapping ===
def emotion_label_from_id(eid):
    return {
        0: "angry", 1: "fear", 2: "happy",
        3: "neutral", 4: "sad"
    }.get(eid, "surprise")

# === Gender mapping ===
gender_mapping = {'male': 'MALE', 'female': 'FEMALE', 'other': 'OTHER'}

# === Sentence segmentation model ===
tokenizer_seg = AutoTokenizer.from_pretrained(
    "KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation"
)
model_seg = AutoModelForTokenClassification.from_pretrained(
    "KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation"
).to(DEVICE).eval()

# === Model head for age/gender ===
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

# === Dataset & collate ===
class AudioManifestDataset(Dataset):
    def __init__(self, manifest_path, max_samples=None):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            recs = [json.loads(l) for l in f]
        self.records = recs[:max_samples] if max_samples else recs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sig, _ = librosa.load(rec["audio_filepath"], sr=16000)
        return sig, rec

def collate_fn(batch):
    sigs, recs = zip(*batch)
    return list(sigs), list(recs)

# === Load models ===
processor_ag = Wav2Vec2Processor.from_pretrained(
    "audeering/wav2vec2-large-robust-24-ft-age-gender"
)
model_ag = AgeGenderModel.from_pretrained(
    "audeering/wav2vec2-large-robust-24-ft-age-gender"
).to(DEVICE).eval()
if torch.cuda.device_count() > 1:
    model_ag = torch.nn.DataParallel(model_ag)

feature_extractor_emo = AutoFeatureExtractor.from_pretrained(
    "xmj2002/hubert-base-ch-speech-emotion-recognition"
)
model_emo = AutoModelForAudioClassification.from_pretrained(
    "xmj2002/hubert-base-ch-speech-emotion-recognition"
).to(DEVICE).eval()
if torch.cuda.device_count() > 1:
    model_emo = torch.nn.DataParallel(model_emo)

# === Inference ===
dataset = AudioManifestDataset(INPUT_JSONL, max_samples=MAX_SAMPLES)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
)

raw_gender = ["female", "male", "other"]
results = []

for signals, recs in tqdm(loader, desc="Processing"):
    # Age/Gender
    inputs_ag = processor_ag(
        signals, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_values.to(DEVICE)
    with torch.no_grad():
        log_age, log_gender = model_ag(inputs_ag)
    ages = (log_age.squeeze(-1).cpu().numpy() * 100).tolist()
    gids = log_gender.argmax(dim=1).cpu().tolist()

    # Emotion
    inputs_em = feature_extractor_emo(
        signals, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_values.to(DEVICE)
    with torch.no_grad():
        emo_logits = model_emo(inputs_em).logits
    eids = emo_logits.argmax(dim=1).cpu().tolist()

    for rec, age, gid, eid in zip(recs, ages, gids, eids):
        rec["age"] = map_age_to_bracket(age)
        rec["gender"] = gender_mapping[raw_gender[gid]]
        rec["emotion"] = emotion_label_from_id(eid)

        # Sentence segmentation
        text = rec.get("text", "")
        if text:
            toks = tokenizer_seg.encode(text, return_tensors="pt").to(DEVICE)
            seg_logits = model_seg(toks).logits
            labs = torch.argmax(seg_logits, dim=2)[0].tolist()[1:-1]
            labels = [model_seg.config.id2label[i] for i in labs]
            seg_text = "".join(
                ch + "。" if lab in ("E", "S") else ch
                for ch, lab in zip(text, labels)
            )
            rec["segmented_text"] = seg_text

        results.append(rec)

# === Save as JSONL ===
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
    for rec in results:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved predictions & segmentation to {OUTPUT_JSONL}")

