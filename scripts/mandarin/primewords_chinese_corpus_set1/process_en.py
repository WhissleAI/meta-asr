import os
import json
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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

# === Configuration ===
INPUT_JSONL   = "/external4/datasets/Mandarin/MAGICDATA/manifest_dev.jsonl"
OUTPUT_JSONL  = "/external4/datasets/Mandarin/MAGICDATA/test_dev_checkpoint.jsonl"
BATCH_SIZE    = 32
NUM_WORKERS   = 16
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES   = None  # or int to limit total samples

# === Age/Gender model definitions (omitted for brevity, same as before) ===
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
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
        self.age      = ModelHead(config, 1)
        self.gender   = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

# === Dataset ===
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
    signals, recs = zip(*batch)
    return list(signals), list(recs)

# === Load processed filepaths to skip ===
processed = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                rec = json.loads(line)
                processed.add(rec.get("audio_filepath"))
            except json.JSONDecodeError:
                continue

# === Load models ===
processor_ag = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-24-ft-age-gender")
model_ag = AgeGenderModel.from_pretrained("audeering/wav2vec2-large-robust-24-ft-age-gender")\
            .to(DEVICE).eval()
if torch.cuda.device_count() > 1:
    model_ag = torch.nn.DataParallel(model_ag)

# === UPDATED: switch to r-f/wav2vec-english-speech-emotion-recognition ===
feature_extractor_emo = AutoFeatureExtractor.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
)
model_emo = AutoModelForAudioClassification.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
).to(DEVICE).eval()
if torch.cuda.device_count() > 1:
    model_emo = torch.nn.DataParallel(model_emo)

tokenizer_seg = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation")
model_seg     = AutoModelForTokenClassification.from_pretrained(
                    "KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation"
                ).to(DEVICE).eval()

# === Prepare dataset & DataLoader ===
full_ds = AudioManifestDataset(INPUT_JSONL, max_samples=MAX_SAMPLES)
indices = [i for i, rec in enumerate(full_ds.records) if rec["audio_filepath"] not in processed]
ds      = Subset(full_ds, indices)
loader  = DataLoader(ds,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     collate_fn=collate_fn,
                     pin_memory=torch.cuda.is_available())

# === Mappings ===
age_brackets = [
    (18, "0_18"),
    (30, "18_30"),
    (45, "30_45"),
    (60, "45_60"),
    (float('inf'), "60PLUS")
]
def map_age(a):
    for m, l in age_brackets:
        if a <= m:
            return l
    return "60PLUS"

# UPDATED: 7-class emotion
def emo_label(eid):
    return {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }.get(eid, "unknown")

gender_map = ["female", "male", "other"]
gender_out = {'male':'MALE','female':'FEMALE','other':'OTHER'}

# === Process & checkpoint ===
with open(OUTPUT_JSONL, 'a', encoding='utf-8') as fout:
    for signals, recs in tqdm(loader, desc="Processing"):
        # Age/Gender
        inputs_ag = processor_ag(signals, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(DEVICE)
        with torch.no_grad():
            log_age, log_gender = model_ag(inputs_ag)
        ages = (log_age.squeeze(-1).cpu().numpy() * 100).tolist()
        gids = log_gender.argmax(dim=1).cpu().tolist()

        # Emotion (now 7 classes)
        inputs_em = feature_extractor_emo(signals, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(DEVICE)
        with torch.no_grad():
            logits_em = model_emo(inputs_em).logits
        eids = logits_em.argmax(dim=1).cpu().tolist()

        # Write each record immediately
        for rec, age, gid, eid in zip(recs, ages, gids, eids):
            rec["age"]     = map_age(age)
            rec["gender"]  = gender_out[gender_map[gid]]
            rec["emotion"] = emo_label(eid)
            # segmentation
            txt = rec.get("text", "")
            if txt:
                toks       = tokenizer_seg.encode(txt, return_tensors="pt").to(DEVICE)
                seg_logits = model_seg(toks).logits
                labs       = torch.argmax(seg_logits, dim=2)[0].tolist()[1:-1]
                seg = "".join(ch + "。" if lab in ("E","S") else ch
                              for ch, lab in zip(txt, labs))
                rec["segmented_text"] = seg

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

print("✅ Processing complete.")
