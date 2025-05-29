import json
import os
import librosa
import torch
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
INPUT_JSONL  = r"C:\Users\kodur\Downloads\Whissle\primewords chinese corpus set 1\manifest1.jsonl"
OUTPUT_JSONL = r"C:\Users\kodur\Downloads\Whissle\primewords chinese corpus set 1\intermediate.jsonl"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Resume support: read already-processed audio_filepaths ===
processed_paths = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                rec = json.loads(line)
                processed_paths.add(rec.get("audio_filepath"))
            except json.JSONDecodeError:
                continue

# === Age brackets & mapping ===
age_brackets = [
    (18,   "0_18"),
    (30,  "18_30"),
    (45,  "30_45"),
    (60,  "45_60"),
    (float('inf'), "60PLUS"),
]
def map_age_to_bracket(age: float) -> str:
    for max_age, label in age_brackets:
        if age <= max_age:
            return label
    return "60PLUS"

# === Emotion mapping ===
def emotion_label_from_id(eid: int) -> str:
    return {0:"angry", 1:"fear", 2:"happy", 3:"neutral", 4:"sad"}.get(eid, "surprise")

# === Gender mapping ===
gender_mapping = {'male':'MALE','female':'FEMALE','other':'OTHER'}

# === Load sentence-segmentation model ===
tokenizer_seg = AutoTokenizer.from_pretrained(
    "KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation"
)
model_seg = AutoModelForTokenClassification.from_pretrained(
    "KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation"
).to(DEVICE).eval()

# === Define Age/Gender model head ===
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

# === Load processors & models ===
processor_ag = Wav2Vec2Processor.from_pretrained(
    "audeering/wav2vec2-large-robust-24-ft-age-gender"
)
model_ag = AgeGenderModel.from_pretrained(
    "audeering/wav2vec2-large-robust-24-ft-age-gender"
).to(DEVICE).eval()

feature_extractor_emo = AutoFeatureExtractor.from_pretrained(
    "xmj2002/hubert-base-ch-speech-emotion-recognition"
)
model_emo = AutoModelForAudioClassification.from_pretrained(
    "xmj2002/hubert-base-ch-speech-emotion-recognition"
).to(DEVICE).eval()

# === Load input manifest ===
with open(INPUT_JSONL, 'r', encoding='utf-8') as fin:
    all_records = [json.loads(l) for l in fin]

raw_gender = ["female", "male", "other"]

# === Open output for append so each record is flushed immediately ===
with open(OUTPUT_JSONL, 'a', encoding='utf-8') as fout:
    for rec in tqdm(all_records, desc="Processing"):
        path = rec.get("audio_filepath")
        if path in processed_paths:
            continue  # already done

        # --- load audio ---
        sig, _ = librosa.load(path, sr=16000)

        # --- age/gender inference ---
        inp_ag = processor_ag(sig, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(DEVICE)
        with torch.no_grad():
            log_age, log_gender = model_ag(inp_ag)
        age_val = log_age.squeeze(-1).cpu().item() * 100
        g_id    = log_gender.argmax(dim=1).cpu().item()
        rec["age"]    = map_age_to_bracket(age_val)
        rec["gender"] = gender_mapping[ raw_gender[g_id] ]

        # --- emotion inference ---
        inp_em = feature_extractor_emo(sig, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(DEVICE)
        with torch.no_grad():
            emo_logits = model_emo(inp_em).logits
        rec["emotion"] = emotion_label_from_id(emo_logits.argmax(dim=1).cpu().item())

        # --- sentence segmentation ---
        text = rec.get("text", "").strip()
        if text:
            toks       = tokenizer_seg.encode(text, return_tensors="pt").to(DEVICE)
            seg_logits = model_seg(toks).logits
            labs       = torch.argmax(seg_logits, dim=2)[0].tolist()[1:-1]
            labels     = [model_seg.config.id2label[i] for i in labs]
            seg_text   = "".join(
                ch + "。" if lab in ("E","S") else ch
                for ch, lab in zip(text, labels)
            )
            rec["segmented_text"] = seg_text

        # --- write & flush ---
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

        processed_paths.add(path)

print(f"\n✅ Done! Output saved/appended to:\n   {OUTPUT_JSONL}")
