#!/usr/bin/env python3
import os
import json
import torch
from transformers import pipeline
from tqdm import tqdm

# === Configuration ===
SPLIT_PATHS = {
    "train": "/external3/databases/ai4bharat_indicvoices/urdu/train_manifest.jsonl",
    "valid": "/external3/databases/ai4bharat_indicvoices/urdu/valid_manifest.jsonl",
}
MODEL_NAME  = "superb/hubert-large-superb-er"
BATCH_SIZE  = 16  # adjust as you like

def normalize_age_group(age_group: str) -> str:
    return age_group.replace("-", "_")

def normalize_gender(gender: str) -> str:
    return gender.upper()

def get_processed_filepaths(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    seen = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(rec.get("audio_filepath"))
            except json.JSONDecodeError:
                continue
    return seen

def process_split(split_name: str, input_path: str, audio_classifier):
    split_dir   = os.path.dirname(input_path)
    output_path = os.path.join(split_dir, f"{split_name}_intermediate.jsonl")
    processed   = get_processed_filepaths(output_path)

    # Count for tqdm
    with open(input_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    batch = []
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "a", encoding="utf-8") as outfile:

        for line in tqdm(infile, total=total, desc=split_name):
            item = json.loads(line)
            fp   = item.get("audio_filepath")
            if not fp or fp in processed:
                continue

            batch.append({
                "filepath": fp,
                "sentence": item.get("text", ""),
                "age_grp":  normalize_age_group(item.get("age_group", "")),
                "gender":   normalize_gender(item.get("gender", "")),
                "duration": item.get("duration")
            })
            processed.add(fp)

            if len(batch) >= BATCH_SIZE:
                _write_batch(batch, audio_classifier, outfile)
                batch.clear()

        if batch:
            _write_batch(batch, audio_classifier, outfile)
            batch.clear()

    print(f"✅ {split_name}: wrote {len(processed)} records to {output_path}")

def _write_batch(batch, audio_classifier, outfile):
    paths = [e["filepath"] for e in batch]
    results = audio_classifier(
        paths,
        return_all_scores=False
    )

    for e, res in zip(batch, results):
        # res might be a dict or a list of dicts
        top = res[0] if isinstance(res, list) else res
        label = top["label"].upper()
        

        new_text = f"{e['sentence']} AGE_{e['age_grp']} GENDER_{e['gender']} EMOTION_{label}"
        rec = {
            "audio_filepath": e["filepath"],
            "text":           new_text,
            "duration":       e["duration"]
        }
        outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    device = 1 if torch.cuda.is_available() else -1
    audio_classifier = pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=device,
        return_all_scores=False,
    )

    for split_name, input_path in SPLIT_PATHS.items():
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing manifest: {input_path}")
        process_split(split_name, input_path, audio_classifier)

if __name__ == "__main__":
    main()
