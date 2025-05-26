from datasets import load_dataset, Audio
import os
import json
import numpy as np
import soundfile as sf          # pip install soundfile
from tqdm.auto import tqdm

# 0. Configure your base output directory
BASE_OUT = "/external3/databases/ai4bharat_indicvoices/urdu"

# 1. Point at your Parquet shards
data_files = {
    "train": os.path.join(BASE_OUT, "train-*.parquet"),
    "valid": os.path.join(BASE_OUT, "valid-*.parquet"),
}

# 2. Load metadata-only and cast audio_filepath → {array, sampling_rate, path}
ds = load_dataset("parquet", data_files=data_files)
ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

# 3. Columns to keep in JSONL
columns_to_keep = [
    "audio_filepath",
    "text",
    "duration",
    "task_name",
    "gender",
    "age_group",
    "state"
]

# 4. Process each split, skipping already-done files and any erroring samples
for split in ("train", "valid"):
    out_wav_dir = os.path.join(BASE_OUT, f"wavs_{split}")
    os.makedirs(out_wav_dir, exist_ok=True)

    jsonl_path = os.path.join(BASE_OUT, f"{split}_manifest.jsonl")
    mode = "a" if os.path.exists(jsonl_path) else "w"
    new_count = 0

    with open(jsonl_path, mode, encoding="utf-8") as fout:
        for idx in tqdm(range(len(ds[split])), desc=f"Processing {split}", unit="ex"):
            try:
                # A) Load example and decode audio
                example = ds[split][idx]
                audio = example["audio_filepath"]        # dict with array, sr, path
                arr   = np.asarray(audio["array"])
                sr    = audio["sampling_rate"]
                # force mono into shape (n,1)
                if arr.ndim == 1:
                    arr = arr[:, None]

                # B) Build stable output path
                stem = os.path.splitext(os.path.basename(audio["path"]))[0]
                dst  = os.path.join(out_wav_dir, f"{stem}.wav")

                # C) Skip if already written
                if os.path.exists(dst):
                    continue

                # D) Write WAV
                sf.write(dst, arr, sr)

                # E) Assemble filtered JSON record
                record = {}
                for k in columns_to_keep:
                    record[k] = dst if k == "audio_filepath" else example[k]

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_count += 1

            except Exception as e:
                # skip this one, log index & cause
                print(f"Skipping {split}[{idx}] due to {type(e).__name__}: {e}")
                continue

    print(f"{split}: +{new_count} new WAVs → {out_wav_dir}/")
    print(f"{split}: metadata appended → {jsonl_path}")


#screen -r 3332087.pts-33.hydra2