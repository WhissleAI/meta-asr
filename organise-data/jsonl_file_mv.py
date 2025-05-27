import json
import shutil
import os

# Define paths
jsonl_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/avspeech-en.jsonl"  # Change this to your actual JSONL file path
new_location = "/path/to/new/location"  # Change this to the destination folder

# Ensure destination directory exists
os.makedirs(new_location, exist_ok=True)

# Process JSONL file
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        audio_path = data.get("audio_filepath")
        
        if audio_path and os.path.exists(audio_path):
            filename = os.path.basename(audio_path)
            dest_path = os.path.join(new_location, filename)
            
            shutil.move(audio_path, dest_path)
            print(f"Moved: {audio_path} -> {dest_path}")
        else:
            print(f"File not found: {audio_path}")
