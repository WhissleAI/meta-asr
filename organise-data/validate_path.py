import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import hashlib

def validate_audio_files(input_jsonl_path, base_path):
    # Read the input JSONL file
    data = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
   
    # Create output filename
    input_filename = Path(input_jsonl_path).stem
    output_json_path = f"{input_filename}_validated.json"
   
    # List to store valid entries and track duplicates
    valid_entries = []
    seen_filepaths = set()
    seen_content = set()
    filepath_duplicate_count = 0
    content_duplicate_count = 0
   
    # Process each entry with a progress bar
    print(f"Validating {len(data)} audio files...")
    for entry in tqdm(data):
        # Construct full path by joining base_path and audio_filepath
        full_path = os.path.join(base_path, entry['audio_filepath'])
        
        # Check if the file exists
        if os.path.isfile(full_path):
            # Check for filepath duplicates
            if entry['audio_filepath'] in seen_filepaths:
                filepath_duplicate_count += 1
                continue
                
            # Create a content hash to detect duplicate data
            # Using transcript text as the key content indicator 
            # (modify this if you need to use different fields)
            if 'text' in entry:
                content_hash = hashlib.md5(entry['text'].encode('utf-8')).hexdigest()
            else:
                # If no text field, create hash from all available fields
                content_hash = hashlib.md5(json.dumps(entry, sort_keys=True).encode('utf-8')).hexdigest()
                
            # Check for content duplicates
            if content_hash in seen_content:
                content_duplicate_count += 1
            else:
                valid_entries.append(entry)
                seen_filepaths.add(entry['audio_filepath'])
                seen_content.add(content_hash)
        else:
            print(f"File not found: {full_path}")
   
    # Save valid entries to new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=2)
   
    # Print summary
    print(f"\nTotal entries: {len(data)}")
    print(f"Valid entries: {len(valid_entries)}")
    print(f"Duplicate filepaths removed: {filepath_duplicate_count}")
    print(f"Duplicate content removed: {content_duplicate_count}")
    print(f"Invalid entries (files not found): {len(data) - len(valid_entries) - filepath_duplicate_count - content_duplicate_count}")
    print(f"Results saved to: {output_json_path}")

if __name__ == "__main__":
    base_path = "/external4/datasets/Shrutilipi/hindi/hn/audio"
    json_path = "/external4/datasets/Shrutilipi/hindi/process_data/hindi_decoded_fixed.jsonl"
   
    validate_audio_files(json_path, base_path)