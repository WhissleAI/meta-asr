#!/usr/bin/env python3
"""
Script to remove entity tags and their end tags from JSONL files.
Usage: python remove_entity_tags.py [input_directory] [output_directory]

Output directory will be created if it doesn't exist.
"""

import os
import json
import sys
import re
import glob
from pathlib import Path
from tqdm import tqdm

def remove_entity_tags(text):
    """
    Remove entity tags (like ENTITY_LOCATION) and their END tags from the text.
    Example: "text with ENTITY_LOCATION New York END city" -> "text with New York city"
    """
    # Pattern to match entity tags and their end tags
    pattern = r'ENTITY_[A-Z_]+\s+(.+?)\s+END'
    
    # Replace entity tags with just the content between them
    processed_text = re.sub(pattern, r'\1', text)
    
    return processed_text

def process_jsonl_file(input_file, output_file):
    """Process a JSONL file, removing entity tags from the 'text' field."""
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    processed_lines = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            if 'text' in data:
                data['text'] = remove_entity_tags(data['text'])
            processed_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON line in {input_file}")
            processed_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(processed_lines)

def main():
   
    
    input_dir = "/external4/datasets/Processed_manifest/madASR/vaani"
    output_dir = "/external4/datasets/Processed_manifest/madASR/vaani_entity_tags_removed"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .jsonl files in the input directory
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(jsonl_files)} .jsonl files to process.")
    
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, file_name)
        process_jsonl_file(file_path, output_file)
    
    print(f"Processing complete. Output saved to {output_dir}")

if __name__ == "__main__":
    main()