#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
import argparse

def clean_text(text):
    # Remove all tags with angle brackets like <PAUSE>, <noise>, <horn>, etc.
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove text enclosed in square brackets like [breathing], [inhaling]
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Remove any content with curly braces (often English translations)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Modified pattern that excludes special tags from removal
    # This will preserve ENTITY_ tags, END tags, and metadata tags
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Keep words if they contain non-Latin characters (Bengali)
        # or if they are special tags we want to preserve
        if (not re.match(r'^[a-zA-Z]+$', word) or 
            word == 'END' or 
            word.startswith('ENTITY_') or
            word.startswith('AGE_') or
            word.startswith('GENDER_') or
            word.startswith('EMOTION_') or
            word.startswith('INTENT_')):
            cleaned_words.append(word)
    
    # Join the cleaned words back into text
    text = ' '.join(cleaned_words)
    
    # Remove double spaces, leading/trailing spaces, etc.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_jsonl_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        line_count = 0
        for line in infile:
            try:
                # Parse the JSON object
                data = json.loads(line)
                
                if 'text' in data:
                    # Clean the text content
                    data['text'] = clean_text(data['text'])
                    
                    # Write the cleaned JSON object
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    line_count += 1
            except json.JSONDecodeError:
                print(f"Error parsing line: {line}")
                continue
    
    return line_count

def main():
   
    
    
    input_dir = Path("/external4/datasets/Processed_manifest/madASR/vaani")
    output_dir =Path("/external4/datasets/Processed_manifest/madASR/vaani_cleaned")
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    total_lines = 0
    
    for jsonl_file in input_dir.glob('*.jsonl'):
        input_path = jsonl_file
        output_path = output_dir / jsonl_file.name
        
        print(f"Processing {input_path}...")
        lines_processed = process_jsonl_file(input_path, output_path)
        print(f"✓ Cleaned {lines_processed} entries in {output_path}")
        
        total_files += 1
        total_lines += lines_processed
    
    print(f"\nSummary: Processed {total_files} files with a total of {total_lines} entries")
    print(f"Cleaned files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()