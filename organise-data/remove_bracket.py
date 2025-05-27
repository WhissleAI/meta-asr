#!/usr/bin/env python3
# filepath: process_jsonl.py

import json
import re
import argparse

def clean_text(text):
    """
    Remove brackets and parentheses, capitalizing any text that was inside them.
    """
    # Function to capitalize words inside brackets/parentheses before removing them
    def capitalize_and_remove(match):
        # Extract the content without the brackets/parentheses
        content = match.group(0)[1:-1]
        # Capitalize each word
        capitalized = ' '.join(word.upper() for word in content.split())
        return ' ' + capitalized + ' '
    
    # Apply for each type of bracket/parenthesis
    patterns = [r'<[^>]*>', r'\[[^\]]*\]', r'\{[^\}]*\}', r'\([^\)]*\)']
    
    for pattern in patterns:
        text = re.sub(pattern, capitalize_and_remove, text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_jsonl_file(input_file, output_file):
    """
    Process each line in the JSONL file, clean the text field, and write to output.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        line_count = 0
        for line in infile:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    data['text'] = clean_text(data['text'])
                
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                line_count += 1
                
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
    
    print(f"Completed processing {line_count} lines.")

if __name__ == "__main__":
    input_file = '/external4/datasets/Vaani_Hindi/hi_corpus_annotated_fixed.jsonl'
    output_file = '/external4/datasets/Vaani_Hindi/hi_corpus_annotated_cleaned.jsonl'
    
    print(f"Processing file: {input_file}")
    process_jsonl_file(input_file,output_file)
    print(f"Output saved to: {output_file}")