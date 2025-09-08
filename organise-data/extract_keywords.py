#!/usr/bin/env python3
"""
Script to extract unique tags from JSONL files and create a keywords.txt file.
Extracts tags like AGE_X_Y, GENDER_X, EMOTION_X, INTENT_X, NER_X, etc.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Set, List
import glob

def extract_tags_from_text(text: str) -> Set[str]:
    """
    Extract all uppercase tags from text.
    Tags are typically at the end of the text and follow patterns like:
    - AGE_X_Y
    - GENDER_X
    - EMOTION_X
    - INTENT_X
    - NER_X
    - SPEAKER_CHANGE
    - END
    """
    # Pattern to match uppercase words/tags (including underscores and numbers)
    tag_pattern = r'\b[A-Z][A-Z0-9_]*\b'
    
    # Find all matches
    matches = re.findall(tag_pattern, text)
    
    # Filter out common words that aren't tags and very short matches
    excluded_words = {'THE', 'AND', 'OR', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'WITH', 'BY', 'FROM', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'CAN', 'MUST', 'SHALL', 'A', 'AN', 'IT', 'HE', 'SHE', 'WE', 'THEY', 'YOU', 'I', 'ME', 'HIM', 'HER', 'US', 'THEM', 'MY', 'YOUR', 'HIS', 'HER', 'OUR', 'THEIR', 'THIS', 'THAT', 'THESE', 'THOSE', 'BUT', 'SO', 'IF', 'WHEN', 'WHERE', 'WHO', 'WHAT', 'HOW', 'WHY', 'WHICH', 'WHILE', 'SINCE', 'UNTIL', 'UNLESS', 'BECAUSE', 'ALTHOUGH', 'THOUGH', 'EVEN', 'STILL', 'YET', 'JUST', 'ONLY', 'ALSO', 'TOO', 'VERY', 'MUCH', 'MORE', 'MOST', 'SOME', 'ANY', 'ALL', 'EACH', 'EVERY', 'NO', 'NOT', 'YES'}
    
    # Filter tags - keep only those that look like our target tags
    valid_tags = set()
    for match in matches:
        # Skip single letters and common words
        if len(match) <= 2 or match in excluded_words:
            continue
            
        # Keep tags that match our patterns
        if (match.startswith(('AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'NER_', 'SPEAKER_')) or 
            match in ['END', 'SPEAKER_CHANGE'] or
            re.match(r'^[A-Z]+_[A-Z0-9_]+$', match)):
            valid_tags.add(match)
    
    return valid_tags

def process_jsonl_file(file_path: str) -> Set[str]:
    """Process a single JSONL file and extract all unique tags."""
    tags = set()
    
    print(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        text_tags = extract_tags_from_text(data['text'])
                        tags.update(text_tags)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in {file_path} at line {line_num}: {e}")
                    continue
                
                # Progress indicator for large files
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num} lines, found {len(tags)} unique tags so far")
                    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    print(f"  Completed: {file_path}, found {len(tags)} total unique tags")
    return tags

def main():
    parser = argparse.ArgumentParser(description='Extract unique tags from JSONL files to create keywords.txt')
    parser.add_argument('--input', '-i', 
                        default='/external4/datasets/bucket_data/wellness/wellness_fitness_annotated.jsonl',
                        help='Input JSONL file path or pattern (supports wildcards)')
    parser.add_argument('--output', '-o', 
                        default='keywords_wellness.txt',
                        help='Output keywords file path (default: keywords_wellness.txt)')
    parser.add_argument('--recursive', '-r', 
                        action='store_true',
                        help='Search for JSONL files recursively in directories')
    
    args = parser.parse_args()
    
    # Use the specific file path
    if args.input:
        # Handle glob patterns
        jsonl_files = glob.glob(args.input, recursive=args.recursive)
        
        if not jsonl_files:
            print(f"No files found matching pattern: {args.input}")
            return
    else:
        print("Please specify input file with --input argument.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process:")
    for f in jsonl_files[:10]:  # Show first 10 files
        print(f"  {f}")
    if len(jsonl_files) > 10:
        print(f"  ... and {len(jsonl_files) - 10} more files")
    
    # Process all files and collect unique tags
    all_tags = set()
    
    for file_path in jsonl_files:
        file_tags = process_jsonl_file(file_path)
        all_tags.update(file_tags)
    
    # Sort tags for better organization
    sorted_tags = sorted(all_tags)
    
    # Write to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for tag in sorted_tags:
            f.write(f"{tag}\n")
    
    print(f"\n✅ Successfully extracted {len(sorted_tags)} unique tags!")
    print(f"📄 Tags written to: {output_path}")
    
    # Show some statistics
    tag_types = {}
    for tag in sorted_tags:
        if '_' in tag:
            prefix = tag.split('_')[0]
        else:
            prefix = 'OTHER'
        tag_types[prefix] = tag_types.get(prefix, 0) + 1
    
    print("\n📊 Tag statistics:")
    for tag_type, count in sorted(tag_types.items()):
        print(f"  {tag_type}: {count} tags")
    
    # Show sample tags
    print(f"\n🔍 Sample tags (first 20):")
    for tag in sorted_tags[:20]:
        print(f"  {tag}")
    
    if len(sorted_tags) > 20:
        print(f"  ... and {len(sorted_tags) - 20} more tags")

if __name__ == "__main__":
    main()
