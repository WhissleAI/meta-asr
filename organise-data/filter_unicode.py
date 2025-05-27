import json
import os
import argparse
import sys

def filter_unicode_json(input_file, output_file):
    """
    Filter out JSON lines that contain non-ASCII Unicode characters in the text field.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    line_count = 0
    filtered_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            line_count += 1
            try:
                # Parse the JSON object
                data = json.loads(line.strip())
                
                # Check if the text field contains Unicode characters
                if 'text' in data and not data['text'].isascii():
                    filtered_count += 1
                    continue  # Skip this line
                
                # Write the valid line to the output file
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {str(e)}")
                print(f"Problematic line: {line[:80]}..." if len(line) > 80 else f"Problematic line: {line}")
    
    print(f"Processed {line_count} lines")
    print(f"Filtered out {filtered_count} lines with Unicode characters ({filtered_count/line_count*100:.2f}%)")
    print(f"Output saved to {output_file}")

def filter_directory(input_dir, output_dir):
    """
    Process all JSONL files in a directory and filter Unicode lines.
    
    Args:
        input_dir (str): Input directory containing JSONL files
        output_dir (str): Output directory for filtered files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_files = 0
    processed_files = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            total_files += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"\nProcessing file {input_path}...")
            try:
                filter_unicode_json(input_path, output_path)
                processed_files += 1
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    
    print(f"\nSummary: Processed {processed_files} of {total_files} files")

def main():
    input_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/uni_vils_data.jsonl'
    output_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/uni_vils_data_filtered.jsonl'
    filter_unicode_json(input_file, output_file)

if __name__ == "__main__":
    main() 