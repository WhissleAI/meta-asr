import json
import re
import os
import argparse

def examine_file(file_path, num_lines=5):
    """
    Examine the first few lines of a file to understand its format.
    
    Args:
        file_path (str): Path to the file to examine
        num_lines (int): Number of lines to print
    """
    print(f"Examining first {num_lines} lines of {file_path}:")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if i > num_lines:
                    break
                print(f"Line {i}: {line[:100]}..." if len(line) > 100 else f"Line {i}: {line}")
                try:
                    json.loads(line.strip())
                    print("  ✓ Valid JSON")
                except json.JSONDecodeError as e:
                    print(f"  ✗ Invalid JSON: {str(e)}")
    except Exception as e:
        print(f"Error opening file: {str(e)}")

def attempt_fix_json(line):
    """
    Attempt to fix common JSON errors.
    
    Args:
        line (str): The problematic JSON line
        
    Returns:
        tuple: (fixed_json_str, success_bool)
    """
    # Try to fix common issues
    fixed_line = line.strip()
    
    # 1. Fix unclosed quotes
    quote_count = fixed_line.count('"')
    if quote_count % 2 == 1:
        fixed_line += '"'
    
    # 2. Fix missing closing braces
    open_braces = fixed_line.count('{')
    close_braces = fixed_line.count('}')
    if open_braces > close_braces:
        fixed_line += '}' * (open_braces - close_braces)
    
    # 3. Fix trailing commas in objects
    fixed_line = re.sub(r',\s*}', '}', fixed_line)
    
    # Try parsing the fixed line
    try:
        json.loads(fixed_line)
        return fixed_line, True
    except json.JSONDecodeError:
        return line, False

def clean_repetitions(input_file, output_file, fix_json=False, skip_errors=False):
    """
    Clean repetitive words in JSONL data text fields.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        fix_json (bool): Whether to attempt to fix JSON errors
        skip_errors (bool): Whether to skip lines with errors
    """
    # Pattern to match consecutive repetitions of the same word
    pattern = r'\b(\w+)(\s+\1){2,}\b'
    
    # Pattern to identify metadata tags we want to preserve
    metadata_pattern = r'(AGE_\w+|GER_\w+|EMOTION_\w+|INTENT_\w+)'
    
    error_count = 0
    line_count = 0
    fixed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(output_file + ".errors", 'w', encoding='utf-8') as errorfile:
        
        for line_num, line in enumerate(infile, 1):
            line_count += 1
            processed = False
            
            try:
                # Parse the JSON object
                data = json.loads(line.strip())
                processed = True
                
                if 'text' in data:
                    text = data['text']
                    
                    # Extract metadata tags to preserve them
                    metadata_tags = re.findall(metadata_pattern, text)
                    
                    # Replace repetitive words with a single instance
                    while re.search(pattern, text):
                        text = re.sub(pattern, r'\1', text)
                    
                    # Make sure metadata tags are preserved and only appear once
                    # Remove any existing tags first
                    for tag in set(metadata_tags):
                        text = re.sub(tag + r'\s+', '', text)
                    
                    # Append metadata tags at the end
                    text = re.sub(r'\s+', ' ', text).strip()
                    if metadata_tags:
                        text += " " + " ".join(set(metadata_tags))
                    
                    data['text'] = text
                
                # Write the cleaned data to the output file
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Error parsing JSON at line {line_num}: {str(e)}")
                print(f"Problematic line: {line[:80]}..." if len(line) > 80 else f"Problematic line: {line}")
                
                if fix_json:
                    fixed_line, success = attempt_fix_json(line)
                    if success:
                        fixed_count += 1
                        try:
                            data = json.loads(fixed_line)
                            if 'text' in data:
                                # Apply the same cleaning as for valid lines
                                text = data['text']
                                metadata_tags = re.findall(metadata_pattern, text)
                                while re.search(pattern, text):
                                    text = re.sub(pattern, r'\1', text)
                                for tag in set(metadata_tags):
                                    text = re.sub(tag + r'\s+', '', text)
                                text = re.sub(r'\s+', ' ', text).strip()
                                if metadata_tags:
                                    text += " " + " ".join(set(metadata_tags))
                                data['text'] = text
                            
                            outfile.write(json.dumps(data) + '\n')
                            processed = True
                            print(f"Fixed line {line_num}")
                        except Exception as ex:
                            print(f"Error processing fixed line: {str(ex)}")
                
                if not processed:
                    errorfile.write(f"Line {line_num}: {line}")
                    if skip_errors:
                        continue
                    else:
                        raise Exception(f"Stopped processing at line {line_num} due to JSON error. Use --skip-errors to continue.")
    
    print(f"Processed {line_count} lines with {error_count} errors")
    if fix_json:
        print(f"Fixed {fixed_count} of {error_count} errors")
    print(f"Processed file saved to {output_file}")
    if error_count > 0:
        print(f"Error lines saved to {output_file}.errors")

def process_directory(input_dir, output_dir, fix_json=False, skip_errors=False):
    """
    Process all JSONL files in a directory.
    
    Args:
        input_dir (str): Input directory containing JSONL files
        output_dir (str): Output directory for cleaned files
        fix_json (bool): Whether to attempt to fix JSON errors
        skip_errors (bool): Whether to skip lines with errors
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
                clean_repetitions(input_path, output_path, fix_json=fix_json, skip_errors=skip_errors)
                processed_files += 1
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    
    print(f"\nSummary: Processed {processed_files} of {total_files} files")

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
    print(f"Filtered out {filtered_count} lines with Unicode characters")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean repetitive words in JSONL data text fields.")
    parser.add_argument('--input', '-i', type=str, help="Input JSONL file path")
    parser.add_argument('--output', '-o', type=str, help="Output JSONL file path")
    parser.add_argument('--examine', '-e', action='store_true', help="Examine the input file before processing")
    parser.add_argument('--fix-json', '-f', action='store_true', help="Attempt to fix JSON errors")
    parser.add_argument('--skip-errors', '-s', action='store_true', help="Skip lines with JSON errors")
    parser.add_argument('--dir', '-d', action='store_true', help="Process a directory of JSONL files")
    parser.add_argument('--input-dir', type=str, help="Input directory containing JSONL files")
    parser.add_argument('--output-dir', type=str, help="Output directory for cleaned files")
    parser.add_argument('--filter-unicode', '-u', action='store_true', help="Filter out lines with Unicode characters")
    
    args = parser.parse_args()
    
    if args.dir:
        if args.input_dir and args.output_dir:
            process_directory(args.input_dir, args.output_dir, fix_json=args.fix_json, skip_errors=args.skip_errors)
        else:
            print("Error: Both --input-dir and --output-dir are required when using --dir")
    else:
        input_file = args.input if args.input else "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/uni_vils_data.jsonl"
        output_file = args.output if args.output else "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/cleaned_vils_data.jsonl"
        
        # Check if file exists
        if os.path.exists(input_file):
            if args.examine:
                examine_file(input_file)
                response = input("Do you want to proceed with cleaning the file? (y/n): ")
                if response.lower() != 'y':
                    print("Operation cancelled.")
                    exit(0)
            
            if args.filter_unicode:
                filter_unicode_json(input_file, output_file)
            else:
                clean_repetitions(input_file, output_file, fix_json=args.fix_json, skip_errors=args.skip_errors)
        else:
            print(f"File {input_file} not found.")