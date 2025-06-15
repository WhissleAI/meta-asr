import json
import sys
from pathlib import Path

def modify_jsonl_paths(input_jsonl, output_jsonl, prefix_path):
    """
    Read JSONL file, modify audio_filepath by adding prefix, and save to new file.
    Also logs problematic lines to a separate file.
    Handles different file encodings to prevent UnicodeDecodeError.
    
    Args:
        input_jsonl (str): Path to input JSONL file
        output_jsonl (str): Path to output JSONL file
        prefix_path (str): Path to prepend to audio_filepath
    """
    # Ensure prefix path ends with forward slash
    if not prefix_path.endswith('/'):
        prefix_path += '/'
    
    # Create output directory if it doesn't exist
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    # Create error log file path
    error_log = Path(output_jsonl).parent / 'error_lines.log'
    
    lines_processed = 0
    errors = 0
    
    try:
        with open(input_jsonl, 'rb') as fin, \
             open(output_jsonl, 'w', encoding='utf-8') as fout, \
             open(error_log, 'w', encoding='utf-8') as error_file:
            
            # Read all lines at once in binary mode
            all_lines_bytes = fin.readlines()
            total_lines = len(all_lines_bytes)
            
            for line_number, line_bytes in enumerate(all_lines_bytes, 1):
                try:
                    # Skip empty lines
                    if not line_bytes.strip():
                        continue
                    
                    # Try to decode with different encodings
                    try:
                        line = line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        line = line_bytes.decode('latin-1')  # Fallback encoding
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to parse the line
                    data = json.loads(line)
                    
                    # Modify the audio_filepath
                    if 'audio_filepath' in data:
                        data['audio_filepath'] = prefix_path + data['audio_filepath']
                        
                        # Write modified JSON object to new file
                        json.dump(data, fout, ensure_ascii=False)
                        fout.write('\n')
                        lines_processed += 1
                    else:
                        error_msg = f"Line {line_number}: Missing 'audio_filepath' field"
                        print(error_msg)
                        error_file.write(f"{error_msg}\nLine content: {line}\n\n")
                        errors += 1
                        
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_number}: JSON parsing error - {str(e)}"
                    print(error_msg)
                    error_file.write(f"{error_msg}\nLine content: {line}\n\n")
                    errors += 1
                    continue
                except Exception as e:
                    error_msg = f"Line {line_number}: Unexpected error - {str(e)}"
                    print(error_msg)
                    error_file.write(f"{error_msg}\n\n")
                    errors += 1
                    continue
                
                # Print progress every 10000 lines
                if line_number % 10000 == 0:
                    print(f"Processed {line_number}/{total_lines} lines...")
                
    except FileNotFoundError:
        print(f"Error: Input file '{input_jsonl}' not found")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied while accessing files")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
    
    print(f"\nProcessing complete:")
    print(f"- Total lines in input file: {total_lines}")
    print(f"- Lines processed successfully: {lines_processed}")
    print(f"- Errors encountered: {errors}")
    print(f"- Modified JSONL file saved to: {output_jsonl}")
    print(f"- Error log saved to: {error_log}")

# Example usage
if __name__ == "__main__":
    input_jsonl = "/external4/datasets/MADASR/process_train_large/gemini_process/bhojpuri_output.jsonl"
    output_jsonl = "/external4/datasets/MADASR/process_train_large/gemini_process/bhojpuri_path_fixed.jsonl"
    prefix_path = "/external4/datasets/MADASR/IISc_RESPIN_dev/"
    
    modify_jsonl_paths(input_jsonl, output_jsonl, prefix_path)