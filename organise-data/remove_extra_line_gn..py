import json
from pathlib import Path

def clean_complex_data(input_file, output_file, error_file):
    """
    Cleans complex JSONL data by extracting audio_filepath, text and duration fields.
    Handles extra data errors and unusual JSON structures.
    
    Args:
        input_file (str): Path to input complex JSONL file
        output_file (str): Path to output cleaned JSONL file
        error_file (str): Path to output errors JSONL file
    """
    stats = {
        'total_processed': 0,
        'successfully_cleaned': 0,
        'errors': 0,
        'structure_fixed': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(error_file, 'w', encoding='utf-8') as f_err:
        
        for line_number, line in enumerate(f_in, 1):
            stats['total_processed'] += 1
            line = line.strip()
            
            try:
                # Special handling for the problematic format you identified
                # Check if the line has this pattern: {...}, "duration": X.XX}
                if '}, "duration":' in line:
                    # Fix the structure by moving the duration inside
                    fixed_line = line.replace('}, "duration":', ', "duration":')
                    try:
                        record = json.loads(fixed_line)
                        stats['structure_fixed'] += 1
                    except json.JSONDecodeError:
                        # If that didn't work, try another approach
                        try:
                            # Try to extract parts manually
                            main_part = line.split('}, "duration":')[0] + '}'
                            duration_part = line.split('}, "duration":')[1].strip('}')
                            
                            # Parse the main part
                            record = json.loads(main_part)
                            # Add duration manually
                            record['duration'] = float(duration_part.strip())
                            stats['structure_fixed'] += 1
                        except Exception as e:
                            raise json.JSONDecodeError(f"Failed to fix structure: {str(e)}", line, 0)
                else:
                    # Regular parsing
                    record = json.loads(line)
                
                # Try to get duration from different possible locations
                duration = 0
                if 'duration' in record:
                    duration = record['duration']
                
                # Extract fields
                cleaned_record = {
                    'audio_filepath': record['audio_filepath'],
                    'text': record['text'],
                    'duration': duration
                }
                
                # Write cleaned record
                json.dump(cleaned_record, f_out, ensure_ascii=False)
                f_out.write('\n')
                stats['successfully_cleaned'] += 1
                
            except json.JSONDecodeError as e:
                # Handle JSON decode errors
                stats['errors'] += 1
                error_info = {
                    'line_number': line_number,
                    'error': str(e),
                    'snippet': line[:250] + "..." if len(line) > 250 else line
                }
                json.dump(error_info, f_err, ensure_ascii=False)
                f_err.write('\n')
                print(f"Error processing line {line_number}: {str(e)}")
            
            except KeyError as e:
                # Handle missing required keys
                stats['errors'] += 1
                error_info = {
                    'line_number': line_number,
                    'error': f"Missing required key: {str(e)}",
                    'snippet': line[:250] + "..." if len(line) > 250 else line
                }
                json.dump(error_info, f_err, ensure_ascii=False)
                f_err.write('\n')
                print(f"Error processing line {line_number}: Missing required key {str(e)}")
    
    # Print statistics
    print("\nCleaning Statistics:")
    print(f"Total lines processed: {stats['total_processed']}")
    print(f"Successfully cleaned: {stats['successfully_cleaned']}")
    print(f"Structure issues fixed: {stats['structure_fixed']}")
    print(f"Errors encountered: {stats['errors']}")

def main():
    # File paths
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/portetuge.jsonl"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/1_gem.jsonl"
    error_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/parsing_errors.jsonl"
    
    # Ensure input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(error_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    clean_complex_data(input_file, output_file, error_file)
    print(f"\nCleaning complete. Cleaned data written to: {output_file}")
    print(f"Error information written to: {error_file}")

if __name__ == '__main__':
    main()