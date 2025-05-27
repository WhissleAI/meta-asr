import json
import re

def process_data(input_filepath, output_filepath, error_filepath, no_duration_filepath):
    """
    Processes a JSONL file containing audio data, cleans the 'text' field,
    and writes the cleaned data to different files based on structure.

    Args:
        input_filepath: Path to the input JSONL file.
        output_filepath: Path to the standard output JSONL file (with duration).
        error_filepath: Path to the file to store records that couldn't be processed.
        no_duration_filepath: Path to store valid records that lack duration field.
    """
    standard_count = 0
    no_duration_count = 0
    error_count = 0
    total_count = 0

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
            open(output_filepath, 'w', encoding='utf-8') as outfile, \
            open(error_filepath, 'w', encoding='utf-8') as errorfile, \
            open(no_duration_filepath, 'w', encoding='utf-8') as no_duration_file:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                data = json.loads(line)

                # Extract audio_filepath
                if 'audio_filepath' not in data:
                    raise KeyError("Missing 'audio_filepath' field")
                audio_filepath = data['audio_filepath']
                
                # Extract text
                if 'text' not in data:
                    raise KeyError("Missing 'text' field")
                
                if isinstance(data['text'], str):
                    # Case 1: text is a simple string
                    text = data['text']
                elif isinstance(data['text'], dict):
                    # Case 2: text is a dictionary
                    if 'sentence' in data['text']:
                        text = data['text']['sentence']
                    elif 'text' in data['text']:  # Handle nested text field
                        text = data['text']['text']
                    elif 'annotated_text' in data: # if it is present in the first level
                        text = data['annotated_text']
                    elif 'annotated_text' in data['text']: # if it is in nested level
                        text = data['text']['annotated_text']
                    else:
                        # Handle cases where the dictionary structure is unexpected
                        error_msg = f"Unexpected 'text' structure"
                        raise ValueError(error_msg)
                else:
                    # Handle cases where 'text' has an unexpected type
                    error_msg = f"Unexpected 'text' type: {type(data['text'])}"
                    raise ValueError(error_msg)

                # Check if duration exists
                has_duration = 'duration' in data
                
                # Create the cleaned data dict
                cleaned_data = {
                    'audio_filepath': audio_filepath,
                    'text': text
                }
                
                # Add duration if it exists
                if has_duration:
                    cleaned_data['duration'] = data['duration']
                    # Write to standard output file
                    json.dump(cleaned_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    standard_count += 1
                else:
                    # Write to the no-duration file
                    json.dump(cleaned_data, no_duration_file, ensure_ascii=False)
                    no_duration_file.write('\n')
                    no_duration_count += 1

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                error_count += 1
                error_entry = {
                    'line_number': line_num,
                    'content': line.strip(),
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                json.dump(error_entry, errorfile, ensure_ascii=False)
                errorfile.write('\n')
                print(f"Error at line {line_num}: {type(e).__name__}: {e}")

    print(f"Processing complete:")
    print(f"Total lines processed: {total_count}")
    print(f"Standard records (with duration): {standard_count}")
    print(f"Records without duration: {no_duration_count}")
    print(f"Error records: {error_count}")
    print(f"Success rate: {(standard_count + no_duration_count)/total_count*100:.2f}%")
    print(f"Standard data saved to: {output_filepath}")
    print(f"No-duration data saved to: {no_duration_filepath}")
    print(f"Error data saved to: {error_filepath}")

# --- Example Usage ---
input_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/salvic/bel.jsonl'  # Replace with your input file
output_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/salvic/belarussian.jsonl'  # For records with duration
no_duration_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/salvic/belarussian_no_duration.jsonl'  # For records without duration
error_file = '/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/salvic/belarussian_error.jsonl'  # For error records

process_data(input_file, output_file, error_file, no_duration_file)