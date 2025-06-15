import json
import os

def process_audio_data(input_file, output_file, no_annotation_file, error_file):
    """
    Process audio data JSON files:
    1. For entries with annotated_sentence: Extract only audio_filepath, annotated_sentence, and duration
    2. For entries without annotated_sentence: Keep the entire original structure
    3. For entries that cause JSON parsing errors: Store in a separate error file
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file for simplified entries with annotations
        no_annotation_file (str): Path to output JSON file for original entries without annotations
        error_file (str): Path to output JSON file for lines that couldn't be parsed
    """
    with_annotations = []
    without_annotations = []
    error_lines = []
    
    # Read input file line by line (assuming JSON Lines format)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                data = json.loads(line)
                
                # Check if annotated_sentence exists
                has_annotation = False
                annotated_sentence = None
                if "text" in data and isinstance(data["text"], dict):
                    if "annotated_sentence" in data["text"] and data["text"]["annotated_sentence"]:
                        has_annotation = True
                        annotated_sentence = data["text"]["annotated_sentence"]
                
                if has_annotation:
                    # For entries with annotation, create simplified structure
                    simplified_entry = {
                        "audio_filepath": data.get("audio_filepath", ""),
                        "annotated_sentence": annotated_sentence,
                        "duration": data.get("duration", 0)
                    }
                    with_annotations.append(simplified_entry)
                else:
                    # For entries without annotation, keep original structure
                    without_annotations.append(data)
                    
            except json.JSONDecodeError as e:
                # Store error information
                error_entry = {
                    "line_number": line_number,
                    "error_message": str(e),
                    "line_content": line[:500] + ("..." if len(line) > 500 else "")  # Store first 500 chars
                }
                error_lines.append(error_entry)
                print(f"Warning: Could not parse line {line_number}: {line[:50]}... - Error: {e}")
    
    # Write entries with annotations to output file (simplified format)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in with_annotations:
            f.write(json.dumps(entry) + '\n')
    
    # Write entries without annotations to separate file (original format)
    with open(no_annotation_file, 'w', encoding='utf-8') as f:
        for entry in without_annotations:
            f.write(json.dumps(entry) + '\n')
    
    # Write error entries to error file
    with open(error_file, 'w', encoding='utf-8') as f:
        for entry in error_lines:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(with_annotations)} entries with annotations (simplified)")
    print(f"Processed {len(without_annotations)} entries without annotations (original structure)")
    print(f"Found {len(error_lines)} lines with JSON parsing errors")

if __name__ == "__main__":
    input_file = "/external4/datasets/jsonl_data/people_speech_processed.jsonl"
    output_file = "/external4/datasets/jsonl_data/annotated_data.jsonl"
    no_annotation_file = "/external4/datasets/jsonl_data/no_annotation_data.jsonl"
    error_file = "/external4/datasets/jsonl_data/error_lines.jsonl"
    
    process_audio_data(input_file, output_file, no_annotation_file, error_file)