import json
import os
import shutil
from pathlib import Path

def read_jsonl_file(jsonl_path):
    """
    Read a JSONL file and return the data, trying multiple encodings
    
    Args:
        jsonl_path (str): Path to JSONL file
        
    Returns:
        list: List of dictionaries containing the JSONL data
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    data = []
    
    for encoding in encodings:
        try:
            with open(jsonl_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {line[:100]}...")
                            print(f"Error message: {str(e)}")
                            continue
                if data:  # If we successfully read data, break the loop
                    print(f"Successfully read file using {encoding} encoding")
                    return data
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            print(f"Error reading file {jsonl_path}: {str(e)}")
            continue
            
    print("Failed to read file with any encoding")
    return []

def move_audio_files(input_jsonl, destination_folder, create_new_json=True):
    """
    Move audio files listed in a JSONL file to a new destination folder
    and update the file paths in the JSONL.
    
    Args:
        input_jsonl (str): Path to input JSONL file
        destination_folder (str): Path to destination folder
        create_new_json (bool): Whether to create a new JSONL file with updated paths
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Prepare output JSONL path
    input_path = Path(input_jsonl)
    output_jsonl = input_path.parent / f"updated_{input_path.name}"
    
    # Track statistics
    total_files = 0
    moved_files = 0
    failed_files = 0
    
    # Read the JSONL file
    print("Reading JSONL file...")
    data_lines = read_jsonl_file(input_jsonl)
    
    if not data_lines:
        print("No valid data found in the JSONL file.")
        return
    
    # Process each entry
    print("Starting to process files...")
    updated_data = []
    
    for data in data_lines:
        total_files += 1
        try:
            # Get the audio file path
            audio_path = Path(data['audio_filepath'])
            
            # Create new destination path
            new_path = Path(destination_folder) / audio_path.name
            
            # Move the file if it exists
            if audio_path.exists():
                shutil.move(str(audio_path), str(new_path))
                # Update the path in the data
                data['audio_filepath'] = str(new_path)
                moved_files += 1
                print(f"Moved: {audio_path.name}")
            else:
                print(f"File not found: {audio_path}")
                failed_files += 1
            
            updated_data.append(data)
            
        except Exception as e:
            print(f"Error processing entry {total_files}: {str(e)}")
            failed_files += 1
            updated_data.append(data)  # Keep original data in case of error
        
        # Print progress every 100 files
        if total_files % 100 == 0:
            print(f"Processed {total_files} files...")
    
    # Write updated JSONL
    if create_new_json and updated_data:
        try:
            with open(output_jsonl, 'w') as f_out:
                for data in updated_data:
                    f_out.write(json.dumps(data) + '\n')
            print(f"Created updated JSONL file: {output_jsonl}")
        except Exception as e:
            print(f"Error writing output file: {str(e)}")
    
    # Print final statistics
    print(f"\nProcessing complete:")
    print(f"Total entries processed: {total_files}")
    print(f"Successfully moved: {moved_files}")
    print(f"Failed: {failed_files}")

# Example usage
if __name__ == "__main__":
    input_jsonl = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/shrutilipi/tamil_uni.jsonl"  # Note the .jsonl extension
    destination_folder = "/external4/datasets/Shrutilipi/tamil/tm/audio1"
    move_audio_files(input_jsonl, destination_folder)