import os
import json
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_audio_files_from_json(json_file):
    """Extract audio file paths from a JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Handle both single object and list of objects
        if isinstance(data, dict):
            data = [data]
            
        audio_files = []
        for item in data:
            if 'audio_filepath' in item:
                # Extract just the filename from the full path
                audio_path = item['audio_filepath']
                audio_filename = os.path.basename(audio_path)
                audio_files.append(audio_filename)
                
        return audio_files
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {json_file}")
        return []
    except Exception as e:
        logging.error(f"Error processing {json_file}: {str(e)}")
        return []

def move_audio_files(json_dir, audio_dir, output_dir):
    """
    Move audio files referenced in JSON files to output directory
    
    Args:
        json_dir (str): Directory containing JSON files
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory where audio files will be moved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Keep track of processed files
    processed_files = set()
    moved_count = 0
    error_count = 0
    
    # Process each JSON file
    for json_file in Path(json_dir).glob('*.json'):
        logging.info(f"Processing JSON file: {json_file.name}")
        
        # Get list of audio files referenced in this JSON
        audio_files = get_audio_files_from_json(json_file)
        
        # Move each referenced audio file
        for audio_filename in audio_files:
            if audio_filename in processed_files:
                continue
                
            source_path = Path(audio_dir) / audio_filename
            dest_path = Path(output_dir) / audio_filename
            
            try:
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    processed_files.add(audio_filename)
                    moved_count += 1
                    logging.info(f"Moved: {audio_filename}")
                else:
                    logging.warning(f"Audio file not found: {audio_filename}")
                    error_count += 1
            except Exception as e:
                logging.error(f"Error moving {audio_filename}: {str(e)}")
                error_count += 1
    
    # Print summary
    logging.info(f"\nSummary:")
    logging.info(f"Total files moved: {moved_count}")
    logging.info(f"Errors encountered: {error_count}")

def main():
    """Main function to run the script"""
    setup_logging()
    
    # Get directory paths from user
    json_dir = "output1/data"
    audio_dir = "output1/audio_chunks_unprocessed"
    output_dir = "output1/audio_data"
    
    # Validate directories
    if not all(Path(p).exists() for p in [json_dir, audio_dir]):
        logging.error("One or more input directories do not exist!")
        return
    
    # Process files
    move_audio_files(json_dir, audio_dir, output_dir)

if __name__ == "__main__":
    main()


import os
import shutil
from pathlib import Path

def find_unprocessed_chunks(processed_dir, all_chunks_dir, output_dir):
    """
    Identifies and moves unprocessed audio chunks to a new directory.
    
    Args:
        processed_dir (str): Directory containing processed audio chunks
        all_chunks_dir (str): Directory containing all audio chunks
        output_dir (str): Directory where unprocessed chunks will be moved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get set of processed audio filenames
    processed_files = set()
    for filename in os.listdir(processed_dir):
        if filename.endswith(('.wav', '.mp3', '.ogg')):  # Add more audio extensions if needed
            processed_files.add(filename)
    
    # Find and move unprocessed chunks
    unprocessed_count = 0
    for filename in os.listdir(all_chunks_dir):
        if filename.endswith(('.wav', '.mp3', '.ogg')):
            if filename not in processed_files:
                source_path = os.path.join(all_chunks_dir, filename)
                dest_path = os.path.join(output_dir, filename)
                shutil.move(source_path, dest_path)
                unprocessed_count += 1
    
    return unprocessed_count

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual directory paths
    PROCESSED_DIR = "output2/audio_data"
    ALL_CHUNKS_DIR = "output2/audio_chunks"
    UNPROCESSED_OUTPUT_DIR = "output2/audio_chunks_unprocessed"
    
    try:
        num_moved = find_unprocessed_chunks(PROCESSED_DIR, ALL_CHUNKS_DIR, UNPROCESSED_OUTPUT_DIR)
        print(f"Successfully moved {num_moved} unprocessed audio chunks to {UNPROCESSED_OUTPUT_DIR}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")