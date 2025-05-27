import pandas as pd
import soundfile as sf
import os
import io
import json
from pathlib import Path
import argparse

def process_parquet_files(input_dir, output_dir, batch_size=10, start_batch=0, processed_log_file="processed_files.txt"):
    """
    Process parquet files in batches, extracting audio and all metadata.
    
    Args:
        input_dir (str): Directory containing parquet files
        output_dir (str): Base directory for output files
        batch_size (int): Number of parquet files to process in one run
        start_batch (int): Which batch to start from (0-based)
        processed_log_file (str): File to track processed parquet files
    """
    # Create output directories
    audio_dir = os.path.join(output_dir, "audio")
    text_dir = os.path.join(output_dir, "text")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load list of previously processed files
    processed_files = set()
    log_path = os.path.join(output_dir, processed_log_file)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            processed_files = set(f.read().splitlines())
    
    # Get all unprocessed parquet files
    all_parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    unprocessed_files = [f for f in all_parquet_files 
                        if os.path.join(input_dir, f) not in processed_files]
    
    # Calculate batch information
    total_files = len(unprocessed_files)
    total_batches = (total_files + batch_size - 1) // batch_size
    
    if start_batch >= total_batches:
        print(f"No more batches to process. Total batches: {total_batches}")
        return
    
    # Get current batch of files
    batch_start = start_batch * batch_size
    batch_end = min(batch_start + batch_size, total_files)
    current_batch = unprocessed_files[batch_start:batch_end]
    
    print(f"Processing batch {start_batch + 1} of {total_batches}")
    print(f"Files in this batch: {len(current_batch)}")
    
    # Process each parquet file in the batch
    for parquet_file in current_batch:
        full_path = os.path.join(input_dir, parquet_file)
        print(f"\nProcessing: {parquet_file}")
        
        try:
            # Load parquet file
            df = pd.read_parquet(full_path)
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Check if the audio field exists
                    if 'audio' not in row:
                        print(f"Error processing row {index} in {parquet_file}: 'audio' field not found")
                        continue
                    
                    # Get audio bytes from the 'audio' field, handling different data types
                    audio_data = row["audio"]
                    
                    # Handle case where audio is a dictionary (could have different formats)
                    if isinstance(audio_data, dict):
                        if 'bytes' in audio_data:
                            audio_bytes = audio_data['bytes']
                        else:
                            print(f"Error processing row {index} in {parquet_file}: audio dictionary does not contain 'bytes' key")
                            continue
                    else:
                        # Assume it's directly bytes
                        audio_bytes = audio_data
                    
                    audio_io = io.BytesIO(audio_bytes)
                    
                    # Define file paths using safer naming
                    base_name = f"{Path(parquet_file).stem}_{index}"
                    audio_path = os.path.join(audio_dir, f"{base_name}.flac")
                    text_path = os.path.join(text_dir, f"{base_name}.txt")
                    metadata_path = os.path.join(metadata_dir, f"{base_name}.json")
                    
                    # Save audio
                    with sf.SoundFile(audio_io) as f:
                        data = f.read(dtype="int16")
                        sf.write(audio_path, data, f.samplerate)
                    
                    # Save text
                    with open(text_path, "w", encoding="utf-8") as text_file:
                        text_file.write(row["transcript"])
                    
                    # Extract and save all metadata
                    metadata = {}
                    for col in row.index:
                        if col != 'audio':  # Skip audio bytes
                            try:
                                # Convert various data types to JSON-serializable format
                                value = row[col]
                                if isinstance(value, (str, int, float, bool)) or value is None:
                                    metadata[col] = value
                                else:
                                    # For non-standard types, convert to string representation
                                    metadata[col] = str(value)
                            except Exception as e:
                                print(f"Warning: Could not process field {col}: {str(e)}")
                                metadata[col] = None
                    
                    # Save metadata as JSON
                    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
                    
                    print(f"Saved: {base_name}")
                    
                except Exception as e:
                    print(f"Error processing row {index} in {parquet_file}: {str(e)}")
                    continue
            
            # Mark file as processed
            with open(log_path, 'a') as f:
                f.write(f"{full_path}\n")
            
        except Exception as e:
            print(f"Error processing file {parquet_file}: {str(e)}")
            continue
    
    print(f"\nBatch {start_batch + 1} complete!")
    print(f"Processed {len(current_batch)} files")
    print(f"Next batch number to process: {start_batch + 1}")
    print(f"Remaining batches: {total_batches - (start_batch + 1)}")

if __name__ == "__main__":

    input_dir = "/external4/datasets/madasr/Vaani-transcription-part/Hindi"
    output_dir = "/external4/datasets/madasr/Vaani-transcription-part/Hindi"
    batch_size = 100
    batch_number = 0
    process_parquet_files(
        input_dir,
        output_dir,
        batch_size,
        batch_number
    )