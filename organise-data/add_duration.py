import json
from pydub import AudioSegment
import os
from pathlib import Path
import subprocess
import tempfile

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    Supports multiple audio formats including OPUS.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    # Get file extension
    extension = Path(file_path).suffix.lower()
    
    try:
        # Special handling for opus files
        if extension == '.opus':
            # Use ffprobe directly to get duration for opus files
            cmd = [
                'ffprobe', 
                '-i', file_path, 
                '-show_entries', 'format=duration', 
                '-v', 'quiet', 
                '-of', 'csv=p=0'
            ]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            try:
                duration = float(output)
                return round(duration, 2)
            except ValueError:
                print(f"Could not parse duration from ffprobe output: {output}")
                return None
        
        # For other standard formats, use pydub as before
        elif extension in ['.mp3', '.wav', '.ogg', '.flac']:
            if extension == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif extension == '.wav':
                audio = AudioSegment.from_wav(file_path)
            elif extension == '.ogg':
                audio = AudioSegment.from_ogg(file_path)
            elif extension == '.flac':
                audio = AudioSegment.from_file(file_path, format="flac")
        
        # Handle M4A, MP4, and WEBM files
        elif extension in ['.m4a', '.mp4', '.webm']:
            audio = AudioSegment.from_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # For pydub processed files
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        return round(duration, 2)
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Alternative approach for opus: temporarily convert to wav
        if extension == '.opus':
            try:
                print(f"Trying alternative approach for {file_path}...")
                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                    # Convert opus to wav using ffmpeg
                    cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-y',  # Overwrite output file if it exists
                        temp_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Get duration from the converted wav file
                    audio = AudioSegment.from_wav(temp_path)
                    duration = len(audio) / 1000.0
                    return round(duration, 2)
            except Exception as e2:
                print(f"Alternative approach also failed for {file_path}: {str(e2)}")
        
        return None

def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file containing audio file paths, read their durations,
    and create a new JSONL file with duration information added.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    # Track processing statistics
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            total_files += 1
            
            # Parse the JSON line
            data = json.loads(line.strip())
            audio_path = data['audio_filepath']
            
            # Get duration
            duration = get_audio_duration(audio_path)
            
            if duration is not None:
                data['duration'] = duration
                processed_files += 1
            else:
                failed_files += 1
            
            # Write the updated data
            f_out.write(json.dumps(data) + '\n')
            
            # Print progress every 100 files
            if total_files % 100 == 0:
                print(f"Processed {total_files} files...")
    
    # Print final statistics
    print(f"\nProcessing complete:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")


if __name__ == "__main__":
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/vils_data_n.jsonl"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/vils_data.jsonl"
    process_jsonl_file(input_file, output_file)