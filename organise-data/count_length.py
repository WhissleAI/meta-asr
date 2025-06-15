# import librosa
# import os

# directory = "/external4/datasets/madasr/Vaani-transcription-part/Hindi/audio"

# total_duration = 0

# for file in os.listdir(directory):
#     if file.endswith(".flac"):  # Change extension if needed
#         filepath = os.path.join(directory, file)
#         y, sr = librosa.load(filepath, sr=None)
#         duration = librosa.get_duration(y=y, sr=sr)
#         total_duration += duration

# print(f"Total Duration: {total_duration/3600:.2f} seconds")
import json

def calculate_total_duration(jsonl_file_path):
    total_duration = 0.0
    file_count = 0
    
    try:
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    # Parse each line as JSON
                    data = json.loads(line.strip())
                    
                    # Add the duration if it exists
                    if 'duration' in data:
                        total_duration += float(data['duration'])
                        file_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
                except (KeyError, ValueError):
                    print(f"Warning: Could not extract duration from line: {line[:50]}...")
    
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_file_path}")
        return 0, 0
    
    # Convert total seconds to hours, minutes, seconds format
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60
    
    return {
        'total_seconds': total_duration,
        'formatted_time': f"{hours:02d}:{minutes:02d}:{seconds:.2f}",
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'file_count': file_count
    }

# Use this function with your JSONL file
jsonl_file_path = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/nptel-test/nptel_english_annotated.jsonl"
 # Replace with your actual file path
result = calculate_total_duration(jsonl_file_path)

print(f"Total audio duration: {result['formatted_time']} (or {result['total_seconds']:.2f} seconds)")
print(f"That's approximately {result['hours']} hours, {result['minutes']} minutes, and {result['seconds']:.2f} seconds")
print(f"Processed {result['file_count']} audio files")