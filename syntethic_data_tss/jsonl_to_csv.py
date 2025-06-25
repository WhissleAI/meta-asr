import json
import csv
import os

def jsonl_to_csv(input_jsonl_path, output_csv_path):
    """
    Convert JSONL file to CSV format with columns: audio_file|text|speaker_name|emotion_name
    """
    with open(input_jsonl_path, 'r', encoding='utf-8') as jsonl_file, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        
        writer = csv.writer(csv_file, delimiter='|')
        
        # Write header
        writer.writerow(['audio_file', 'text', 'speaker_name', 'emotion_name'])
        
        for line in jsonl_file:
            data = json.loads(line.strip())
            
            # Extract filename from the full path
            audio_filepath = data['audio_filepath']
            audio_file = os.path.basename(audio_filepath)
            
            # Extract speaker from filename (e.g., m0005 from m0005_us_m0005_00026.wav)
            speaker_name = audio_file.split('_')[0]  # Gets 'm0005' or 'f0002' etc.
            
            # Get transcription and emotion
            text = data['transcription']
            emotion_name = data['emotion']
            
            # Write row to CSV
            writer.writerow(["wavs/"+audio_file, text, speaker_name, emotion_name])

# Usage
input_file = "E:\Meta_asr\datasets\data\output.jsonl"
output_file = "E:/Meta_asr/datasets/converted_data.csv"

jsonl_to_csv(input_file, output_file)
print(f"Conversion complete! CSV saved to: {output_file}")
