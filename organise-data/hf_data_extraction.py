import json
import os

def extract_transcriptions(jsonl_file_path, transcription_txt_path, output_jsonl_path):
    """
    Extract audio file paths from JSONL and find their transcriptions in a text file.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file containing audio metadata
        transcription_txt_path (str): Path to the text file with clean transcriptions
        output_jsonl_path (str): Path to save the output with matched transcriptions
    """
    # Read the transcription.txt file into memory
    with open(transcription_txt_path, 'r') as f:
        txt_content = f.readlines()
    
    # Create a dictionary mapping paths to transcriptions
    path_to_transcription = {}
    for line in txt_content:
        if ' | ' in line:
            path, transcription = line.strip().split(' | ', 1)
            path_to_transcription[path] = transcription
    
    # Process the JSONL file and extract relevant information
    extracted_data = []
    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line.strip())
            
            # Extract the audio file path
            audio_path = data.get('audio_file_path', '')
            
            # Find the matching transcription
            clean_transcription = path_to_transcription.get(audio_path, '')
            
            # Create a simplified entry with just the path and transcription
            entry = {
                'audio_file_path': audio_path,
                'clean_transcription': clean_transcription
            }
            
            extracted_data.append(entry)
    
    # Write the extracted data to a new JSONL file
    with open(output_jsonl_path, 'w') as output_file:
        for entry in extracted_data:
            output_file.write(json.dumps(entry) + '\n')
    
    print(f"Extraction complete. Found {len(extracted_data)} entries. Output saved to {output_jsonl_path}")

# Example usage
if __name__ == "__main__":

    
    

# Example usage

    # Replace these paths with your actual file paths
    jsonl_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/comp_hf_data.jsonl"
    transcription_txt = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/transcriptions_hf_data.txt"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/cleaned_hf_data.jsonl"
    
    extract_transcriptions(jsonl_file, transcription_txt, output_file)