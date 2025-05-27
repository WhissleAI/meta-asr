import json

def process_files(json_path, txt_path):
    """
    Read JSON file, extract audio filepath, and remove corresponding line from TXT file.
    
    Args:
        json_path (str): Path to the JSON file
        txt_path (str): Path to the TXT file
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Handle both single dictionary and list of dictionaries
    if isinstance(json_data, list):
        # If it's a list, process each item
        audio_ids = [item['audio_filepath'].split('/')[-1].replace('.flac', '') 
                    for item in json_data]
    else:
        # If it's a single dictionary
        audio_ids = [json_data['audio_filepath'].split('/')[-1].replace('.flac', '')]
    
    # Read all lines from the TXT file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out the lines that start with any of the audio IDs
    filtered_lines = []
    for line in lines:
        should_keep = True
        for audio_id in audio_ids:
            if line.startswith(audio_id):
                should_keep = False
                print(f"Removing line with ID {audio_id}")
                break
        if should_keep:
            filtered_lines.append(line)
    
    # Write the filtered lines back to the TXT file
    with open(txt_path, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"Processed {len(audio_ids)} entries and updated {txt_path}")

# Example usage
if __name__ == "__main__":
    json_path = "/external2/datasets/json_jata/spanish/train/1.json"
    txt_path = "/external2/datasets/librespeech/mls_spanish_opus/train/transcripts.txt"
    
    process_files(json_path, txt_path)