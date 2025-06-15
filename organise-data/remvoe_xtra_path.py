import json
import re

def clean_filepath(filepath):
    """Clean the filepath by removing redundant part after audio_chunks/ up to the next /"""
    # Find the part after audio_chunks/ up to the next /
    pattern = r'audio_chunks/[^/]+/'
    match = re.search(pattern, filepath)
    if match:
        # Replace with just audio_chunks/
        return filepath.replace(match.group(0), 'audio_chunks/')
    return filepath

def process_jsonl(input_file, output_file):
    """Process JSONL file and clean audio filepaths"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                if 'audio_filepath' in data:
                    # Clean the audio filepath
                    data['audio_filepath'] = clean_filepath(data['audio_filepath'])
                # Write the modified JSON line
                f_out.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

# Example usage
if __name__ == "__main__":
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/all-ytdata/merged_data_yt.jsonl"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/all-ytdata/merged_data_clean.jsonl"
    process_jsonl(input_file, output_file)