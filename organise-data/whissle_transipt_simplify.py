import json

def extract_transcripts(input_file, output_file):
    """
    Extract file paths and transcripts from a complex JSON file and
    create a new JSON file with a simplified format containing only paths and transcripts.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file to be created
    """
    # Load the original JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a new dictionary with only file paths and transcripts
    simplified_data = {}
    for file_path, file_info in data.items():
        simplified_data[file_path] = file_info.get('transcript', '')
    
    # Write the simplified data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    print(f"Extracted transcripts saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/whissle_python_api/transcription_results.json"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/whissle_python_api/simplified_transcripts.json"
    extract_transcripts(input_file, output_file)