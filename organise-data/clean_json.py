import json
import os
from pathlib import Path

def clean_text(text):
    """Clean text by removing extra spaces and trimming."""
    return ' '.join(text.split())

def process_json_file(file_path):
    """Process a single JSON file and clean all text fields."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single object and list of objects
        if isinstance(data, list):
            for item in data:
                if 'text' in item:
                    item['text'] = clean_text(item['text'])
        else:
            if 'text' in data:
                data['text'] = clean_text(data['text'])
        
        # Write back to the same file with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_folder(folder_path):
    """Process all JSON files in the given folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist!")
        return
    
    processed = 0
    failed = 0
    
    # Process all JSON files in the folder
    for file_path in folder.glob('*.json'):
        print(f"Processing {file_path.name}...")
        if process_json_file(file_path):
            processed += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} files")
    print(f"Failed to process: {failed} files")

# Example usage
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "Data_store", "youtube_data/data")
    base_dir = os.path.abspath(base_dir)
    process_folder(base_dir)


import json
import os
from pathlib import Path

def convert_to_jsonl(input_file, output_file):
    """Convert JSON array to JSONL format."""
    try:
        # Read input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Write each object as a separate line in the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                # Clean the text field
                if 'text' in item:
                    item['text'] = ' '.join(item['text'].split())
                # Write the object as a single line
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """Process all JSON files in the input folder and save to output folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    failed = 0
    
    # Process all JSON files
    for file_path in input_path.glob('*.json'):
        print(f"Processing {file_path.name}...")
        
        # Create output file path
        output_file = output_path / f"{file_path.stem}_processed.jsonl"
        
        if convert_to_jsonl(file_path, output_file):
            processed += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} files")
    print(f"Failed to process: {failed} files")
    print(f"Processed files are saved in: {output_path.absolute()}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "meta-asr", "train-data")
    base_dir = os.path.abspath(base_dir)
    output_folder = "opensr_data_jsonl"
    process_folder(base_dir, output_folder)