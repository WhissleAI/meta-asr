import json
import os
import re

# Load the main JSON file with correct information
def load_main_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping of file IDs to gender, age, and other values
    info_mapping = {}
    for file_id, info in data.items():
        info_mapping[file_id] = {
            'gender': info.get('gender', 'NA'),
            'age_group': info.get('age_group', 'NA')
        }
    
    return info_mapping

# Process JSONL files and correct gender, age, and emotion information
def process_jsonl_files(jsonl_path, info_mapping, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    corrected_lines = []
    corrections = {'gender': 0, 'age': 0, 'emotion': 0}
    
    for line in lines:
        entry = json.loads(line.strip())
        
        # Extract file ID from the audio filepath
        audio_path = entry.get('audio_filepath', '')
        filename = os.path.basename(audio_path)
        file_id = os.path.splitext(filename)[0]
        
        # Get text field
        text = entry.get('text', '')
        
        # Fix gender if needed
        if file_id in info_mapping:
            gender = info_mapping[file_id]['gender']
            if gender == "MALE":
                correct_gender = "GER_M"
            elif gender == "FEMALE":
                correct_gender = "GER_F"
            else:
                correct_gender = f"GER_{gender}"
            
            # Replace gender pattern
            if re.search(r'GER_NA', text):
                text = re.sub(r'GER_NA', correct_gender, text)
                corrections['gender'] += 1
            elif re.search(r'GENDER_NA', text):
                text = re.sub(r'GENDER_NA', correct_gender, text)
                corrections['gender'] += 1
        
        # Fix age if needed
        if file_id in info_mapping:
            age_group = info_mapping[file_id]['age_group']
            if age_group != "NA":
                correct_age = f"AGE_{age_group}"
                
                # Replace age pattern
                if re.search(r'AGE_NA', text):
                    text = re.sub(r'AGE_NA', correct_age, text)
                    corrections['age'] += 1
        
        # Fix emotion errors
        if "EMOTION_OOM_ERROR" in text:
            text = text.replace("EMOTION_OOM_ERROR", "EMOTION_NEU")
            corrections['emotion'] += 1
        
        # Update the text field
        entry['text'] = text
        
        corrected_lines.append(json.dumps(entry, ensure_ascii=False))
    
    # Write the corrected lines to a new file
    output_file = os.path.join(output_path, os.path.basename(jsonl_path))
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in corrected_lines:
            f.write(line + '\n')
    
    return len(corrected_lines), corrections

# Main function
def main():
    # Paths - update these to match your file locations
    main_json_path = "/external4/datasets/MADASR/IISc_RESPIN_train_small/data/meta_train_small.json"
    jsonl_dir = "/external4/datasets/MADASR/process_train_small/end_fixz"
    output_dir = "/external4/datasets/MADASR/IISc_RESPIN_train_small/manifest_data"
    
    # Load information mapping from main JSON
    print("Loading data mapping from main JSON file...")
    info_mapping = load_main_json(main_json_path)
    print(f"Loaded information for {len(info_mapping)} entries")
    
    # Process each JSONL file
    total_processed = 0
    total_corrections = {'gender': 0, 'age': 0, 'emotion': 0}
    
    if os.path.isfile(jsonl_dir) and jsonl_dir.endswith('.jsonl'):
        # If jsonl_dir is actually a file
        count, corrections = process_jsonl_files(jsonl_dir, info_mapping, output_dir)
        filename = os.path.basename(jsonl_dir)
        print(f"Processed {count} entries in {filename}")
        print(f"Corrections in {filename}: Gender: {corrections['gender']}, Age: {corrections['age']}, Emotion: {corrections['emotion']}")
        
        total_processed += count
        for key in total_corrections:
            total_corrections[key] += corrections[key]
    else:
        # Process directory of JSONL files
        for filename in os.listdir(jsonl_dir):
            if filename.endswith('.jsonl'):
                jsonl_path = os.path.join(jsonl_dir, filename)
                count, corrections = process_jsonl_files(jsonl_path, info_mapping, output_dir)
                
                print(f"Processed {count} entries in {filename}")
                print(f"Corrections in {filename}: Gender: {corrections['gender']}, Age: {corrections['age']}, Emotion: {corrections['emotion']}")
                
                total_processed += count
                for key in total_corrections:
                    total_corrections[key] += corrections[key]
    
    print("\nSummary:")
    print(f"Total processed: {total_processed} entries")
    print(f"Total corrections: Gender: {total_corrections['gender']}, Age: {total_corrections['age']}, Emotion: {total_corrections['emotion']}")

if __name__ == "__main__":
    main()