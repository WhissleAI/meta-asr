import json
import os

# Define the output directory where files will be saved
output_dir = "/external4/datasets/MADASR/IISc_RESPIN_train_small"  # You can change this to your preferred directory

path = "/external4/datasets/MADASR/IISc_RESPIN_train_small/meta_train_small.json"
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of languages to track
languages = ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"]

# Initialize dictionaries for each language
language_data = {lang: {} for lang in languages}

with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Separate data by language
for key, value in data.items():
    # Extract language from the wav_path field
    if 'wav_path' in value:
        # Assuming wav_path format like "bh/D1/10846/..."
        language = value['wav_path'].split('/')[0]
        if language in languages:
            language_data[language][key] = value
    elif 'lid' in value:
        # Fallback to lid if wav_path is not present
        language = value['lid']
        if language in languages:
            language_data[language][key] = value
    else:
        # If both wav_path and lid not present, try to extract from the key
        parts = key.split('_')
        if len(parts) > 2 and parts[2] in languages:
            language = parts[2]
            language_data[language][key] = value

# Save each language data to a separate file in the output directory
for lang, lang_data in language_data.items():
    if lang_data:  # Only create files for languages that have data
        output_filename = os.path.join(output_dir, f"{lang}_meta_data.json")
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(lang_data, outfile, ensure_ascii=False, indent=2)
        print(f"Created {output_filename} with {len(lang_data)} entries")
    else:
        print(f"No data found for language: {lang}")

print(f"Data separation complete! Files saved to {os.path.abspath(output_dir)}")