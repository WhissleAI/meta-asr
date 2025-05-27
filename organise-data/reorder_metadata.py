import json
import re
import os

def extract_metadata(text):
    """Extract metadata tags from text and return cleaned text and metadata tags."""
    # Regular expression to find metadata tags (uppercase words with underscores)
    metadata_pattern = r'\b(INTENT_\w+|AGE_\w+|GER_\w+|EMOTION_\w+)\b'
    
    # Find all metadata tags
    metadata_tags = re.findall(metadata_pattern, text)
    
    # Remove metadata tags from text
    cleaned_text = re.sub(metadata_pattern, '', text).strip()
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text, metadata_tags

def sort_and_deduplicate_metadata(metadata_tags):
    """Sort metadata tags according to specified order and remove duplicates."""
    # Create dictionaries to store metadata by type
    age_tags = set()
    gender_tags = set()
    emotion_tags = set()
    intent_tags = set()
    
    # Categorize each tag
    for tag in metadata_tags:
        if tag.startswith('AGE_'):
            age_tags.add(tag)
        elif tag.startswith('GER_'):
            gender_tags.add(tag)
        elif tag.startswith('EMOTION_'):
            emotion_tags.add(tag)
        elif tag.startswith('INTENT_'):
            intent_tags.add(tag)
    
    # Combine tags in the specified order: age, gender, emotion, intent
    sorted_tags = list(age_tags) + list(gender_tags) + list(emotion_tags) + list(intent_tags)
    
    return sorted_tags

def process_jsonl_file(input_file, output_file):
    """Process each line of the JSONL file and write corrected records to output file."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                # Parse the JSON object
                record = json.loads(line.strip())
                
                # Extract metadata from text
                text = record['text']
                cleaned_text, metadata_tags = extract_metadata(text)
                
                # Sort and deduplicate metadata
                sorted_tags = sort_and_deduplicate_metadata(metadata_tags)
                
                # Update the text field with cleaned text and sorted metadata
                record['text'] = cleaned_text + " " + " ".join(sorted_tags)
                
                # Write the updated record to the output file
                outfile.write(json.dumps(record) + '\n')
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {e}")

def main():
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/new_vils_data.jsonl"  # Change this to your input file path
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/corrected_vils_data.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
    
    process_jsonl_file(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()