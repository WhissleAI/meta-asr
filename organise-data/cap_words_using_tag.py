#!/usr/bin/env python3
import json
import re
import sys

def capitalize_entity_contents(text):
    """
    Finds all entity tags and capitalizes the first letter of each word inside them.
    Specifically handles ENTITY_PERSON_NAME and location-related tags (ENTITY_LOCATION, etc.)
    """
    # First, fix cases where END is merged with the last word (no space)
    # This pattern finds ENTITY tags where END is directly attached to the last word
    fix_pattern = r'(ENTITY_\w+\s+[\w\s]+)END'
    text = re.sub(fix_pattern, r'\1 END', text)
    
    # Define patterns for different entity types
    person_pattern = r"(ENTITY_PERSON_NAME\s+)([^E][^N][^D]*)(\s+END|\s+\w)"
    location_pattern = r"(ENTITY_(?:LOCATION|CITY|COUNTRY|STATE)\w*\s+)([^E][^N][^D]*)(\s+END|\s+\w)"
    
    # Function to capitalize each word in a matched entity
    def capitalize_match(match):
        prefix = match.group(1)  # The tag
        content = match.group(2)  # The content to capitalize
        suffix = match.group(3)   # END tag or suffix
        
        # Capitalize each word in the content
        capitalized_content = ' '.join(word.capitalize() for word in content.split())
        
        # Return the reconstructed string
        return prefix + capitalized_content + suffix
    
    # Apply capitalization for person names
    text = re.sub(person_pattern, capitalize_match, text)
    
    # Apply capitalization for location entities
    text = re.sub(location_pattern, capitalize_match, text)
    
    return text

def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file, capitalizing entity contents, and write to output file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Process the text field if it exists
                if 'text' in data:
                    data['text'] = capitalize_entity_contents(data['text'])
                
                # Write the processed data back as JSON
                outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                # Handle invalid JSON lines
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                continue

def process_single_json(json_string):
    """
    Process a single JSON string, capitalizing entity contents.
    """
    try:
        data = json.loads(json_string)
        if 'text' in data:
            data['text'] = capitalize_entity_contents(data['text'])
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return "Error: Invalid JSON string"

if __name__ == "__main__":
    # Example usage
 
        input_file = "/external1/datasets/manifest_hf/euro/test/test_peoplespeech_en.json"
        output_file = "/external1/datasets/manifest_hf/euro/test/test_peoplespeech_en_cap.jsonl"
        process_jsonl_file(input_file, output_file)
    #    '{"audio_filepath": "/peoplespeech_audio/train-00561-of-00804_1596.flac", "text": "the board didn't rely on that theory and i guess what i was wondering is is it the board's conclusion that for some claims mr ENTITY_PERSON_NAME kit lens END alleged adverse acts were due to his military status whereas other AGE_30_45 GER_MALE EMOTION_NEU", "duration": 14.65}'''
        
        print("Original JSON:")
        
        print("\nProcessed JSON:")
        
        print("\nUsage: python capitalize_entities.py input.jsonl output.jsonl")