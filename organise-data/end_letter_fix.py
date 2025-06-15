import json
import re
import os

def fix_entity_tags_and_reorganize(input_file, output_file):
    fixed_count = 0
    end_tag_removed_count = 0
    error_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'rb') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line_bytes in enumerate(f_in, 1):
            try:
                # Try to decode with various encodings
                try:
                    line = line_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    line = line_bytes.decode('latin-1')  # Almost always works
                
                data = json.loads(line.strip())
                
                # Fix patterns like "ENTITY_CIT Y" to "ENTITY_CITY"
                # And "ENTITY_PERSON_NAM E" to "ENTITY_PERSON_NAME"
                text = data['text']
                
                # Pattern to match entity tags with spaces in the tag format ENTITY_WORD_NAM E
                pattern1 = r'(ENTITY_[A-Z]+)_([A-Z]+)\s+([A-Z]+)'
                text = re.sub(pattern1, r'\1_\2\3', text)
                
                # Pattern to match entity tags with spaces like ENTITY_CIT Y
                pattern2 = r'(ENTITY_[A-Z]+)\s+([A-Z]+)'
                text = re.sub(pattern2, r'\1\2', text)
                
                # Additional pattern to match cases like ENTITY_PERSON_NAM E specifically
                pattern3 = r'(ENTITY_[A-Z]+)_([A-Z]{2,})\s+([A-Z])'
                text = re.sub(pattern3, r'\1_\2\3', text)
                
                # Fix for any remaining spaces in entity tags
                pattern4 = r'(ENTITY_[A-Z_]+)\s+([A-Z]+)'
                text = re.sub(pattern4, r'\1\2', text)
                
                # Remove standalone END tags that don't have corresponding opening entity tags
                # First, check if there are any ENTITY_ tags in the text
                has_entity_tags = bool(re.search(r'ENTITY_[A-Z_]+', text))
                
                # If there are no entity tags, remove all END tags
                if not has_entity_tags:
                    old_text = text
                    text = re.sub(r'\bEND\b', '', text)
                    if old_text != text:
                        end_tag_removed_count += 1
                else:
                    # If there are entity tags, only keep the first END tag after each ENTITY_ tag
                    # Split the text by ENTITY_ tags
                    parts = re.split(r'(ENTITY_[A-Z_]+)', text)
                    for i in range(1, len(parts), 2):
                        # For each part that follows an ENTITY_ tag, keep only the first END
                        if i+1 < len(parts):
                            # Count ENDs in this part
                            end_count = parts[i+1].count('END')
                            if end_count > 1:
                                # Replace all instances of END except the first one
                                first_end_pos = parts[i+1].find('END')
                                parts[i+1] = parts[i+1][:first_end_pos+3] + parts[i+1][first_end_pos+3:].replace('END', '')
                                end_tag_removed_count += end_count - 1
                    text = ''.join(parts)
                
                if text != data['text']:
                    fixed_count += 1
                    
                data['text'] = text
                
                # Create a new reorganized dictionary with the desired order
                reorganized_data = {
                    'audio_filepath': data['audio_filepath'],
                    'text': data['text'],
                    'duration': data['duration']
                }
                
                # Remove any other fields that might be in the original data
                f_out.write(json.dumps(reorganized_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 3:  # Show only first few errors to avoid flooding the console
                    print(f"JSON decode error at line {line_num}: {e}")
                    try:
                        problematic_part = line[max(0, e.pos-20):min(len(line), e.pos+20)]
                        print(f"Context: ...{problematic_part}... (error at position {e.pos})")
                    except:
                        print(f"Could not show context for the error")
                elif error_count == 4:
                    print("Additional JSON errors found. Not showing all errors.")
            except Exception as e:
                error_count += 1
                print(f"Error at line {line_num}: {e}")
    
    return fixed_count, end_tag_removed_count, error_count


input_file = '/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/nptel-test/nptel_english_annotated_v2_lowercase.jsonl'
output_file = '/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/nptel-test/nptel_english_annotated.jsonl'

fixed, end_removed, errors = fix_entity_tags_and_reorganize(input_file, output_file)
print(f"Fixed {fixed} entity tag issues")
print(f"Removed {end_removed} unnecessary END tags")
print(f"Encountered {errors} JSON decode errors")
if errors > 0:
    print("\nSuggestions to fix JSON errors:")
    print("1. Check for missing or extra commas in the JSON file")
    print("2. Look for unescaped quotes inside string values")
    print("3. Inspect for invalid control characters")
    print("4. Ensure all field names and values are properly quoted")