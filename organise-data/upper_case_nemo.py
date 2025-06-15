import json
import re

def fix_and_capitalize_tags(text):
    # Fix overlapping tags issues
    overlap_pattern = r'Entity_([a-z_]+)ENTITY_'
    text = re.sub(overlap_pattern, r'ENTITY_', text)

    # Handle multi-part entity tags like "entity_person_name"
    # This pattern will match the entire entity tag with any number of underscore-separated parts
    entity_pattern = r'(?:^|\s|[.,:;!?])([Ee]ntity)_((?:[a-zA-Z]+_?)+)'

    def process_entity(match):
        prefix = match.group(0)[:match.group(0).find(match.group(1))]
        entity_prefix = match.group(1).upper()  # "entity" -> "ENTITY"
        entity_parts = match.group(2).upper() # Uppercase the entire entity_parts

        return f"{prefix}{entity_prefix}_{entity_parts}"

    text = re.sub(entity_pattern, process_entity, text)

    # Replace "Entity_" with "ENTITY_" for any remaining instances (after processing multi-part entities)
    text = text.replace("Entity_", "ENTITY_")

    # Uppercase END tags
    text = re.sub(r'\b[Ee][Nn][Dd]\b', 'END', text)

    return text

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        data = json.loads(line.strip())

        # Fix and capitalize entity tags in the text field
        if 'text' in data:
            data['text'] = fix_and_capitalize_tags(data['text'])

        processed_lines.append(json.dumps(data))

    with open(output_file, 'w') as f:
        for line in processed_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    import sys

    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/avspeech-en-nemo_processed.jsonl"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/avspeech-en.jsonl"

    process_file(input_file, output_file)
    print(f"Processed file saved to {output_file}")

# Test the function with examples
if __name__ == "__main__" and len(sys.argv) == 1:
    test_examples = [
        "Entity_person_name Dave, End can to talk about identity",
        "Entity_distancENTITY_DISTANCE twenty three miles, End",
        "entity_organization Moses Brown faculty staff end."
    ]

    for example in test_examples:
        fixed_text = fix_and_capitalize_tags(example)
        print(f"Original: {example}")
        print(f"Fixed:    {fixed_text}")
        print()