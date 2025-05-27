import json

def clean_ner_tags(text):
    words = text.split()
    cleaned_words = []
    seen_tags = set()
    
    for word in words:
        if word.startswith("NER_") or word == "END":
            if word not in seen_tags:
                seen_tags.add(word)
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            seen_tags.clear()  # Reset tags after non-NER word
    
    return ' '.join(cleaned_words)

def clean_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                data['text'] = clean_ner_tags(data['text'])
                outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error message: {e}")
# to clean up the jsonl format

# def clean_jsonl_file(input_file, output_file):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line_number, line in enumerate(infile, start=1):
#             try:
#                 # Attempt to parse the line as JSON
#                 data = json.loads(line)
#                 # Write the cleaned JSON object to the output file
#                 outfile.write(json.dumps(data) + '\n')
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON on line {line_number}: {line.strip()}")
#                 print(f"Error message: {e}")
#                 # Optionally, you can write the line to a separate error file for further inspection
#                 with open('error_lines.jsonl', 'a') as error_file:
#                     error_file.write(line)

# Usage

# Usage
input_file = '/hydra2-prev/home/compute/workspace_himanshu/meta-asr/cleaned_error_lines.jsonl'
output_file = '/hydra2-prev/home/compute/workspace_himanshu/meta-asr/extra.jsonl'
clean_jsonl_file(input_file, output_file)