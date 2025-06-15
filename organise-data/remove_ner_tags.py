# import json
# import re
# import sys

# def process_jsonl_file(input_file, output_file, empty_paths_file):
#     """
#     Process a JSONL file to:
#     1. Remove all capital letter tags (like AGE_30_45)
#     2. Remove extra spaces from text
#     3. Log paths of entries with empty text fields
#     """
#     # Regular expression to match all-caps tags (words with underscores)
#     pattern = r'\b[A-Z]+(?:_[A-Z0-9]+)+\b'
    
#     # Count of processed entries and empty entries
#     processed_count = 0
#     empty_count = 0
    
#     with open(input_file, 'r', encoding='utf-8') as fin, \
#          open(output_file, 'w', encoding='utf-8') as fout, \
#          open(empty_paths_file, 'w', encoding='utf-8') as empty_file:
        
#         for line in fin:
#             try:
#                 # Parse the JSON line
#                 data = json.loads(line.strip())
#                 processed_count += 1
                
#                 # Check if text field exists and is empty
#                 if 'text' in data and (data['text'] is None or data['text'].strip() == ''):
#                     empty_count += 1
#                     # Log the audio filepath of entries with empty text
#                     if 'audio_filepath' in data:
#                         empty_file.write(f"{data['audio_filepath']}\n")
#                     else:
#                         empty_file.write(f"Unknown path (entry #{processed_count})\n")
                
#                 # Process all string values in the data
#                 for key in data:
#                     if isinstance(data[key], str):
#                         # Remove capital letter tags
#                         cleaned_text = re.sub(pattern, '', data[key])
#                         # Remove extra spaces (multiple spaces to single space)
#                         cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
#                         # Strip leading/trailing spaces
#                         data[key] = cleaned_text.strip()
                
#                 # Write the processed data back as a JSON line
#                 fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
#             except json.JSONDecodeError:
#                 # If the line is not valid JSON, just process it as text
#                 cleaned_line = re.sub(pattern, '', line)
#                 cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
#                 fout.write(cleaned_line + '\n')
    
#     print(f"Processed {processed_count} entries")
#     print(f"Found {empty_count} entries with empty text fields")
#     print(f"Paths of empty text entries saved to {empty_paths_file}")

# if __name__ == "__main__":
#     input_file = '/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle_hf.jsonl'
#     output_file = '/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle2_hf.jsonl'
#     empty_paths_file = '/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/empty_paths.txt'
#     process_jsonl_file(input_file, output_file, empty_paths_file)

# # Example usage

import json

input_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle2_hf.jsonl"  # Change to your actual JSONL file path
output_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle1_hf.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if data.get("text", "").strip():  # Keep only if text is not empty
            outfile.write(json.dumps(data) + "\n")

print("Filtered JSONL saved to", output_file)