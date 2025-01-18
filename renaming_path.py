import os
import json

# Directory containing the JSONL files
directory = "yt_jsonl_fixed"

# Prefix to add to the file paths
prefix = "/external1/datasets/asr-himanshu/"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):  # Process only JSONL files
        filepath = os.path.join(directory, filename)
        
        # Read and process the JSONL file
        updated_lines = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    record = json.loads(line.strip())  # Parse each JSON line
                    if "audio_filepath" in record:
                        record["audio_filepath"] = prefix + record["audio_filepath"]
                    updated_lines.append(json.dumps(record))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")
        
        # Write the updated lines back to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("\n".join(updated_lines))

# import json
# import os
# from pathlib import Path
# import shutil

# def fix_audio_paths(input_file, output_file):
#     """
#     Fix duplicate paths in JSONL files by removing redundant directory names
#     while preserving the original data in a new file.
#     """
#     # Create a list to store fixed data
#     fixed_data = []
    
#     # Read the input JSONL file line by line
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 # Parse each line as a JSON object
#                 entry = json.loads(line.strip())
#                 new_entry = entry.copy()  # Create a copy of the current entry
                
#                 if 'audio_filepath' in entry:
#                     path = entry['audio_filepath']
                    
#                     # Split the path into components
#                     parts = path.split('/')
                    
#                     if len(parts) >= 3:  # Only process if we have enough parts
#                         audio_chunks = parts[0]  # Should be "audio_chunks"
#                         folder_name = parts[1]   # The main folder name
                        
#                         # Find the last component (filename)
#                         filename = parts[-1]
                        
#                         # Create new path: audio_chunks/original_filename
#                         new_path = f"{audio_chunks}/{filename}"
                        
#                         # Update the path in the new entry
#                         new_entry['audio_filepath'] = new_path
                
#                 fixed_data.append(new_entry)
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing line: {line.strip()}")
#                 print(f"Error details: {e}")
#                 continue

#     # Write the fixed data to a new JSONL file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for entry in fixed_data:
#             # Write each entry as a separate line
#             f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# def process_directory(input_dir, output_dir):
#     """
#     Process all JSONL files in a directory and its subdirectories.
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process all JSONL files
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith('.jsonl'):
#                 input_path = os.path.join(root, file)
                
#                 # Create corresponding output path
#                 relative_path = os.path.relpath(input_path, input_dir)
#                 output_path = os.path.join(output_dir, relative_path)
                
#                 # Create necessary subdirectories
#                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
#                 # Process the file
#                 fix_audio_paths(input_path, output_path)

# # Example usage
# if __name__ == "__main__":
#     # For single file
#     # input_file = "outputs/processed/1_Second_of_Every_Show_from_Cartoon_Network_as_of_We_Baby_Bears_UPDATED_16k_chunk_0_audio_text_pairs_processed.jsonl"
#     # output_file = "outputs/test.jsonl"
#     # fix_audio_paths(input_file, output_file)
    
#     # For processing entire directory
#     base_dir = os.path.join(os.path.dirname(__file__), "..", "Processed_Data", "model_ready_manifest/yt_data_jsonl")
#     base_dir = os.path.abspath(base_dir)
#     output_directory = "yt_jsonl_fixed"
#     process_directory(base_dir, output_directory)

