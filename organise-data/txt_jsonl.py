import json

# Input and output file paths
input_txt_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/transcriptions_hf_data.txt"  # Replace with your actual input file path
output_jsonl_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/txt_jsonl_hf.jsonl"  # Replace with your desired output file path

# Read the TXT file and process each line
with open(input_txt_file, "r", encoding="utf-8") as infile, open(output_jsonl_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split(" | ", 1)  # Split into filepath and text
        if len(parts) == 2:
            audio_filepath, text = parts
            json_obj = {"audio_filepath": audio_filepath, "text": text}
            outfile.write(json.dumps(json_obj) + "\n")

print("Conversion complete. JSONL file saved at:", output_jsonl_file)