import json

# Update these paths as needed
input_json_path = '/hydra2-prev/home/compute/workspace_himanshu/wellness_fitness_annotations.json'
output_jsonl_path = '/external4/datasets/bucket_data/wellness/wellness_fitness_annotated.jsonl'

with open(input_json_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
    for item in data:
        output_obj = {
            'audio_filepath': item.get('audio_filepath'),
            'text': item.get('text'),
            'duration': item.get('duration')  # Default to 0 if duration is not present
        }
        outfile.write(json.dumps(output_obj, ensure_ascii=False) + '\n')

print(f"Converted {input_json_path} to {output_jsonl_path} with only 'audio_filepath' and 'text' fields.")
