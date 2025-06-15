import json

input_file = 'error_lines.jsonl'
output_file = 'cleaned_error_lines.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    data = infile.read()
    # Split the data on '}{' and add the missing brackets
    json_objects = data.split('}{')
    json_objects = [obj + '}' if not obj.endswith('}') else obj for obj in json_objects]
    json_objects = ['{' + obj if not obj.startswith('{') else obj for obj in json_objects]
    
    for obj in json_objects:
        try:
            # Validate JSON
            json.loads(obj)
            outfile.write(obj + '\n')
        except json.JSONDecodeError:
            print(f"Invalid JSON object skipped: {obj}")

print("Cleaning completed. Check the cleaned_error_lines.jsonl file.")