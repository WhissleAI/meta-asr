import json
import re

def process_error_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                # Parse the error entry
                error_entry = json.loads(line.strip())
                
                # Get the problematic line content
                line_content = error_entry.get('line_content', '')
                
                # Extract duration from the error entry structure
                duration_match = re.search(r'}, "duration": ([\d\.]+)}$', line_content)
                duration = float(duration_match.group(1)) if duration_match else 0
                
                # Try to extract the main JSON object
                content_match = re.match(r'(\{.*?\})(, "duration": [\d\.]+})?$', line_content)
                if content_match:
                    content_json = content_match.group(1)
                else:
                    content_json = line_content
                
                try:
                    # Try to parse the extracted content
                    content_data = json.loads(content_json)
                    
                    # Get the text content
                    text_content = ""
                    if "annotated_sentence" in content_data:
                        text_content = content_data["annotated_sentence"]
                    else:
                        text_content = content_data.get("text", "")
                    
                    # Create output entry with requested format and order
                    output_entry = {
                        "audio_filepath": content_data.get("audio_filepath", ""),
                        "text": text_content,
                        "duration": duration
                    }
                    
                    # Write the cleaned entry to output file
                    fout.write(json.dumps(output_entry) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error in line {line_num}, could not parse content: {e}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                
    print(f"Processing complete. Results written to {output_file}")

# Example usage
input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/parsing_errors.jsonl"
output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/cleaned_data.jsonl"
process_error_file(input_file, output_file)