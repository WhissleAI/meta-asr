import json
import codecs

input_file = '/external4/datasets/train_opus/manifest.jsonl'
output_file = '/external4/datasets/train_opus/manifest_raw.jsonl'

def decode_jsonl_file(input_file_path, output_file_path="decoded_data.jsonl"):
    """
    Decode Unicode text in a JSONL file and write to a new JSONL file.
    
    Args:
        input_file_path: Path to the input JSONL file
        output_file_path: Path to the output JSONL file
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as in_file, \
             open(output_file_path, 'w', encoding='utf-8') as out_file:
            
            line_count = 0
            error_count = 0
            
            for i, line in enumerate(in_file, 1):
                line = line.strip()
                if not line: 
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # No additional parsing needed - the json.loads already handles Unicode escape sequences
                    # Just write it back with ensure_ascii=False for readable Unicode characters
                    out_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                    line_count += 1
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: Could not decode JSON on line {i}: {e}")
                    # Skip this problematic line and continue processing
                    continue
                
                # Print progress every 100,000 lines
                if line_count % 100000 == 0:
                    print(f"Processed {line_count} lines...")
                    
            print(f"Successfully processed {line_count} lines from {input_file_path}")
            print(f"Skipped {error_count} problematic lines")
            print(f"Output saved to {output_file_path}")
            
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    decode_jsonl_file(input_file, output_file)