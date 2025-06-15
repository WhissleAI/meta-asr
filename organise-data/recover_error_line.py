import json
from pathlib import Path

def fix_malformed_json(error_line):
    """
    Fixes and recovers data from malformed JSON with specific entity structure issues.
    
    Args:
        error_line (str): The problematic JSON line
    Returns:
        dict: Recovered JSON object
    """
    try:
        # If the input is a dictionary containing error info, extract the problematic line
        if isinstance(error_line, dict) and 'problematic_line' in error_line:
            error_line = error_line['problematic_line']
            
        # First try to parse the line directly
        return json.loads(error_line)
    except json.JSONDecodeError:
        # If that fails, try to reconstruct the entities array
        try:
            # Extract the base information before the entities array
            base_info = error_line.split('"entities": [')[0] + '"entities": ['
            
            # Extract the entities part
            entities_part = error_line.split('"entities": [')[1].split('], "duration":')[0]
            
            # Extract the duration part
            duration_part = '], "duration":' + error_line.split('], "duration":')[1]
            
            # Split entities into individual objects and fix them
            entity_strings = entities_part.split('{')[1:]  # Skip the first empty split
            fixed_entities = []
            
            for entity_str in entity_strings:
                # Remove any trailing commas and clean up the entity string
                entity_str = entity_str.strip().rstrip(',')
                if not entity_str.endswith('}'):
                    entity_str += '}'
                
                # Parse the individual entity
                try:
                    entity = json.loads('{' + entity_str)
                    fixed_entities.append(entity)
                except json.JSONDecodeError:
                    continue
            
            # Reconstruct the full JSON string
            fixed_json_str = base_info + json.dumps(fixed_entities) + duration_part
            
            return json.loads(fixed_json_str)
            
        except Exception as e:
            print(f"Failed to recover JSON: {str(e)}")
            return None

def process_error_lines(error_file, recovered_output, remaining_errors):
    """
    Process the error_lines.jsonl file and attempt to recover the data.
    
    Args:
        error_file (str): Path to the error lines file
        recovered_output (str): Path to write recovered records
        remaining_errors (str): Path to write still problematic records
    """
    stats = {
        'total_lines': 0,
        'recovered': 0,
        'unrecoverable': 0
    }
    
    print(f"Starting to process error lines from {error_file}")
    
    with open(error_file, 'r', encoding='utf-8') as f_in, \
         open(recovered_output, 'w', encoding='utf-8') as f_recovered, \
         open(remaining_errors, 'w', encoding='utf-8') as f_errors:
        
        for line_number, line in enumerate(f_in, 1):
            stats['total_lines'] += 1
            try:
                # Parse the error line entry
                error_entry = json.loads(line.strip())
                
                # Attempt to recover the data
                recovered_data = fix_malformed_json(error_entry)
                
                if recovered_data:
                    # Write recovered data
                    json.dump(recovered_data, f_recovered, ensure_ascii=False)
                    f_recovered.write('\n')
                    stats['recovered'] += 1
                    print(f"Successfully recovered line {line_number}")
                else:
                    # Write back to errors if recovery failed
                    json.dump(error_entry, f_errors, ensure_ascii=False)
                    f_errors.write('\n')
                    stats['unrecoverable'] += 1
                    print(f"Failed to recover line {line_number}")
                
            except Exception as e:
                print(f"Error processing line {line_number}: {str(e)}")
                stats['unrecoverable'] += 1
                
            # Print progress every 100 lines
            if line_number % 100 == 0:
                print(f"Processed {line_number} lines...")
    
    # Print final statistics
    print("\nProcessing Complete!")
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Successfully recovered: {stats['recovered']}")
    print(f"Unrecoverable lines: {stats['unrecoverable']}")

def main():
    # Define file paths
    error_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/error_lines.jsonl"
    recovered_output = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/recovered_lines.jsonl"
    remaining_errors = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/unrecovered_lines.jsonl"
    
    # Ensure input file exists
    if not Path(error_file).exists():
        print(f"Error: Input file '{error_file}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    Path(recovered_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the error lines
    process_error_lines(error_file, recovered_output, remaining_errors)

if __name__ == '__main__':
    main()