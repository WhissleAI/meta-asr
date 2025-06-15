import json
import os
import glob
from pathlib import Path

def analyze_file_paths(input_jsonl, sample_size=20):
    """
    Analyze file paths in the JSONL to understand their structure and existence.
    
    Args:
        input_jsonl (str): Path to the input JSONL file
        sample_size (int): Number of sample paths to print
    """
    total_files = 0
    files_without_duration = 0
    sample_paths = []
    
    print(f"Analyzing file paths in {input_jsonl}...")
    
    with open(input_jsonl, 'r') as f:
        for line in f:
            total_files += 1
            
            # Parse the JSON line
            data = json.loads(line.strip())
            audio_path = data.get('audio_filepath')
            
            # Check if duration is missing or None
            if 'duration' not in data or data['duration'] is None:
                files_without_duration += 1
                
                # Collect sample paths for analysis
                if len(sample_paths) < sample_size and audio_path:
                    sample_paths.append(audio_path)
            
            # Print progress every 100000 files
            if total_files % 100000 == 0:
                print(f"Processed {total_files} entries, found {files_without_duration} without duration...")
    
    # Print sample paths for analysis
    print("\nSample file paths without duration (first few):")
    common_prefix = os.path.commonprefix(sample_paths) if sample_paths else ""
    for path in sample_paths:
        print(f"  Path: {path}")
        print(f"  Basename: {os.path.basename(path)}")
        print(f"  Exists: {os.path.exists(path)}")
        print(f"  Directory exists: {os.path.exists(os.path.dirname(path))}")
        print("  ----")
    
    print(f"\nCommon prefix of paths: {common_prefix}")
    
    # Print statistics
    print(f"\nAnalysis complete:")
    print(f"Total files examined: {total_files}")
    print(f"Files without duration: {files_without_duration}")

def create_unprocessed_jsonl(input_jsonl, output_jsonl):
    """
    Creates a new JSONL file with only the entries that don't have duration tags.
    
    Args:
        input_jsonl (str): Path to the input JSONL file
        output_jsonl (str): Path to the output JSONL file
    """
    total_files = 0
    files_without_duration = 0
    
    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for line in f_in:
            total_files += 1
            
            # Parse the JSON line
            data = json.loads(line.strip())
            
            # Check if duration is missing or None
            if 'duration' not in data or data['duration'] is None:
                files_without_duration += 1
                f_out.write(json.dumps(data) + '\n')
            
            # Print progress every 100000 files
            if total_files % 100000 == 0:
                print(f"Processed {total_files} entries...")
    
    # Print final statistics
    print(f"\nUnprocessed JSONL creation complete:")
    print(f"Total files examined: {total_files}")
    print(f"Files without duration extracted: {files_without_duration}")
    print(f"Output file created: {output_jsonl}")

def search_for_files(search_pattern, max_results=10):
    """
    Search for files matching a pattern to help locate missing audio files.
    
    Args:
        search_pattern (str): File pattern to search for
        max_results (int): Maximum number of results to return
    """
    print(f"Searching for files matching pattern: {search_pattern}")
    matching_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(matching_files)} matching files.")
    if matching_files:
        print("Sample matches:")
        for i, file_path in enumerate(matching_files[:max_results]):
            print(f"  {file_path}")
        if len(matching_files) > max_results:
            print(f"  ... and {len(matching_files) - max_results} more")
    
    return matching_files

if __name__ == "__main__":
    # Path configurations - MODIFY THESE
    input_jsonl = "/external2/datasets/json_data/people_speech2/final_duration_people_speech.jsonl"
    
    # Path for new JSONL file with only unprocessed entries
    unprocessed_jsonl = "/external2/datasets/json_data/people_speech2/unprocessed_people_speech.jsonl"
    
    # 1. Analyze file paths to understand the structure
    analyze_file_paths(input_jsonl)
    
    # 2. Create a JSONL file with only the unprocessed entries for later processing
    print("\nCreating JSONL file with only unprocessed entries...")
    create_unprocessed_jsonl(input_jsonl, unprocessed_jsonl)
    
    print("\nAll operations completed successfully!")
    print("\nRecommendations:")
    print("1. Review the sample file paths to understand if they're valid.")
    print("2. If the original audio files have been moved, update the paths in the unprocessed JSONL.")
    print("3. Use the unprocessed JSONL file to reprocess only the files without duration.")