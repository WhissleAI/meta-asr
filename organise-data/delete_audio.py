import os
import glob

def delete_chunked_files(directory, pattern):
    """
    Delete all numbered chunk files matching the base filename pattern in the specified directory

    Args:
        directory (str): Path to the directory to search in
        pattern (str): A filename containing '_chunk_' to use as pattern base

    Returns:
        int: Number of files deleted
    """
    if '_chunk_' not in pattern:
        raise ValueError("Pattern must contain '_chunk_' to identify chunk files")

    # Extract the base name before '_chunk_'
    base_name = pattern.split('_chunk_')[0]

    # Create pattern to match all chunk files
    search_pattern = os.path.join(directory, f"{base_name}_chunk_*.json")

    # Find all matching files
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"No matching files found in {directory}")
        return 0

    # Show preview of files to be deleted
    print("\nFiles to be deleted:")
    for file_path in matching_files:
        print(f"- {file_path}")

    # Ask for confirmation
    confirm = input("\nAre you sure you want to delete these files? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled")
        return 0

    # Delete each file
    count = 0
    for file_path in matching_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            count += 1
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")

    return count

# Example usage
directory = "outputs/data"  # Replace with the target directory path
pattern = "TV_par_live_debate_-_The_Kapil_Sharma_Show_-_Episode_8_-_15th_May_2016_16k_chunk_8_chunk_0.json"  # Replace with an example filename containing '_chunk_'

try:
    files_deleted = delete_chunked_files(directory, pattern)
    print(f"\nTotal files deleted: {files_deleted}")
except ValueError as e:
    print(f"Error: {e}")
import os
import shutil

# Define the range and the exclusions
dir_range = range(200, 2001)  # Range from 20 to 2000 inclusive

#  # Update this path as needed
base_dir = os.path.join(os.path.dirname(__file__), "..", "Data-store", "train-other-500")
base_dir = os.path.abspath(base_dir)
# Iterate through the range and delete directories
for i in dir_range:
    dir_path = os.path.join(base_dir, str(i))
    try:
            shutil.rmtree(dir_path)
            print(f"Deleted: {dir_path}")
    except Exception as e:
            print(f"Error deleting {dir_path}: {e}")
