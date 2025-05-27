import os
import argparse
from pathlib import Path

def cleanup_processed_files(processed_log_file, dry_run=True):

    if not os.path.exists(processed_log_file):
        print(f"Error: Log file {processed_log_file} not found!")
        return
        
    # Read the processed files list
    with open(processed_log_file, 'r') as f:
        processed_files = f.read().splitlines()
    
    if not processed_files:
        print("No files found in the processed log.")
        return
        
    print(f"Found {len(processed_files)} files to process")
    
   
    successfully_deleted = []
    failed_deletions = []
 
    for file_path in processed_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} - already deleted or moved")
            failed_deletions.append(file_path)
            continue
            
        try:
            if dry_run:
                print(f"Would delete: {file_path}")
                successfully_deleted.append(file_path)
            else:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                successfully_deleted.append(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
            failed_deletions.append(file_path)
    
    if not dry_run and successfully_deleted:
       
        with open(processed_log_file, 'w') as f:
            for file_path in failed_deletions:
                f.write(f"{file_path}\n")
        
        print(f"\nSummary:")
        print(f"Successfully deleted: {len(successfully_deleted)} files")
        print(f"Failed to delete: {len(failed_deletions)} files")
        print(f"Updated {processed_log_file} to remove successfully deleted files")
    elif dry_run:
        print(f"\nDry run summary:")
        print(f"Would delete: {len(successfully_deleted)} files")
        print(f"Would fail: {len(failed_deletions)} files")
        print("No files were actually deleted (dry run)")
    
if __name__ == "__main__":
    file = "/external4/datasets/Shrutilipi/maithili/mt/processed_files.txt"
    cleanup_processed_files(
        file,False
    )