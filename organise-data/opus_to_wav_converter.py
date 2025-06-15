#!/usr/bin/env python3
"""
Script to convert .opus audio files to .wav format.
Preserves the original directory structure.
"""

import os
import subprocess
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

def convert_opus_to_wav(opus_file, output_dir=None, ffmpeg_path=None):
    """
    Convert a single opus file to wav format
    
    Args:
        opus_file (str): Path to the opus file
        output_dir (str, optional): Base directory to save the wav file. If None, saves in the same directory.
        ffmpeg_path (str, optional): Path to ffmpeg executable. If None, assumes ffmpeg is in PATH.
    
    Returns:
        str: Path to the created wav file
    """
    opus_path = Path(opus_file)
    
    # Determine output directory, preserving the directory structure
    if output_dir:
        # Get the relative path from the current working directory
        cwd = Path.cwd()
        try:
            rel_path = opus_path.parent.relative_to(cwd)
            # Create target directory preserving structure
            out_dir = Path(output_dir) / rel_path
        except ValueError:
            # If opus_path is not relative to cwd, use just the opus filename
            out_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
    else:
        # If no output directory specified, use the same directory as the opus file
        out_dir = opus_path.parent
    
    # Create output filename (same name but with .wav extension)
    wav_file = out_dir / f"{opus_path.stem}.wav"
    
    # Determine ffmpeg command
    if ffmpeg_path:
        ffmpeg_cmd = ffmpeg_path
    else:
        # Try to find ffmpeg in the workspace
        ffmpeg_workspace_path = "/hydra2-prev/home/compute/workspace_himanshu/ffmpeg-7.0.2-amd64-static/ffmpeg"
        if os.path.exists(ffmpeg_workspace_path):
            ffmpeg_cmd = ffmpeg_workspace_path
        else:
            ffmpeg_cmd = "ffmpeg"  # Assume it's in PATH
    
    try:
        # Run ffmpeg to convert the file
        cmd = [
            ffmpeg_cmd,
            "-i", str(opus_path),
            "-acodec", "pcm_s16le",  # 16-bit PCM encoding for WAV
            "-ar", "16000",          # Sample rate of 16kHz (common for speech)
            "-ac", "1",              # Mono channel
            "-y",                    # Overwrite output file if it exists
            str(wav_file)
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {opus_file} → {wav_file}")
        
        return str(wav_file)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {opus_file}: {e}")
        return None

def process_directory(directory, output_dir=None, workers=4, ffmpeg_path=None, recursive=True):
    """
    Process an entire directory of .opus files
    
    Args:
        directory (str): Directory containing opus files
        output_dir (str, optional): Base directory to save wav files
        workers (int): Number of parallel workers
        ffmpeg_path (str, optional): Path to ffmpeg executable
        recursive (bool): Whether to search recursively in subdirectories
    
    Returns:
        dict: Mapping of opus files to their converted wav files
    """
    dir_path = Path(directory)
    
    # Find all opus files
    if recursive:
        opus_files = list(dir_path.glob('**/*.opus'))
    else:
        opus_files = list(dir_path.glob('*.opus'))
    
    print(f"Found {len(opus_files)} .opus files to convert")
    
    # Create conversion mapping
    conversion_map = {}
    
    # If output_dir is specified and we want to preserve structure
    if output_dir:
        # Prepare output directory structure
        input_base_dir = os.path.abspath(directory)
        output_base_dir = os.path.abspath(output_dir)
        
        # For each opus file, calculate its relative path from the input directory
        for opus_file in opus_files:
            abs_opus_path = os.path.abspath(str(opus_file))
            rel_path = os.path.relpath(abs_opus_path, input_base_dir)
            rel_dir = os.path.dirname(rel_path)
            
            # Create corresponding output directory
            target_dir = os.path.join(output_base_dir, rel_dir)
            os.makedirs(target_dir, exist_ok=True)
    
    # Convert files in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Build a list of tasks
        tasks = []
        for opus_file in opus_files:
            # For each opus file, we need to pass the specific output directory
            if output_dir:
                abs_opus_path = os.path.abspath(str(opus_file))
                rel_path = os.path.relpath(abs_opus_path, input_base_dir)
                rel_dir = os.path.dirname(rel_path)
                target_dir = os.path.join(output_base_dir, rel_dir)
                tasks.append(executor.submit(convert_opus_to_wav, str(opus_file), target_dir, ffmpeg_path))
            else:
                tasks.append(executor.submit(convert_opus_to_wav, str(opus_file), None, ffmpeg_path))
        
        # Process results as they complete
        for i, future in enumerate(tasks):
            try:
                wav_file = future.result()
                if wav_file:
                    conversion_map[str(opus_files[i])] = wav_file
            except Exception as e:
                print(f"Error processing {opus_files[i]}: {e}")
    
    return conversion_map

def update_manifest(manifest_file, conversion_map, output_manifest=None):
    """
    Update a manifest file replacing .opus files with .wav files
    
    Args:
        manifest_file (str): Path to the original manifest JSON or JSONL file
        conversion_map (dict): Mapping of opus files to wav files
        output_manifest (str, optional): Path to save updated manifest
    """
    if not output_manifest:
        # Create default output name
        manifest_path = Path(manifest_file)
        output_manifest = str(manifest_path.parent / f"{manifest_path.stem}_wav{manifest_path.suffix}")
    
    try:
        # Check if it's a JSONL file (one JSON object per line)
        if manifest_file.endswith('.jsonl'):
            # Process JSONL file
            with open(manifest_file, 'r') as f_in, open(output_manifest, 'w') as f_out:
                for line in f_in:
                    try:
                        item = json.loads(line.strip())
                        if 'audio_filepath' in item and item['audio_filepath'] in conversion_map:
                            item['audio_filepath'] = conversion_map[item['audio_filepath']]
                        f_out.write(json.dumps(item) + '\n')
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
        else:
            # Process regular JSON file
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            # If manifest is a list of dictionaries
            if isinstance(manifest_data, list):
                for item in manifest_data:
                    if 'audio_filepath' in item and item['audio_filepath'] in conversion_map:
                        item['audio_filepath'] = conversion_map[item['audio_filepath']]
            
            # Write updated manifest
            with open(output_manifest, 'w') as f:
                json.dump(manifest_data, f, indent=2)
        
        print(f"Updated manifest saved to: {output_manifest}")
    except Exception as e:
        print(f"Error updating manifest: {e}")

def main():
    # parser = argparse.ArgumentParser(description='Convert .opus files to .wav format')
    # parser.add_argument('input_dir', help='Directory containing .opus files')
    # parser.add_argument('--output-dir', '-o', help='Base directory to save .wav files with the same structure')
    # parser.add_argument('--workers', '-w', type=int, default=4, help='Number of parallel workers')
    # parser.add_argument('--ffmpeg-path', help='Path to ffmpeg executable')
    # parser.add_argument('--manifest', '-m', help='Path to manifest.json or manifest.jsonl file to update')
    # parser.add_argument('--output-manifest', help='Path to save updated manifest')
    # parser.add_argument('--no-recursive', action='store_true', help='Do not search recursively in subdirectories')
    
    input_dir = "/external4/datasets/russian_cv_test"
    output_dir = "/external4/datasets/russian_opensr_cv/test_wav"
    ffmpeg_path = "/hydra2-prev/home/compute/workspace_himanshu/ffmpeg-7.0.2-amd64-static/ffmpeg"
    manifest = "/external4/datasets/golos-data/test_opus/manifest.jsonl"
    output_manifest = "/external4/datasets/golos-data/test_wav/manifest.jsonl"
   
    
    # Convert files
    conversion_map = process_directory(
        input_dir, 
        output_dir, 
        4, 
        ffmpeg_path,
        
    )
    
    # Update manifest if provided
    if args.manifest and conversion_map:
        update_manifest(manifest, conversion_map, output_manifest)
    
    print(f"Conversion complete. Converted {len(conversion_map)} files.")

if __name__ == "__main__":
    main()