from pydub import AudioSegment
import os

def convert_and_trim_audio_with_overlap(input_folder, output_folder, target_format="mp3", segment_length_sec=30, overlap_sec=10):
    """
    Converts and trims audio files from input_folder into overlapping segments.

    Args:
        input_folder (str): Directory with input audio files.
        output_folder (str): Directory where output segments will be saved.
        target_format (str): Output format (e.g., "mp3", "wav").
        segment_length_sec (int): Segment duration in seconds.
        overlap_sec (int): Overlap between segments in seconds.
    """
    os.makedirs(output_folder, exist_ok=True)

    segment_duration_ms = segment_length_sec * 1000
    overlap_duration_ms = overlap_sec * 1000
    step_ms = segment_duration_ms - overlap_duration_ms

    for filename in os.listdir(input_folder):
        input_filepath = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)

        if ext[1:].lower() == target_format.lower():
            print(f"Skipping {filename}: Already in {target_format} format.")
            continue

        try:
            audio = AudioSegment.from_file(input_filepath)
            duration_ms = len(audio)

            if duration_ms <= segment_duration_ms:
                output_filename = f"{name}_segment_1.{target_format}"
                output_filepath = os.path.join(output_folder, output_filename)
                audio.export(output_filepath, format=target_format)
                print(f"Exported {output_filename} (short audio)")
                continue

            num_segments = max(1, (duration_ms - segment_duration_ms) // step_ms + 1)

            print(f"Processing {filename}: {duration_ms} ms total, creating {num_segments} segments with {overlap_sec}s overlap.")

            for i in range(num_segments):
                start_ms = i * step_ms
                end_ms = min(start_ms + segment_duration_ms, duration_ms)

                segment = audio[start_ms:end_ms]
                output_filename = f"{name}_segment_{i+1}.{target_format}"
                output_filepath = os.path.join(output_folder, output_filename)
                segment.export(output_filepath, format=target_format)
                print(f"  - Exported {output_filename}")

            # Optional: Handle final chunk if there's leftover audio at the end
            if (duration_ms - (num_segments * step_ms)) > 0 and (duration_ms > segment_duration_ms):
                start_ms = duration_ms - segment_duration_ms
                segment = audio[start_ms:duration_ms]
                output_filename = f"{name}_segment_{num_segments+1}.{target_format}"
                output_filepath = os.path.join(output_folder, output_filename)
                segment.export(output_filepath, format=target_format)
                print(f"  - Exported {output_filename} (final segment)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Run the updated function
input_audio_folder = '/external4/datasets/bucket_data/wellness'
output_audio_folder = '/external4/datasets/bucket_data/wellness/overlap'
print("Starting audio processing with overlap...")
convert_and_trim_audio_with_overlap(input_audio_folder, output_audio_folder, target_format="mp3", segment_length_sec=30, overlap_sec=10)
print("Audio processing complete.")