import pandas as pd
import pprint

# Path to the parquet file
parquet_file = "/external4/datasets/madasr/Vaani-transcription-part/audio/Bhojpuri/train-00002-of-00004.parquet"

# Load the parquet file
df = pd.read_parquet(parquet_file)

# Print basic information
print(f"DataFrame shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Check the first row
first_row = df.iloc[0]
print(f"\nFirst row keys: {list(first_row.keys())}")

# Check the audio field of the first row
audio_field = first_row['audio']
print(f"\nType of audio field: {type(audio_field)}")

# Check if the audio field is a dictionary
if hasattr(audio_field, 'keys'):
    print(f"Audio field keys: {list(audio_field.keys())}")
else:
    print("Audio field is not a dictionary")
    print(f"Audio field attributes: {dir(audio_field)}")

# Check the transcript field
transcript_field = first_row['transcript']
print(f"\nType of transcript field: {type(transcript_field)}")
print(f"Transcript content: {transcript_field}")

# Check the structure of a row that causes an error
try:
    error_row = df.iloc[2637]
    print("\nRow 2637 (error row):")
    print(f"Keys: {list(error_row.keys())}")
    print(f"Has audio field: {'audio' in error_row}")
    if 'audio' in error_row:
        print(f"Type of audio field: {type(error_row['audio'])}")
except Exception as e:
    print(f"Error examining row 2637: {str(e)}") 