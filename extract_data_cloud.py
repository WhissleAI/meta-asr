from google.cloud import storage
import os

# Initialize Google Cloud Storage client
client = storage.Client()

# Specify the bucket name
bucket_name = "stream2action-audio"
bucket = client.get_bucket(bucket_name)

# Set download directory
download_path = "./batch_processing"
os.makedirs(download_path, exist_ok=True)

# Initialize processed count
processed_count = 0

# List all blobs in the bucket
blobs = list(bucket.list_blobs())

# Process only the first video
for blob in blobs:
    if blob.name.endswith(('.mp4', '.avi', '.mov')):
        # Construct the local file path
        local_file_path = os.path.join(download_path, blob.name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        print(f"Downloading {blob.name}...")
        blob.download_to_filename(local_file_path)

        print(f"Processing {local_file_path}...")
        # Add your video processing logic here if needed

        processed_count += 1
        break  # Exit the loop after processing the first video

print(f"Total videos processed: {processed_count}")
