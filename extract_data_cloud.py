from google.cloud import storage
import os

# Initialize the client
client = storage.Client()

# Specify your bucket name
bucket_name = "stream2action-audio"

# Create a local directory to store downloaded files
download_path = "./stream2action_videos"
os.makedirs(download_path, exist_ok=True)

# Access the bucket
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Iterate through the blobs and download videos
for blob in blobs:
    # Check if the blob is a video file (adjust extension check as needed)
    if blob.name.endswith(('.mp4', '.avi', '.mov','.wav','.mp3')):
        # Determine local file path
        file_path = os.path.join(download_path, blob.name)
        
        # Ensure directory structure is created locally
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download the file
        print(f"Downloading {blob.name} to {file_path}...")
        blob.download_to_filename(file_path)

print("Download complete.")
