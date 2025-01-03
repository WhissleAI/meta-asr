from google.cloud import storage
import os

# Initialize the client
client = storage.Client()

# Specify your bucket name
bucket_name = "avspeech-data"

# Create a local directory to store downloaded files
download_path = "./avspeech-data"
os.makedirs(download_path, exist_ok=True)

bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Iterate through the blobs and download videos
for blob in blobs:
        # Determine local file path
        file_path = os.path.join(download_path, blob.name)
        
        # Ensure directory structure is created locally
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download the file
        print(f"Downloading {blob.name} to {file_path}...")
        blob.download_to_filename(file_path)

print("Download complete.")
