from google.cloud import storage
import os


client = storage.Client()


bucket_name = "stream2action-audio"
bucket = client.get_bucket(bucket_name)

download_path = "./batch_processing"
os.makedirs(download_path, exist_ok=True)

processed_count = 0

blobs = list(bucket.list_blobs())

for blob in blobs:
    if blob.name.endswith(('.mp4', '.avi', '.mov')):
      
        local_file_path = os.path.join(download_path, blob.name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        print(f"Downloading {blob.name}...")
        blob.download_to_filename(local_file_path)


        processed_count += 1
        break  

print(f"Total videos processed: {processed_count}")
