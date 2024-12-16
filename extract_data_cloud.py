from google.cloud import storage

def list_public_bucket_objects(bucket_name, prefix=None):
    """Lists objects in a public Google Cloud Storage bucket."""
    # Create a client without credentials for public buckets
    storage_client = storage.Client.create_anonymous_client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # List objects with an optional prefix
    blobs = bucket.list_blobs(prefix=prefix)
    
    print(f"Objects in bucket '{bucket_name}':")
    for blob in blobs:
        print(blob.name)

# Define the bucket name and optional prefix
bucket_name = "stream2action-audio"
prefix = "youtube-videos"

# Call the function
list_public_bucket_objects(bucket_name, prefix)
