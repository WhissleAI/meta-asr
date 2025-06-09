# applications/gcs_utils.py
import os
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage
from config import logger, GOOGLE_APPLICATION_CREDENTIALS_PATH, TEMP_DOWNLOAD_DIR

_storage_client: Optional[storage.Client] = None

def get_gcs_client() -> Optional[storage.Client]:
    """Initializes and returns a GCS storage client if credentials are set."""
    global _storage_client
    if _storage_client:
        return _storage_client

    if GOOGLE_APPLICATION_CREDENTIALS_PATH and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS_PATH):
        try:
            _storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS_PATH)
            logger.info("Google Cloud Storage client initialized successfully using service account.")
            return _storage_client
        except Exception as e:
            logger.error(f"Failed to initialize GCS client from service account json {GOOGLE_APPLICATION_CREDENTIALS_PATH}: {e}", exc_info=True)
            return None
    else:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS_PATH is not set or file does not exist. GCS client not initialized.")
        return None

def parse_gcs_path(gcs_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Parses a GCS path (gs://bucket_name/blob_name) into bucket and blob names."""
    if not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path format: {gcs_path}. Must start with gs://")
        return None, None
    try:
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else None
        if not bucket_name or not blob_name: # Blob name can be empty if it's just the bucket, but we need a blob.
            logger.error(f"Invalid GCS path: {gcs_path}. Could not extract bucket or blob name.")
            return None, None
        return bucket_name, blob_name
    except Exception as e:
        logger.error(f"Error parsing GCS path '{gcs_path}': {e}")
        return None, None

def download_gcs_blob(bucket_name: str, blob_name: str) -> Optional[Path]:
    """Downloads a blob from GCS to the configured temporary directory.
    Returns the local path to the downloaded file, or None on failure.
    """
    client = get_gcs_client()
    if not client:
        logger.error("GCS client not available. Cannot download blob.")
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.error(f"Blob gs://{bucket_name}/{blob_name} does not exist.")
            return None

        # Construct a unique-ish local file name to avoid collisions if multiple users download same file name
        # For simplicity, using blob_name directly. For more robustness, add a unique ID or user ID.
        # The TEMP_DOWNLOAD_DIR should be cleaned periodically by a separate process.
        local_file_name = Path(blob_name).name # Get the base file name
        destination_path = Path(TEMP_DOWNLOAD_DIR) / local_file_name
        
        # Ensure parent directory of destination_path exists (TEMP_DOWNLOAD_DIR itself is created in config.py)
        # destination_path.parent.mkdir(parents=True, exist_ok=True) # TEMP_DOWNLOAD_DIR is the parent

        logger.info(f"Attempting to download gs://{bucket_name}/{blob_name} to {destination_path}")
        blob.download_to_filename(str(destination_path))
        logger.info(f"Successfully downloaded gs://{bucket_name}/{blob_name} to {destination_path}")
        return destination_path
    except Exception as e:
        logger.error(f"Failed to download blob gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

# Example usage (for testing this module directly):
# if __name__ == "__main__":
#     # Make sure .env is loaded if you run this directly and GOOGLE_APPLICATION_CREDENTIALS_PATH is set
#     # from dotenv import load_dotenv
#     # load_dotenv('D:/z-whissle/meta-asr/.env') # Adjust path as needed
#     # print(f"Using GCS Key: {GOOGLE_APPLICATION_CREDENTIALS_PATH}")
#     # print(f"Temp dir: {TEMP_DOWNLOAD_DIR}")

#     test_gcs_path = "gs://your-public-bucket-name/path/to/some-audio.wav" # Replace with a real public path for testing
#     bucket, blob_name_path = parse_gcs_path(test_gcs_path)
#     if bucket and blob_name_path:
#         print(f"Bucket: {bucket}, Blob: {blob_name_path}")
#         downloaded_file = download_gcs_blob(bucket, blob_name_path)
#         if downloaded_file:
#             print(f"File downloaded to: {downloaded_file}")
#         else:
#             print("Download failed.")
#     else:
#         print("Invalid GCS path for testing.")
