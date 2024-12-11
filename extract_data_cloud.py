from google.cloud import storage
import cv2
import numpy as np

def process_single_video_from_bucket(bucket_name, video_name):
    """Processes a single video directly from a Google Cloud Storage bucket."""
    # Create a client without credentials for public buckets
    storage_client = storage.Client.create_anonymous_client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Get the blob (video file)
    blob = bucket.blob(video_name)
    
    print(f"Accessing {video_name} from bucket '{bucket_name}'...")
    
    # Stream video bytes from the cloud
    video_data = blob.download_as_bytes()
    
    # Convert bytes to a NumPy array for OpenCV processing
    video_array = np.frombuffer(video_data, np.uint8)
    
    # Save video to a temporary file for OpenCV to access
    temp_video_file = "temp_video.mp4"
    with open(temp_video_file, "wb") as f:
        f.write(video_data)
    
    # Open video using OpenCV
    cap = cv2.VideoCapture(temp_video_file)

    if not cap.isOpened():
        print(f"Failed to open {video_name}")
        return
    
    print("Displaying video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Display frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Define the bucket name and video name
bucket_name = "stream2action-audio"
video_name = "youtube-videos/sample_video.mp4"  # Replace with an actual video name in your bucket

# Call the function
process_single_video_from_bucket(bucket_name, video_name)
