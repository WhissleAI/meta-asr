from google.cloud import storage
import cv2
import numpy as np

def process_single_video_from_bucket(bucket_name, video_name):
    storage_client = storage.Client.create_anonymous_client()
    
    bucket = storage_client.bucket(bucket_name)
    

    blob = bucket.blob(video_name)
    
    print(f"Accessing {video_name} from bucket '{bucket_name}'...")
    
    video_data = blob.download_as_bytes()
    

    video_array = np.frombuffer(video_data, np.uint8)
    
    temp_video_file = "temp_video.mp4"
    with open(temp_video_file, "wb") as f:
        f.write(video_data)
    
 
    cap = cv2.VideoCapture(temp_video_file)

    if not cap.isOpened():
        print(f"Failed to open {video_name}")
        return
    
    print("Displaying video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


bucket_name = "stream2action-audio"
video_name = "youtube-videos/sample_video.mp4"  


process_single_video_from_bucket(bucket_name, video_name)
