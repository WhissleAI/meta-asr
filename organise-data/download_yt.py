import yt_dlp
import os

# Define the folder to save the downloaded video
output_folder = '/external2/datasets/yt_data1'

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set yt_dlp options to save the video in the specified folder and use cookies
ydl_opts = {
    'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
    'cookiefile': '/hydra2-prev/home/compute/workspace_himanshu/meta-asr/organise-data/cookies.txt',  # Ensure this path is correct
}

# Download the video
links = ["https://youtu.be/uKO8tk3_itQ?si=UvOvGBRay9jQy_vK","https://youtu.be/7qZl_5xHoBw?si=qCAjdv_xq3OAaWCF","https://youtu.be/F-MEegYIWaM?si=b-KY_oa9Uq4O-DMS"]
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(links)
