import yt_dlp
import os

# Where to save files
output_folder = '/external4/datasets/youtube_videos/'
os.makedirs(output_folder, exist_ok=True)

links = ["https://youtu.be/6oaOoZZdtZ8?si=H68615oCY-a-HQwj"]

ydl_opts = {
    'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
    'noplaylist': True,  # don't accidentally download a whole playlist
    'cookiefile': '/hydra2-prev/home/compute/workspace_himanshu/meta-asr/organise-data/cookies.txt',

    # Get best quality video+audio
    'format': 'bv*+ba/b',

    # Subtitles
    'writesubtitles': True,        # download creator-uploaded subs
    'writeautomaticsub': True,     # also allow auto-generated (ASR) subs
    'subtitleslangs': ['en', 'en-US', 'en-GB'],  # change or use ['all'] to grab everything

    # Grab the best sub format from YouTube (usually .vtt) and convert to .srt
    'subtitlesformat': 'best',
    'postprocessors': [
        {'key': 'FFmpegSubtitlesConvertor', 'format': 'srt'},  # vtt → srt
    ],

    # Optional hygiene
    'restrictfilenames': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(links)
print("Download complete!")