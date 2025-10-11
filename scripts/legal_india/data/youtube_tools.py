#!/usr/bin/env python3
"""
Tools for the YouTube Data Agent

This module provides the core functionalities for searching videos,
extracting metadata, and downloading transcripts, designed to be used
as tools within a Google ADK agent.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party libraries, assuming they are installed via requirements.txt
from googleapiclient.discovery import build
from google.cloud import storage
import yt_dlp

# Configure logging
logger = logging.getLogger(__name__)

# --- Helper Functions & Classes ---

def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitizes a string to be a valid filename and truncates it."""
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Truncate to max_length
    return sanitized[:max_length]

class YouTubeAPIClient:
    """Client for interacting with the YouTube Data API v3."""
    _instance = None

    def __new__(cls, api_key: str):
        if cls._instance is None:
            cls._instance = super(YouTubeAPIClient, cls).__new__(cls)
            try:
                cls._instance.youtube = build('youtube', 'v3', developerKey=api_key)
                logger.info("YouTube API client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to build YouTube service: {e}")
                cls._instance.youtube = None
        return cls._instance

    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def get_channel_id(self, channel_handle: str) -> Optional[str]:
        """Resolves a channel handle to a channel ID."""
        try:
            # YouTube API expects handles to start with '@' in search queries
            q_handle = channel_handle if channel_handle.startswith('@') else f'@{channel_handle}'
            
            search_response = self.youtube.search().list(
                part='id',
                q=q_handle,
                type='channel',
                maxResults=1
            ).execute()
            
            if search_response.get("items"):
                return search_response["items"][0]["id"]["channelId"]
            return None
        except Exception as e:
            print(f"Error resolving channel handle '{channel_handle}': {e}")
            return None

    def search_videos(self, query, max_results=50, region_code=None, channel_id=None):
        """
        Searches for videos, handling pagination to retrieve up to max_results.
        """
        all_videos = []
        next_page_token = None

        while len(all_videos) < max_results:
            page_size = min(50, max_results - len(all_videos))
            try:
                # If a channel ID is provided, we search within that channel.
                # The 'q' parameter can be omitted to get all videos from the channel.
                request = self.youtube.search().list(
                    part="snippet",
                    q=query, # Always use the user's query
                    type="video",
                    # videoCaption="closedCaption", # This filter is too restrictive.
                    maxResults=page_size,
                    regionCode=region_code,
                    channelId=channel_id,
                    pageToken=next_page_token,
                    order="date" # Fetch most recent videos first
                )
                response = request.execute()

                video_ids = []
                for item in response.get("items", []):
                    # Safely extract videoId, skipping any non-video results
                    if item.get("id", {}).get("kind") == "youtube#video":
                        video_id = item["id"].get("videoId")
                        if video_id:
                            video_ids.append(video_id)
                
                if not video_ids:
                    break # No more video IDs found on this page

                # Get detailed video information for the collected IDs
                video_details_request = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids)
                ).execute()

                for item in video_details_request.get("items", []):
                    all_videos.append({
                        "video_id": item["id"],
                        "title": item["snippet"].get("title", "No Title Provided"),
                        "description": item["snippet"].get("description", ""),
                        "published_at": item["snippet"].get("publishedAt"),
                        "channel": item["snippet"].get("channelTitle", "N/A"),
                        "duration": item["contentDetails"].get("duration"),
                        "view_count": int(item["statistics"].get("viewCount", 0)),
                        "like_count": int(item["statistics"].get("likeCount", 0)),
                        "comment_count": int(item["statistics"].get("commentCount", 0)),
                        "url": f"https://www.youtube.com/watch?v={item['id']}"
                    })

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            except Exception as e:
                print(f"An error occurred during API call: {e}")
                # Stop paginating if an error occurs
                break
        return all_videos


# --- Internal Worker Functions (Not exposed as tools) ---

def _download_transcript_worker(video_info: Dict[str, Any], language: str, output_dir: Path, cookies_file: Optional[str] = None) -> Dict[str, Any]:
    """Worker to download a single transcript."""
    video_id = video_info['video_id']
    title = video_info.get('title', 'no_title')
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    sanitized_title = _sanitize_filename(title)
    video_output_dir = output_dir / f"{video_id}-{sanitized_title}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'quiet': True, 'no_warnings': True, 'skip_download': True,
        'writeautomaticsub': True, 'writesubtitles': True,
        'subtitleslangs': [language], 'subtitlesformat': 'vtt',
        'outtmpl': str(video_output_dir / f'{sanitized_title}')
    }
    if cookies_file:
        ydl_opts['cookiefile'] = cookies_file
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        expected_file = video_output_dir / f"{sanitized_title}.{language}.vtt"
        if expected_file.exists():
            return {"video_id": video_id, "status": "success", "path": str(expected_file)}
        return {"video_id": video_id, "status": "error", "message": "File not found after download (likely no transcript available)."}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": str(e)}

def _download_video_worker(video_info: Dict[str, Any], output_dir: Path, cookies_file: Optional[str] = None) -> Dict[str, Any]:
    """Worker to download a single video file and its metadata."""
    video_id = video_info['video_id']
    title = video_info.get('title', 'no_title')
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    sanitized_title = _sanitize_filename(title)
    video_output_dir = output_dir / f"{video_id}-{sanitized_title}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata to a JSON file
    metadata_path = video_output_dir / "metadata.json"
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save metadata for {video_id}: {e}")


    ydl_opts = {
        'quiet': True, 'no_warnings': True,
        'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': str(video_output_dir / f'{sanitized_title}.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    if cookies_file:
        ydl_opts['cookiefile'] = cookies_file
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if Path(downloaded_path).exists():
                return {"video_id": video_id, "status": "success", "path": downloaded_path}
        return {"video_id": video_id, "status": "error", "message": "File not found after download."}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": str(e)}

# --- New GCS Worker ---
def _upload_to_gcs_worker(local_path: str, bucket_name: str, gcs_prefix: str) -> Dict[str, str]:
    """Worker to upload a single file to GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        file_path = Path(local_path)
        # Use the parent directory (video_id) and filename for a clean GCS path
        destination_blob_name = f"{gcs_prefix}/{file_path.parent.name}/{file_path.name}"
        
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        logger.info(f"Successfully uploaded {local_path} to {gcs_uri}")
        return {"local_path": local_path, "status": "success", "gcs_uri": gcs_uri}
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {e}")
        return {"local_path": local_path, "status": "error", "message": str(e)}

# --- Agent Tools ---

def search_youtube_videos(query: str, max_results: int = 5, region_code: Optional[str] = None, channel_handle: Optional[str] = None) -> str:
    """
    Searches YouTube for videos with closed captions based on a query, optionally
    biasing results for a specific country. Can be limited to a specific channel.

    Args:
        query (str): The search term.
        max_results (int): The maximum number of results to return.
        region_code (Optional[str]): The ISO 3166-1 alpha-2 country code (e.g., IN, US).
        channel_handle (Optional[str]): The handle of the YouTube channel to search within (e.g., "SupremeCourtofIndia-1950").

    Returns:
        str: A JSON string containing a list of video details or an error.
    """
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        return json.dumps({"error": "YOUTUBE_API_KEY is not set."})

    client = YouTubeAPIClient(api_key=YOUTUBE_API_KEY)

    channel_id = None
    if channel_handle:
        print(f"Resolving channel handle: '{channel_handle}'...", flush=True)
        channel_id = client.get_channel_id(channel_handle)
        if not channel_id:
            return json.dumps({"error": f"Could not find a channel with handle '{channel_handle}'."})
        print(f"Found channel ID: {channel_id}", flush=True)

    try:
        videos = client.search_videos(
            query,
            max_results=max_results,
            region_code=region_code,
            channel_id=channel_id
        )
        return json.dumps(videos, indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred during YouTube search: {e}"})


def download_media_in_parallel(videos_json: str, download_target: str, language: str, output_dir: str = "./downloads", max_workers: int = 5, cookies_file: Optional[str] = None) -> str:
    """
    Downloads video files and/or transcripts in parallel for a list of YouTube videos.

    Args:
        videos_json (str): A JSON string representing a list of video objects. Each object
                           must have 'video_id' and 'title'.
        download_target (str): What to download. Must be one of 'video', 'transcript', or 'both'.
        language (str): The language code for transcripts (e.g., 'en').
        output_dir (str): The root directory to save downloaded files.
        max_workers (int): The maximum number of parallel download workers.

    Returns:
        str: A JSON string summarizing the results of all download operations.
    """
    print(f"--- Tool: download_media_in_parallel called for {download_target} ---")
    try:
        videos = json.loads(videos_json)
        if not isinstance(videos, list):
            return json.dumps({"status": "error", "error_message": "Invalid videos_json format; expected a list."})
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "error_message": "Failed to decode videos_json string."})

    output_path = Path(output_dir)
    results = {"video_downloads": [], "transcript_downloads": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if download_target in ['video', 'both']:
            print("Submitting video download tasks...", flush=True)
            for video in videos:
                futures.append(executor.submit(_download_video_worker, video, output_path, cookies_file))
        
        if download_target in ['transcript', 'both']:
            print("Submitting transcript download tasks...", flush=True)
            for video in videos:
                futures.append(executor.submit(_download_transcript_worker, video, language, output_path, cookies_file))

        for future in as_completed(futures):
            result = future.result()
            # Distinguish between video and transcript results by checking the path/key
            if "transcript_path" in str(result): # Simple heuristic
                 results["transcript_downloads"].append(result)
            else:
                 results["video_downloads"].append(result)

    return json.dumps(results, indent=2)


def upload_files_to_gcs_in_parallel(local_files_json: str, bucket_name: str, gcs_prefix: str = "youtube_data", max_workers: int = 5) -> str:
    """
    Uploads a list of local files to a Google Cloud Storage bucket in parallel.

    Args:
        local_files_json (str): A JSON string of a list of local file paths to upload.
        bucket_name (str): The name of the GCS bucket.
        gcs_prefix (str): A prefix (folder) to use within the GCS bucket.
        max_workers (int): The maximum number of parallel upload workers.

    Returns:
        str: A JSON string summarizing the results of the upload operations.
    """
    print(f"--- Calling function upload_files_to_gcs_in_parallel for bucket: {bucket_name} ---")
    try:
        local_files = json.loads(local_files_json)
        if not isinstance(local_files, list):
            return json.dumps({"status": "error", "message": "Invalid JSON format; expected a list of file paths."})
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Failed to decode JSON."})

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_upload_to_gcs_worker, file_path, bucket_name, gcs_prefix) for file_path in local_files]
        for future in as_completed(futures):
            results.append(future.result())

    return json.dumps(results, indent=2)
