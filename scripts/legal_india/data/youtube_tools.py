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
import yt_dlp

# Configure logging
logger = logging.getLogger(__name__)

# --- Helper Functions & Classes ---

def _sanitize_filename(filename: str) -> str:
    """Sanitize a string to be a valid filename."""
    sanitized = re.sub(r'[^\w\-_.]', '_', str(filename).replace(' ', '_'))
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')[:100]

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

    def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos with pagination and return structured metadata."""
        if not self.youtube:
            return [{"status": "error", "error_message": "YouTube client not initialized."}]

        videos = []
        next_page_token = None

        try:
            while len(videos) < max_results:
                # API max is 50 per page
                results_to_fetch = min(max_results - len(videos), 50)

                search_request = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=results_to_fetch,
                    type='video',
                    videoCaption='closedCaption',
                    pageToken=next_page_token
                )
                search_response = search_request.execute()

                video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
                if not video_ids:
                    break  # No more results

                video_response = self.youtube.videos().list(
                    part='id,snippet,contentDetails,statistics',
                    id=','.join(video_ids)
                ).execute()

                for item in video_response.get('items', []):
                    videos.append({
                        "video_id": item["id"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "duration": item["contentDetails"]["duration"],
                        "view_count": int(item["statistics"].get("viewCount", 0)),
                        "url": f"https://www.youtube.com/watch?v={item['id']}"
                    })

                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break  # No more pages

            return videos[:max_results] # Return exactly the number requested

        except Exception as e:
            logger.error(f"An error occurred during YouTube search: {e}")
            return [{"status": "error", "error_message": str(e)}]


# --- Internal Worker Functions (Not exposed as tools) ---

def _download_transcript_worker(video_info: Dict[str, Any], language: str, output_dir: Path) -> Dict[str, Any]:
    """Worker to download a single transcript."""
    video_id = video_info['video_id']
    title = video_info['title']
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    sanitized_title = _sanitize_filename(title)
    
    video_output_dir = output_dir / _sanitize_filename(f"{video_id}_{title}")
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'quiet': True, 'no_warnings': True, 'skip_download': True,
        'writeautomaticsub': True, 'writesubtitles': True,
        'subtitleslangs': [language], 'subtitlesformat': 'vtt',
        'outtmpl': str(video_output_dir / f'{sanitized_title}')
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        expected_file = video_output_dir / f"{sanitized_title}.{language}.vtt"
        if expected_file.exists():
            return {"video_id": video_id, "status": "success", "path": str(expected_file)}
        return {"video_id": video_id, "status": "error", "message": "File not found after download."}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": str(e)}

def _download_video_worker(video_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Worker to download a single video file."""
    video_id = video_info['video_id']
    title = video_info['title']
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    sanitized_title = _sanitize_filename(title)

    video_output_dir = output_dir / _sanitize_filename(f"{video_id}_{title}")
    video_output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'quiet': True, 'no_warnings': True,
        'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': str(video_output_dir / f'{sanitized_title}.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if Path(downloaded_path).exists():
                return {"video_id": video_id, "status": "success", "path": downloaded_path}
        return {"video_id": video_id, "status": "error", "message": "File not found after download."}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": str(e)}

# --- Agent Tools ---

def search_youtube_videos(query: str, max_results: int = 5) -> str:
    """
    Searches YouTube for videos with closed captions based on a query.

    Args:
        query (str): The search term (e.g., "Supreme Court hearings").
        max_results (int): The maximum number of video results to return.

    Returns:
        str: A JSON string containing a list of found videos with their metadata,
             or an error message.
    """
    print(f"--- Tool: search_youtube_videos called for query: {query} ---")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return json.dumps({"status": "error", "error_message": "YOUTUBE_API_KEY not set."})
    
    client = YouTubeAPIClient(api_key)
    results = client.search_videos(query, max_results)
    return json.dumps(results)


def download_media_in_parallel(videos_json: str, download_target: str, language: str, output_dir: str = "./downloads", max_workers: int = 10) -> str:
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
            for video in videos:
                futures.append(executor.submit(_download_video_worker, video, output_path))
        
        if download_target in ['transcript', 'both']:
            for video in videos:
                futures.append(executor.submit(_download_transcript_worker, video, language, output_path))

        for future in as_completed(futures):
            result = future.result()
            # Distinguish between video and transcript results by checking the path/key
            if "transcript_path" in str(result): # Simple heuristic
                 results["transcript_downloads"].append(result)
            else:
                 results["video_downloads"].append(result)

    return json.dumps(results, indent=2)
