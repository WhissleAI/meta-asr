#!/usr/bin/env python3
"""
CLI Runner for the Google ADK YouTube Agent

This script provides a command-line interface to interact with the YouTube
downloader agent, handle arguments, and manage API keys.
"""

import os
import asyncio
import argparse
import logging
from dotenv import load_dotenv

# --- Main Agent Logic ---
# Ensure the agent script is importable
try:
    from youtube_adk_agent import run_youtube_downloader_agent
except ImportError:
    print("Error: Could not import 'run_youtube_downloader_agent'.")
    print("Ensure 'youtube_adk_agent.py' is in the same directory.")
    exit(1)

def setup_api_keys(args: argparse.Namespace):
    """Load API keys from .env, environment, or command-line arguments."""
    load_dotenv() # Load from .env file if it exists

    # Google Gemini API Key
    if args.google_api_key:
        os.environ["GOOGLE_API_KEY"] = args.google_api_key
    elif not os.environ.get("GOOGLE_API_KEY"):
        print("Error: Google Gemini API Key is required.")
        print("Set GOOGLE_API_KEY in your environment, .env file, or use --google-api-key.")
        exit(1)
    
    # YouTube Data API Key
    if args.youtube_api_key:
        os.environ["YOUTUBE_API_KEY"] = args.youtube_api_key
    elif not os.environ.get("YOUTUBE_API_KEY"):
        print("Error: YouTube Data API Key is required.")
        print("Set YOUTUBE_API_KEY in your environment, .env file, or use --youtube-api-key.")
        exit(1)
    
    # Configure ADK to use API keys directly
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

    print("API Keys configured successfully.")

def main():
    """Main function to parse arguments and run the agent."""
    parser = argparse.ArgumentParser(
        description='CLI Runner for the Google ADK YouTube Agent.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python run_adk_agent.py "live supreme court hearing" --max-results 10 --download both --output-dir "/path/to/downloads" --gcs-bucket "my-bucket" --region IN --channel "SupremeCourtofIndia-1950" --max-workers 5 --cookies "/path/to/cookies.txt"
"""
    )

    parser.add_argument('query', type=str, help='The search query for YouTube videos.')
    parser.add_argument('--language', type=str, default='en', help='Language for the transcript (e.g., "en", "es").')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum number of videos to search for.')
    parser.add_argument('--download', type=str, choices=['transcript', 'video', 'both'], default='transcript',
                        help='Specify what to download: transcript, video, or both.')
    parser.add_argument('--output-dir', type=str, default='./downloads', help='Directory to save downloaded files.')
    parser.add_argument('--gcs-bucket', type=str, help='Google Cloud Storage bucket name to upload files to. If provided, uploads are enabled.')
    parser.add_argument('--region', type=str, help='Region code to prioritize search results (e.g., IN, US).')
    parser.add_argument('--channel', type=str, help='YouTube channel handle (e.g., "SupremeCourtofIndia-1950") to search within.')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel download workers.')
    parser.add_argument('--cookies', type=str, help='Path to a cookies file to authenticate yt-dlp downloads.')
    parser.add_argument('--google-api-key', type=str, help='Google Gemini API Key.')
    parser.add_argument('--youtube-api-key', type=str, help='YouTube Data API v3 Key.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging for debugging.')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('google.api_core').setLevel(logging.ERROR)
    
    setup_api_keys(args)

    print("\n🚀 Starting YouTube ADK Agent...")
    print(f"   Query: '{args.query}'")
    print(f"   Language: '{args.language}'")
    print(f"   Max Results: {args.max_results}")
    print(f"   Download Target: '{args.download}'")
    print(f"   Output Directory: '{args.output_dir}'")
    if args.gcs_bucket:
        print(f"   GCS Upload Bucket: '{args.gcs_bucket}'")
    if args.region:
        print(f"   Search Region: '{args.region}'")
    if args.channel:
        print(f"   Search Channel: '{args.channel}'")
    if args.cookies:
        print(f"   Cookies File: '{args.cookies}'")
    print("-" * 40)

    try:
        # The new workflow is a single, direct execution
        asyncio.run(run_youtube_downloader_agent(
            query=args.query,
            language=args.language,
            max_results=args.max_results,
            download_target=args.download,
            output_dir=args.output_dir,
            gcs_bucket=args.gcs_bucket,
            region_code=args.region,
            channel_handle=args.channel,
            max_workers=args.max_workers,
            cookies_file=args.cookies
        ))
        print("\n" + "-" * 40)
        print("✅ Agent workflow completed successfully.")
    except KeyboardInterrupt:
        print("\n🛑 Agent run interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        logging.error("Agent run failed.", exc_info=True)

if __name__ == '__main__':
    main()
