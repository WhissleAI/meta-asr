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
Examples:
  1. Basic search (downloads transcripts by default):
     python run_adk_agent.py "live supreme court hearings"

  2. Download only video files:
     python run_adk_agent.py "python tutorial" --download video --max-results 3

  3. Download both videos and transcripts:
     python run_adk_agent.py "data science" --download both --language "en"

Note: API keys can also be set as environment variables or in a .env file.
(GOOGLE_API_KEY, YOUTUBE_API_KEY)
"""
    )

    parser.add_argument('query', type=str, help='The search query for YouTube videos.')
    parser.add_argument('--language', type=str, default='en', help='Language for the transcript (e.g., "en", "es").')
    parser.add_argument('--max-results', type=int, default=200, help='Maximum number of videos to search for.')
    parser.add_argument('--download', type=str, choices=['transcript', 'video', 'both'], default='transcript',
                        help='Specify what to download: transcript, video, or both.')
    parser.add_argument('--output-dir', type=str, default='./downloads', help='Directory to save downloaded files.')
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
    print("-" * 40)

    try:
        asyncio.run(run_youtube_downloader_agent(
            query=args.query,
            language=args.language,
            max_results=args.max_results,
            download_target=args.download,
            output_dir=args.output_dir
        ))
        print("\n" + "-" * 40)
        print("✅ Agent run completed successfully.")
    except KeyboardInterrupt:
        print("\n🛑 Agent run interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        logging.error("Agent run failed.", exc_info=True)

if __name__ == '__main__':
    main()
