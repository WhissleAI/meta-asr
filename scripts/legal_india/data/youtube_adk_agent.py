#!/usr/bin/env python3
"""
Google ADK Agent for YouTube Data Collection

This script defines and runs a Gemini-powered agent that uses the Google
Agent Development Kit (ADK) to search for YouTube videos and download their
transcripts.
"""

import os
import asyncio
import logging
import json
from typing import Optional

# --- ADK and Gemini Imports ---
try:
    from google.adk.agents import Agent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types as genai_types
except ImportError:
    print("Error: Google ADK not installed. Please run 'pip install google-adk'")
    exit(1)

# --- Import Agent Tools & Helper Functions ---
from youtube_tools import search_youtube_videos, download_media_in_parallel, upload_files_to_gcs_in_parallel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('google.api_core').setLevel(logging.ERROR) # Quieten noisy logs

# Use a model that supports tool calling. More models are listed here:
# https://ai.google.dev/gemini-api/docs/models/gemini
AGENT_MODEL = "gemini-2.5-pro"

# --- Define Agent ---
youtube_agent = Agent(
    name="youtube_agent_v1",
    model=AGENT_MODEL,
    description="Finds YouTube videos and returns the raw JSON data.",
    instruction="""
    You are a data-focused assistant for YouTube. Your ONLY job is to find videos
    and return the results as raw, unformatted JSON.

    Workflow:
    1.  When the user gives you a query, immediately call the `search_youtube_videos` tool.
    2.  Take the JSON output from the `search_youtube_videos` tool.
    3.  Your final response to the user MUST be ONLY the raw, complete, and unformatted
        JSON string that the tool provided.

    **CRITICAL RULES**:
    - DO NOT be conversational.
    - DO NOT provide any text before or after the JSON.
    - DO NOT summarize the results.
    - DO NOT ask the user what they want to do next.
    - Your entire output must be the JSON from the tool.
    """,
    tools=[search_youtube_videos],
)


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and returns the final text response."""
    from google.genai import types
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    return final_response_text


async def run_youtube_downloader_agent(query: str, language: str, max_results: int, download_target: str, output_dir: str, gcs_bucket: str = None, region_code: str = None, channel_handle: str = None, max_workers: int = 5, cookies_file: Optional[str] = None):
    """
    Main function to orchestrate the YouTube downloader agent.
    """
    # --- Step 1: Initialize Runner and Session ---
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    import uuid

    session_service = InMemorySessionService()
    runner = Runner(
        agent=youtube_agent,
        app_name="youtube_downloader_app",
        session_service=session_service
    )

    user_id = "user__default"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # Explicitly create the session in the session service
    await session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"Session created: {session_id}")

    # --- Step 2: Call the Agent to Get Search Results ---
    initial_query = f"Please find the top {max_results} videos about '{query}'"
    if channel_handle:
        initial_query += f" within the YouTube channel with handle '{channel_handle}'"
    if region_code:
        initial_query += f" with a preference for results from the region '{region_code}'."
    
    print(f"🤖 Calling agent with prompt: \"{initial_query}\"")
    search_results_text = await call_agent_async(initial_query, runner, user_id, session_id)
    
    print("\n--- Agent Raw Response ---", flush=True)
    print(search_results_text, flush=True)
    print("--- End Agent Raw Response ---\n", flush=True)

    try:
        # The agent sometimes wraps the JSON in markdown, so we strip it.
        cleaned_json = search_results_text.strip().removeprefix("```json").removesuffix("```")
        video_list = json.loads(cleaned_json)

        # Check if the tool returned an error before proceeding
        if isinstance(video_list, dict) and 'error' in video_list:
            print(f"❌ Tool returned an error: {video_list['error']}. Aborting.", flush=True)
            return
        if isinstance(video_list, list) and video_list and 'error' in video_list[0]:
             print(f"❌ Tool returned an error: {video_list[0]['error']}. Aborting.", flush=True)
             return

        if not isinstance(video_list, list) or not all(isinstance(item, dict) for item in video_list):
            print("\n❌ Agent did not return a valid JSON list of videos. Aborting.", flush=True)
            print(f"Received: {search_results_text}", flush=True)
            return
    except json.JSONDecodeError:
        print("❌ Failed to decode the agent's response as JSON. Aborting.", flush=True)
        print(f"Received: {search_results_text}", flush=True)
        return

    print(f"✅ Agent returned {len(video_list)} videos. Proceeding to download...", flush=True)

    # Call the download tool directly
    download_results_json = download_media_in_parallel(
        videos_json=json.dumps(video_list), # Pass the cleaned and parsed list
        download_target=download_target,
        language=language,
        output_dir=output_dir,
        max_workers=max_workers,
        cookies_file=cookies_file
    )
    download_results = json.loads(download_results_json)
    print("\n--- Download Summary ---")
    print(json.dumps(download_results, indent=2))

    # If a GCS bucket is specified, call the upload tool directly
    if gcs_bucket:
        successful_downloads = []
        if download_results.get("video_downloads"):
            successful_downloads.extend([item['path'] for item in download_results["video_downloads"] if item['status'] == 'success'])
        if download_results.get("transcript_downloads"):
            successful_downloads.extend([item['path'] for item in download_results["transcript_downloads"] if item['status'] == 'success'])

        if successful_downloads:
            print(f"\n🚀 Uploading {len(successful_downloads)} files to GCS bucket: {gcs_bucket}...", flush=True)
            upload_results_json = upload_files_to_gcs_in_parallel(
                local_files_json=json.dumps(successful_downloads),
                bucket_name=gcs_bucket,
                max_workers=max_workers
            )
            upload_results = json.loads(upload_results_json)
            print("\n--- Upload Summary ---")
            print(json.dumps(upload_results, indent=2))
        else:
            print("\n⚠️ No files were downloaded successfully, skipping upload.")


if __name__ == '__main__':
    # This is a simple example for direct execution.
    # A separate run_adk_agent.py will handle command-line arguments.
    
    # --- IMPORTANT: Set API Keys ---
    # Make sure to set GOOGLE_API_KEY and YOUTUBE_API_KEY in your environment
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("YOUTUBE_API_KEY"):
        print("ERROR: Please set 'GOOGLE_API_KEY' and 'YOUTUBE_API_KEY' environment variables.")
    else:
        print("API keys found. Running example conversation...")
        
        # --- Example Conversation ---
        try:
            asyncio.run(run_youtube_downloader_agent(
                query="Supreme Court of India live hearing",
                language="en",
                max_results=3,
                download_target="transcript",
                output_dir="./downloads",
                gcs_bucket=None, # Or a bucket name to test upload e.g., "my-youtube-data-bucket"
                region_code="IN"
            ))
        except Exception as e:
            print(f"An error occurred during the agent run: {e}")
