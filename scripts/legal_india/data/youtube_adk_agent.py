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

# --- ADK and Gemini Imports ---
try:
    from google.adk.agents import Agent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types as genai_types
except ImportError:
    print("Error: Google ADK not installed. Please run 'pip install google-adk'")
    exit(1)

# --- Import Agent Tools ---
from youtube_tools import search_youtube_videos, download_media_in_parallel, upload_files_to_gcs_in_parallel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('google.api_core').setLevel(logging.ERROR) # Quieten noisy logs

# Use a model that supports tool calling. More models are listed here:
# https://ai.google.dev/gemini-api/docs/models/gemini
AGENT_MODEL = "gemini-2.5-pro"

# --- Agent Definition ---

youtube_agent = Agent(
    name="youtube_agent_v1",
    model=AGENT_MODEL,
    description="Finds YouTube videos and downloads their transcripts or video files in parallel.",
    instruction="""
    You are an intelligent assistant for downloading YouTube data and uploading it to Google Cloud Storage.

    Workflow:
    1.  **Search**: When a user provides a topic, use `search_youtube_videos` to find relevant videos. If the user specifies a country or region, pass the corresponding two-letter ISO 3166-1 alpha-2 country code to the `region_code` argument.
    2.  **Confirm**: Present the search results to the user and ask what they want to download.
    3.  **Download**: Use the `download_media_in_parallel` tool, passing the entire JSON list of videos.
    4.  **Extract File Paths & Upload**: After downloads are complete, you must parse the JSON output from the download tool to create a new JSON list containing only the local file paths of the successfully downloaded files. Then, if the user requested an upload, you must call the `upload_files_to_gcs_in_parallel` tool, passing this new list of file paths to it.
    5.  **Summarize**: Provide a final summary of both the download and upload operations.

    **Important Rules**:
    - Always `search_youtube_videos` first.
    - If a region is mentioned, you must use the `region_code` argument in your search tool call.
    - Always use `download_media_in_parallel` for downloading.
    - **Crucially, after downloading, you must extract the `path` from each successful result and create a new JSON list to pass to `upload_files_to_gcs_in_parallel`.**
    - Do not try to upload files that failed to download.
    """,
    tools=[search_youtube_videos, download_media_in_parallel, upload_files_to_gcs_in_parallel],
)

# --- Runner & Session Setup ---

session_service = InMemorySessionService()
runner = Runner(
    agent=youtube_agent,
    app_name="youtube_adk_downloader",
    session_service=session_service
)

# --- Agent Interaction Logic ---

async def call_agent_async(query: str, user_id: str, session_id: str):
    """
    Sends a query to the agent, streams events, and prints the final response.
    """
    print(f"\n>>> User: {query}")
    print("--- Agent is thinking... ---")

    content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                break
    except Exception as e:
        final_response_text = f"An unexpected error occurred: {e}"
        logging.error(final_response_text)

    print(f"\n<<< Agent: {final_response_text}")
    return final_response_text

async def run_youtube_downloader_agent(query: str, language: str, max_results: int, download_target: str, output_dir: str, gcs_bucket: str = None, region_code: str = None):
    """
    Orchestrates a full conversation with the YouTube agent to find and download
    transcripts.
    """
    # --- Setup Session ---
    user_id = "user_main"
    session_id = f"session_{os.getpid()}"
    await session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"Session created: {session_id}")

    # --- Step 1: Initial Search Query ---
    initial_query = (
        f"Please find the top {max_results} videos about '{query}'"
    )
    if region_code:
        initial_query += f" with a preference for results from the region '{region_code}'."
    initial_query += " Do not download anything yet, just show me the results."
    
    search_results_text = await call_agent_async(initial_query, user_id, session_id)

    # --- Step 2: Extract Video Info & Formulate Download Query ---
    # In a real interactive app, you'd parse this properly. Here, we'll
    # assume the agent provided a list and we'll proceed.
    # This is a simplification for a non-interactive script.
    
    # We create a follow-up prompt that leverages the context from the first turn.
    download_query = (
        f"Great. Now please download the {download_target} in '{language}' for all the videos you just found. "
        f"Save everything into the '{output_dir}' directory."
    )

    # Conditionally add upload instruction if a bucket is specified
    if gcs_bucket:
        download_query += (
            f" After the download is complete, please upload all successfully downloaded files to the GCS bucket named '{gcs_bucket}'."
        )
    
    # --- Step 3: Download and a half: Upload ---
    await call_agent_async(download_query, user_id, session_id)

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
