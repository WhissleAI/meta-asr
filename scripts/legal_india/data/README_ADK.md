# Google ADK YouTube Data Downloader

This project provides an intelligent, agent-based system for searching and downloading YouTube data using Google's Agent Development Kit (ADK) and the Gemini family of models.

It has been designed as a robust data collection tool that can search for videos (either across all of YouTube or within a specific channel), download media files (video and/or transcripts), and automatically upload them to a Google Cloud Storage bucket.

## 🚀 Features

-   **Agentic Workflow**: Utilizes Google ADK for a robust, multi-step, tool-based workflow.
-   **Gemini-Powered**: Leverages the tool-calling capabilities of the `gemini-2.5-pro` model to orchestrate complex tasks.
-   **Advanced Search**:
    -   Search globally across YouTube with a text query.
    -   Restrict searches to a specific channel using its handle (e.g., `SupremeCourtofIndia-1950`).
    -   Filter search results by country/region (e.g., `IN` for India).
-   **Flexible Downloads**:
    -   Download video files (`.mp4`), transcripts (`.vtt`), or both.
    -   Control the number of parallel downloads to avoid rate-limiting issues.
-   **Rich Metadata**:
    -   Fetches detailed metadata for each video (description, view count, likes, etc.).
    -   Saves this data in a `metadata.json` file alongside each downloaded video.
-   **Cloud Integration**: Automatically uploads all downloaded files (videos, transcripts, and metadata) to a specified Google Cloud Storage (GCS) bucket.

## 📋 Requirements

### System
-   Python 3.9+
-   Google Cloud SDK (`gcloud` CLI) for GCS authentication.
-   `ffmpeg` is required by `yt-dlp` for some video processing operations.

### Python Dependencies
All required Python packages are listed in `requirements_agent.txt`.

## 🛠️ Setup & Installation

1.  **Clone the Repository**
    If you haven't already, get the code on your local machine.

2.  **Install Dependencies**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements_agent.txt
    ```

3.  **Configure API Keys (Crucial Step)**
    For detailed instructions, see `API_KEY_SETUP.md`. You will need:
    -   **Google API Key** (for Gemini)
    -   **YouTube Data API v3 Key** (for searching videos)

    The recommended way is to create a `.env` file in this directory:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY_HERE"
    ```

4.  **Authenticate for GCS Uploads (Important)**
    To enable uploads to Google Cloud Storage, you must authenticate your local machine using the `gcloud` CLI.
    ```bash
    gcloud auth application-default login
    ```
    Follow the prompts in your browser to log in. This command saves your credentials in a standard location, allowing the script to automatically use them.

5.  **Export Browser Cookies (Optional, but Recommended)**
    To avoid "Sign in to confirm you’re not a bot" errors from YouTube during large downloads, it's highly recommended to use a cookies file. This makes your requests look like they are coming from a logged-in browser session.

    -   **How to Export**: Use a browser extension to export your cookies in the `Netscape` format and save them as `cookies.txt`.
    -   **Recommended Guide**: Follow the official `yt-dlp` instructions here: [How to Export Browser Cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)

## ▶️ How to Run the Agent

Use the `run_adk_agent.py` script to start the agent. It offers a wide range of options to customize your data collection.

### Command-Line Arguments

-   `query`: The search query (e.g., "live supreme court hearing").
-   `--language`: Language for the transcript (default: `en`).
-   `--max-results`: Maximum number of videos to find (default: `100`).
-   `--download`: What to download: `transcript`, `video`, or `both` (default: `transcript`).
-   `--output-dir`: Local directory to save files (default: `./downloads`).
-   `--gcs-bucket`: GCS bucket name to enable uploads.
-   `--region`: ISO country code to filter search results (e.g., `IN`, `US`).
-   `--channel`: YouTube channel handle to search within (e.g., `SupremeCourtofIndia-1950`).
-   `--max-workers`: Number of parallel download threads (default: `5`).
-   `--cookies`: Path to a `cookies.txt` file to authenticate downloads.

### Examples

**1. Simple Search (Transcripts Only)**
```bash
python run_adk_agent.py "latest advancements in AI" --max-results 10
```

**2. Advanced Search within a Specific Channel**
This is the recommended usage for targeted data collection. This command will:
-   Search for "Supreme Court of India" videos.
-   Only within the `@SupremeCourtofIndia-1950` channel.
-   Prioritize results from India (`IN`).
-   Download both the video and transcript for up to 100 results.
-   Save them to a specific local directory.
-   Upload everything to the `legalai-supremecourt-india-youtube` GCS bucket.
-   Use a maximum of 5 parallel download workers to avoid timeouts.

```bash
python run_adk_agent.py "Supreme Court of India Court" \
  --max-results 100 \
  --download both \
  --output-dir "/Users/karan/Desktop/work/whissle/data/legal_india2" \
  --gcs-bucket "legalai-supremecourt-india-youtube" \
  --region IN \
  --channel "SupremeCourtofIndia-1950" \
  --max-workers 5 \
  --cookies "/path/to/your/cookies.txt"
```

## ⚙️ How It Works

The system is a single-shot agent orchestrator. When you run the script, it constructs a detailed prompt for the Gemini model, which then calls the `search_youtube_videos` tool. The main Python script takes the structured JSON output from the agent and then directly calls the `download` and `upload` tools in parallel to complete the workflow. This is more efficient and reliable than a multi-turn conversational approach.

-   **`run_adk_agent.py`**: The command-line interface and entry point.
-   **`youtube_adk_agent.py`**: Defines the Gemini agent and orchestrates the tool calls.
-   **`youtube_tools.py`**: Contains the functions that interact with external services (YouTube API, `yt-dlp`, Google Cloud Storage).

## 📁 Output Structure

Downloaded files are saved in a structured and descriptive format.

```
<output_dir>/
└── <VIDEO_ID>-<SANITIZED_VIDEO_TITLE>/
    ├── metadata.json
    ├── <SANITIZED_VIDEO_TITLE>.mp4
    └── <SANITIZED_VIDEO_TITLE>.<LANGUAGE>.vtt
```

**Example:**
```
/Users/karan/Desktop/work/whissle/data/legal_india2/
└── mCHbwDTsads-Supreme_Court_of_India_-_Court_1/
    ├── metadata.json
    ├── Supreme_Court_of_India_-_Court_1.mp4
    └── Supreme_Court_of_India_-_Court_1.en.vtt
```
