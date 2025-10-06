# Google ADK YouTube Agent

This project provides an intelligent, agent-based system for searching YouTube and downloading video transcripts using Google's Agent Development Kit (ADK) and the Gemini family of models.

The agent follows a conversational workflow to first find relevant videos based on a user's query and then, upon user confirmation, downloads the transcripts for those videos.

## 🚀 Features

-   **Agentic Workflow**: Utilizes Google ADK for a robust, multi-step, tool-based workflow.
-   **Gemini-Powered**: Leverages the tool-calling capabilities of Gemini models to orchestrate tasks.
-   **Conversational Interaction**: Engages in a simple, two-step conversation to search and then download.
-   **Specialized Tools**: Clean separation of concerns with dedicated tools for YouTube search and transcript downloading.
-   **Easy to Use**: Simplified command-line interface for easy execution.
-   **Secure**: Manages API keys via environment variables or a `.env` file.

## 📋 Requirements

### System
-   Python 3.9+
-   `yt-dlp` requires `ffmpeg` to be installed for some operations.

### Python Dependencies
All required Python packages are listed in `requirements_agent.txt`.

## 🛠️ Setup & Installation

1.  **Clone the Repository**
    If you haven't already, get the code on your local machine.

2.  **Install Dependencies**
    Install all the necessary Python libraries.
    ```bash
    pip install -r requirements_agent.txt
    ```

3.  **Configure API Keys (Crucial Step)**
    The agent requires two API keys to function:
    -   **Google API Key** (for Gemini)
    -   **YouTube Data API v3 Key** (for searching videos)

    The recommended way to set them is by creating a `.env` file in the same directory:

    **`.env` file:**
    ```
    # Get from Google AI Studio: https://aistudio.google.com/app/apikey
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

    # Get from Google Cloud Console: https://console.cloud.google.com/apis/credentials
    YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY_HERE"
    ```

    Alternatively, you can set them as environment variables or pass them as command-line arguments.

## ▶️ How to Run the Agent

Use the `run_adk_agent.py` script to start the agent. You just need to provide a search query.

### Basic Example
This command will search for the top 5 videos related to the query and download their English transcripts.

```bash
python run_adk_agent.py "Supreme Court of India live hearing"
```

### Advanced Example
Specify the number of results and the desired transcript language.

```bash
python run_adk_agent.py "latest advancements in AI" --max-results 10 --language "en"
```

### Providing API Keys via Command Line
If you don't want to use a `.env` file, you can provide the keys directly.

```bash
python run_adk_agent.py "data science tutorials" --google-api-key "AIza..." --youtube-api-key "AIza..."
```

## ⚙️ How It Works

The system is composed of three main files:

1.  **`run_adk_agent.py` (CLI Runner)**
    -   This is the entry point.
    -   It parses user commands (like the query, language, etc.).
    -   It securely loads and configures the necessary API keys.
    -   It kicks off the agent's main task.

2.  **`youtube_adk_agent.py` (The Agent)**
    -   Defines the `Agent` using Google ADK.
    -   Contains the **instructions** (the "brain") that tell the Gemini model how to behave and in what order to use its tools.
    -   Specifies which `tools` the agent has access to.
    -   Manages the conversational flow (search first, then download).

3.  **`youtube_tools.py` (The Tools)**
    -   Contains the actual functions that perform the work.
    -   `search_youtube_videos()`: Communicates with the YouTube Data API to find videos.
    -   `download_video_transcript()`: Uses the `yt-dlp` library to fetch transcripts for a specific video.

### The Agent's Workflow

1.  The user runs the `run_adk_agent.py` script with a query (e.g., "AI ethics").
2.  The agent receives its first instruction: "Please find the top 5 videos about 'AI ethics'".
3.  The Gemini model, following its instructions, determines that it needs to use the `search_youtube_videos` tool. It calls the tool with the query.
4.  The tool returns a JSON list of videos, which the agent presents to the user.
5.  The `run_adk_agent.py` script immediately sends a follow-up instruction: "Great. Now please download the transcripts...".
6.  The agent receives this new prompt. Because ADK maintains conversation history, the agent remembers the videos it just found.
7.  It iterates through the list of videos and calls the `download_video_transcript` tool for each one, using the `video_id` and `title` from the search results.
8.  After all tool calls are finished, the agent provides a final summary of its work.

## 📁 Output Structure

Downloaded transcripts are saved in a structured format:

```
./downloads/
└── <VIDEO_ID>_<SANITIZED_VIDEO_TITLE>/
    └── <SANITIZED_VIDEO_TITLE>.<LANGUAGE>.vtt
```

**Example:**
```
./downloads/
└── dQ_w4w9WgXcQ_Rick_Astley_Never_Gonna_Give_You_Up/
    └── Rick_Astley_Never_Gonna_Give_You_Up.en.vtt
```
