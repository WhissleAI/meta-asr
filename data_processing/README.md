# Data Processing Server and Jupyter Notebook Usage

This document outlines how to use the Python server for data processing, specifically through the provided Jupyter notebooks for handling audio files stored in Google Cloud Storage (GCS).

## Overview

The setup consists of:

- A **Python FastAPI server** (`data_processing/python_server`) that exposes endpoints for processing audio files.
- **Jupyter notebooks** (`data_processing/jupiter`) that provide a client interface to interact with the server for tasks like transcription and annotation.

The primary workflow involves sending requests from a Jupyter notebook to the FastAPI server, pointing to audio data in GCS.

## Setup

1.  **Run the Python Server**:
    Navigate to the `data_processing/python_server` directory and start the server.
    ```bash
    cd data_processing/python_server
    python main.py
    ```
    The server will typically run on `http://localhost:8000`.

## Usage with Jupyter Notebooks

The Jupyter notebooks in `data_processing/jupiter/` are designed for processing audio data from GCS.

### 1. Processing a Single GCS Audio File

The `soccer_single_file_finetuning.ipynb` notebook demonstrates how to process a single audio file from a GCS bucket.

**Endpoint**: `POST /process_gcs_file/`

**Example Request Payload**:
The notebook sends a POST request to the server with a JSON payload. The key fields are:

- `user_id`: A unique identifier for the user making the request.
- `gcs_path`: The full `gs://` path to the single audio file.
- `model_choice`: The transcription model to use (e.g., "gemini").
- `output_jsonl_path`: An absolute path on the **server's local filesystem** where the output `.jsonl` file will be saved.
- `annotations`: A list of annotations to perform (e.g., "entity", "intent").
- `prompt`: A custom prompt string to guide the annotation model.

**Jupyter Notebook Snippet**:

```python
import requests
import json

# Load a custom prompt from a file
with open("soccer_prompt.txt") as f:
    custom_prompt = f.read()

# Build the request payload
payload = {
    "user_id": "user_123",
    "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data/England_v_Italy_-_Watch_the_full_2012_penalty_shoot-out_16k.wav",
    "model_choice": "gemini",
    "output_jsonl_path": "/home/dchauhan/workspace/meta-asr/data_processing/hello",
    "annotations": ["age", "gender", "emotion", "entity", "intent"],
    "prompt": custom_prompt
}

# Send the request to the server
url = "http://localhost:8000/process_gcs_file/"
response = requests.post(url, json=payload)

# Print the server's response
print(json.dumps(response.json(), indent=2))
```

### 2. Processing a GCS Directory

The `soccer_data_directory_finetuning.ipynb` notebook shows how to process all audio files within a specified GCS directory (folder).

**Endpoint**: `POST /process_gcs_directory/`

**Example Request Payload**:
The payload is identical to the single file request, but the `gcs_path` points to a directory instead of a file. The server will find and process all compatible audio files in that directory.

**Jupyter Notebook Snippet**:

```python
import requests
import json

# Load a custom prompt from a file
with open("soccer_prompt.txt") as f:
    custom_prompt = f.read()

# Build the request payload
payload = {
    "user_id": "user_123",
    "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data", # Path to the GCS directory
    "model_choice": "gemini",
    "output_jsonl_path": "/home/dchauhan/workspace/meta-asr/data_processing/hello",
    "annotations": ["age", "gender", "emotion", "entity", "intent"],
    "prompt": custom_prompt
}

# Send the request to the server
url = "http://localhost:8000/process_gcs_directory/"
response = requests.post(url, json=payload)

# Print the server's response
print(json.dumps(response.json(), indent=2))
```
