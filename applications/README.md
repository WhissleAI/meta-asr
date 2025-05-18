# Audio Processing API

A FastAPI-based server for audio file processing, transcription, and advanced audio analysis.

## Features

- **Audio Transcription**: Transcribe audio files using either:
  - Google's Gemini API
  - Whissle API
- **Advanced Audio Analysis**:
  - Age and gender prediction
  - Emotion detection
  - Intent classification
  - Entity recognition
- **Output Format**: Results are saved in JSONL format for easy integration with other systems

## Prerequisites

- Python 3.8+
- Required Python packages (see [Installation](#installation))
- API Keys:
  - Google API Key (for Gemini)
  - Whissle Auth Token (optional)

## Installation

1. Clone this repository or navigate to the project folder

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
WHISSLE_AUTH_TOKEN=your_whissle_auth_token_here
```

## Starting the Server

### Basic Usage

```bash
cd /path/to/meta-asr/fast_api
python main.py
```

By default, the server starts on `127.0.0.1:8000` with reload mode enabled.

### Environment Variables

You can customize the server behavior using the following environment variables:

- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)
- `RELOAD`: Enable hot reload for development (default: "true")

Example:

```bash
HOST=0.0.0.0 PORT=5000 RELOAD=false python main.py
```

### Alternative Start Using Uvicorn

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Using the API

### Web Interface

The server provides a simple web UI accessible at:

```
http://localhost:8000/
```

### API Documentation

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

### API Endpoints

1. **Status Check**
   ```
   GET /status
   ```
   Returns information about the API version and available models.

2. **Create Transcription-Only Manifest**
   ```
   POST /create_transcription_manifest/
   ```
   Transcribes audio files and creates a simple JSONL manifest with just the transcriptions.

3. **Create Annotated Manifest**
   ```
   POST /create_annotated_manifest/
   ```
   Processes audio files with full analysis (transcription, age/gender detection, emotion analysis, entity/intent annotation).

### Request Format

Both main endpoints accept the same request format:

```json
{
  "directory_path": "/absolute/path/to/audio/directory",
  "model_choice": "gemini",  // or "whissle"
  "output_jsonl_path": "/absolute/path/for/results.jsonl"
}
```

### Response Format

The API returns a response with processing statistics:

```json
{
  "message": "Processing summary message",
  "output_file": "/path/to/output/file.jsonl",
  "processed_files": 10,
  "saved_records": 9,
  "errors": 1
}
```

### JSONL Output Format

Each record in the output JSONL file follows this format:

```json
{
  "audio_filepath": "/path/to/audio/file.flac",
  "text": "transcription with AGE_25_34 GENDER_MALE EMOTION_NEUTRAL ENTITY_PERSON_NAME John Smith END INTENT_SOCIAL_CHITCHAT",
  "duration": 5.32,
  "model_used_for_transcription": "gemini",
  "error": null
}
```

## Example Usage

### Using curl

```bash
curl -X 'POST' \
  'http://localhost:8000/create_annotated_manifest/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "directory_path": "/path/to/audio/files",
  "model_choice": "gemini",
  "output_jsonl_path": "/path/to/output/results.jsonl"
}'
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/create_annotated_manifest/"
payload = {
  "directory_path": "/path/to/audio/files",
  "model_choice": "gemini",
  "output_jsonl_path": "/path/to/output/results.jsonl"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)

## Troubleshooting

- **Model Loading Failures**: Check console logs for detailed error messages.
- **API Key Issues**: Verify that your `.env` file contains valid API keys.
- **Whissle SDK Issues**: Make sure the Whissle Python package is installed correctly.
- **Memory Errors**: For large audio files, consider processing files in smaller batches.

## License

[Your License Information]

## Contact

[Your Contact Information]