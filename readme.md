# Audio Processing Pipeline

This repository contains a set of tools for processing audio files, including format conversion, segmentation, speaker diarization, transcription, annotation, and natural language processing tasks.

## 📚 **Pipeline Overview**

### 1. **mp4_to_wav.py**
- **Purpose:** Converts `.mp4` files into `.mp3` format.
- **Input:** Folder containing `.mp4` files.
- **Output:** Folder containing `.mp3` files.

### 2. **audioChunker.py**
- **Purpose:** Segments `.mp3` files into smaller audio chunks of a specified maximum length.
- **MaxLength:** 20 seconds (default).
- **Input:** Folder containing `.mp3` files.
- **Output:** Folder containing chunked audio segments.

### 3. **pipeline.py**
- **Purpose:** Processes chunked audio files for speaker segmentation and optional transcription/annotation.
- **Input:** Folder containing chunked audio segments.
- **Output:** JSON file with the structure:
  ```json
  {
    "audio_filepath": "path_to_audio",
    "text": "processed_text"
  }
  ```
- **Flags:**
  - `ENABLE_SPEAKER_CHANGE`: Enables speaker change detection.
  - `ENABLE_TRANSCRIPTION`: Enables transcription processing.

### 4. **audio_processing_pipeline.py**
- **Purpose:** Applies NER (Named Entity Recognition) and audio classification on single-speaker segments with available transcription.
- **Conditions:**
  - Transcription must be available.
  - Segments must be ≤ MaxLength.
- **Input:** JSON file with the structure:
  ```json
  {
    "audio_filepath": "path_to_audio",
    "text": "transcription_text"
  }
  ```
- **Output:** JSON file with the structure:
  ```json
  {
    "audio_filepath": "path_to_audio",
    "text": "annotated_text"
  }
  ```

## 📂 **Data Flow**
1. **mp4_to_wav.py** → Converts `.mp4` to `.mp3`.
2. **audioChunker.py** → Chunks `.mp3` files.
3. **pipeline.py** → Processes chunked audio (segmentation,NER annotation ,Speaker segmentation, transcription).
4. **audio_processing_pipeline.py(still working on this)** → Applies NER and audio classification(emotion and age prediction).
5. **Final Output:** Annotated JSON files.

## 🛠️ **Setup Instructions**
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update `.env` file with required credentials (e.g., Hugging Face token).

## 🚀 **Run the Pipeline**
1. Run each script sequentially:
   ```bash
   python mp4_to_wav.py
   python audioChunker.py
   python pipeline.py
   python audio_processing_pipeline.py
   ```
2. Verify outputs at each stage in the corresponding output folders.

## 📑 **Example JSON Outputs**
### JSON File 1 (Transcription Output)
```json
{
  "audio_filepath": "path_to_audio",
  "text": "This is a transcription."
}
```

### JSON File 2 (Annotated Output)
```json
{
  "audio_filepath": "path_to_audio",
  "text": "This is an annotated transcription with NER tags."
}
```



