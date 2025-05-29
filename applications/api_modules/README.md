# API Modules Documentation

This directory contains the modularized components of the Audio Processing API, refactored from the original monolithic `main.py` file.

## Module Structure

### Core Files
- **`main.py`** - Main FastAPI application entry point that orchestrates all modules
- **`main_backup.py`** - Backup of the original monolithic main.py file

### API Modules (`api_modules/`)

#### 1. `config.py`
- **Purpose**: Configuration, constants, and environment setup
- **Contains**:
  - Environment variable loading (API keys)
  - Global constants (ENTITY_TYPES, INTENT_TYPES, AUDIO_EXTENSIONS)
  - Device configuration (CUDA/CPU)
  - API service configuration setup (Gemini, Whissle, Deepgram)
- **Key Functions**: `setup_api_configurations()`

#### 2. `models.py`
- **Purpose**: Pydantic models and data structures
- **Contains**:
  - Request/Response models (`ProcessRequest`, `ProcessResponse`)
  - Data record models (`TranscriptionJsonlRecord`, `AnnotatedJsonlRecord`)
  - Enums (`ModelChoice`)
  - BIO annotation structures (`BioAnnotation`)

#### 3. `age_gender_model.py`
- **Purpose**: Age and gender prediction functionality
- **Contains**:
  - Custom PyTorch model classes (`AgeGenderModel`, `ModelHead`)
  - Model loading and prediction functions
  - Age group categorization and gender string conversion
- **Key Functions**: 
  - `load_age_gender_model()`
  - `predict_age_gender()`
  - `get_age_group()`, `get_gender_string()`

#### 4. `emotion_model.py`
- **Purpose**: Emotion recognition from audio
- **Contains**:
  - Emotion model loading and prediction
  - Emotion label formatting
- **Key Functions**:
  - `load_emotion_model()`
  - `predict_emotion()`
  - `format_emotion_label()`

#### 5. `transcription.py`
- **Purpose**: Audio transcription services
- **Contains**:
  - Multiple transcription providers (Whissle, Gemini, Deepgram)
  - MIME type detection for audio files
  - Async transcription functions
- **Key Functions**:
  - `transcribe_with_whissle_single()`
  - `transcribe_with_gemini_single()`
  - `transcribe_with_deepgram_single()`

#### 6. `prompt.py`
- **Purpose**: AI prompt generation and text annotation
- **Contains**:
  - BIO tagging prompt generation
  - Gemini-based text annotation
  - Intent and entity extraction
- **Key Functions**:
  - `get_annotation_prompt()`
  - `annotate_text_structured_with_gemini()`

#### 7. `audio_utils.py`
- **Purpose**: Audio file processing utilities
- **Contains**:
  - Audio file discovery and validation
  - Audio loading and resampling
  - Duration extraction
  - Path validation
- **Key Functions**:
  - `discover_audio_files()`
  - `load_audio()`
  - `get_audio_duration()`
  - `validate_paths()`

## Benefits of Modularization

### 1. **Maintainability**
- Each module has a single responsibility
- Easier to locate and fix bugs
- Cleaner code organization

### 2. **Scalability**
- Easy to add new transcription providers
- Simple to extend with new annotation types
- Modular testing capabilities

### 3. **Reusability**
- Individual modules can be imported independently
- Functions can be reused across different applications
- Clear separation of concerns

### 4. **Development Efficiency**
- Multiple developers can work on different modules simultaneously
- Faster development cycles
- Easier code reviews

## API Endpoints

The refactored API maintains the same endpoints as before:

- **`POST /create_transcription_manifest/`** - Transcription-only processing
- **`POST /create_annotated_manifest/`** - Full annotation processing with optional age/gender/emotion/entity/intent
- **`GET /status`** - API status and model information
- **`GET /`** - Web interface

## Usage

### Running the API
```bash
cd E:\Meta_asr\meta-asr\applications
python main.py
```

### Importing Modules
```python
from api_modules.transcription import transcribe_with_gemini_single
from api_modules.age_gender_model import predict_age_gender
from api_modules.config import setup_api_configurations
```

## Configuration

The API requires the following environment variables:
- `GOOGLE_API_KEY` - For Gemini transcription and annotation
- `WHISSLE_AUTH_TOKEN` - For Whissle transcription
- `DEEPGRAM_API_KEY` - For Deepgram transcription

## Frontend Integration

The modular structure maintains full compatibility with the existing frontend application. All API endpoints and response formats remain unchanged, ensuring seamless integration with the Next.js frontend.

## Migration Notes

- Original `main.py` is backed up as `main_backup.py`
- All functionality remains identical to the original implementation
- No breaking changes to API contracts
- Same performance characteristics maintained
