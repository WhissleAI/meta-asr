# YAML Configuration Schema Reference

Quick reference for creating domain-specific configurations for `universal_process.py`.

## Complete Schema Template

```yaml
# ============================================================================
# DOMAIN METADATA
# ============================================================================
domain:
  name: "string"              # Unique domain identifier
  description: "string"       # Brief description of domain

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
models:
  age_gender: "string"        # HuggingFace model ID for age/gender detection
                              # Default: "audeering/wav2vec2-large-robust-6-ft-age-gender"
  
  whisper: "string"           # HuggingFace Whisper model ID
                              # Default: "openai/whisper-large-v3"
                              # Options: whisper-tiny, small, medium, large-v2, large-v3
  
  gemini: "string"            # Google Gemini model name
                              # Default: "gemini-2.0-flash"
                              # Options: gemini-pro, gemini-2.0-flash
  
  deepgram: "string"          # Deepgram model name
                              # Default: "nova-2-general"
                              # Options: nova-2, nova-2-general, enhanced

# ============================================================================
# FEATURE FLAGS
# ============================================================================
features:
  diarization: boolean        # Enable speaker diarization (true/false)
                              # Requires: HF_TOKEN, pyannote.audio
  
  entity_intent_extraction: boolean    # Enable Gemini entity/intent extraction
                                       # Requires: GEMINI_API_KEY
  
  structured_summary: boolean          # Enable Gemini structured summaries
                                       # Requires: GEMINI_API_KEY
  
  gemini_transcription: boolean        # Use Gemini for transcription
                                       # Requires: GEMINI_API_KEY
  
  deepgram_transcription: boolean      # Use Deepgram for transcription
                                       # Requires: DEEPGRAM_API_KEY

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
processing:
  chunk_duration: float       # Duration of audio chunks in seconds
                              # Default: 30.0
                              # Range: 10.0 - 60.0 (lower = more chunks, higher memory)
  
  overlap_duration: float     # Overlap between chunks in seconds
                              # Default: 10.0
                              # Range: 0.0 - 15.0 (helps maintain context)
  
  min_speakers: integer       # Minimum number of speakers to detect
                              # Default: 1
                              # Range: 1 - 10
  
  max_speakers: integer|null  # Maximum number of speakers (null = auto-detect)
                              # Default: null
                              # Range: 1 - 10 or null

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
language: "string"            # Target language for transcription
                              # Default: "english"
                              # Options: english, spanish, french, german, etc.

# ============================================================================
# INPUT CONFIGURATION
# ============================================================================
input:
  directory: "string"         # Default input directory path
                              # Default: "."
                              # Can be absolute or relative
  
  extensions: [strings]       # Supported file extensions
                              # Default: [".wav", ".mp3", ".mp4"]
                              # Options: .wav, .mp3, .mp4, .webm, .m4a, .flac, .mkv

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
output:
  directory: "string"         # Output directory path
                              # Default: "./output"
                              # Can be absolute or relative
  
  per_file_subdirs: boolean   # Create subdirs per file in same location
                              # true = outputs next to input files
                              # false = all outputs in output.directory

# ============================================================================
# PROMPTS (Domain-Specific)
# ============================================================================
prompts:
  # Gemini transcription prompt (used if features.gemini_transcription: true)
  gemini_transcription: |
    Multi-line string with your transcription instructions.
    Tell Gemini how to transcribe audio for your domain.
    Example: "Transcribe this audio focusing on technical terminology."
  
  # Entity and intent extraction prompt
  gemini_entities_intents: |
    Multi-line string defining:
    1. Entity types to extract (with descriptions)
    2. Intent categories to identify (with descriptions)
    3. JSON output format
    
    Example structure:
    Extract entities and intents from this text.
    
    Entities:
    - ENTITY_TYPE_1: description
    - ENTITY_TYPE_2: description
    
    Intents:
    - INTENT_1: description
    - INTENT_2: description
    
    Return JSON:
    {
      "entities": [{"text": "...", "type": "..."}],
      "intents": ["...", "..."]
    }
    
    Text:
  
  # Structured summary extraction prompt
  gemini_structured: |
    Multi-line string defining:
    1. Structured fields to extract
    2. JSON output format with field descriptions
    
    Example:
    Extract structured information from this conversation.
    
    Return JSON:
    {
      "field1": "description",
      "field2": ["list", "of", "items"],
      "field3": "value"
    }
    
    Text:

# ============================================================================
# ENTITY TYPES (For Reference/Documentation)
# ============================================================================
entity_types:
  - "ENTITY_TYPE_1"           # List of entity types your domain uses
  - "ENTITY_TYPE_2"           # Used for documentation and validation
  - "ENTITY_TYPE_3"           # Should match types in prompts.gemini_entities_intents
  # Add as many as needed

# ============================================================================
# INTENT CATEGORIES (For Reference/Documentation)
# ============================================================================
intent_categories:
  - "INTENT_CATEGORY_1"       # List of intent categories your domain uses
  - "INTENT_CATEGORY_2"       # Used for documentation and validation
  - "INTENT_CATEGORY_3"       # Should match intents in prompts.gemini_entities_intents
  # Add as many as needed
```

## Field Types & Constraints

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `domain.name` | string | No | "generic" | Unique identifier |
| `domain.description` | string | No | "" | Human-readable description |
| `models.age_gender` | string | No | "audeering/..." | HF model ID |
| `models.whisper` | string | No | "openai/whisper-large-v3" | HF model ID |
| `models.gemini` | string | No | "gemini-2.0-flash" | Gemini model name |
| `models.deepgram` | string | No | "nova-2-general" | Deepgram model |
| `features.diarization` | boolean | No | true | Requires HF_TOKEN |
| `features.entity_intent_extraction` | boolean | No | false | Requires GEMINI_API_KEY |
| `features.structured_summary` | boolean | No | false | Requires GEMINI_API_KEY |
| `features.gemini_transcription` | boolean | No | false | Requires GEMINI_API_KEY |
| `features.deepgram_transcription` | boolean | No | false | Requires DEEPGRAM_API_KEY |
| `processing.chunk_duration` | float | No | 30.0 | Seconds, 10-60 range |
| `processing.overlap_duration` | float | No | 10.0 | Seconds, 0-15 range |
| `processing.min_speakers` | integer | No | 1 | 1-10 range |
| `processing.max_speakers` | integer/null | No | null | 1-10 or null |
| `language` | string | No | "english" | ISO language name |
| `input.directory` | string | No | "." | Path to input folder |
| `input.extensions` | list[string] | No | [".wav", ".mp3", ".mp4"] | File extensions |
| `output.directory` | string | No | "./output" | Path to output folder |
| `output.per_file_subdirs` | boolean | No | true | Output location strategy |
| `prompts.gemini_transcription` | string | No | "" | Multi-line prompt |
| `prompts.gemini_entities_intents` | string | No | "" | Multi-line prompt |
| `prompts.gemini_structured` | string | No | "" | Multi-line prompt |
| `entity_types` | list[string] | No | [] | Documentation list |
| `intent_categories` | list[string] | No | [] | Documentation list |

## Minimal Valid Config

```yaml
# Bare minimum - uses all defaults
domain:
  name: "minimal"

features:
  diarization: true
  entity_intent_extraction: false
  structured_summary: false
```

## Typical Production Config

```yaml
domain:
  name: "production_domain"
  description: "Production-ready configuration"

models:
  whisper: "openai/whisper-large-v3"  # Best quality
  gemini: "gemini-2.0-flash"          # Fast API model

features:
  diarization: true                   # Speaker separation
  entity_intent_extraction: true      # Domain NER
  structured_summary: true            # Structured data
  gemini_transcription: false         # Use Whisper (faster)
  deepgram_transcription: false       # Optional backup

processing:
  chunk_duration: 30.0
  overlap_duration: 10.0
  min_speakers: 2
  max_speakers: 4

language: "english"

input:
  directory: "/path/to/input"
  extensions: [".wav", ".mp3", ".mp4"]

output:
  directory: "/path/to/output"
  per_file_subdirs: true

prompts:
  gemini_entities_intents: |
    [Your domain-specific prompt here]

entity_types:
  - "TYPE_1"
  - "TYPE_2"

intent_categories:
  - "INTENT_1"
  - "INTENT_2"
```

## Common Configurations

### High-Performance (GPU Available)
```yaml
processing:
  chunk_duration: 60.0        # Larger chunks
  overlap_duration: 15.0

models:
  whisper: "openai/whisper-large-v3"  # Best model

features:
  diarization: true           # All features enabled
  entity_intent_extraction: true
  structured_summary: true
```

### Memory-Constrained (CPU Only)
```yaml
processing:
  chunk_duration: 15.0        # Smaller chunks
  overlap_duration: 5.0

models:
  whisper: "openai/whisper-small"  # Lighter model

features:
  diarization: false          # Disable heavy features
  entity_intent_extraction: false
```

### Quick Transcription Only
```yaml
features:
  diarization: false
  entity_intent_extraction: false
  structured_summary: false
  gemini_transcription: false
  deepgram_transcription: false

processing:
  chunk_duration: 30.0
  overlap_duration: 0.0       # No overlap for speed
```

## Environment Variables Required

Based on enabled features:

| Feature | Required Environment Variable | Where to Get |
|---------|-------------------------------|--------------|
| `diarization: true` | `HF_TOKEN` | https://huggingface.co/settings/tokens |
| `entity_intent_extraction: true` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | https://ai.google.dev/ |
| `structured_summary: true` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | https://ai.google.dev/ |
| `gemini_transcription: true` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | https://ai.google.dev/ |
| `deepgram_transcription: true` | `DEEPGRAM_API_KEY` | https://deepgram.com/ |

Optional:
- `FORCE_CPU=1` - Force CPU mode even if GPU available
- `GEMINI_DEBUG=1` - Enable Gemini API debug logging

## Validation Checklist

Before running with a new config:

- [ ] `domain.name` is unique and descriptive
- [ ] All enabled features have required API keys set
- [ ] `input.directory` exists and contains media files
- [ ] `input.extensions` matches your file types
- [ ] `output.directory` is writable
- [ ] `prompts` are complete if features enabled
- [ ] `entity_types` match types in prompt
- [ ] `intent_categories` match intents in prompt
- [ ] `processing.chunk_duration` suits your audio length
- [ ] `processing.min_speakers` / `max_speakers` are reasonable

## Testing a New Config

```bash
# 1. Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# 2. Dry run with single file
python universal_process.py configs/my_config.yaml /path/to/single/file.wav

# 3. Check outputs
ls output/results/

# 4. Verify JSON structure
python -m json.tool output/results/*_processed_data.json

# 5. Full batch processing
python universal_process.py configs/my_config.yaml /path/to/all/files/
```

## Tips for Writing Prompts

### Entity Extraction Prompt
```yaml
gemini_entities_intents: |
  Extract [DOMAIN] entities and intents.
  
  Entities (with examples):
  - TYPE_1: description [example: "specific instance"]
  - TYPE_2: description [example: "another example"]
  
  Intents (with examples):
  - INTENT_1: when user is doing X [example: "user utterance"]
  - INTENT_2: when user wants Y [example: "another utterance"]
  
  Return ONLY valid JSON:
  {"entities": [{"text": "...", "type": "TYPE_1"}], "intents": ["INTENT_1"]}
  
  Text:
```

### Structured Summary Prompt
```yaml
gemini_structured: |
  Extract structured data from [DOMAIN] conversation.
  
  Return JSON with these fields (omit if not present):
  {
    "field1": "single value",
    "field2": ["list", "of", "values"],
    "field3": {
      "nested": "object",
      "if": "needed"
    }
  }
  
  Be concise. Only include fields with actual data.
  
  Text:
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Config file not found" | Use absolute path or verify file exists |
| "No media files found" | Check `input.extensions` matches your files |
| "CUDA out of memory" | Reduce `chunk_duration` or enable `FORCE_CPU=1` |
| "Diarization failed" | Verify `HF_TOKEN` and accept model terms |
| "Gemini returns empty" | Check `GEMINI_API_KEY` and prompt format |
| "Invalid JSON output" | Improve prompt to request valid JSON only |
| "Missing entity types" | Add more examples in prompt |
