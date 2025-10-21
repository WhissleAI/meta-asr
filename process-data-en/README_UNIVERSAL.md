# Universal Audio Processing Pipeline

A fully configurable audio processing pipeline that consolidates domain-specific processing scripts into a single universal tool. Configure different domains (automotive, kitchen, interview, etc.) through YAML files instead of maintaining separate Python scripts.

## 🎯 Purpose

Previously, we had multiple nearly-identical scripts (`creative_design.py`, `interview.py`, `bucket_process.py`) that differed only in:
- Gemini prompts (entity/intent extraction)
- Entity type lists
- Intent categories  
- Default file paths

This universal script eliminates code duplication by making everything configurable through YAML.

## ⚙️ Features

- **Multi-domain support**: Automotive, Kitchen/Cooking, Interviews, Generic
- **Speaker diarization**: Automatic speaker detection and separation
- **Multi-ASR support**: Whisper (primary), Gemini, Deepgram
- **Age/Gender detection**: Per-speaker demographic analysis
- **Emotion recognition**: Emotion classification per segment
- **Entity extraction**: Domain-specific named entity recognition via Gemini
- **Intent classification**: Automatic intent detection
- **Structured summaries**: Domain-specific structured data extraction
- **Flexible output**: JSON, JSONL, CSV formats

## 📋 Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch
# - transformers
# - librosa
# - soundfile
# - pyannote.audio==3.1
# - google-generativeai
# - pyyaml
# - python-dotenv
# - resampy
# - moviepy
```

## 🔑 API Keys Setup

Create a `.env` file:

```bash
# Required for diarization
HF_TOKEN=your_huggingface_token

# Optional - for Gemini features
GEMINI_API_KEY=your_gemini_api_key
# OR
GOOGLE_API_KEY=your_google_api_key

# Optional - for Deepgram transcription
DEEPGRAM_API_KEY=your_deepgram_api_key

# Force CPU if GPU issues
FORCE_CPU=0
```

## 🚀 Usage

### Basic Usage

```bash
# Process files with automotive config
python universal_process.py configs/automotive.yaml /path/to/audio/files

# Process with interview config
python universal_process.py configs/interview.yaml /path/to/interviews

# Process with kitchen config
python universal_process.py configs/kitchen.yaml /path/to/cooking/videos

# Process with generic config (no domain-specific extraction)
python universal_process.py configs/generic.yaml /path/to/audio
```

### Input Directory Auto-detection

If you don't specify an input directory, it uses the `input.directory` from the YAML config:

```bash
python universal_process.py configs/automotive.yaml
```

## 📁 File Structure

```
process-data-en/
├── universal_process.py          # Main universal script
├── configs/                       # Domain configurations
│   ├── automotive.yaml           # Automotive/creative design
│   ├── interview.yaml            # Professional interviews
│   ├── kitchen.yaml              # Cooking/kitchen content
│   └── generic.yaml              # General-purpose
├── output/                        # Default output location
│   ├── audio_chunks/             # Chunked audio files
│   ├── audio_converted/          # Converted WAV files
│   └── results/                  # Processed results
│       ├── *_processed_data.json      # Full segment data
│       ├── *_speaker_mapping.csv      # Speaker statistics
│       └── *_audio_text_pairs.jsonl   # Audio-text pairs
└── README.md                     # This file
```

## 📝 Configuration File Structure

Each YAML config has these sections:

### Domain Info
```yaml
domain:
  name: "automotive_creative_design"
  description: "Processing automotive and creative design content"
```

### Models
```yaml
models:
  age_gender: "audeering/wav2vec2-large-robust-6-ft-age-gender"
  whisper: "openai/whisper-large-v3"
  gemini: "gemini-2.0-flash"
  deepgram: "nova-2-general"
```

### Features (Enable/Disable)
```yaml
features:
  diarization: true
  entity_intent_extraction: true
  structured_summary: true
  gemini_transcription: false
  deepgram_transcription: false
```

### Processing Parameters
```yaml
processing:
  chunk_duration: 30.0      # seconds
  overlap_duration: 10.0    # seconds
  min_speakers: 2
  max_speakers: null        # auto-detect
```

### Input/Output
```yaml
input:
  directory: "."
  extensions: [".wav", ".mp3", ".mp4"]

output:
  directory: "./output"
  per_file_subdirs: true    # Create outputs next to input files
```

### Prompts (Domain-Specific)
```yaml
prompts:
  gemini_transcription: |
    Your transcription prompt here...
  
  gemini_entities_intents: |
    Your entity/intent extraction prompt...
  
  gemini_structured: |
    Your structured summary prompt...
```

### Entity Types & Intents
```yaml
entity_types:
  - "VEHICLE"
  - "VEHICLE_PART"
  - "FEATURE"
  # ... more types

intent_categories:
  - "PURCHASE_INQUIRY"
  - "COMPARISON"
  - "TECHNICAL_QUESTION"
  # ... more intents
```

## 🎨 Domain Examples

### Automotive (configs/automotive.yaml)
- **Entities**: VEHICLE, VEHICLE_PART, FEATURE, DESIGN_ELEMENT, COLOR, MATERIAL, BRAND
- **Intents**: PURCHASE_INQUIRY, COMPARISON, TECHNICAL_QUESTION, PROBLEM_REPORT, DESIGN_FEEDBACK
- **Use cases**: Car reviews, design walkthroughs, customer inquiries

### Interview (configs/interview.yaml)
- **Entities**: PERSON, ROLE, COMPANY, SKILL, TECHNOLOGY, QUALIFICATION, PROJECT
- **Intents**: SKILL_ASSESSMENT, EXPERIENCE_INQUIRY, BEHAVIORAL_QUESTION, TECHNICAL_CHALLENGE
- **Use cases**: Job interviews, screening calls, technical assessments

### Kitchen (configs/kitchen.yaml)
- **Entities**: FOOD_ITEM, UTENSIL_TOOL, ACTION_COOKING, MEASUREMENT, TEMPERATURE, TIME_DURATION
- **Intents**: RECIPE_INSTRUCTION, INGREDIENT_INQUIRY, TECHNIQUE_EXPLANATION, SUBSTITUTION_SUGGESTION
- **Use cases**: Cooking videos, recipe tutorials, kitchen conversations

### Generic (configs/generic.yaml)
- **Entities**: PERSON, ORGANIZATION, LOCATION, DATE, TIME, PRODUCT, EVENT
- **Intents**: QUESTION, STATEMENT, REQUEST, INSTRUCTION, GREETING
- **Use cases**: General audio transcription without domain-specific extraction

## 📊 Output Formats

### 1. Full Segment Data (JSON)
`<filename>_processed_data.json`

Contains all segments with full metadata:
```json
[
  {
    "start_time": 0.5,
    "end_time": 5.2,
    "speaker": "speaker_0",
    "speaker_index": 0,
    "age": 0.34,
    "gender": 1,
    "whisper_text": "transcript...",
    "gemini_text": "",
    "emotion": "neutral",
    "entities": [...],
    "intents": [...],
    "structured_summary": {...},
    "audio_file_path": "/path/to/chunk.wav",
    "speaker_change_tag": "speaker_change_0"
  }
]
```

### 2. Speaker Mapping (CSV)
`<filename>_speaker_mapping.csv`

Speaker statistics:
```csv
speaker_index,speaker_tag,first_start_time,cumulative_duration,segment_count
0,speaker_0,0.5,45.3,12
1,speaker_1,5.8,38.7,10
```

### 3. Audio-Text Pairs (JSONL)
`<filename>_audio_text_pairs.jsonl`

Training-ready format:
```jsonl
{"audio_filepath": "/path/chunk_0.wav", "whisper_text": "...", "gemini_text": "", "tagged_text": "... AGE_18_30 GENDER_MALE EMOTION_NEUTRAL", "duration": 30.0}
{"audio_filepath": "/path/chunk_1.wav", "whisper_text": "...", "gemini_text": "", "tagged_text": "...", "duration": 28.5}
```

## 🔧 Creating Custom Configurations

To create a new domain configuration:

1. Copy an existing config:
```bash
cp configs/generic.yaml configs/my_domain.yaml
```

2. Update the domain info:
```yaml
domain:
  name: "my_custom_domain"
  description: "Description of your domain"
```

3. Define your entity types:
```yaml
entity_types:
  - "MY_ENTITY_1"
  - "MY_ENTITY_2"
  # Add more...
```

4. Define your intents:
```yaml
intent_categories:
  - "MY_INTENT_1"
  - "MY_INTENT_2"
  # Add more...
```

5. Write domain-specific prompts:
```yaml
prompts:
  gemini_entities_intents: |
    Extract entities and intents specific to my domain...
    
    Entities:
    - MY_ENTITY_1: description
    - MY_ENTITY_2: description
    
    Intents:
    - MY_INTENT_1: description
    - MY_INTENT_2: description
    
    Text:
```

6. Run with your config:
```bash
python universal_process.py configs/my_domain.yaml /path/to/data
```

## 🐛 Troubleshooting

### CUDA/GPU Issues
```bash
# Force CPU mode
export FORCE_CPU=1
python universal_process.py configs/automotive.yaml
```

### Diarization Fails
- Check HF_TOKEN is set correctly
- Verify pyannote.audio version: `pip show pyannote.audio` (should be 3.1)
- Accept pyannote model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

### Gemini Features Not Working
- Verify GEMINI_API_KEY or GOOGLE_API_KEY is set
- Enable features in config:
```yaml
features:
  entity_intent_extraction: true
  structured_summary: true
```

### Out of Memory
Reduce chunk duration:
```yaml
processing:
  chunk_duration: 15.0  # Smaller chunks
  overlap_duration: 5.0
```

## 🔄 Migration from Old Scripts

If you were using domain-specific scripts before:

| Old Script | New Command |
|------------|-------------|
| `python creative_design.py` | `python universal_process.py configs/automotive.yaml` |
| `python interview.py` | `python universal_process.py configs/interview.yaml` |
| `python bucket_process.py` | `python universal_process.py configs/kitchen.yaml` |

**Benefits:**
- ✅ Single codebase to maintain
- ✅ Easy to add new domains (just create YAML)
- ✅ Consistent output formats
- ✅ Share improvements across all domains
- ✅ No code duplication

## 📈 Performance Tips

1. **Use GPU**: Significantly faster processing (10-50x speedup)
2. **Batch processing**: Process multiple files at once
3. **Adjust chunk sizes**: Balance between memory usage and processing speed
4. **Disable unused features**: Turn off Gemini/Deepgram if not needed
5. **Use Whisper only**: Fastest transcription option

## 🤝 Contributing

To add a new domain:
1. Create a YAML config in `configs/`
2. Test with sample audio
3. Document entity types and intents in this README

## 📄 License

[Your license here]

## 🙋 Support

For issues or questions:
- Check existing configs for examples
- Verify API keys are set correctly
- Test with `generic.yaml` first to isolate domain-specific issues
