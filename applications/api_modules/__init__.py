"""
API Modules package for Audio Processing API
"""

# Import specific functions/classes to avoid conflicts
from .config import (
    setup_api_configurations, device, logger,
    GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED,
    ENTITY_TYPES, INTENT_TYPES, TARGET_SAMPLE_RATE
)
from .models import (
    ModelChoice, ProcessRequest, ProcessResponse,
    TranscriptionJsonlRecord, AnnotatedJsonlRecord, BioAnnotation
)
from .age_gender_model import (
    load_age_gender_model, predict_age_gender, 
    get_age_group, get_gender_string, age_gender_model
)
from .emotion_model import (
    load_emotion_model, predict_emotion, format_emotion_label, emotion_model
)
from .transcription import (
    transcribe_with_whissle_single, transcribe_with_gemini_single,
    transcribe_with_deepgram_single
)
from .prompt import (
    annotate_text_structured_with_gemini, get_annotation_prompt
)
from .audio_utils import (
    discover_audio_files, load_audio, get_audio_duration, validate_paths
)

__all__ = [
    # Config
    'setup_api_configurations', 'device', 'logger',
    'GEMINI_CONFIGURED', 'WHISSLE_CONFIGURED', 'DEEPGRAM_CONFIGURED',
    'ENTITY_TYPES', 'INTENT_TYPES', 'TARGET_SAMPLE_RATE',
    
    # Models
    'ModelChoice', 'ProcessRequest', 'ProcessResponse',
    'TranscriptionJsonlRecord', 'AnnotatedJsonlRecord', 'BioAnnotation',
    
    # Age/Gender
    'load_age_gender_model', 'predict_age_gender', 'get_age_group', 'get_gender_string',
    'age_gender_model',
    
    # Emotion
    'load_emotion_model', 'predict_emotion', 'format_emotion_label', 'emotion_model',
    
    # Transcription
    'transcribe_with_whissle_single', 'transcribe_with_gemini_single',
    'transcribe_with_deepgram_single',
    
    # Prompt/Annotation
    'annotate_text_structured_with_gemini', 'get_annotation_prompt',
    
    # Audio Utils
    'discover_audio_files', 'load_audio', 'get_audio_duration', 'validate_paths',
]
