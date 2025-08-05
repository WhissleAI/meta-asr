# applications/models.py
import torch
from transformers import (
    AutoModelForAudioClassification, AutoFeatureExtractor,
    Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
)
import torch.nn as nn
from config import logger, GOOGLE_API_KEY, DEEPGRAM_API_KEY, WHISSLE_AUTH_TOKEN, device


# Globals for model availability
GEMINI_AVAILABLE = False # Renamed from GEMINI_CONFIGURED, indicates library availability
WHISSLE_AVAILABLE = False # Indicates library availability, WHISSLE_CONFIGURED removed as it was token dependent
DEEPGRAM_AVAILABLE = False # Renamed from DEEPGRAM_CONFIGURED, indicates library availability

# Model instances
age_gender_model = None
age_gender_processor = None
emotion_model = None
emotion_feature_extractor = None

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        return hidden_states, logits_age, logits_gender

# Load models and configure APIs
try:
    logger.info("Attempting to import google.generativeai for Gemini...")
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("Gemini (google.generativeai) library is available.")

    # Configure Gemini API if GOOGLE_API_KEY is present
    GEMINI_CONFIGURED = False
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Gemini API configured successfully.")
            GEMINI_CONFIGURED = True
            GEMINI_AVAILABLE = True
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
            GEMINI_AVAILABLE = False
    else:
        logger.warning("GOOGLE_API_KEY not set. Gemini API will not be configured.")
except ImportError:
    logger.warning("google.generativeai library not found. Gemini features will be unavailable.")
    GEMINI_AVAILABLE = False
except Exception as e:
    logger.error(f"Error during Gemini (google.generativeai) import or initial setup: {e}. Gemini features will be unavailable.")
    GEMINI_AVAILABLE = False


try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
    logger.info("WhissleClient SDK found and available.")
except ImportError:
    logger.warning("WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass # Keep shim for type hinting if used elsewhere

# Check for Deepgram SDK availability
try:
    from deepgram import DeepgramClient as DeepgramSDKClient # Alias to avoid confusion if we had a global
    DEEPGRAM_AVAILABLE = True
    logger.info("Deepgram SDK is available.")
except ImportError:
    logger.warning("Deepgram SDK not found. Deepgram features will be unavailable.")
    DEEPGRAM_AVAILABLE = False
    class DeepgramSDKClient: pass # Shim for type hinting

# Ensure DEEPGRAM_CONFIGURED (old name) is not used later, or update its usage.
# For now, we'll rely on DEEPGRAM_AVAILABLE.
# The actual configuration with a key will happen at request time.