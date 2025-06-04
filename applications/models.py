# applications/models.py
import torch
from transformers import (
    AutoModelForAudioClassification, AutoFeatureExtractor,
    Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
)
import torch.nn as nn
import google.generativeai as genai
from deepgram import DeepgramClient
from applications.config import logger, GOOGLE_API_KEY, DEEPGRAM_API_KEY, WHISSLE_AUTH_TOKEN, device

# Globals for model availability
GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
DEEPGRAM_CONFIGURED = bool(DEEPGRAM_API_KEY)

# Model instances
age_gender_model = None
age_gender_processor = None
emotion_model = None
emotion_feature_extractor = None
DEEPGRAM_CLIENT = None

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
    logger.info("Loading Age/Gender model...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
    age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
    age_gender_model.eval()
    logger.info("Age/Gender model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Age/Gender model: {e}", exc_info=True)

try:
    logger.info("Loading Emotion model...")
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
    emotion_model.eval()
    logger.info("Emotion model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Emotion model: {e}", exc_info=True)

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API configured successfully.")
        GEMINI_CONFIGURED = True
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")

try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
    if WHISSLE_AUTH_TOKEN:
        logger.info("Whissle Auth Token found.")
        WHISSLE_CONFIGURED = True
except ImportError:
    logger.warning("WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass

if DEEPGRAM_CONFIGURED:
    try:
        DEEPGRAM_CLIENT = DeepgramClient(DEEPGRAM_API_KEY)
        logger.info("Deepgram client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Deepgram client: {e}")
        DEEPGRAM_CONFIGURED = False
else:
    logger.warning("Deepgram API key not set. Deepgram transcription disabled.")