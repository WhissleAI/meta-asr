"""
Emotion prediction model
"""
import torch
import numpy as np
import logging
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from typing import Tuple, Optional
from .config import device

logger = logging.getLogger(__name__)

# Global model variables
emotion_model = None
emotion_feature_extractor = None

def load_emotion_model():
    """Load the emotion prediction model"""
    global emotion_model, emotion_feature_extractor
    
    try:
        logger.info("Loading Emotion model...")
        emotion_model_name = "superb/hubert-large-superb-er"
        emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model.eval()
        logger.info("Emotion model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load Emotion model: {e}", exc_info=True)
        return False

def predict_emotion(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[str], Optional[str]]:
    """Predict emotion from audio data"""
    if emotion_model is None or emotion_feature_extractor is None:
        return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, "Empty audio data provided for Emotion."
    
    min_length = int(sampling_rate * 0.1)
    if len(audio_data) < min_length:
        return "SHORT_AUDIO", None
    
    try:
        inputs = emotion_feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")
        return emotion_label, None
    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=False)
        return None, f"Emotion prediction failed: {type(e).__name__}"

def format_emotion_label(emotion_label: str) -> str:
    """Format emotion label for display"""
    if emotion_label == "SHORT_AUDIO":
        return "Short Audio"
    return emotion_label.replace("_", " ").title()
