"""
Age and Gender prediction models
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from typing import Tuple, Optional
from .config import device

logger = logging.getLogger(__name__)

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
        self.gender = ModelHead(config, 3) # Assuming 3 genders (Male, Female, Other/Unknown)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1) # Pool over sequence length
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        return hidden_states, logits_age, logits_gender

# Global model variables
age_gender_model = None
age_gender_processor = None

def load_age_gender_model():
    """Load the age and gender prediction model"""
    global age_gender_model, age_gender_processor
    
    try:
        logger.info("Loading Age/Gender model...")
        age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
        age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        logger.info("Age/Gender model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load Age/Gender model: {e}", exc_info=True)
        return False

def predict_age_gender(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Predict age and gender from audio data"""
    if age_gender_model is None or age_gender_processor is None:
        return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, None, "Empty audio data provided for Age/Gender."
    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = age_gender_model(input_values)
        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]
        return float(age_pred), int(gender_pred_idx), None
    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=False)
        return None, None, f"Age/Gender prediction failed: {type(e).__name__}"

def get_age_group(age_pred: float) -> str:
    """Convert age prediction to age group"""
    try:
        actual_age = round(age_pred, 1)
        age_brackets = [
            (18, "0-17"), (25, "18-24"), (35, "25-34"),
            (45, "35-44"), (55, "45-54"), (65, "55-64"),
            (float('inf'), "65+")
        ]
        age_group = "Unknown"
        for threshold, bracket in age_brackets:
            if actual_age < threshold:
                age_group = bracket
                break
        return age_group
    except Exception as age_e:
        logger.error(f"Error formatting age_group: {age_e}")
        return "Error"

def get_gender_string(gender_idx: int) -> str:
    """Convert gender index to string"""
    gender_str = "Unknown"
    if gender_idx == 1:
        gender_str = "Male"
    elif gender_idx == 0:
        gender_str = "Female"
    return gender_str
