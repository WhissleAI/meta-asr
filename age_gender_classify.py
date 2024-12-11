import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""

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
    r"""Speech age and gender classifier."""

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
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # Normalize signal using processor
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # Run through the model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # Convert to NumPy array
    y = y.detach().cpu().numpy()
    return y


audio_path = "test_audio/test.mp3" 
signal, sr = librosa.load(audio_path, sr=16000)
sampling_rate = 16000

signal = signal[:sampling_rate]  
signal = signal.reshape(1, -1).astype(np.float32)  

# Process the audio
output = process_func(signal, sampling_rate)
print(output)