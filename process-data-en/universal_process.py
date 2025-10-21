#!/usr/bin/env python3
"""
Universal Audio Processing Pipeline
Fully configurable via YAML for different domains (automotive, kitchen, interview, creative, etc.)
"""
import os
import json
import torch
import librosa
import numpy as np
import csv
import moviepy as mp
import torch.nn as nn
import soundfile as sf
import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from pyannote.audio import Pipeline
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as transformers_pipeline
)

load_dotenv()

# Fix for numpy.NAN deprecation
if not hasattr(np, 'NAN'):
    np.NAN = np.nan
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'Infinity'):
    np.Infinity = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'infty'):
    np.infty = np.inf


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, 'final_dropout', 0.1))
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
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.randn(1).to('cuda')
        if not torch.backends.cudnn.is_available():
            print("[device] cuDNN not available despite CUDA; falling back to CPU.")
            return False
        _ = torch.backends.cudnn.version()
        return True
    except Exception as e:
        print(f"[device] CUDA reported but unusable ({e}); will fall back to CPU.")
        return False


# Global config placeholder
CONFIG = {}

# Initialize device
FORCE_CPU = os.getenv('FORCE_CPU', '0') == '1'
USE_CUDA = _cuda_usable() and not FORCE_CPU
device = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"[device] Selected device: {device}. FORCE_CPU={FORCE_CPU} USE_CUDA={USE_CUDA}")

# Global model placeholders
processor = None
model = None
pipeline = None
_whisper_model = None
_whisper_pipeline = None
_gemini_ready = False
_deepgram_ready = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_models(config: dict):
    """Initialize all required models based on config."""
    global processor, model, pipeline, device
    
    # Age/Gender model
    model_name = config.get('models', {}).get('age_gender', 'audeering/wav2vec2-large-robust-6-ft-age-gender')
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    try:
        model = AgeGenderModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"[age-gender] Initial load failed on {device} ({e}). Forcing CPU fallback.")
        device = torch.device('cpu')
        model = AgeGenderModel.from_pretrained(model_name).to(device)
    
    # Diarization pipeline
    if config.get('features', {}).get('diarization', True):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HF_TOKEN")
            )
        except Exception as e:
            print(f"[diarization] Init failed ({e}); retrying CPU fallback.")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HF_TOKEN")
                )
            except Exception as e2:
                print(f"[diarization] CPU init also failed: {e2}. Diarization disabled.")
                pipeline = None


def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    """Extract emotion from audio segment."""
    model_name = "superb/hubert-large-superb-er"
    
    if not hasattr(extract_emotion, 'model'):
        extract_emotion.model = AutoModelForAudioClassification.from_pretrained(model_name)
        extract_emotion.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    if audio_data is None or len(audio_data) == 0:
        return "No Audio"
    if len(audio_data) < sampling_rate:
        return "Audio Too Short"
    
    inputs = extract_emotion.feature_extractor(
        audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    outputs = extract_emotion.model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    return extract_emotion.model.config.id2label.get(predicted_class_idx, "Unknown")


def convert_mp4_to_wav(media_path: str, preferred_out_dir: str | None = None) -> str:
    """Convert video to mono 16k wav."""
    file_ext = os.path.splitext(media_path)[1].lower()
    if file_ext not in ['.mp4', '.webm', '.mkv']:
        return media_path
    
    candidate_dirs = []
    if preferred_out_dir:
        candidate_dirs.append(preferred_out_dir)
    parent_dir = os.path.dirname(media_path)
    candidate_dirs.append(os.path.join(parent_dir, "converted_wav"))
    candidate_dirs.append("/tmp")
    
    wav_basename = os.path.splitext(os.path.basename(media_path))[0] + ".wav"
    wav_path = None
    for d in candidate_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".__perm_test")
            with open(test_path, 'w') as t:
                t.write('ok')
            os.remove(test_path)
            wav_path = os.path.join(d, wav_basename)
            break
        except Exception:
            continue
    
    if wav_path is None:
        return media_path
    if os.path.exists(wav_path):
        return wav_path
    
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_path, "-y"
        ]
        subprocess.run(cmd, check=True)
        return wav_path
    except Exception as e:
        print(f"[convert] FFmpeg failed ({e}); attempting moviepy fallback")
        try:
            video = mp.VideoFileClip(media_path)
            video.audio.write_audiofile(wav_path, fps=16000, verbose=False, logger=None)
            video.close()
            return wav_path
        except Exception as e2:
            print(f"[convert] MoviePy failed: {e2}. Using original: {media_path}")
            return media_path


# Whisper transcription functions
def _transcribe_chunk_transformers(audio_path: str, config: dict) -> List[Dict[str, Any]]:
    """Transcribe using transformers Whisper pipeline."""
    global _whisper_pipeline
    
    try:
        if _whisper_pipeline is None:
            model_id = config.get('models', {}).get('whisper', 'openai/whisper-large-v3')
            force_cpu = os.getenv('FORCE_CPU', '0') == '1'
            device_local = 'cpu' if force_cpu or not torch.cuda.is_available() else 'cuda'
            torch_dtype = torch.float16 if device_local == 'cuda' else torch.float32
            
            print(f"[whisper-transformers] Loading model '{model_id}' on {device_local}")
            
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            whisper_model.to(device_local)
            whisper_processor = AutoProcessor.from_pretrained(model_id)
            
            _whisper_pipeline = transformers_pipeline(
                task="automatic-speech-recognition",
                model=whisper_model,
                tokenizer=whisper_processor.tokenizer,
                feature_extractor=whisper_processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_local,
                chunk_length_s=30,
                stride_length_s=5,
            )
        
        result = _whisper_pipeline(
            audio_path,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": config.get('language', 'english'),
            }
        )
        
        words: List[Dict[str, Any]] = []
        if 'chunks' in result and result['chunks']:
            for chunk in result['chunks']:
                text = chunk.get('text', '').strip()
                if not text:
                    continue
                timestamp = chunk.get('timestamp', (0.0, 0.0))
                if timestamp and len(timestamp) == 2:
                    start, end = timestamp
                    start = float(start) if start is not None else 0.0
                    end = float(end) if end is not None else start + 1.0
                    words.append({'text': text, 'start': start, 'end': end})
        elif 'text' in result:
            words.append({'text': result['text'].strip(), 'start': 0.0, 'end': 0.0})
        
        return words
    except Exception as e:
        print(f"[whisper-transformers] Transcription failed: {e}")
        return []


def _init_gemini():
    """Initialize Gemini API."""
    global _gemini_ready
    if _gemini_ready:
        return True
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_ready = True
        return True
    except Exception as e:
        print(f"[gemini] Init failed: {e}")
        return False


def gemini_transcribe_segment(audio_path: str, config: dict) -> str:
    """Transcribe audio segment using Gemini."""
    if not _init_gemini():
        return ""
    try:
        import google.generativeai as genai
        model_name = config.get('models', {}).get('gemini', 'gemini-2.0-flash')
        model = genai.GenerativeModel(model_name)
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        prompt = config.get('prompts', {}).get('gemini_transcription', 
            "Transcribe this audio to English text. Output only the transcript.")
        
        response = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_bytes},
            prompt
        ])
        return (getattr(response, 'text', None) or "").strip()
    except Exception as e:
        if os.getenv('GEMINI_DEBUG', '0') == '1':
            print(f"[gemini] Transcription failed: {e}")
        return ""


def gemini_entities_intents(text: str, config: dict) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract entities and intents using Gemini."""
    if not text or not text.strip():
        return [], []
    if not _init_gemini():
        return [], []
    try:
        import json as _json
        import google.generativeai as genai
        model_name = config.get('models', {}).get('gemini', 'gemini-2.0-flash')
        model = genai.GenerativeModel(model_name)
        truncated = text[:6000]
        
        prompt = config.get('prompts', {}).get('gemini_entities_intents', 
            'Extract entities and intents from this text as JSON: {"entities":[{"text":"...","type":"..."}],"intents":["..."]}'
        )
        escaped_text = truncated.replace('"', '\\"')
        prompt = f"{prompt}\n\nText: \"{escaped_text}\""
        
        resp = model.generate_content(prompt)
        raw = getattr(resp, 'text', '') or '{}'
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            candidate = raw[start:end+1]
        else:
            candidate = '{}'
        
        parsed = _json.loads(candidate)
        entities = parsed.get('entities', []) if isinstance(parsed, dict) else []
        intents = parsed.get('intents', []) if isinstance(parsed, dict) else []
        return entities, intents
    except Exception as e:
        if os.getenv('GEMINI_DEBUG', '0') == '1':
            print(f"[gemini] entity/intent extraction failed: {e}")
        return [], []


def gemini_structured_summary(text: str, config: dict) -> Dict[str, Any]:
    """Extract structured domain-specific summary using Gemini."""
    if not text or not text.strip():
        return {}
    if not _init_gemini():
        return {}
    try:
        import json as _json
        import google.generativeai as genai
        model_name = config.get('models', {}).get('gemini', 'gemini-2.0-flash')
        model = genai.GenerativeModel(model_name)
        truncated = text[:6000]
        
        prompt = config.get('prompts', {}).get('gemini_structured', 
            'Return JSON structured summary from this text.'
        )
        escaped_text = truncated.replace('"', '\\"')
        prompt = f"{prompt}\n\nText: \"{escaped_text}\""
        
        resp = model.generate_content(prompt)
        raw = getattr(resp, 'text', '') or '{}'
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            candidate = raw[start:end+1]
        else:
            candidate = '{}'
        
        parsed = _json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        if os.getenv('GEMINI_DEBUG', '0') == '1':
            print(f"[gemini] structured extraction error: {e}")
        return {}


def _init_deepgram():
    """Initialize Deepgram API."""
    global _deepgram_ready
    if _deepgram_ready:
        return True
    if not os.getenv('DEEPGRAM_API_KEY'):
        return False
    _deepgram_ready = True
    return True


def deepgram_transcribe_segment(audio_path: str, config: dict) -> str:
    """Transcribe segment using Deepgram."""
    if not _init_deepgram():
        return ""
    api_key = os.getenv('DEEPGRAM_API_KEY')
    model = config.get('models', {}).get('deepgram', 'nova-2-general')
    try:
        import requests
        with open(audio_path, 'rb') as f:
            resp = requests.post(
                'https://api.deepgram.com/v1/listen',
                params={'model': model, 'language': 'en', 'smart_format': 'true'},
                headers={'Authorization': f'Token {api_key}'},
                data=f
            )
        if resp.status_code != 200:
            return ""
        j = resp.json()
        text = j.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
        return text.strip()
    except Exception:
        return ""


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: int
    text: str
    whisper_text: str
    gemini_text: str
    emotion: str
    chunk_filename: str
    deepgram_text: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    structured_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""
    duration_secs: float = 0.0

    def get_formatted_text(self, config: dict) -> str:
        """Format segments with metadata tags."""
        texts: List[str] = []
        current_speaker = None
        last_text = None
        for segment in self.segments:
            age_bucket = self.get_age_bucket(segment.age)
            gender_text = "MALE" if segment.gender == 1 else "FEMALE"
            emotion_text = segment.emotion.upper().replace(" ", "_")
            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
            if current_speaker != segment.speaker:
                metadata += " SPEAKER_CHANGE"
                current_speaker = segment.speaker
            raw = (segment.text or "").lower().strip()
            if raw == last_text and raw != "":
                continue
            last_text = raw
            combined = f"{raw} {metadata}".strip() if raw else metadata
            texts.append(combined)
        return " ".join(texts)

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100, 2)
        age_brackets = [(18, "0_18"), (30, "18_30"), (45, "30_45"), (60, "45_60"), (float('inf'), "60PLUS")]
        for threshold, bracket in age_brackets:
            if actual_age < threshold:
                return bracket
        return "60PLUS"


def process_large_audio(
    audio_path: str,
    config: dict,
    output_base_dir: str = "output",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Main audio processing pipeline - fully configurable."""
    chunk_duration = config.get('processing', {}).get('chunk_duration', 30.0)
    overlap_duration = config.get('processing', {}).get('overlap_duration', 10.0)
    
    chunks_dir = os.path.abspath(os.path.join(output_base_dir, "audio_chunks"))
    results_dir = os.path.abspath(os.path.join(output_base_dir, "results"))
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    processed_audio_path = convert_mp4_to_wav(audio_path, preferred_out_dir=os.path.join(output_base_dir, "audio_converted"))
    print(f"[process] Using audio: {processed_audio_path}")

    try:
        import soundfile as sf
        signal, sr = sf.read(processed_audio_path)
        if isinstance(signal, np.ndarray) and signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        if sr != 16000:
            import resampy
            signal = resampy.resample(signal, sr, 16000)
            sr = 16000
    except Exception as e:
        print(f"Error loading audio file {processed_audio_path}: {e}")
        return [], []

    all_data = []
    chunk_data: Dict[str, ChunkData] = defaultdict(ChunkData)
    chunk_size = int(chunk_duration * sr)
    overlap_size = int(max(0.0, overlap_duration) * sr)
    stride = max(1, chunk_size - overlap_size)

    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    global_speaker_id_map: Dict[str, int] = {}
    global_next_spk_id = 0

    min_speakers = config.get('processing', {}).get('min_speakers', 2)
    max_speakers = config.get('processing', {}).get('max_speakers')

    try:
        chunk_counter = 0
        for chunk_start in range(0, len(signal), stride):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            end = min(chunk_start + chunk_size, len(signal))
            chunk = signal[chunk_start:end]

            if len(chunk) / sr < 1.0:
                continue

            chunk_index = chunk_counter
            chunk_filename = f"{base_filename}_chunk_{chunk_index}.wav"
            chunk_path = os.path.join(chunks_dir, chunk_filename)

            if not os.path.exists(chunk_path):
                sf.write(chunk_path, chunk, sr)

            true_chunk_duration = float(len(chunk)) / float(sr)
            chunk_data[chunk_filename].duration_secs = true_chunk_duration
            chunk_data[chunk_filename].filepath = os.path.abspath(chunk_path)

            try:
                if pipeline is None:
                    print("[process] Skipping diarization (pipeline unavailable).")
                    speaker_changes = []
                else:
                    if max_speakers is not None:
                        diarization = pipeline(chunk_path, min_speakers=min_speakers, max_speakers=max_speakers)
                    else:
                        diarization = pipeline(chunk_path, min_speakers=min_speakers)
                    speaker_changes = [
                        (turn.start, turn.end, speaker)
                        for turn, _, speaker in diarization.itertracks(yield_label=True)
                    ]

                speaker_id_map = global_speaker_id_map
                next_spk_id = global_next_spk_id
                whisper_words = _transcribe_chunk_transformers(chunk_path, config)

                for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                    if speaker not in speaker_id_map:
                        speaker_id_map[speaker] = next_spk_id
                        next_spk_id += 1
                    numeric_speaker = speaker_id_map[speaker]
                    global_next_spk_id = next_spk_id

                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    speaker_segment = chunk[start_sample:end_sample]

                    if len(speaker_segment) / sr < 1.5:
                        continue

                    y = processor(speaker_segment, sampling_rate=sr)
                    y = y['input_values'][0]
                    y = y.reshape(1, -1)
                    y = torch.from_numpy(y).to(device)

                    with torch.no_grad():
                        model_output = model(y)
                        age = float(model_output[1].detach().cpu().numpy()[0][0])
                        gender = np.argmax(model_output[2].detach().cpu().numpy())

                    temp_segment_path = os.path.join(chunks_dir, f"temp_segment_{chunk_index}_{speaker_idx}.wav")
                    sf.write(temp_segment_path, speaker_segment, sr)

                    try:
                        speaker_segment_audio, _ = librosa.load(temp_segment_path, sr=16000)
                        emotion = extract_emotion(speaker_segment_audio)

                        seg_start, seg_end = start_time, end_time
                        seg_words = []
                        for w in whisper_words:
                            w_mid = (w['start'] + w['end']) / 2.0
                            if w_mid >= seg_start and w_mid <= seg_end:
                                seg_words.append(w['text'])
                        whisper_text = ' '.join(seg_words).strip()

                        gemini_text = ""
                        deepgram_text = ""
                        if config.get('features', {}).get('gemini_transcription', False):
                            gemini_text = gemini_transcribe_segment(temp_segment_path, config)
                        if config.get('features', {}).get('deepgram_transcription', False):
                            deepgram_text = deepgram_transcribe_segment(temp_segment_path, config)

                        base_text_for_ner = gemini_text if gemini_text else whisper_text
                        entities, intents = [], []
                        structured = {}
                        if config.get('features', {}).get('entity_intent_extraction', False):
                            entities, intents = gemini_entities_intents(base_text_for_ner, config)
                        if config.get('features', {}).get('structured_summary', False):
                            structured = gemini_structured_summary(base_text_for_ner, config)

                        segment = AudioSegment(
                            start_time=start_time,
                            end_time=end_time,
                            speaker=f"speaker_{numeric_speaker}",
                            age=float(age),
                            gender=int(gender),
                            text=whisper_text,
                            whisper_text=whisper_text,
                            gemini_text=gemini_text,
                            deepgram_text=deepgram_text,
                            emotion=emotion,
                            chunk_filename=chunk_filename,
                            entities=entities,
                            intents=intents,
                            structured_summary=structured
                        )

                        segment_data = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'speaker': f"speaker_{numeric_speaker}",
                            'speaker_index': numeric_speaker,
                            'speaker_tag': f"speaker_{numeric_speaker}",
                            'age': float(age),
                            'gender': int(gender),
                            'whisper_text': whisper_text,
                            'gemini_text': gemini_text,
                            'deepgram_text': deepgram_text,
                            'entities': entities,
                            'intents': intents,
                            'emotion': emotion,
                            'structured_summary': structured,
                            'audio_file_path': os.path.abspath(chunk_path)
                        }

                        all_data.append(segment_data)
                        chunk_data[chunk_filename].segments.append(segment)

                    except Exception as e:
                        print(f"Error processing segment in chunk {chunk_filename}: {e}")
                        continue
                    finally:
                        if os.path.exists(temp_segment_path):
                            os.remove(temp_segment_path)

            except Exception as e:
                print(f"Error processing chunk {chunk_filename}: {e}")
                continue
            finally:
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            chunk_counter += 1

        if all_data:
            last_speaker = None
            speaker_change_counter = {}
            seq = []
            speaker_stats: Dict[int, Dict[str, Any]] = {}

            for seg in all_data:
                spk = seg['speaker']
                if spk != last_speaker:
                    if spk not in speaker_change_counter:
                        speaker_change_counter[spk] = len(speaker_change_counter)
                    change_id = speaker_change_counter[spk]
                    seg['speaker_change_tag'] = f"speaker_change_{change_id}"
                    last_speaker = spk
                else:
                    seg['speaker_change_tag'] = f"speaker_change_{speaker_change_counter[spk]}"

                spk_index = seg.get('speaker_index')
                if spk_index is not None:
                    entry = speaker_stats.setdefault(spk_index, {
                        'speaker_index': spk_index,
                        'speaker_tag': seg.get('speaker_tag', spk),
                        'first_start_time': seg['start_time'],
                        'cumulative_duration': 0.0,
                        'segment_count': 0
                    })
                    duration = float(seg['end_time'] - seg['start_time'])
                    entry['cumulative_duration'] += max(0.0, duration)
                    entry['segment_count'] += 1
                    if seg['start_time'] < entry['first_start_time']:
                        entry['first_start_time'] = seg['start_time']
                seq.append(seg)

            json_path = os.path.join(results_dir, f"{base_filename}_processed_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(seq, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON to: {json_path}")

            if speaker_stats:
                csv_path = os.path.join(results_dir, f"{base_filename}_speaker_mapping.csv")
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                        writer = csv.writer(cf)
                        writer.writerow(["speaker_index", "speaker_tag", "first_start_time", "cumulative_duration", "segment_count"])
                        for idx in sorted(speaker_stats.keys()):
                            s = speaker_stats[idx]
                            writer.writerow([
                                s['speaker_index'],
                                s['speaker_tag'],
                                round(s['first_start_time'], 3),
                                round(s['cumulative_duration'], 3),
                                s['segment_count']
                            ])
                    print(f"Saved speaker mapping CSV to: {csv_path}")
                except Exception as e:
                    print(f"[speaker-csv] Failed writing CSV: {e}")

            jsonl_path = os.path.join(results_dir, f"{base_filename}_audio_text_pairs.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as jf:
                for chunk_name, cdata in chunk_data.items():
                    if cdata.segments:
                        duration = float(cdata.duration_secs) if getattr(cdata, 'duration_secs', 0.0) else 0.0
                        whisper_join = ' '.join([seg.whisper_text for seg in cdata.segments if seg.whisper_text])
                        gemini_join = ' '.join([seg.gemini_text for seg in cdata.segments if seg.gemini_text])
                        deepgram_join = ' '.join([seg.deepgram_text for seg in cdata.segments if getattr(seg, 'deepgram_text', '')])
                        tagged_text = cdata.get_formatted_text(config)
                        entry = {
                            "audio_filepath": cdata.filepath,
                            "whisper_text": whisper_join.strip(),
                            "gemini_text": gemini_join.strip(),
                            "deepgram_text": deepgram_join.strip(),
                            "tagged_text": tagged_text,
                            "duration": duration
                        }
                        jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Saved JSONL pairs to: {jsonl_path}")

            return all_data, []
        else:
            print("[process] No segments extracted.")
            return [], []

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return [], []


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python universal_process.py <config.yaml> [input_directory]")
        print("Example: python universal_process.py configs/automotive.yaml /path/to/audio/files")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Initialize models
    init_models(config)
    
    # Determine input directory
    if len(sys.argv) > 2:
        input_dir = sys.argv[2]
    else:
        input_dir = config.get('input', {}).get('directory', '.')
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = config.get('output', {}).get('directory', input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing root: {input_dir}")
    print(f"Outputs saved to: {output_dir}")
    
    # Get file extensions from config
    target_extensions = set(config.get('input', {}).get('extensions', ['.wav', '.mp3', '.mp4']))
    
    # Find all media files
    media_files: List[str] = []
    for dirpath, _, filenames in os.walk(input_dir):
        for fn in sorted(filenames):
            if any(fn.lower().endswith(ext) for ext in target_extensions):
                media_files.append(os.path.join(dirpath, fn))
    
    if not media_files:
        print(f"[exit] No media files found with extensions: {target_extensions}")
        sys.exit(0)
    
    total = len(media_files)
    print(f"Discovered {total} media files to process.")
    
    for idx, audio_path in enumerate(media_files, start=1):
        parent_dir = os.path.dirname(audio_path)
        filename = os.path.basename(audio_path)
        print(f"\n[{idx}/{total}] Processing: {filename}")
        print(f"Path: {audio_path}")
        
        output_base = config.get('output', {}).get('per_file_subdirs', True)
        if output_base:
            file_output_dir = parent_dir
        else:
            file_output_dir = output_dir
        
        df, _ = process_large_audio(audio_path, config, output_base_dir=file_output_dir)
        print(f"Finished {filename}. Segments: {len(df)}")
    
    print("\n✅ All files processed successfully!")


if __name__ == "__main__":
    main()
