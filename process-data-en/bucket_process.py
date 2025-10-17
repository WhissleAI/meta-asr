import os
import json
import torch
import asyncio 
import librosa
import numpy as np
import pandas as pd
import csv
import moviepy as mp
import torch.nn as nn
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv

from pyannote.audio import Pipeline
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import pipeline as transformers_pipeline  # Alias to avoid conflict with pyannote.Pipeline
load_dotenv()

# Fix for numpy.NAN deprecation (needed for pyannote and other audio libraries)
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



"""Updated pipeline (multi transcription edition):
1. Performs speaker diarization per chunk, maps numeric speaker ids (speaker_0, speaker_1, ...).
2. High quality Whisper (faster-whisper OR transformers) chunk-level transcription, then assigns words to speaker segments.
3. Optional Gemini transcription per segment (gemini_text) if GEMINI_API_KEY / GOOGLE_API_KEY set.
4. Optional Deepgram transcription per segment (deepgram_text) if DEEPGRAM_API_KEY set.
5. Gemini-based entity + intent extraction tailored for animated content (characters, actions, SFX, etc.).
6. Age / Gender / Emotion tagging per speaker segment.
7. speaker_change_X tag added on first occurrence of a speaker in chronological order.
8. Outputs:
    - Consolidated JSON with all segment metadata including whisper_text, gemini_text, deepgram_text, entities, intents.
    - JSONL of chunks: audio_filepath, whisper_text, gemini_text, deepgram_text, duration, combined tagged text.
Note: External subtitle ingestion (VTT/SRT) has been removed to simplify the pipeline and avoid
dependencies on sidecar files. All annotations are now derived from ASR outputs only.
"""


import re
from datetime import timedelta
import sys

SRT_TIME_PATTERN = re.compile(r"(?P<h>\d{2}):(\d{2}):(\d{2}),(\d{3})")

def _srt_timestamp_to_seconds(ts: str) -> float:
    match = SRT_TIME_PATTERN.match(ts.strip())
    if not match:
        return 0.0
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0


def _merge_vtt_cues_texts(cue_texts: List[str]) -> str:
    """Merge a list of VTT cue texts while removing overlap-based repetition.
    Uses token-level suffix/prefix matching on lowercase tokens to detect overlaps,
    but preserves the original tokens in the final output.
    """
    result_raw: List[str] = []
    result_norm: List[str] = []
    for text in cue_texts:
        if not text:
            continue
        tokens_raw = text.strip().split()
        if not tokens_raw:
            continue
        tokens_norm = [t.lower() for t in tokens_raw]
        if not result_raw:
            result_raw = tokens_raw[:]
            result_norm = tokens_norm[:]
            continue
        max_k = min(len(result_norm), len(tokens_norm))
        # Cap overlap search to a reasonable window to avoid O(n^2) cost on huge segments
        max_k = min(max_k, 30)
        overlap = 0
        for k in range(max_k, 0, -1):
            if result_norm[-k:] == tokens_norm[:k]:
                overlap = k
                break
        # Append only the non-overlapping tail from the raw tokens
        if overlap < len(tokens_raw):
            result_raw.extend(tokens_raw[overlap:])
            result_norm.extend(tokens_norm[overlap:])
    return " ".join(result_raw)



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
        # Quick allocation + cuDNN probe
        _ = torch.randn(1).to('cuda')
        if not torch.backends.cudnn.is_available():
            print("[device] cuDNN not available despite CUDA; falling back to CPU.")
            return False
        _ = torch.backends.cudnn.version()
        return True
    except Exception as e:
        print(f"[device] CUDA reported but unusable ({e}); will fall back to CPU.")
        return False

FORCE_CPU = os.getenv('FORCE_CPU', '0') == '1'
USE_CUDA = _cuda_usable() and not FORCE_CPU
device = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"[device] Selected device: {device}. FORCE_CPU={FORCE_CPU} USE_CUDA={USE_CUDA}")
if USE_CUDA:
    try:
        print(f"[device] torch CUDA version: {getattr(torch.version, 'cuda', 'unknown')}; cuDNN: {torch.backends.cudnn.version()}")
    except Exception:
        pass

model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(model_name)

try:
    model = AgeGenderModel.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"[age-gender] Initial load failed on {device} ({e}). Forcing CPU fallback.")
    FORCE_CPU = True
    USE_CUDA = False
    device = torch.device('cpu')
    os.environ['FORCE_CPU'] = '1'
    model = AgeGenderModel.from_pretrained(model_name).to(device)

# Diarization pipeline init with fallback if CUDA/cuDNN broken
pipeline = None
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"[diarization] CUDA pipeline init failed ({e}); retrying CPU fallback.")
    # Hide GPUs for second attempt
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_TOKEN"))
        print("[diarization] Successfully initialized pipeline on CPU.")
    except Exception as e2:
        print(f"[diarization] CPU pipeline init also failed: {e2}. Diarization disabled.")
        pipeline = None

def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0].detach().cpu().numpy()
        else:
            age_logits = y[1].detach().cpu().numpy()
            gender_logits = y[2].detach().cpu().numpy()
            y = [age_logits, gender_logits]

    return y


def get_speaker_changes(audio_path: str):
    diarization = pipeline({'audio': audio_path})
    speaker_changes = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_changes.append((turn.start, turn.end, speaker))
    return speaker_changes





def split_audio_by_speaker(audio_path: str, speaker_changes: list, output_dir: str = "spk_dir", max_duration: float = 20.0):
    os.makedirs(output_dir, exist_ok=True)

    signal, sr = librosa.load(audio_path, sr=16000)
    audio_segments = []

    for idx, (start, end, speaker) in enumerate(speaker_changes):
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        while start_sample < end_sample:
            segment_end_sample = min(start_sample + int(max_duration * sr), end_sample)
            segment = signal[start_sample:segment_end_sample]

            segment_filename = os.path.join(output_dir, f"speaker_{speaker}_segment_{idx}.wav")
            sf.write(segment_filename, segment, sr)
            audio_segments.append(segment_filename)

            start_sample = segment_end_sample

    return audio_segments
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    model_name = "superb/hubert-large-superb-er"

    if not hasattr(extract_emotion, 'model'):
        extract_emotion.model = AutoModelForAudioClassification.from_pretrained(model_name)
        extract_emotion.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    if audio_data is None or len(audio_data) == 0:
        return "No Audio"

    if len(audio_data) < sampling_rate:
        return "Audio Too Short"

    inputs = extract_emotion.feature_extractor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
    )

    outputs = extract_emotion.model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()

    return extract_emotion.model.config.id2label.get(predicted_class_idx, "Unknown")


def convert_mp4_to_wav(media_path: str, preferred_out_dir: str | None = None) -> str:
    """Convert mp4/webm/mkv to mono 16k wav. On failure returns original path.
    Handles permission errors by falling back to a temp directory under the same parent.
    """
    file_ext = os.path.splitext(media_path)[1].lower()
    if file_ext not in ['.mp4', '.webm', '.mkv']:
        return media_path

    # Choose output directory (prefer user-provided or same directory if writable)
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
        print("[convert_mp4_to_wav] Could not find writable directory, using original file directly.")
        return media_path

    # If already exists, reuse
    if os.path.exists(wav_path):
        return wav_path

    # Attempt ffmpeg extraction
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
        print(f"[convert_mp4_to_wav] FFmpeg failed ({e}); attempting moviepy fallback -> {wav_path}")
        try:
            video = mp.VideoFileClip(media_path)
            video.audio.write_audiofile(wav_path, fps=16000, verbose=False, logger=None)
            video.close()
            return wav_path
        except Exception as e2:
            print(f"[convert_mp4_to_wav] MoviePy fallback failed: {e2}. Using original media file: {media_path}")
            return media_path
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: int
    text: str  # Backward compatibility: will store whisper_text
    whisper_text: str
    gemini_text: str
    emotion: str
    chunk_filename: str
    deepgram_text: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)

def extract_entities_and_intents(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Placeholder for gimnin entity+intent extraction. Return empty for now."""
    return [], []

# ---------------- Whisper & Gemini Integration ---------------- #
def _can_load_ctranslate2_gpu() -> bool:
    """
    Probes if the CTranslate2 backend used by faster-whisper can likely load a GPU model
    without a hard crash by checking for the presence of the specific cuDNN libraries it needs.
    This is a workaround for segfaults that can't be caught by try/except.
    """
    if not USE_CUDA:
        return False
    try:
        import ctypes.util
        # CTranslate2/faster-whisper specifically looks for one of these .so files.
        # If none can be found by the dynamic linker, it will almost certainly crash.
        cudnn_libs = [
            "libcudnn_ops.so.9", "libcudnn_ops.so.8", "libcudnn.so.9", "libcudnn.so.8"
        ]
        found_a_cudnn_lib = any(ctypes.util.find_library(lib) for lib in cudnn_libs)
        if not found_a_cudnn_lib:
            print("[probe] Pre-emptive check failed: No suitable cuDNN library found for faster-whisper. Will use CPU fallback.")
            return False
        return True
    except Exception as e:
        print(f"[probe] Pre-emptive library check failed with an error: {e}. Assuming GPU is not safe for faster-whisper.")
        return False

_whisper_model = None
_whisper_pipeline = None  # Cache for transformers pipeline
_whisper_backend = os.getenv('WHISPER_BACKEND', 'faster').lower().strip()
_whisper_model_name_env = os.getenv('WHISPER_MODEL')  # optional override

# Pre-flight check to prevent hard crash
if _whisper_backend == 'faster' and not _can_load_ctranslate2_gpu():
    print("[probe] Switching to 'transformers' backend due to unsafe GPU environment for faster-whisper.")
    _whisper_backend = 'transformers'
    # transformers backend will use GPU by default


def _transcribe_chunk_openai_whisper(audio_path: str) -> List[Dict[str, Any]]:
    """Fallback using openai-whisper library. Returns list of dicts with text,start,end.
    Word-level timestamps not guaranteed; we use segment-level granularity.
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        print("[whisper-openai] Package not installed. Install with: pip install --upgrade openai-whisper")
        return []
    model_name = _whisper_model_name_env or 'large'
    # Use CUDA if available and not explicitly forced to CPU
    force_cpu = os.getenv('FORCE_CPU', '0') == '1'
    device_local = 'cpu' if force_cpu or not torch.cuda.is_available() else 'cuda'
    try:
        print(f"[whisper-openai] Loading model '{model_name}' on {device_local}")
        model = whisper.load_model(model_name, device=device_local)
        result = model.transcribe(audio_path, verbose=False)
        words: List[Dict[str, Any]] = []
        for seg in result.get('segments', []):
            # segment-level timing
            words.append({
                'text': seg.get('text','').strip(),
                'start': float(seg.get('start',0.0)),
                'end': float(seg.get('end', seg.get('start',0.0)))
            })
        return words
    except Exception as e:
        print(f"[whisper-openai] Transcription failed: {e}")
        return []

def _transcribe_chunk_transformers(audio_path: str) -> List[Dict[str, Any]]:
    """Transcribe using Hugging Face transformers pipeline with openai/whisper-large-v3.
    This properly utilizes GPU and provides chunk-level timestamps.
    """
    global _whisper_pipeline
    
    try:
        # Create pipeline once and reuse
        if _whisper_pipeline is None:
            model_id = _whisper_model_name_env or "openai/whisper-large-v3"
            
            force_cpu = os.getenv('FORCE_CPU', '0') == '1'
            device = 'cpu' if force_cpu or not torch.cuda.is_available() else 'cuda'
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
            
            print(f"[whisper-transformers] Loading model '{model_id}' on {device}")
            
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            model.to(device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline - use transformers_pipeline to avoid conflict with pyannote
            # Use chunk_length_s to ensure proper timestamp generation
            _whisper_pipeline = transformers_pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
                chunk_length_s=30,  # Process in 30-second chunks for better timestamp accuracy
                stride_length_s=5,   # 5-second overlap between chunks
            )
        
        # Transcribe - use chunk-level timestamps (more reliable than word-level)
        # Setting return_timestamps=True gives us chunk-level timestamps
        result = _whisper_pipeline(
            audio_path, 
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "english",  # Adjust if needed
            }
        )
        
        words: List[Dict[str, Any]] = []
        
        # Extract chunk-level timestamps
        if 'chunks' in result and result['chunks']:
            for chunk in result['chunks']:
                text = chunk.get('text', '').strip()
                if not text:
                    continue
                    
                timestamp = chunk.get('timestamp', (0.0, 0.0))
                if timestamp and len(timestamp) == 2:
                    start, end = timestamp
                    # Handle None values
                    start = float(start) if start is not None else 0.0
                    end = float(end) if end is not None else start + 1.0  # Fallback: 1 second duration
                    
                    words.append({
                        'text': text,
                        'start': start,
                        'end': end
                    })
                else:
                    # No timestamp info for this chunk
                    words.append({
                        'text': text,
                        'start': 0.0,
                        'end': 1.0
                    })
        elif 'text' in result:
            # Fallback: no chunks, just full text
            words.append({
                'text': result['text'].strip(),
                'start': 0.0,
                'end': 0.0
            })
        
        return words
        
    except Exception as e:
        print(f"[whisper-transformers] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return []
        print(f"[whisper-transformers] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_whisper_model():
    global _whisper_model, _whisper_backend
    if _whisper_backend in ['openai', 'transformers']:
        return None  # handled separately by their respective functions
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[whisper] faster-whisper not installed. Install with: pip install faster-whisper")
        return None
    force_cpu_local = os.getenv('FORCE_CPU', '0') == '1'
    use_cuda_local = torch.cuda.is_available() and not force_cpu_local
    device = 'cuda' if use_cuda_local else 'cpu'
    # choose compute type conservatively if GPU unstable
    if use_cuda_local:
        compute_type = 'float16'
    else:
        # int8 for CPU for memory; could switch to int8_float16 if supported
        compute_type = 'int8'
    print(f"[whisper] Loading model on {device} compute_type={compute_type} (FORCE_CPU={force_cpu_local})")
    model_choice = _whisper_model_name_env or 'large-v3'
    try:
        _whisper_model = WhisperModel(model_choice, device=device, compute_type=compute_type)
    except Exception as e:
        # cuDNN failure? fallback to transformers
        e_lower = str(e).lower()
        if 'cudnn' in e_lower or 'libcudnn' in e_lower:
            print(f"[whisper] cuDNN-related load failure for '{model_choice}': {e}. Falling back to transformers backend.")
            _whisper_backend = 'transformers'
            return None
        print(f"[whisper] Failed loading {model_choice} ({e}), trying medium.")
        try:
            _whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)
        except Exception as e2:
            print(f"[whisper] Medium model load also failed: {e2}. Switching to transformers backend.")
            _whisper_backend = 'transformers'
            return None
    return _whisper_model

def transcribe_chunk_whisper(audio_path: str) -> List[Dict[str, Any]]:
    """Return list of word/segment dicts: {text,start,end} for the chunk (backend-dependent)."""
    global _whisper_backend
    if _whisper_backend == 'openai':
        return _transcribe_chunk_openai_whisper(audio_path)
    if _whisper_backend == 'transformers':
        return _transcribe_chunk_transformers(audio_path)
    model = load_whisper_model()
    if model is None:
        # Possibly switched backend due to failure
        if _whisper_backend == 'openai':
            return _transcribe_chunk_openai_whisper(audio_path)
        if _whisper_backend == 'transformers':
            return _transcribe_chunk_transformers(audio_path)
        return []
    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        words = []
        for seg in segments:
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    words.append({
                        'text': w.word.strip(),
                        'start': float(w.start),
                        'end': float(w.end)
                    })
            else:
                words.append({'text': seg.text.strip(), 'start': float(seg.start), 'end': float(seg.end)})
        return words
    except Exception as e:
        e_lower = str(e).lower()
        if 'cudnn' in e_lower or 'libcudnn' in e_lower:
            print(f"[whisper] Detected cuDNN error during transcription: {e}. Switching to transformers backend.")
            _whisper_backend = 'transformers'
            return _transcribe_chunk_transformers(audio_path)
        print(f"[whisper] Transcription failed on {audio_path}: {e}")
        return []

_gemini_ready = False
_gemini_model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')  # 1.5 flash deprecated -> use 2.0
_deepgram_ready = False

def _init_deepgram():
    """Initialize Deepgram lazily. Returns True if usable."""
    global _deepgram_ready
    if _deepgram_ready:
        return True
    if not os.getenv('DEEPGRAM_API_KEY'):
        return False
    _deepgram_ready = True
    return True

def deepgram_transcribe_segment(audio_path: str) -> str:
    if not _init_deepgram():
        return ""
    api_key = os.getenv('DEEPGRAM_API_KEY')
    model = os.getenv('DEEPGRAM_MODEL', 'nova-2-general')
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
            if os.getenv('DEEPGRAM_DEBUG','0') == '1':
                print(f"[deepgram] HTTP {resp.status_code}: {resp.text[:160]}")
            return ""
        j = resp.json()
        text = j.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
        return text.strip()
    except Exception as e:
        if os.getenv('DEEPGRAM_DEBUG','0') == '1':
            print(f"[deepgram] Error: {e}")
        return ""

def _init_gemini():
    """Initialise Gemini. Accepts GEMINI_API_KEY or GOOGLE_API_KEY.
    Sets _gemini_ready flag. Returns True if usable.
    """
    global _gemini_ready
    if _gemini_ready:
        return True
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        if os.getenv('GEMINI_DEBUG','0') == '1':
            print('[gemini] No GEMINI_API_KEY or GOOGLE_API_KEY found; skipping Gemini features.')
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # quick lightweight sanity request (model list) – optional, guarded
        if os.getenv('GEMINI_DEBUG','0') == '1':
            try:
                _ = list(genai.list_models())[:1]
                print(f"[gemini] Initialization ok. Using model: {_gemini_model_name}")
            except Exception as se:
                print(f"[gemini] Model list probe failed (non-fatal): {se}")
        _gemini_ready = True
        return True
    except Exception as e:
        print(f"[gemini] Init failed: {e}")
        return False

def gemini_transcribe_segment(audio_path: str) -> str:
    """Transcribe an audio segment using Gemini.
    Returns empty string on failure. Adds optional debug prints if GEMINI_DEBUG=1.
    """
    if not _init_gemini():
        return ""
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(_gemini_model_name)
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        # Prompt tailored for animated content and character dialogue
        prompt = (
            "You are transcribing dialogue from animated content. "
            "Output a faithful, verbatim transcript in ENGLISH ONLY. "
            "Preserve colloquial speech, interjections, and expressive sounds (e.g., uh, hmm, ah, gasp, laughter) as words; "
            "do NOT describe them in brackets. Do not include speaker names, character labels, timestamps, or annotations. "
            "Do not summarize or paraphrase; keep the original phrasing. "
            "If the speech is in another language, return an English transliteration or natural translation inline, without brackets. "
            "Return only the transcript text."
        )
        response = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_bytes},
            prompt
        ])
        text = (getattr(response, 'text', None) or "").strip()
        # If non-ASCII detected, attempt a forced-English rephrase once
        if text and any(ord(c) > 127 for c in text):
            if os.getenv('GEMINI_DEBUG','0') == '1':
                print(f"[gemini] Non-English detected, retrying English enforcement: {text}")
            retry_prompt = (
                "Provide an English-only transliteration or translation of the previous transcript. "
                "Do not include any original script characters or explanations. Only the English text."
            )
            try:
                retry_resp = model.generate_content([text, retry_prompt])
                retry_text = (getattr(retry_resp, 'text', None) or '').strip()
                if retry_text and all(ord(c) < 128 for c in retry_text):
                    text = retry_text
            except Exception as _:
                pass
        if not text and os.getenv('GEMINI_DEBUG','0') == '1':
            print(f"[gemini] Empty transcription for segment {audio_path} (response: {response})")
        return text
    except Exception as e:
        if os.getenv('GEMINI_DEBUG','0') == '1':
            print(f"[gemini] Transcription failed for {audio_path}: {e}")
        return ""

def gemini_entities_intents(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract entities and intents for animated content using Gemini.
    Output contract:
      - entities: list of {"text": str, "type": str} where type ∈ {CHARACTER, LOCATION, OBJECT, ACTION, EMOTION, SFX, CATCHPHRASE}
      - intents: list of labels ∈ {QUESTION, COMMAND, REQUEST, EXCLAMATION, HUMOR, SARCASM, THREAT, APOLOGY, THANKS, GREETING, NARRATION, TRANSITION, AFFECTION, SURPRISE}
    Robust JSON recovery + debug logging retained.
    """
    if not text or not text.strip():
        return [], []
    if not _init_gemini():
        return [], []
    try:
        import json as _json
        import google.generativeai as genai
        model = genai.GenerativeModel(_gemini_model_name)
        truncated = text[:6000]  # safety limit
        prompt = (
            "You are an NLP annotator for ANIMATED CONTENT (cartoons/anime).\n"
            "Given the transcript text, extract: \n"
            "  - Entities with types from: CHARACTER, LOCATION, OBJECT, ACTION, EMOTION, SFX, CATCHPHRASE.\n"
            "    • CHARACTER: names or clear references to specific characters (e.g., 'Tom', 'the wizard').\n"
            "    • LOCATION: places or settings (e.g., 'castle', 'forest', 'school').\n"
            "    • OBJECT: key items (e.g., 'magic wand', 'sword', 'remote').\n"
            "    • ACTION: significant actions described or implied by the dialogue (verbs or short phrases).\n"
            "    • EMOTION: explicit emotional states (e.g., 'angry', 'excited', 'scared').\n"
            "    • SFX: notable sound effects mentioned as words (e.g., 'boom', 'whoosh', 'thud', 'laughter').\n"
            "    • CATCHPHRASE: recurring, signature lines.\n"
            "  - Intents from: QUESTION, COMMAND, REQUEST, EXCLAMATION, HUMOR, SARCASM, THREAT, APOLOGY, THANKS, GREETING, NARRATION, TRANSITION, AFFECTION, SURPRISE.\n"
            "Return ONLY strict JSON: {\"entities\":[{\"text\":\"...\",\"type\":\"...\"}],\"intents\":[\"...\"]}.\n"
            "Use empty arrays if nothing found.\n"
            "Text: \"" + truncated.replace('"','\\"') + "\""
        )
        resp = model.generate_content(prompt)
        raw = getattr(resp, 'text', '') or '{}'
        # Isolate possible JSON
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            candidate = raw[start:end+1]
        else:
            candidate = '{}'
        try:
            parsed = _json.loads(candidate)
        except Exception:
            if os.getenv('GEMINI_DEBUG','0') == '1':
                print(f"[gemini] JSON parse failed, raw response: {raw}")
            return [], []
        entities = parsed.get('entities', []) if isinstance(parsed, dict) else []
        intents = parsed.get('intents', []) if isinstance(parsed, dict) else []
        if not isinstance(entities, list):
            entities = []
        if not isinstance(intents, list):
            intents = []
        return entities, intents
    except Exception as e:
        if os.getenv('GEMINI_DEBUG','0') == '1':
            print(f"[gemini] entity/intent extraction failed: {e}")
        return [], []


@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""
    duration_secs: float = 0.0  # true chunk duration (len(chunk)/sr)

    def get_formatted_text(self) -> str:
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
                continue  # skip duplicate consecutive text
            last_text = raw
            combined = f"{raw} {metadata}".strip() if raw else metadata
            texts.append(combined)
        return " ".join(texts)

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age: float = round(age*100, 2)


        age_brackets: List[tuple[float, str]] = [
            (18, "0_18"),
            (30, "18_30"),
            (45, "30_45"),
            (60, "45_60"),
            (float('inf'), "60PLUS")
        ]

        for threshold, bracket in age_brackets:
            if actual_age < threshold:
                return bracket
        return "60PLUS"

def create_output_directories(base_path: str) -> Tuple[str, str]:
    chunks_dir = os.path.join(base_path, "audio_chunks")
    results_dir = os.path.join(base_path, "results")

    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return chunks_dir, results_dir

def process_large_audio(
    audio_path: str,
    chunk_duration: float = 30.0,
    output_base_dir: str = "output",
    overlap_duration: float = 10.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    chunks_dir = os.path.abspath(os.path.join(output_base_dir, "audio_chunks"))
    results_dir = os.path.abspath(os.path.join(output_base_dir, "results"))

    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    processed_audio_path = convert_mp4_to_wav(audio_path, preferred_out_dir=os.path.join(output_base_dir, "audio_converted"))
    print(f"[process_large_audio] Using audio: {processed_audio_path}")
    # External subtitle ingestion removed: proceed with ASR-only pipeline
    # Precompute for quick lookup
    # We'll do a simple linear scan (subtitle count usually small); could index if large
    try:
        # Replace librosa.load with soundfile
        import soundfile as sf
        signal, sr = sf.read(processed_audio_path)
        # Convert to mono if necessary
        if isinstance(signal, np.ndarray) and signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        # If the sample rate is not 16000, resample it
        if sr != 16000:
            import resampy
            signal = resampy.resample(signal, sr, 16000)
            sr = 16000
    except Exception as e:
        print(f"Error loading audio file {processed_audio_path}: {str(e)}")
        return [], []

    all_data = []
    chunk_data: Dict[str, ChunkData] = defaultdict(ChunkData)
    # Overlapped chunking: 30s chunks with 10s overlap => stride = 20s
    chunk_size = int(chunk_duration * sr)
    overlap_size = int(max(0.0, overlap_duration) * sr)
    stride = max(1, chunk_size - overlap_size)

    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Persistent speaker id mapping across ALL chunks (so speaker_0 remains same voice if pyannote stable)
    global_speaker_id_map: Dict[str, int] = {}
    global_next_spk_id = 0

    # Speaker constraint parameters (user tunable via env)
    min_speakers_env = os.getenv('MIN_SPEAKERS')
    max_speakers_env = os.getenv('MAX_SPEAKERS')
    try:
        min_speakers = int(min_speakers_env) if min_speakers_env else 2  # default assume dialogue
    except ValueError:
        min_speakers = 2
    try:
        max_speakers = int(max_speakers_env) if max_speakers_env else None
    except ValueError:
        max_speakers = None

    if max_speakers is not None and max_speakers < min_speakers:
        # sanity swap
        max_speakers = None

    try:
        # Iterate with overlap using stride
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

            # Record true chunk duration for JSONL output
            true_chunk_duration = float(len(chunk)) / float(sr)
            chunk_data[chunk_filename].duration_secs = true_chunk_duration
            chunk_data[chunk_filename].filepath = os.path.abspath(chunk_path)

            try:
                if pipeline is None:
                    print("[process_large_audio] Skipping diarization (pipeline unavailable).")
                    speaker_changes = []
                else:
                    diarization_kwargs = {'audio': chunk_path}
                    # pyannote Pipeline expects positional path OR dict; for constraints we call with path string
                    # Use path form when supplying min/max to leverage built-in segmentation clustering
                    if max_speakers is not None:
                        diarization = pipeline(
                            chunk_path,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                    else:
                        diarization = pipeline(
                            chunk_path,
                            min_speakers=min_speakers
                        )
                    speaker_changes = [
                        (turn.start, turn.end, speaker)
                        for turn, _, speaker in diarization.itertracks(yield_label=True)
                    ]
                # Assign incremental speaker ids mapping original labels to numeric order of appearance
                # Use global maps so a speaker label repeated in later chunks keeps same numeric id
                speaker_id_map = global_speaker_id_map
                next_spk_id = global_next_spk_id
                # Whisper chunk transcription (once per chunk)
                whisper_words = transcribe_chunk_whisper(chunk_path)

                for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                    if speaker not in speaker_id_map:
                        speaker_id_map[speaker] = next_spk_id
                        next_spk_id += 1
                    numeric_speaker = speaker_id_map[speaker]
                    # Update global next id after potential assignment
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

                    temp_segment_path = os.path.join(
                        chunks_dir,
                        f"temp_segment_{chunk_index}_{speaker_idx}.wav"
                    )
                    sf.write(temp_segment_path, speaker_segment, sr)

                    try:
                        speaker_segment_audio, _ = librosa.load(temp_segment_path, sr=16000)
                        emotion = extract_emotion(speaker_segment_audio)

                        # Build whisper text for this speaker segment from word midpoints
                        seg_start, seg_end = start_time, end_time
                        seg_words = []
                        for w in whisper_words:
                            w_mid = (w['start'] + w['end']) / 2.0
                            if w_mid >= seg_start and w_mid <= seg_end:
                                seg_words.append(w['text'])
                        whisper_text = ' '.join(seg_words).strip()

                        # Gemini & Deepgram transcription (always call when keys set)
                        gemini_text = gemini_transcribe_segment(temp_segment_path)
                        deepgram_text = deepgram_transcribe_segment(temp_segment_path)

                        # Entities & intents: prefer Gemini text, fall back to Whisper
                        base_text_for_ner = gemini_text if gemini_text else whisper_text
                        entities, intents = gemini_entities_intents(base_text_for_ner)

                        # Backward compatibility: set text to whisper_text
                        segment = AudioSegment(
                            start_time=start_time,
                            end_time=end_time,
                            speaker=f"speaker_{numeric_speaker}",  # legacy field
                            age=float(age),
                            gender=int(gender),
                            text=whisper_text,
                            whisper_text=whisper_text,
                            gemini_text=gemini_text,
                            deepgram_text=deepgram_text,
                            emotion=emotion,
                            chunk_filename=chunk_filename,
                            entities=entities,
                            intents=intents
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
                            'audio_file_path': os.path.abspath(chunk_path)
                        }
                        
                        all_data.append(segment_data)

                        chunk_data[chunk_filename].segments.append(segment)

                    except Exception as e:
                        print(f"Error processing segment in chunk {chunk_filename}: {str(e)}")
                        continue

                    finally:
                        if os.path.exists(temp_segment_path):
                            os.remove(temp_segment_path)

            except Exception as e:
                print(f"Error processing chunk {chunk_filename}: {str(e)}")
                continue

            finally:
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # increment chunk counter after processing
            chunk_counter += 1

        if all_data:
            # Build speaker change tag sequence for output
            last_speaker = None
            speaker_change_counter = {}
            seq = []
            # Aggregation for speaker CSV
            speaker_stats: Dict[int, Dict[str, Any]] = {}
            for seg in all_data:
                spk = seg['speaker']
                if spk != last_speaker:
                    # increment counter per new speaker appearance order
                    if spk not in speaker_change_counter:
                        speaker_change_counter[spk] = len(speaker_change_counter)
                    change_id = speaker_change_counter[spk]
                    seg['speaker_change_tag'] = f"speaker_change_{change_id}"
                    last_speaker = spk
                else:
                    seg['speaker_change_tag'] = f"speaker_change_{speaker_change_counter[spk]}"
                # Collect stats
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
                    # Keep earliest start
                    if seg['start_time'] < entry['first_start_time']:
                        entry['first_start_time'] = seg['start_time']
                seq.append(seg)

            # Save consolidated JSON
            json_path = os.path.join(results_dir, f"{base_filename}_processed_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(seq, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON to: {json_path}")

            # Write speaker mapping CSV
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
                    print(f"[speaker-csv] Failed writing speaker mapping CSV: {e}")

            # Build JSONL of audio_filepath, text, duration
            jsonl_path = os.path.join(results_dir, f"{base_filename}_audio_text_pairs.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as jf:
                for chunk_name, cdata in chunk_data.items():
                        # Use true chunk duration (not derived from diarization overlap)
                    if cdata.segments:
                        duration = float(cdata.duration_secs) if getattr(cdata, 'duration_secs', 0.0) else 0.0
                        whisper_join = ' '.join([seg.whisper_text for seg in cdata.segments if seg.whisper_text])
                        gemini_join = ' '.join([seg.gemini_text for seg in cdata.segments if seg.gemini_text])
                        deepgram_join = ' '.join([seg.deepgram_text for seg in cdata.segments if getattr(seg, 'deepgram_text', '')])
                        tagged_text = cdata.get_formatted_text()
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

            # Skip previous chunk_texts generation since format changed
            return all_data, []

        print("[process_large_audio] No segments extracted (empty diarization or subtitle overlap); nothing saved.")
        return [], []

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {str(e)}")
        return [], []


# WebVTT parsing removed from active use; external subtitle ingestion is no longer supported.


if __name__ == "__main__":
    # Allow passing a directory via CLI; default remains legacy path.
    if len(sys.argv) > 1:
        download_dir = sys.argv[1]
    else:
        download_dir = "/external4/datasets/strem2action-audio/animated_characters"
    output_dir = download_dir  # default: save results alongside each media
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing root: {download_dir}")
    print(f"Outputs saved alongside each media folder")

    # As requested, focus on MP4 files present in the directory
    target_extension = '.wav'

    # No VTT matching is attempted; processing is ASR-only.

    # Recursively find all .mp4 files within the given directory
    mp4_files: List[str] = []
    for dirpath, _, filenames in os.walk(download_dir):
        for fn in sorted(filenames):
            if fn.lower().endswith(target_extension):
                mp4_files.append(os.path.join(dirpath, fn))

    if not mp4_files:
        print("[warn] No .mp4 files found recursively. Falling back to flat scan for common audio/video extensions.")
        entries = sorted(os.listdir(download_dir))
        fallback_exts = ['.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg', '.webm', '.mkv']
        for filename in entries:
            audio_path = os.path.join(download_dir, filename)
            if os.path.isfile(audio_path) and any(filename.lower().endswith(ext) for ext in fallback_exts):
                mp4_files.append(audio_path)

    if not mp4_files:
        print("[exit] No media files found to process.")
        sys.exit(0)

    total = len(mp4_files)
    print(f"Discovered {total} media files to process.")

    for idx, audio_path in enumerate(mp4_files, start=1):
        parent_dir = os.path.dirname(audio_path)
        filename = os.path.basename(audio_path)
        print(f"\n[{idx}/{total}] Processing audio file: {filename}")
        print(f"Path: {audio_path}")
        df, _ = process_large_audio(audio_path, output_base_dir=parent_dir)
        print(f"Finished {filename}. Segments: {len(df)}")
        print("Output JSON saved alongside in results directory.")