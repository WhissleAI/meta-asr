# applications/audio_utils.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import resampy
from config import AUDIO_EXTENSIONS, TARGET_SAMPLE_RATE, logger
from typing import Tuple, Optional, List
from fastapi import HTTPException
from pydub import AudioSegment # Add pydub import

def validate_paths(dir_path_str: str, output_path_str: str) -> Tuple[Path, Path]:
    dir_path = Path(dir_path_str)
    output_jsonl_path = Path(output_path_str)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {dir_path_str}")
    if not output_jsonl_path.parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Output directory does not exist: {output_jsonl_path.parent}")
    return dir_path, output_jsonl_path

def discover_audio_files(directory_path: Path) -> List[Path]:
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory_path.glob(f"*{ext}"))
        audio_files.extend(directory_path.glob(f"*{ext.upper()}"))
    audio_files.sort()
    logger.info(f"Discovered {len(audio_files)} audio files in {directory_path}")
    return audio_files

def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=False)
        return None, None, f"Failed to load audio: {type(e).__name__}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as le:
            logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=False)
            return None

def trim_audio(audio_path: Path, segment_length_ms: int, output_dir: Path) -> List[Path]:
    """
    Trims an audio file into segments of a specified length.

    Args:
        audio_path: Path to the input audio file.
        segment_length_ms: Desired length of each segment in milliseconds.
        output_dir: Directory to save the trimmed audio segments.

    Returns:
        A list of paths to the trimmed audio segments.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        trimmed_files = []
        # Ensure segment_length_ms is an integer for slicing
        step = int(segment_length_ms)
        for i, chunk in enumerate(audio[::step]):
            trimmed_file_path = output_dir / f"{audio_path.stem}_segment_{i}{audio_path.suffix}"
            chunk.export(trimmed_file_path, format=audio_path.suffix[1:])
            trimmed_files.append(trimmed_file_path)
        return trimmed_files
    except Exception as e:
        logger.error(f"Error trimming audio file {audio_path}: {e}", exc_info=True)
        return []