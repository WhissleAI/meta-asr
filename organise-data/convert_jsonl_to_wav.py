#!/usr/bin/env python3
"""
Convert audio files referenced in a JSONL to WAV and write a new JSONL with updated paths.

Each line of the input JSONL should be a JSON object containing at least:
  - "audio_filepath": string, path to the source audio file
  - "text": string, transcript or text (preserved)
  - "duration": number (optional; if present, may be replaced by probed duration)

Example input line:
{"audio_filepath": "/external4/datasets/bucket_data/wellness/overlap/trimmed.mp3", "text": "hello world", "duration": 30.0}

Outputs:
  - WAV files in the specified output directory (optionally preserving a relative structure)
  - A new JSONL file with updated "audio_filepath" pointing to the new WAVs and duration updated via ffprobe (unless --keep-duration is set)

Requirements:
  - ffmpeg and ffprobe available on PATH, or provide --ffmpeg and --ffprobe paths.
  - The repository may already include a static ffmpeg at: ./ffmpeg-7.0.2-amd64-static/ffmpeg

Usage:
  python convert_jsonl_to_wav.py \
    --input /path/to/input.jsonl \
    --out-jsonl /path/to/output.jsonl \
    --out-dir /path/to/wav_dir \
    --sr 16000 --channels 1 --jobs 4

Optional:
  --prefix-strip /external4/datasets    # strip this prefix from source paths to preserve relative structure under out-dir
  --overwrite                           # overwrite existing wavs
  --keep-duration                       # do not probe wav duration; keep existing duration if present
  --reencode-wav                        # if source is already .wav, still re-encode to target format (otherwise copy)
"""

from __future__ import annotations
from __future__ import annotations
import concurrent.futures as futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


# ======================
# USER CONFIG - EDIT ME
# ======================
# Provide paths below. Set these to your files/folders before running the script.
INPUT_JSONL: str = "/external4/datasets/bucket_data/wellness/wellness_fitness_annotated.jsonl"
OUTPUT_JSONL: str = "/external3/databases/wellness-wav/wellness_wav.jsonl"
OUTPUT_DIR: str = "/external3/databases/wellness-wav"

# Audio parameters
SAMPLE_RATE: int = 16000
CHANNELS: int = 1

# Parallel workers (set to 1 to disable parallelism)
JOBS: int = min(4, (os.cpu_count() or 2))

# Behavior flags
OVERWRITE: bool = False          # Overwrite existing WAV files
KEEP_DURATION: bool = False      # Keep original duration from JSONL; if False, probe new duration
REENCODE_WAV: bool = False       # If source is already .wav, re-encode (True) or just copy (False)

# If you want to preserve source subdirectories under OUTPUT_DIR, set a prefix to strip.
# Example: if sources are under /external4/datasets, set PREFIX_STRIP = "/external4/datasets"
PREFIX_STRIP: Optional[str] = None

# Optional explicit paths to ffmpeg/ffprobe. Leave as None to auto-detect (including bundled static binaries in repo).
FFMPEG_PATH: Optional[str] = None
FFPROBE_PATH: Optional[str] = None


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def which(program: str) -> Optional[str]:
    """Return full path to an executable if found in PATH, else None."""
    path = shutil.which(program)
    return path


def find_ffmpeg_candidates(script_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Try to locate ffmpeg and ffprobe. Prefer repo-local static binaries if present."""
    candidates_ffmpeg = [
        script_dir / "ffmpeg-7.0.2-amd64-static/ffmpeg",
        Path("ffmpeg-7.0.2-amd64-static/ffmpeg"),
    ]
    candidates_ffprobe = [
        script_dir / "ffmpeg-7.0.2-amd64-static/ffprobe",
        Path("ffmpeg-7.0.2-amd64-static/ffprobe"),
    ]

    ffmpeg = None
    for c in candidates_ffmpeg:
        if c.exists() and os.access(c, os.X_OK):
            ffmpeg = str(c.resolve())
            break
    if ffmpeg is None:
        ffmpeg = which("ffmpeg")

    ffprobe = None
    for c in candidates_ffprobe:
        if c.exists() and os.access(c, os.X_OK):
            ffprobe = str(c.resolve())
            break
    if ffprobe is None:
        ffprobe = which("ffprobe")

    return ffmpeg, ffprobe


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def derive_out_path(src: Path, out_dir: Path, prefix_strip: Optional[Path]) -> Path:
    """Compute destination WAV path given a source path.

    If prefix_strip is provided and src is under it, preserve the relative path under out_dir.
    Otherwise, place the file at out_dir/<stem>.wav.
    """
    stem = src.stem
    if prefix_strip:
        try:
            rel = src.relative_to(prefix_strip)
            rel = rel.with_suffix(".wav")
            return (out_dir / rel).resolve()
        except ValueError:
            pass
    return (out_dir / f"{stem}.wav").resolve()


def convert_to_wav(src: Path, dst: Path, ffmpeg: str, sr: int, channels: int, overwrite: bool) -> Tuple[bool, str]:
    ensure_parent(dst)
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", str(channels),
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = run_cmd(cmd)
    ok = proc.returncode == 0
    if not ok and dst.exists():
        # ffmpeg -n will return 1 if file exists; treat as success if dst exists
        ok = True
    return ok, proc.stderr.strip() or proc.stdout.strip()


def copy_or_reencode_if_wav(src: Path, dst: Path, ffmpeg: str, sr: int, channels: int, overwrite: bool, reencode: bool) -> Tuple[bool, str]:
    if reencode:
        return convert_to_wav(src, dst, ffmpeg, sr, channels, overwrite)
    # If not reencoding, perform a copy when formats match desired PCM 16 LE, otherwise reencode.
    # We cannot cheaply verify encoding without ffprobe; for simplicity, copy if .wav extension and not forcing reencode.
    if overwrite or not dst.exists():
        ensure_parent(dst)
        try:
            shutil.copy2(src, dst)
            return True, "copied"
        except Exception as e:
            return False, str(e)
    return True, "exists"


def probe_duration_seconds(path: Path, ffprobe: Optional[str]) -> Optional[float]:
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except Exception:
        return None


def process_line(
    line: str,
    out_dir: Path,
    prefix_strip: Optional[Path],
    ffmpeg: str,
    ffprobe: Optional[str],
    sr: int,
    channels: int,
    overwrite: bool,
    keep_duration: bool,
    reencode_wav: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (new_json_line, error_message)."""
    line = line.strip()
    if not line:
        return None, None
    try:
        obj: Dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

    src_path = obj.get("audio_filepath")
    if not src_path:
        return None, "Missing 'audio_filepath'"
    src = Path(src_path)
    if not src.exists():
        return None, f"Source not found: {src}"

    dst = derive_out_path(src, out_dir, prefix_strip)

    if src.suffix.lower() == ".wav":
        ok, msg = copy_or_reencode_if_wav(src, dst, ffmpeg, sr, channels, overwrite, reencode_wav)
    else:
        ok, msg = convert_to_wav(src, dst, ffmpeg, sr, channels, overwrite)
    if not ok:
        return None, f"Conversion failed for {src}: {msg}"

    # Build new JSON line
    new_obj = dict(obj)
    new_obj["audio_filepath"] = str(dst)
    if not keep_duration:
        dur = probe_duration_seconds(dst, ffprobe)
        if dur is not None:
            new_obj["duration"] = round(dur, 6)
    try:
        return json.dumps(new_obj, ensure_ascii=False), None
    except Exception as e:
        return None, f"JSON encode error: {e}"


def main():
    # Resolve configuration
    in_path = Path(INPUT_JSONL).resolve()
    out_jsonl_path = Path(OUTPUT_JSONL).resolve()
    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix_strip = Path(PREFIX_STRIP).resolve() if PREFIX_STRIP else None

    script_dir = Path(__file__).resolve().parent
    ffmpeg = FFMPEG_PATH
    ffprobe = FFPROBE_PATH
    if not ffmpeg or not ffprobe:
        ff_d, fp_d = find_ffmpeg_candidates(script_dir)
        ffmpeg = ffmpeg or ff_d
        ffprobe = ffprobe or fp_d

    if not ffmpeg:
        eprint("ffmpeg not found. Install ffmpeg or pass --ffmpeg path.")
        sys.exit(2)
    if not ffprobe and not KEEP_DURATION:
        eprint("ffprobe not found. Duration will not be updated; proceeding with keep-duration behavior.")
        keep_duration_flag = True
    else:
        keep_duration_flag = KEEP_DURATION

    eprint(f"Using ffmpeg: {ffmpeg}")
    eprint(f"Using ffprobe: {ffprobe or 'N/A'}")
    eprint(f"Reading: {in_path}")
    eprint(f"Writing WAVs to: {out_dir}")
    eprint(f"Writing JSONL to: {out_jsonl_path}")

    total = 0
    ok_count = 0
    errors = 0

    # Read all lines first for simpler parallel processing
    try:
        with in_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        eprint(f"Input JSONL not found: {in_path}")
        sys.exit(1)

    # Process in parallel
    def worker(line: str) -> Tuple[Optional[str], Optional[str]]:
        return process_line(
            line=line,
            out_dir=out_dir,
            prefix_strip=prefix_strip,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        sr=SAMPLE_RATE,
        channels=CHANNELS,
        overwrite=OVERWRITE,
        keep_duration=keep_duration_flag,
        reencode_wav=REENCODE_WAV,
        )

    results: list[Tuple[Optional[str], Optional[str]]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, JOBS)) as ex:
        for res in ex.map(worker, lines):
            results.append(res)

    # Write output JSONL; collect and report errors
    with out_jsonl_path.open("w", encoding="utf-8") as out_f:
        for new_line, err in results:
            total += 1
            if new_line:
                out_f.write(new_line)
                out_f.write("\n")
                ok_count += 1
            if err:
                errors += 1
                eprint(err)

    eprint(f"Processed lines: {total}")
    eprint(f"Succeeded: {ok_count}")
    eprint(f"Errors: {errors}")

    if ok_count == 0:
        eprint("No lines succeeded. Exiting with error.")
        sys.exit(3)


if __name__ == "__main__":
    main()
