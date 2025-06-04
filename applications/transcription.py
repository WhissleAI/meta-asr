# applications/transcription.py
from pathlib import Path
import asyncio
from typing import Tuple, Optional
from config import logger, WHISSLE_AUTH_TOKEN
from models import GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED, DEEPGRAM_CLIENT
import google.generativeai as genai
from deepgram import PrerecordedOptions

def get_mime_type(audio_file_path: Path) -> str:
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> Tuple[Optional[str], Optional[str]]:
    if not WHISSLE_CONFIGURED:
        return None, "Whissle is not configured."
    try:
        from whissle import WhissleClient
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        if isinstance(response, dict):
            text = response.get('text')
            if text is not None: return text.strip(), None
            else:
                error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                return None, f"Whissle API error: {error_detail}"
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str):
            return response.transcript.strip(), None
        else: return None, f"Unexpected Whissle response format: {type(response)}"
    except Exception as e:
        return None, f"Whissle SDK error: {type(e).__name__}: {e}"

async def transcribe_with_gemini_single(audio_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    model_name = "models/gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(model_name)
        mime_type = get_mime_type(audio_path)
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini resource {uploaded_file.name}: {del_e}")
            return None, error_msg
        prompt = "Transcribe the audio accurately. Provide only the spoken text. If no speech is detected, return an empty string."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})
        try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
        except Exception as del_e: logger.warning(f"Could not delete Gemini resource {uploaded_file.name} after transcription: {del_e}")
        if response.candidates:
            try:
                if hasattr(response, 'text') and response.text is not None: transcription = response.text.strip()
                elif response.candidates[0].content.parts: transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else: return None, "Gemini response candidate found, but no text content."
                return transcription if transcription else "", None
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                return None, f"Error parsing Gemini transcription response: {resp_e}"
        else:
            error_message = f"No candidates from Gemini transcription for {audio_path.name}."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                error_message += f" Feedback: {feedback}"
            return None, error_message
    except Exception as e:
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error during transcription: {type(e).__name__}: {e}"

async def transcribe_with_deepgram_single(audio_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not DEEPGRAM_CONFIGURED:
        return None, "Deepgram not configured."
    try:
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()
        options = PrerecordedOptions(
            model="nova-2", smart_format=True, diarize=False, language="en"
        )
        response = await asyncio.to_thread(
            DEEPGRAM_CLIENT.listen.prerecorded.v("1").transcribe_file,
            {"buffer": buffer_data, "mimetype": get_mime_type(audio_path)},
            options
        )
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        if not transcript:
            return "", None
        return transcript, None
    except Exception as e:
        logger.error(f"Deepgram transcription error for {audio_path.name}: {e}")
        return None, f"Deepgram error: {str(e)}"