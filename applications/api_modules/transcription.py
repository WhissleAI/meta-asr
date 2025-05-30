"""
Transcription services for different providers
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple
from .config import (
    is_whissle_configured, is_gemini_configured, is_deepgram_configured, # Use getter functions
    WHISSLE_AUTH_TOKEN, DEEPGRAM_API_KEY # These should be fine as they are from os.getenv in config
)

logger = logging.getLogger(__name__)

# Import external libraries conditionally
try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
except ImportError:
    logger.warning("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass  # Dummy class

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    logger.warning("Warning: Google GenerativeAI not available")
    GENAI_AVAILABLE = False

# Remove global DEEPGRAM_CLIENT initialization from here
try:
    from deepgram import DeepgramClient, PrerecordedOptions
    DEEPGRAM_AVAILABLE = True
except ImportError:
    logger.warning("Warning: Deepgram SDK not available")
    DEEPGRAM_AVAILABLE = False
    class DeepgramClient: pass # Dummy
    class PrerecordedOptions: pass # Dummy

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio using Whissle"""
    if not is_whissle_configured(): # Use getter
        return None, "Whissle is not configured."
    if not WHISSLE_AVAILABLE:
        return None, "Whissle SDK not available."
    try:
        # Initialize client inside the function
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        if isinstance(response, dict):
            text = response.get('text')
            if text is not None:
                return text.strip(), None
            else:
                error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                return None, f"Whissle API error: {error_detail}"
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str):
            return response.transcript.strip(), None
        else:
            return None, f"Unexpected Whissle response format: {type(response)}"
    except Exception as e:
        return None, f"Whissle SDK error: {type(e).__name__}: {e}"

def get_mime_type(audio_file_path: Path) -> str:
    """Get MIME type for audio file"""
    ext = audio_file_path.suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4"
    }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_gemini_single(audio_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio using Gemini"""
    if not is_gemini_configured(): # Use getter
        return None, "Gemini API is not configured."
    if not GENAI_AVAILABLE:
        return None, "Google GenerativeAI SDK not available."
    
    model_name = "models/gemini-1.5-flash"  # Or "models/gemini-1.5-pro" for higher quality potential
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return None, f"Error initializing Gemini model: {e}"

    mime_type = get_mime_type(audio_path)
    uploaded_file = None
    try:
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
        
        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            try:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e:
                logger.warning(f"Could not delete failed Gemini resource {uploaded_file.name}: {del_e}")
            return None, error_msg

        prompt = "Transcribe the audio accurately. Provide only the spoken text. If no speech is detected, return an empty string."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})
        
        # Ensure file is deleted after response
        try:
            await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            uploaded_file = None
        except Exception as del_e:
            logger.warning(f"Could not delete Gemini resource {uploaded_file.name} after transcription: {del_e}")

        if response.candidates:
            try:
                if hasattr(response, 'text') and response.text is not None:
                    transcription = response.text.strip()
                elif response.candidates[0].content.parts:
                    transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else:
                    return None, "Gemini response candidate found, but no text content."
                return transcription if transcription else "", None  # Return empty string for no speech
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
            try:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e:
                logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error during transcription: {type(e).__name__}: {e}"
    finally:  # Redundant with above but good practice for ensuring cleanup
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e:
                logger.warning(f"Could not delete Gemini file {uploaded_file.name} in finally: {del_e}")

async def transcribe_with_deepgram_single(audio_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio using Deepgram"""
    if not is_deepgram_configured(): # Use getter
        return None, "Deepgram not configured."
    if not DEEPGRAM_AVAILABLE:
        return None, "Deepgram SDK not available."
    try:
        # Initialize client inside the function
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=False,
            language="en"
        )
        response = await asyncio.to_thread(
            deepgram_client.listen.prerecorded.v("1").transcribe_file,
            {"buffer": buffer_data, "mimetype": get_mime_type(audio_path)},
            options
        )
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        if not transcript:
            return "", None  # Return empty string for no speech, no error
        return transcript, None
    except Exception as e:
        logger.error(f"Deepgram transcription error for {audio_path.name}: {e}")
        return None, f"Deepgram error: {str(e)}"
