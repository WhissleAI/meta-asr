# applications/transcription.py
from pathlib import Path
import asyncio
from typing import Tuple, Optional
from config import logger # Removed WHISSLE_AUTH_TOKEN
from models import GEMINI_AVAILABLE, WHISSLE_AVAILABLE, DEEPGRAM_AVAILABLE # Use *_AVAILABLE flags
# Removed: from models import DEEPGRAM_CLIENT
from session_store import get_user_api_key
import google.generativeai as genai
from deepgram import PrerecordedOptions, DeepgramClient as DeepgramSDKClient # Ensure DeepgramSDKClient is used

def get_mime_type(audio_file_path: Path) -> str:
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_whissle_single(audio_path: Path, user_id: str, model_name="en-US-0.6b") -> Tuple[Optional[str], Optional[str]]:
    if not WHISSLE_AVAILABLE:
        return None, "Whissle SDK is not available."
    
    whissle_auth_token = get_user_api_key(user_id, "whissle")
    if not whissle_auth_token:
        return None, "Whissle API key not found or session expired for user."

    try:
        from whissle import WhissleClient # Keep local import if preferred
        whissle_client = WhissleClient(auth_token=whissle_auth_token)
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

async def transcribe_with_gemini_single(audio_path: Path, user_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not GEMINI_AVAILABLE:
        return None, "Gemini (google.generativeai) library is not available."

    gemini_api_key = get_user_api_key(user_id, "gemini")
    if not gemini_api_key:
        return None, "Gemini API key not found or session expired for user."

    # Critical: genai.configure is a global setting.
    # This can cause issues in concurrent environments if not handled carefully.
    # For true per-request isolation, the Gemini SDK would need to support key-per-client or key-per-request.
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to configure Gemini with user API key: {e}")
        return None, "Failed to configure Gemini API with user key."

    model_name = "models/gemini-1.5-flash"
    uploaded_file = None # Ensure uploaded_file is defined for the finally block
    # ... rest of the function remains largely the same, ensure genai calls use the configured key
    try:
        model = genai.GenerativeModel(model_name)
        mime_type = get_mime_type(audio_path)
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2) # Consider making sleep duration configurable or dynamic
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            # No need to await asyncio.to_thread for genai.delete_file if it's synchronous
            # However, if it's I/O bound and there's a sync version, direct call is fine.
            # Assuming genai.delete_file can be awaited if it's an async operation or wrapped if sync.
            try: 
                # If genai.delete_file is sync, wrap it: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
                # If it's already async: await genai.delete_file(name=uploaded_file.name)
                # For now, assuming it's okay to call directly if it's a quick cleanup.
                # Let's assume it's okay to call directly for cleanup, or it's an async function.
                # If it's a sync blocking call, it should be wrapped.
                # For safety, wrapping:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini resource {uploaded_file.name}: {del_e}")
            return None, error_msg
        
        prompt = "Transcribe the audio accurately. Provide only the spoken text. If no speech is detected, return an empty string."
        # The model.generate_content call will use the globally configured API key
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file], request_options={'timeout': 300})
        
        try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
        except Exception as del_e: logger.warning(f"Could not delete Gemini resource {uploaded_file.name} after transcription: {del_e}")

        if response.candidates:
            try:
                if hasattr(response, 'text') and response.text is not None: transcription = response.text.strip()
                elif response.candidates[0].content.parts: transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else: return None, "Gemini response candidate found, but no text content."
                return transcription if transcription else "", None # Return empty string if transcription is empty
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e: # More specific exception handling
                logger.error(f"Error parsing Gemini transcription response for {audio_path.name}: {resp_e}", exc_info=True)
                return None, f"Error parsing Gemini transcription response: {resp_e}"
        else:
            error_message = f"No candidates from Gemini transcription for {audio_path.name}."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                # Check if block_reason_message exists, otherwise convert feedback to string
                feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                error_message += f" Feedback: {feedback}"
            return None, error_message
    except Exception as e:
        logger.error(f"Gemini API/SDK error during transcription for {audio_path.name}: {e}", exc_info=True)
        if uploaded_file and hasattr(uploaded_file, 'name'): # Check if uploaded_file exists and has a name
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error: {type(e).__name__}: {e}"
    # finally: # Optional: Reset global genai configuration if necessary, though this is tricky
        # genai.configure(api_key=None) # Or to a system key if available. This is not ideal.


async def transcribe_with_deepgram_single(audio_path: Path, user_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not DEEPGRAM_AVAILABLE:
        return None, "Deepgram SDK is not available."

    deepgram_api_key = get_user_api_key(user_id, "deepgram")
    if not deepgram_api_key:
        return None, "Deepgram API key not found or session expired for user."

    try:
        # Initialize Deepgram client with the user-specific key
        deepgram_client = DeepgramSDKClient(api_key=deepgram_api_key)
        
        with open(audio_path, "rb") as audio_file_obj: # Renamed to avoid conflict
            buffer_data = audio_file_obj.read()
        
        source = {"buffer": buffer_data, "mimetype": get_mime_type(audio_path)}
        options = PrerecordedOptions(
            model="nova-2", smart_format=True, diarize=False, language="en" # Ensure language is set if needed
        )
        
        # The Deepgram SDK's transcribe_file method might be synchronous or asynchronous.
        # If synchronous, it should be wrapped with asyncio.to_thread.
        # Assuming listen.prerecorded.v("1").transcribe_file is a synchronous blocking call.
        response = await asyncio.to_thread(
            deepgram_client.listen.prerecorded.v("1").transcribe_file,
            source,
            options
        )
        
        # Robust transcript extraction
        if response and response.results and response.results.channels and \
           response.results.channels[0].alternatives and \
           response.results.channels[0].alternatives[0].transcript:
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip() if transcript else "", None # Return empty string if transcript is empty but present
        else:
            logger.warning(f"Deepgram transcription for {audio_path.name} returned unexpected structure or empty transcript. Response: {response}")
            return None, "Deepgram returned no transcript or unexpected response structure."

    except Exception as e:
        logger.error(f"Deepgram transcription error for {audio_path.name}: {e}", exc_info=True)
        return None, f"Deepgram SDK/API error: {type(e).__name__}: {e}"