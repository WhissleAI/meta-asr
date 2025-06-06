# applications/routes.py
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse
from pathlib import Path
import gc
import torch
import asyncio
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from config import (
    ProcessRequest, ProcessResponse,
    TranscriptionJsonlRecord, AnnotatedJsonlRecord,
    logger, device,
    ModelChoice,
    TARGET_SAMPLE_RATE,
    BioAnnotation,
    InitSessionRequest # Added InitSessionRequest
)
from models import (
    GEMINI_AVAILABLE, WHISSLE_AVAILABLE, DEEPGRAM_AVAILABLE, # Updated to _AVAILABLE flags
    age_gender_model, age_gender_processor,
    emotion_model, emotion_feature_extractor
    # Removed GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED as they are replaced by session checks
)
from audio_utils import validate_paths, discover_audio_files, load_audio, get_audio_duration
from transcription import transcribe_with_whissle_single, transcribe_with_gemini_single, transcribe_with_deepgram_single
from annotation import annotate_text_structured_with_gemini
from session_store import init_user_session, is_user_session_valid, get_user_api_key # Added session_store imports

router = APIRouter()

@router.post("/init_session/", summary="Initialize or Update User API Key Session", status_code=200)
async def init_session_endpoint(session_request: InitSessionRequest):
    try:
        init_user_session(session_request.user_id, session_request.api_keys)
        return {"message": f"Session initialized/updated for user {session_request.user_id}"}
    except Exception as e:
        logger.error(f"Error initializing session for user {session_request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")

def predict_age_gender(audio_data, sampling_rate) -> Tuple[Optional[float], Optional[int], Optional[str]]:
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

def predict_emotion(audio_data, sampling_rate) -> Tuple[Optional[str], Optional[str]]:
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

@router.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = process_request.model_choice
    provider_name = model_choice.value

    # Check service availability and user key
    service_available = False
    if model_choice == ModelChoice.whissle:
        service_available = WHISSLE_AVAILABLE
    elif model_choice == ModelChoice.gemini:
        service_available = GEMINI_AVAILABLE
    elif model_choice == ModelChoice.deepgram:
        service_available = DEEPGRAM_AVAILABLE

    if not service_available:
        raise HTTPException(status_code=400, detail=f"{provider_name.capitalize()} SDK is not available on the server.")

    if not get_user_api_key(user_id, provider_name):
        raise HTTPException(status_code=400, detail=f"API key for {provider_name.capitalize()} not found for user or session expired.")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: Optional[str] = None
                transcription_text: Optional[str] = None
                duration: Optional[float] = None
                logger.info(f"--- Processing {audio_file.name} (Transcription Only for user {user_id}) ---")
                processed_files_count += 1
                try:
                    duration = get_audio_duration(audio_file)
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file, user_id)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file, user_id)
                    elif model_choice == ModelChoice.deepgram:
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file, user_id)
                    else:
                        transcription_error = "Invalid model choice."

                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None: # Explicitly check for None after successful call
                        file_error = "Transcription returned None without an error."

                except Exception as e:
                    logger.error(f"Unexpected error processing {audio_file.name} for user {user_id}: {e}", exc_info=True)
                    file_error = f"Unexpected error: {type(e).__name__}: {e}"
                
                record = TranscriptionJsonlRecord(
                    audio_filepath=str(audio_file.resolve()), 
                    text=transcription_text, 
                    duration=duration, 
                    model_used_for_transcription=model_choice.value, 
                    error=file_error
                )
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    file_error = (file_error + "; " if file_error else "") + f"JSONL write error: {write_e}"
                
                if file_error:
                    error_count += 1
                else:
                    saved_records_count += 1
                gc.collect()
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")
    
    msg = f"Processed {processed_files_count}/{len(audio_files)}. Saved: {saved_records_count}. Errors: {error_count}."
    return ProcessResponse(message=msg, output_file=str(output_jsonl_path), processed_files=processed_files_count, saved_records=saved_records_count, errors=error_count)

@router.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = process_request.model_choice
    transcription_provider_name = model_choice.value

    # Check transcription model availability and user key
    transcription_service_available = False
    if model_choice == ModelChoice.whissle:
        transcription_service_available = WHISSLE_AVAILABLE
    elif model_choice == ModelChoice.gemini:
        transcription_service_available = GEMINI_AVAILABLE
    elif model_choice == ModelChoice.deepgram:
        transcription_service_available = DEEPGRAM_AVAILABLE

    if not transcription_service_available:
        raise HTTPException(status_code=400, detail=f"{transcription_provider_name.capitalize()} SDK for transcription is not available on the server.")
    if not get_user_api_key(user_id, transcription_provider_name):
        raise HTTPException(status_code=400, detail=f"API key for {transcription_provider_name.capitalize()} (transcription) not found for user or session expired.")

    # Check Gemini availability for annotation if needed
    requires_gemini_for_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    if requires_gemini_for_annotation:
        if not GEMINI_AVAILABLE:
            raise HTTPException(status_code=400, detail="Gemini SDK for annotation is not available on the server.")
        if not get_user_api_key(user_id, "gemini"):
            raise HTTPException(status_code=400, detail="Gemini API key for annotation not found for user or session expired.")

    logger.info(f"User {user_id} - Received annotated request with prompt: {process_request.prompt[:100] if process_request.prompt else 'None'}")

    needs_age_gender = process_request.annotations and any(a in ["age", "gender"] for a in process_request.annotations)
    needs_emotion = process_request.annotations and "emotion" in process_request.annotations

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e: 
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error_details: List[str] = []
                record_data: Dict[str, Any] = {"audio_filepath": str(audio_file.resolve()), "task_name": "NER"}
                logger.info(f"--- User {user_id} - Processing {audio_file.name} (Selective Annotation) ---")
                processed_files_count += 1
                record_data["duration"] = get_audio_duration(audio_file)
                audio_data: Optional[np.ndarray] = None # Ensure type hint for audio_data
                sample_rate: Optional[int] = None

                if needs_age_gender or needs_emotion:
                    audio_data, sample_rate, load_err = load_audio(audio_file)
                    if load_err:
                        file_error_details.append(load_err)
                    elif audio_data is None or (sample_rate != TARGET_SAMPLE_RATE if sample_rate is not None else True):
                        file_error_details.append("Audio load/SR mismatch for A/G/E.")
                
                transcription_text: Optional[str] = None
                transcription_error: Optional[str] = None

                if model_choice == ModelChoice.whissle:
                    transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file, user_id)
                elif model_choice == ModelChoice.gemini:
                    transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file, user_id)
                elif model_choice == ModelChoice.deepgram:
                    transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file, user_id)
                else:
                    transcription_error = "Invalid model choice for transcription."

                if transcription_error:
                    file_error_details.append(f"Transcription: {transcription_error}")
                    # transcription_text remains None
                elif transcription_text is None: # Explicitly check for None after successful call
                    file_error_details.append("Transcription returned None without an error.")
                
                record_data["original_transcription"] = transcription_text
                record_data["text"] = transcription_text # This might be overwritten by annotation if successful

                # Age/Gender and Emotion processing (remains largely the same, ensure audio_data and sample_rate are valid)
                if (needs_age_gender or needs_emotion) and audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                    tasks = []
                    if needs_age_gender:
                        if age_gender_model is None: file_error_details.append("A/G_WARN: Age/Gender model not loaded.")
                        else: tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                    if needs_emotion:
                        if emotion_model is None: file_error_details.append("EMO_WARN: Emotion model not loaded.")
                        else: tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result_item in results: # Renamed result to result_item to avoid conflict
                            if isinstance(result_item, Exception):
                                logger.error(f"Error in A/G/E sub-task: {result_item}", exc_info=False)
                                file_error_details.append(f"A_G_E_SubtaskError: {type(result_item).__name__}")
                                continue

                            if isinstance(result_item, tuple) and len(result_item) == 3: # Age/Gender result
                                age_pred, gender_idx, age_gender_err = result_item
                                if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                else:
                                    if process_request.annotations and "age" in process_request.annotations and age_pred is not None:
                                        try:
                                            actual_age = round(age_pred, 1)
                                            age_brackets = [(18, "0-17"), (25, "18-24"), (35, "25-34"), (45, "35-44"), (55, "45-54"), (65, "55-64"), (float('inf'), "65+")]
                                            age_group = "Unknown"
                                            for threshold, bracket in age_brackets:
                                                if actual_age < threshold:
                                                    age_group = bracket
                                                    break
                                            record_data["age_group"] = age_group
                                        except Exception as age_e:
                                            logger.error(f"Error formatting age_group: {age_e}")
                                            record_data["age_group"] = "Error"
                                    if process_request.annotations and "gender" in process_request.annotations and gender_idx is not None:
                                        gender_str = "Unknown"
                                        if gender_idx == 1: gender_str = "Male"
                                        elif gender_idx == 0: gender_str = "Female"
                                        record_data["gender"] = gender_str
                            elif isinstance(result_item, tuple) and len(result_item) == 2: # Emotion result
                                emotion_label, emotion_err = result_item
                                if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
                                elif process_request.annotations and "emotion" in process_request.annotations and emotion_label is not None:
                                    record_data["emotion"] = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"
                
                # Gemini Annotation
                prompt_type_for_gemini: Optional[str] = None
                if requires_gemini_for_annotation and transcription_text and transcription_text.strip() != "":
                    if process_request.annotations and ("entity" in process_request.annotations or "intent" in process_request.annotations):
                        prompt_type_for_gemini = process_request.prompt # Use user's custom prompt if provided
                    
                    tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(
                        transcription_text, 
                        custom_prompt=prompt_type_for_gemini, 
                        user_id=user_id
                    )
                    if gemini_anno_err:
                        file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                        if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "ANNOTATION_FAILED"
                    else:
                        if process_request.annotations and "entity" in process_request.annotations and tokens and tags:
                             record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                        if process_request.annotations and "intent" in process_request.annotations and intent:
                             record_data["gemini_intent"] = intent
                    record_data["prompt_used"] = prompt_type_for_gemini if prompt_type_for_gemini else "default_generated_prompt_behavior" # Clarify what prompt was used
                
                elif requires_gemini_for_annotation: # Case where transcription failed or was empty
                    if not transcription_text or transcription_text.strip() == "":
                        if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "NO_SPEECH_FOR_ANNOTATION"
                    # else: # This case should be covered by transcription_error already
                        # if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "TRANSCRIPTION_FAILED_FOR_ANNOTATION"

                final_error_msg = "; ".join(file_error_details) if file_error_details else None
                record_data["error"] = final_error_msg
                
                # ... (rest of record saving and error counting logic remains similar) ...
                current_errors_before_write = error_count
                try:
                    record = AnnotatedJsonlRecord(**record_data)
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                    final_error_msg = (final_error_msg + "; " if final_error_msg else "") + f"JSONL write error: {write_e}"
                    record_data["error"] = final_error_msg # Update record_data if write fails
                    if error_count == current_errors_before_write: # Ensure error is counted if write fails
                        error_count += 1
                
                if not final_error_msg: # Successfully processed and written
                    saved_records_count += 1
                elif error_count == current_errors_before_write and final_error_msg: # Error occurred before write, or write itself failed
                     error_count +=1 # Ensure error is counted if it happened before write and wasn't already

                del audio_data # Cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ... (final message and return ProcessResponse remains similar) ...
        truly_successful_saves = saved_records_count
        final_message = (
            f"Processed {processed_files_count}/{len(audio_files)} files for selective annotation. "
            f"{truly_successful_saves} records successfully saved (no internal errors). "
            f"{error_count} files encountered errors or warnings (check 'error' field in JSONL)."
        )
        return ProcessResponse(
            message=final_message,
            output_file=str(output_jsonl_path),
            processed_files=processed_files_count,
            saved_records=truly_successful_saves,
            errors=error_count
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

@router.get("/status", summary="API Status")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API (User Session Based)",
        "docs_url": "/docs", "html_interface": "/",
        "endpoints": {
            "init_session": "/init_session/",
            "transcription_only": "/create_transcription_manifest/",
            "full_annotation": "/create_annotated_manifest/"
        },
        "gemini_sdk_available": GEMINI_AVAILABLE,
        "whissle_sdk_available": WHISSLE_AVAILABLE,
        "deepgram_sdk_available": DEEPGRAM_AVAILABLE,
        "age_gender_model_loaded": age_gender_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
    }