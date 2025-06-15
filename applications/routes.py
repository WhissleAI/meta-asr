# applications/routes.py
from fastapi import FastAPI, HTTPException, APIRouter, WebSocket # Added WebSocket
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
    InitSessionRequest,
    UserApiKey,
    GcsProcessRequest, # Added
    SingleFileProcessResponse # Added
)
from models import (
    GEMINI_AVAILABLE, WHISSLE_AVAILABLE, DEEPGRAM_AVAILABLE, # Updated to _AVAILABLE flags
    age_gender_model, age_gender_processor,
    emotion_model, emotion_feature_extractor
    # Removed GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED as they are replaced by session checks
)
from audio_utils import validate_paths, discover_audio_files, load_audio, get_audio_duration, trim_audio # Added trim_audio
from transcription import transcribe_with_whissle_single, transcribe_with_gemini_single, transcribe_with_deepgram_single
from annotation import annotate_text_structured_with_gemini
import json
from session_store import init_user_session, is_user_session_valid, get_user_api_key # Added session_store imports
from gcs_utils import parse_gcs_path, download_gcs_blob # Added
from websocket_utils import manager as websocket_manager # Added WebSocket manager

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

@router.post("/trim_audio_and_transcribe/", response_model=ProcessResponse, summary="Trim Audio Files and Create Transcription Manifest")
async def trim_audio_and_transcribe_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = process_request.model_choice
    provider_name = model_choice.value
    segment_length_sec = process_request.segment_length_sec # Assuming this is added to ProcessRequest

    if not segment_length_sec or segment_length_sec <= 0:
        raise HTTPException(status_code=400, detail="Invalid segment length provided.")
    segment_length_ms = segment_length_sec * 1000

    # Check service availability and user key (similar to create_transcription_manifest_endpoint)
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
    original_audio_files = discover_audio_files(dir_path)

    if not original_audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files found in the directory.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0
    all_trimmed_audio_paths = []

    # Create a subdirectory for trimmed files within the original audio directory
    trimmed_audio_base_dir = dir_path / "trimmed_segments"
    trimmed_audio_base_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in original_audio_files:
        try:
            # Create a unique subdirectory for each original file's segments
            file_specific_trimmed_dir = trimmed_audio_base_dir / audio_file.stem
            trimmed_segments = await asyncio.to_thread(trim_audio, audio_file, segment_length_ms, file_specific_trimmed_dir)
            all_trimmed_audio_paths.extend(trimmed_segments)
        except Exception as e:
            logger.error(f"Error trimming {audio_file.name}: {e}", exc_info=True)
            error_count += 1 # Count error for the original file that failed to trim
    
    if not all_trimmed_audio_paths:
        return ProcessResponse(message=f"No audio segments were created after trimming. Original errors: {error_count}", output_file=str(output_jsonl_path), processed_files=len(original_audio_files), saved_records=0, errors=error_count)

    # Now transcribe the trimmed segments
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_segment_path in all_trimmed_audio_paths:
                file_error: Optional[str] = None
                transcription_text: Optional[str] = None
                duration: Optional[float] = None
                logger.info(f"--- Processing segment {audio_segment_path.name} (Transcription for user {user_id}) ---")
                processed_files_count += 1 # Now counting segments
                try:
                    duration = get_audio_duration(audio_segment_path)
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_segment_path, user_id)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_segment_path, user_id)
                    elif model_choice == ModelChoice.deepgram:
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_segment_path, user_id)
                    else:
                        transcription_error = "Invalid model choice."

                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None:
                        file_error = "Transcription returned None without an error."

                except Exception as e:
                    logger.error(f"Unexpected error processing segment {audio_segment_path.name} for user {user_id}: {e}", exc_info=True)
                    file_error = f"Unexpected error: {type(e).__name__}: {e}"
                
                record = TranscriptionJsonlRecord(
                    audio_filepath=str(audio_segment_path.resolve()), 
                    text=transcription_text, 
                    duration=duration, 
                    model_used_for_transcription=model_choice.value, 
                    error=file_error
                )
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_segment_path.name}: {write_e}", exc_info=True)
                    file_error = (file_error + "; " if file_error else "") + f"JSONL write error: {write_e}"
                
                if file_error:
                    error_count += 1 # Counting errors for segments
                else:
                    saved_records_count += 1
                gc.collect()
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")
    
    msg = f"Processed {processed_files_count}/{len(all_trimmed_audio_paths)} audio segments. Saved: {saved_records_count}. Errors: {error_count}."
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
                    record_data["prompt_used"] = prompt_type_for_gemini[:100] if prompt_type_for_gemini else "default_generated_prompt_behavior" # Clarify what prompt was used
                
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

async def _process_single_downloaded_file(
    local_audio_path: Path,
    user_id: str,
    model_choice: ModelChoice,
    requested_annotations: Optional[List[str]],
    custom_prompt: Optional[str],
    output_jsonl_path: Path, # New parameter
    original_gcs_path: str # New parameter
) -> Dict[str, Any]:
    """
    Processes a single downloaded audio file for transcription and optional annotation.
    Sends progress updates via WebSocket and returns a dictionary of results.
    """
    results: Dict[str, Any] = { # Initialize results structure
        "duration": None, "transcription": None, "age_group": None, "gender": None,
        "emotion": None, "bio_annotation_gemini": None, "gemini_intent": None,
        "prompt_used": None, "error_details": [], "overall_error_summary": None
    }
    await websocket_manager.send_personal_message({"status": "processing_started", "detail": f"Processing file: {local_audio_path.name}"}, user_id)

    try:
        await websocket_manager.send_personal_message({"status": "calculating_duration", "detail": "Calculating audio duration..."}, user_id)
        results["duration"] = await asyncio.to_thread(get_audio_duration, local_audio_path)
        await websocket_manager.send_personal_message({"status": "duration_complete", "detail": f"Duration: {results['duration']:.2f}s" if results["duration"] else "Duration: Unknown"}, user_id)
    except Exception as e:
        logger.error(f"User {user_id} - Failed to get duration for {local_audio_path.name}: {e}")
        results["error_details"].append(f"DurationError: {str(e)}")
        await websocket_manager.send_personal_message({"status": "error", "detail": f"Failed to get duration: {str(e)}"}, user_id)

    # --- Transcription ---
    transcription_text: Optional[str] = None
    transcription_provider_name = model_choice.value
    await websocket_manager.send_personal_message({"status": "transcription_started", "detail": f"Starting transcription with {transcription_provider_name.capitalize()}..."}, user_id)

    if not get_user_api_key(user_id, transcription_provider_name):
        err_msg = f"TranscriptionError: API key for {transcription_provider_name.capitalize()} not found or session expired."
        results["error_details"].append(err_msg)
        await websocket_manager.send_personal_message({"status": "error", "detail": err_msg}, user_id)
    else:
        transcription_error: Optional[str] = None
        try:
            if model_choice == ModelChoice.whissle:
                transcription_text, transcription_error = await transcribe_with_whissle_single(local_audio_path, user_id)
            elif model_choice == ModelChoice.gemini:
                transcription_text, transcription_error = await transcribe_with_gemini_single(local_audio_path, user_id)
            elif model_choice == ModelChoice.deepgram:
                transcription_text, transcription_error = await transcribe_with_deepgram_single(local_audio_path, user_id)
            else:
                transcription_error = "Invalid transcription model choice."

            if transcription_error:
                results["error_details"].append(f"TranscriptionError: {transcription_error}")
                await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": transcription_error}, user_id)
            elif transcription_text is None:
                results["error_details"].append("TranscriptionError: Transcription returned None without an explicit error.")
                await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": "Transcription returned no text."}, user_id)
            else:
                results["transcription"] = transcription_text
                await websocket_manager.send_personal_message({"status": "transcription_complete", "detail": "Transcription successful.", "data": {"transcription": transcription_text[:100] + "..." if transcription_text and len(transcription_text) > 100 else transcription_text}}, user_id)
        except Exception as e:
            logger.error(f"User {user_id} - Transcription failed for {local_audio_path.name} with {model_choice.value}: {e}", exc_info=True)
            err_msg = f"TranscriptionError: Unexpected error - {type(e).__name__}: {str(e)}"
            results["error_details"].append(err_msg)
            await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": err_msg}, user_id)

    # --- Annotations (if transcription was successful and annotations requested) ---
    if transcription_text and requested_annotations:
        await websocket_manager.send_personal_message({"status": "annotation_started", "detail": "Starting annotations..."}, user_id)
        # ... (rest of annotation logic from previous step, with added WebSocket messages for progress/completion/errors for each annotation type)
        # For brevity, I'll show a conceptual placement for a Gemini annotation message.
        # You would add similar messages for age/gender/emotion if they are run.

        audio_data: Optional[np.ndarray] = None
        sample_rate: Optional[int] = None
        needs_local_models = any(a in ["age", "gender", "emotion"] for a in requested_annotations)
        
        if needs_local_models:
            # ... (audio loading logic)
            pass # Add WS messages for audio loading status if it's slow

        # Age/Gender and Emotion (Example for one part)
        if audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
            if "age" in requested_annotations or "gender" in requested_annotations:
                await websocket_manager.send_personal_message({"status": "age_gender_started", "detail": "Processing age/gender..."}, user_id)
                # ... (age/gender processing) ...
                if results.get("age_group") or results.get("gender"):
                    await websocket_manager.send_personal_message({"status": "age_gender_complete", "data": {"age": results.get("age_group"), "gender": results.get("gender")}}, user_id)
                elif any("AgeGenderError" in e for e in results["error_details"]):
                     await websocket_manager.send_personal_message({"status": "age_gender_failed", "detail": "Age/gender processing failed."}, user_id)
            # Similar for emotion
            if "emotion" in requested_annotations:
                await websocket_manager.send_personal_message({"status": "emotion_started", "detail": "Processing emotion..."}, user_id)
                # ... (emotion processing) ...
                if results.get("emotion"):
                    await websocket_manager.send_personal_message({"status": "emotion_complete", "data": {"emotion": results.get("emotion")}}, user_id)
                elif any("EmotionError" in e for e in results["error_details"]):
                    await websocket_manager.send_personal_message({"status": "emotion_failed", "detail": "Emotion processing failed."}, user_id)
        
        requires_gemini_for_annotation = any(a in ["entity", "intent"] for a in requested_annotations)
        if requires_gemini_for_annotation:
            await websocket_manager.send_personal_message({"status": "gemini_annotation_started", "detail": "Starting Gemini entity/intent annotation..."}, user_id)
            # ... (Gemini annotation logic) ...
            if results.get("bio_annotation_gemini") or results.get("gemini_intent"):
                 await websocket_manager.send_personal_message({"status": "gemini_annotation_complete", "data": {"entities_tags_sample": results.get("bio_annotation_gemini"), "intent": results.get("gemini_intent")}}, user_id)
            elif any("GeminiAnnotationError" in e for e in results["error_details"]):
                await websocket_manager.send_personal_message({"status": "gemini_annotation_failed", "detail": "Gemini annotation failed."}, user_id)

        await websocket_manager.send_personal_message({"status": "annotation_complete", "detail": "Annotations finished."}, user_id)

    elif requested_annotations and not transcription_text:
         err_msg = "AnnotationSkipped: Transcription failed or was empty, skipping annotations."
         results["error_details"].append(err_msg)
         await websocket_manager.send_personal_message({"status": "annotation_skipped", "detail": err_msg}, user_id)

    if results["error_details"]:
        results["overall_error_summary"] = "; ".join(results["error_details"])
        await websocket_manager.send_personal_message({"status": "processing_failed", "detail": results["overall_error_summary"], "data": results}, user_id)
    else:
        await websocket_manager.send_personal_message({"status": "processing_complete", "detail": "All processing finished successfully.", "data": results}, user_id)
    
    # Save to JSONL
    try:
        record_to_save = {
            "audio_filepath": original_gcs_path, # Use original GCS path
            "text": results.get("transcription"),
            "duration": results.get("duration"),
            "model_used_for_transcription": model_choice.value,
            "age_group": results.get("age_group"),
            "gender": results.get("gender"),
            "emotion": results.get("emotion"),
            "bio_annotation_gemini": results.get("bio_annotation_gemini"),
            "gemini_intent": results.get("gemini_intent"),
            "prompt_used": results.get("prompt_used"),
            "error": results.get("overall_error_summary")
        }
        # Filter out None values before saving
        final_record = {k: v for k, v in record_to_save.items() if v is not None}

        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
            json.dump(final_record, f_out)
            f_out.write('\n')
        logger.info(f"User {user_id} - Successfully saved GCS processing result to {output_jsonl_path}")
        await websocket_manager.send_personal_message({"status": "result_saved", "detail": f"Result saved to {output_jsonl_path.name}"}, user_id)

    except IOError as e:
        logger.error(f"User {user_id} - Failed to write GCS result to {output_jsonl_path}: {e}")
        results["error_details"].append(f"Failed to save result: {e}")
        results["overall_error_summary"] = "; ".join(results["error_details"])
        await websocket_manager.send_personal_message({"status": "save_failed", "detail": f"Failed to save result to {output_jsonl_path.name}"}, user_id)
    except Exception as e:
        logger.error(f"User {user_id} - Unexpected error saving GCS result to {output_jsonl_path}: {e}", exc_info=True)
        results["error_details"].append(f"Unexpected error saving result: {e}")
        results["overall_error_summary"] = "; ".join(results["error_details"])
        await websocket_manager.send_personal_message({"status": "save_failed", "detail": f"Unexpected error saving result to {output_jsonl_path.name}"}, user_id)

    return results

@router.post("/process_gcs_file/", response_model=SingleFileProcessResponse, summary="Download and Process a Single File from GCS")
async def process_gcs_file_endpoint(request: GcsProcessRequest):
    user_id = request.user_id
    output_jsonl_path_str = request.output_jsonl_path # Get the output path string
    logger.info(f"User {user_id} - Received request to process GCS file: {request.gcs_path}. Output will be saved to: {output_jsonl_path_str}")
    await websocket_manager.send_personal_message({"status": "request_received", "detail": f"Received request for {request.gcs_path}"}, user_id)

    if not is_user_session_valid(user_id):
        logger.warning(f"User {user_id} - Invalid or expired session for GCS processing.")
        # Send error over WebSocket before returning HTTP response
        await websocket_manager.send_personal_message({"status": "error", "detail": "User session is invalid or expired."}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            status_message="User session is invalid or expired. Please re-initialize session.",
            overall_error="AuthenticationError"
        )

    bucket_name, blob_name = parse_gcs_path(request.gcs_path)
    if not bucket_name or not blob_name:
        logger.error(f"User {user_id} - Invalid GCS path provided: {request.gcs_path}")
        await websocket_manager.send_personal_message({"status": "error", "detail": "Invalid GCS path format."}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            status_message="Invalid GCS path format.",
            overall_error="InvalidInput"
        )

    await websocket_manager.send_personal_message({"status": "download_started", "detail": f"Downloading gs://{bucket_name}/{blob_name}..."}, user_id)
    local_audio_path: Optional[Path] = None
    try:
        # Validate and create Path object for output_jsonl_path
        output_jsonl_path = Path(output_jsonl_path_str)
        if not output_jsonl_path.parent.exists():
            try:
                output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"User {user_id} - Created directory for output JSONL: {output_jsonl_path.parent}")
            except Exception as e:
                logger.error(f"User {user_id} - Failed to create directory {output_jsonl_path.parent}: {e}")
                return SingleFileProcessResponse(
                    original_gcs_path=request.gcs_path,
                    status_message=f"Failed to create output directory: {output_jsonl_path.parent}",
                    overall_error="OutputDirectoryError"
                )

        local_audio_path = await asyncio.to_thread(download_gcs_blob, bucket_name, blob_name)
        if not local_audio_path:
            logger.error(f"User {user_id} - Failed to download gs://{bucket_name}/{blob_name}")
            err_msg = f"Failed to download file from GCS path: gs://{bucket_name}/{blob_name}"
            await websocket_manager.send_personal_message({"status": "download_failed", "detail": err_msg}, user_id)
            return SingleFileProcessResponse(
                original_gcs_path=request.gcs_path,
                status_message=err_msg,
                overall_error="DownloadError"
            )
        logger.info(f"User {user_id} - Successfully downloaded to {local_audio_path}")
        await websocket_manager.send_personal_message({"status": "download_complete", "detail": f"Successfully downloaded to {local_audio_path}"}, user_id)

        processing_results = await _process_single_downloaded_file(
            local_audio_path,
            user_id,
            request.model_choice,
            request.annotations,
            request.prompt,
            output_jsonl_path, # Pass the Path object
            request.gcs_path # Pass original GCS path for saving
        )
        
        status_msg = f"File processed. Results saved to {output_jsonl_path.name}"
        if processing_results.get("overall_error_summary"):
            status_msg = f"File processed with errors/warnings: {processing_results['overall_error_summary']}"
        elif not processing_results.get("transcription") and not processing_results.get("overall_error_summary"):
             status_msg = "File processed, but no transcription was generated (and no specific errors reported during transcription)."

        # Final WebSocket message is sent from within _process_single_downloaded_file
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            downloaded_local_path=str(local_audio_path),
            status_message=status_msg,
            duration=processing_results.get("duration"),
            transcription=processing_results.get("transcription"),
            age_group=processing_results.get("age_group"),
            gender=processing_results.get("gender"),
            emotion=processing_results.get("emotion"),
            bio_annotation_gemini=processing_results.get("bio_annotation_gemini"),
            gemini_intent=processing_results.get("gemini_intent"),
            prompt_used=processing_results.get("prompt_used"),
            error_details=processing_results.get("error_details"),
            overall_error=processing_results.get("overall_error_summary")
        )

    except Exception as e:
        logger.error(f"User {user_id} - Unexpected error during GCS file processing for {request.gcs_path}: {e}", exc_info=True)
        err_msg_server = f"An unexpected server error occurred: {type(e).__name__}"
        await websocket_manager.send_personal_message({"status": "error", "detail": err_msg_server}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            downloaded_local_path=str(local_audio_path) if local_audio_path else None,
            status_message=err_msg_server,
            overall_error="ServerError"
        )
    finally:
        if local_audio_path and local_audio_path.exists():
            try:
                logger.info(f"User {user_id} - Attempting final cleanup of: {local_audio_path}")
                local_audio_path.unlink()
                logger.info(f"User {user_id} - Cleaned up temporary file: {local_audio_path}")
            except Exception as e_cleanup:
                logger.error(f"User {user_id} - Error during final cleanup of temporary file {local_audio_path}: {e_cleanup}")

@router.websocket("/ws/gcs_status/{user_id}")
async def gcs_status_websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_manager.connect(websocket, user_id)
    logger.info(f"User {user_id} WebSocket connected for GCS status updates.")
    try:
        while True:
            # Keep the connection alive. The server will push messages.
            # You can optionally receive messages from the client if needed for this WS.
            # For simple status updates, just keeping it open is often enough.
            # A common pattern is to await a receive_text or receive_json call,
            # which will also detect disconnections.
            data = await websocket.receive_text() # Or receive_json if expecting client messages
            # logger.debug(f"Received WebSocket message from {user_id}: {data}") # If you expect client pings or messages
            # If client sends a specific "close" message, you could break here.
            # Or, more commonly, the client just closes the connection, which the 'except WebSocketDisconnect' below handles.
    except Exception as e: # WebSocketDisconnect inherits from Exception, but catch others too.
        # This will catch WebSocketDisconnect when the client closes the connection,
        # and other potential errors during the receive loop.
        logger.info(f"WebSocket for user {user_id} disconnected or error: {type(e).__name__} - {e}")
    finally:
        websocket_manager.disconnect(user_id)
        logger.info(f"User {user_id} WebSocket connection closed and cleaned up.")

# Ensure the main app includes this router, e.g., in applications/main.py:
# from . import routes
# app.include_router(routes.router)

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