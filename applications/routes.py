# applications/routes.py
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse
from pathlib import Path
import gc
import torch
import asyncio
from typing import Dict, Any, List, Tuple
import numpy as np
from config import (
    ProcessRequest, ProcessResponse, 
    TranscriptionJsonlRecord, AnnotatedJsonlRecord, 
    logger, device, 
    ModelChoice, 
    TARGET_SAMPLE_RATE, 
    BioAnnotation 
)
from models import (
    GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED, 
    age_gender_model, age_gender_processor, 
    emotion_model, emotion_feature_extractor,
    WHISSLE_AVAILABLE
)
from audio_utils import validate_paths, discover_audio_files, load_audio, get_audio_duration
from transcription import transcribe_with_whissle_single, transcribe_with_gemini_single, transcribe_with_deepgram_single
from annotation import annotate_text_structured_with_gemini

router = APIRouter()

def predict_age_gender(audio_data, sampling_rate) -> Tuple[float, int, str]:
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

def predict_emotion(audio_data, sampling_rate) -> Tuple[str, str]:
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
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED: 
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    if model_choice == ModelChoice.gemini and not GEMINI_CONFIGURED: 
        raise HTTPException(status_code=400, detail="Gemini not configured.")
    if model_choice == ModelChoice.deepgram and not DEEPGRAM_CONFIGURED:
        raise HTTPException(status_code=400, detail="Deepgram not configured.")
    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)
    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e: 
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)
    processed_files_count = 0; saved_records_count = 0; error_count = 0
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: str = None; transcription_text: str = None; duration: float = None
                logger.info(f"--- Processing {audio_file.name} (Transcription Only) ---")
                processed_files_count += 1
                try:
                    duration = get_audio_duration(audio_file)
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                    else:
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None:
                        file_error = "Transcription returned None."
                except Exception as e:
                    file_error = f"Unexpected error: {type(e).__name__}: {e}"
                record = TranscriptionJsonlRecord(audio_filepath=str(audio_file.resolve()), text=transcription_text, duration=duration, model_used_for_transcription=model_choice.value, error=file_error)
                try: 
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    if not file_error: 
                        file_error = f"JSONL write error: {write_e}"
                    else: 
                        file_error += f"; JSONL write error: {write_e}"
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
    model_choice = process_request.model_choice
    # check if getting prompt
    logger.info(f"Received request with prompt: {process_request.prompt[:100]}")
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    requires_gemini_for_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    if requires_gemini_for_annotation and not GEMINI_CONFIGURED:
        raise HTTPException(status_code=400, detail="Gemini not configured (required for entity/intent annotation).")
    needs_age_gender = process_request.annotations and any(a in ["age", "gender"] for a in process_request.annotations)
    needs_emotion = process_request.annotations and "emotion" in process_request.annotations
    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)
    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _:
                pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(
            message="No audio files. Empty manifest created.",
            output_file=str(output_jsonl_path),
            processed_files=0,
            saved_records=0,
            errors=0
        )
    processed_files_count = 0
    saved_records_count = 0
    error_count = 0
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error_details: List[str] = []
                record_data: Dict[str, Any] = {"audio_filepath": str(audio_file.resolve()), "task_name": "NER"}
                logger.info(f"--- Processing {audio_file.name} (Selective Annotation) ---")
                processed_files_count += 1
                record_data["duration"] = get_audio_duration(audio_file)
                audio_data = None
                sample_rate = None
                if needs_age_gender or needs_emotion:
                    audio_data, sample_rate, load_err = load_audio(audio_file)
                    if load_err:
                        file_error_details.append(load_err)
                    elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                        file_error_details.append("Audio load/SR mismatch for A/G/E.")
                transcription_text: str = None
                transcription_error: str = None
                if model_choice == ModelChoice.whissle:
                    transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                elif model_choice == ModelChoice.gemini:
                    transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                else:
                    transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
                if transcription_error:
                    file_error_details.append(f"Transcription: {transcription_error}")
                    transcription_text = None
                elif transcription_text is None:
                    file_error_details.append("Transcription returned None.")
                record_data["original_transcription"] = transcription_text
                record_data["text"] = transcription_text
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
                        for result in results:
                            if isinstance(result, tuple) and len(result) == 3:
                                age_pred, gender_idx, age_gender_err = result
                                if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                else:
                                    if "age" in process_request.annotations:
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
                                    if "gender" in process_request.annotations:
                                        gender_str = "Unknown"
                                        if gender_idx == 1: gender_str = "Male"
                                        elif gender_idx == 0: gender_str = "Female"
                                        record_data["gender"] = gender_str
                            elif isinstance(result, tuple) and len(result) == 2:
                                emotion_label, emotion_err = result
                                if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
                                else:
                                    if "emotion" in process_request.annotations:
                                        record_data["emotion"] = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"
                if requires_gemini_for_annotation and transcription_text and transcription_text.strip() != "":
                    if "entity" in process_request.annotations or "intent" in process_request.annotations:
                            prompt_type = process_request.prompt if process_request.prompt else None
                    tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(transcription_text, custom_prompt=prompt_type)
                # if "entity" in process_request.annotations or "intent" in process_request.annotations:
                #             prompt_type = process_request.prompt if process_request.prompt else None
                #             try:
                #                 tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(
                #                     transcription_text, custom_prompt=prompt_type
                #                 )
                    if gemini_anno_err:
                        file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                        if "intent" in process_request.annotations: record_data["gemini_intent"] = "ANNOTATION_FAILED"
                    else:
                        if "entity" in process_request.annotations: record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                        if "intent" in process_request.annotations: record_data["gemini_intent"] = intent
                    record_data["prompt_used"] = process_request.prompt  # Add prompt to record
                elif requires_gemini_for_annotation:
                    if transcription_text == "":
                        if "intent" in process_request.annotations: record_data["gemini_intent"] = "NO_SPEECH"
                    else:
                        if "intent" in process_request.annotations: record_data["gemini_intent"] = "TRANSCRIPTION_FAILED"
                final_error_msg = "; ".join(file_error_details) if file_error_details else None
                record_data["error"] = final_error_msg
                current_errors_before_write = error_count
                try:
                    record = AnnotatedJsonlRecord(**record_data)
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                    if not final_error_msg:
                        final_error_msg = f"JSONL write error: {write_e}"
                    if error_count == current_errors_before_write:
                        error_count += 1
                if not final_error_msg:
                    saved_records_count += 1
                elif error_count == current_errors_before_write and final_error_msg:
                    error_count += 1
                del audio_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
        "message": "Welcome to the Audio Processing API v1.5.0",
        "docs_url": "/docs", "html_interface": "/",
        "endpoints": { "transcription_only": "/create_transcription_manifest/", "full_annotation": "/create_annotated_manifest/" },
        "gemini_configured": GEMINI_CONFIGURED, "whissle_available": WHISSLE_AVAILABLE, "whissle_configured": WHISSLE_CONFIGURED,
        "age_gender_model_loaded": age_gender_model is not None, "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
    }