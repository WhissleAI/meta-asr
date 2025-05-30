"""
Main FastAPI application for Audio Processing API
Refactored to use modular components
"""
import os
import gc
import asyncio
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any
import traceback # ADDED FOR DEBUGGING
from dotenv import load_dotenv
load_dotenv("E:/Meta_asr/meta-asr/applications/.env")  # Load environment variables from .env file 

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print("DEBUG: main.py execution started. Attempting custom module imports...")

try:    # Import our custom modules
    from api_modules.config import (
        setup_api_configurations, device, logger,
        is_gemini_configured, is_whissle_configured, is_deepgram_configured,
        get_api_configurations
    )
    from api_modules.models import (
        ModelChoice, ProcessRequest, ProcessResponse,
        TranscriptionJsonlRecord, AnnotatedJsonlRecord, BioAnnotation
    )
    from api_modules.age_gender_model import (
        load_age_gender_model, predict_age_gender, 
        get_age_group, get_gender_string,
        age_gender_model
    )
    from api_modules.emotion_model import (
        load_emotion_model, predict_emotion, format_emotion_label,
        emotion_model
    )
    from api_modules.transcription import (
        transcribe_with_whissle_single, transcribe_with_gemini_single,
        transcribe_with_deepgram_single
    )
    from api_modules.prompt import annotate_text_structured_with_gemini
    from api_modules.audio_utils import (
        discover_audio_files, load_audio, get_audio_duration, validate_paths
    )
    print("DEBUG: All custom modules appear to have been imported by main.py.")
except Exception: # Catching a broad exception to see any import-related error
    print("CRITICAL ERROR: An exception occurred during the import of API modules in main.py.")
    traceback.print_exc()
    raise # Re-raise the exception to halt execution, as 'app' won't be defined.

print("DEBUG: Importing contextlib...")
from contextlib import asynccontextmanager
print("DEBUG: contextlib imported successfully")

print("DEBUG: Defining lifespan function...")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Initializing models...")
    setup_api_configurations()
    load_age_gender_model()
    load_emotion_model()
    logger.info("Model initialization complete.")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Application shutdown")
print("DEBUG: lifespan function defined successfully")

# --- Initialize FastAPI ---
print("DEBUG: Creating FastAPI app...")
try:
    app = FastAPI(
        title="Audio Processing API",
        description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities, "
                    "and saves results to a JSONL manifest file.",
        version="1.5.0",
        lifespan=lifespan
    )
    print("DEBUG: FastAPI app created successfully")
except Exception as e:
    print(f"DEBUG: Error creating FastAPI app: {e}")
    import traceback
    traceback.print_exc()
    raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount static files
current_file_path = Path(__file__).parent
static_dir = current_file_path / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the main HTML interface"""
    index_html_path = current_file_path / "static" / "index.html"
    if not index_html_path.is_file():
        logger.error(f"HTML file not found at: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html_path)

@app.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    """Create a transcription-only JSONL manifest"""
    model_choice = process_request.model_choice
      # Validate model configuration
    if model_choice == ModelChoice.whissle and not is_whissle_configured(): 
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    if model_choice == ModelChoice.gemini and not is_gemini_configured(): 
        raise HTTPException(status_code=400, detail="Gemini not configured.")
    if model_choice == ModelChoice.deepgram and not is_deepgram_configured():
        raise HTTPException(status_code=400, detail="Deepgram not configured.")
    
    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: 
                pass
        except IOError as e: 
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(
            message=f"No audio files. Empty manifest created.", 
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
                file_error = None
                transcription_text = None
                duration = None
                
                logger.info(f"--- Processing {audio_file.name} (Transcription Only) ---")
                processed_files_count += 1
                
                try:
                    duration = get_audio_duration(audio_file)
                    
                    # Transcribe based on model choice
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                    else:  # Deepgram
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
                    
                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None:
                        file_error = "Transcription returned None."
                        
                except Exception as e:
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
    return ProcessResponse(
        message=msg, 
        output_file=str(output_jsonl_path), 
        processed_files=processed_files_count, 
        saved_records=saved_records_count, 
        errors=error_count
    )

@app.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    """Create an annotated JSONL manifest with optional age/gender/emotion/entity/intent annotations"""
    model_choice = process_request.model_choice
      # Validate configurations
    if model_choice == ModelChoice.whissle and not is_whissle_configured():
        raise HTTPException(status_code=400, detail="Whissle not configured.")
    
    # Check if Gemini is required for entity/intent and configured
    requires_gemini_for_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    if requires_gemini_for_annotation and not is_gemini_configured():
        raise HTTPException(status_code=400, detail="Gemini not configured (required for entity/intent annotation).")

    # Check if Age/Gender/Emotion models are needed and loaded
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
                file_error_details = []                # Initialize data for the record
                record_data: Dict[str, Any] = {
                    "audio_filepath": str(audio_file.resolve()),
                    "task_name": "OTHER",  # Changed to "OTHER" as requested
                    "gender": None,
                    "age_group": None,
                    "emotion": "Unk",  # Default emotion value
                    "gemini_intent": None,
                    "bio_annotation_gemini": None,
                    "ollama_intent": None,  # Placeholder for future integration
                    "bio_annotation_ollama": None,  # Placeholder for future integration
                }

                logger.info(f"--- Processing {audio_file.name} (Selective Annotation) ---")
                processed_files_count += 1

                # 1. Get Duration
                record_data["duration"] = get_audio_duration(audio_file)

                # 2. Load Audio for A/G/E if requested
                audio_data = None
                sample_rate = None
                if needs_age_gender or needs_emotion:
                    audio_data, sample_rate, load_err = load_audio(audio_file)
                    if load_err:
                        file_error_details.append(load_err)
                    elif audio_data is None or sample_rate != 16000:  # TARGET_SAMPLE_RATE
                        file_error_details.append("Audio load/SR mismatch for A/G/E.")
                
                # 3. Transcribe Audio
                transcription_text = None
                transcription_error = None
                if model_choice == ModelChoice.whissle:
                    transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                elif model_choice == ModelChoice.gemini:
                    transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)
                else:  # Deepgram
                    transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file)
                
                if transcription_error:
                    file_error_details.append(f"Transcription: {transcription_error}")
                    transcription_text = None  # Mark as failed if error
                elif transcription_text is None:
                    file_error_details.append("Transcription returned None.")
                
                # Store both original transcription and processed text
                record_data["original_transcription"] = transcription_text
                record_data["text"] = transcription_text  # For now, same as original; can be processed later

                # 4. Predict Age/Gender/Emotion if requested
                if (needs_age_gender or needs_emotion) and audio_data is not None and sample_rate == 16000:
                    tasks = []
                    if needs_age_gender:
                        if age_gender_model is None:
                            file_error_details.append("A/G_WARN: Age/Gender model not loaded.")
                        else:
                            tasks.append(asyncio.to_thread(predict_age_gender, audio_data, 16000))
                    if needs_emotion:
                        if emotion_model is None:
                            file_error_details.append("EMO_WARN: Emotion model not loaded.")
                        else:
                            tasks.append(asyncio.to_thread(predict_emotion, audio_data, 16000))
                    
                    if tasks:  # Only run gather if there are tasks
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        for result in results:
                            if isinstance(result, tuple) and len(result) == 3:  # Age/Gender result
                                age_pred, gender_idx, age_gender_err = result
                                if age_gender_err:
                                    file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                else:
                                    if "age" in process_request.annotations:
                                        record_data["age_group"] = get_age_group(age_pred)
                                    if "gender" in process_request.annotations:
                                        record_data["gender"] = get_gender_string(gender_idx)
                                        
                            elif isinstance(result, tuple) and len(result) == 2:  # Emotion result
                                emotion_label, emotion_err = result
                                if emotion_err:
                                    file_error_details.append(f"EMO_WARN: {emotion_err}")
                                else:
                                    if "emotion" in process_request.annotations:
                                        record_data["emotion"] = format_emotion_label(emotion_label)
                                
                # 5. Annotate Entities/Intent with Gemini if requested and transcription exists
                if requires_gemini_for_annotation and transcription_text and transcription_text.strip() != "":
                    tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(transcription_text)
                    if gemini_anno_err:
                        file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                        if "intent" in process_request.annotations:
                            record_data["gemini_intent"] = "ANNOTATION_FAILED"
                    else:
                        if "entity" in process_request.annotations:
                            record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                        if "intent" in process_request.annotations:
                            record_data["gemini_intent"] = intent  # intent is already uppercase from the function
                elif requires_gemini_for_annotation:  # If annotation requested but transcription is empty/failed
                    if transcription_text == "":
                        if "intent" in process_request.annotations:
                            record_data["gemini_intent"] = "NO_SPEECH"
                    else:  # transcription_text is None (transcription failed)
                        if "intent" in process_request.annotations:
                            record_data["gemini_intent"] = "TRANSCRIPTION_FAILED"

                # Set final error message
                final_error_msg = "; ".join(file_error_details) if file_error_details else None
                record_data["error"] = final_error_msg

                # Create Pydantic record and write to file
                current_errors_before_write = error_count
                try:
                    record = AnnotatedJsonlRecord(**record_data)
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                    if not final_error_msg:  # If the error was *only* during write
                        final_error_msg = f"JSONL write error: {write_e}"
                    if error_count == current_errors_before_write:  # Ensure it's counted once
                        error_count += 1
                
                # Determine if record was truly successful (no processing errors, no write errors)
                if not final_error_msg:
                    saved_records_count += 1
                elif error_count == current_errors_before_write and final_error_msg:  # If processing errors caused error_count to increment already
                    error_count += 1

                # Clean up GPU memory
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

@app.get("/status", summary="API Status")
async def get_status():
    """Get API status and model information"""
    config_status = get_api_configurations()
    return {
        "message": "Welcome to the Audio Processing API v1.5.0",
        "docs_url": "/docs", 
        "html_interface": "/",        "endpoints": { 
            "transcription_only": "/create_transcription_manifest/", 
            "full_annotation": "/create_annotated_manifest/" 
        },
        "gemini_configured": config_status["gemini_configured"],
        "whissle_configured": config_status["whissle_configured"],
        "deepgram_configured": config_status["deepgram_configured"],
        "age_gender_model_loaded": age_gender_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    fastapi_script_name = Path(__file__).stem
    app_module_string = f"{fastapi_script_name}:app"
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = "info"
    
    logger.info(f"Starting FastAPI server for '{app_module_string}' on {host}:{port}...")
    logger.info(f"Log Level: {log_level.upper()}, Reload Mode: {'Enabled' if reload else 'Disabled'}")
    logger.info(f"Docs: http://{host}:{port}/docs, UI: http://{host}:{port}/")
    
    uvicorn.run(
        app_module_string, 
        host=host, 
        port=port, 
        reload=reload, 
        reload_dirs=[str(script_dir)] if reload else None, 
        log_level=log_level
    )