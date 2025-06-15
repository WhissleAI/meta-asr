# applications/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import os
from config import logger
from routes import router


# Initialize FastAPI app
app = FastAPI(
    title="Audio Processing API",
    description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities, "
                "and saves results to a JSONL manifest file.",
    version="1.5.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

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
    logger.info(f"API Docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        app_module_string, 
        host=host, 
        port=port, 
        reload=reload, 
        reload_dirs=[str(script_dir)] if reload else None, 
        log_level=log_level
    )