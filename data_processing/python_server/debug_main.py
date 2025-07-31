#!/usr/bin/env python3
"""
Step-by-step debug test to find where main.py execution fails
"""
import sys
import traceback

print("Debug test for main.py execution...")
print("=" * 50)

try:
    print("Step 1: Testing imports one by one...")
    
    # Test basic imports
    print("  - Testing FastAPI import...")
    from fastapi import FastAPI
    print("    ✓ FastAPI imported")
    
    print("  - Testing our modules...")
    from api_modules.config import setup_api_configurations, device, logger
    print("    ✓ config imported")
    
    from api_modules.models import ModelChoice, ProcessRequest
    print("    ✓ models imported")
    
    print("  - Testing model modules...")
    from api_modules.age_gender_model import load_age_gender_model
    print("    ✓ age_gender_model imported")
    
    from api_modules.emotion_model import load_emotion_model
    print("    ✓ emotion_model imported")
    
    print("Step 2: Testing manual app creation...")
    # Try to create an app manually with similar setup
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def test_lifespan(app):
        print("    - Test lifespan startup")
        yield
        print("    - Test lifespan shutdown")
    
    test_app = FastAPI(title="Test", lifespan=test_lifespan)
    print("    ✓ Manual FastAPI app created successfully")
    
    print("Step 3: Testing main.py import with detailed error catching...")
    
    # Import main.py line by line execution simulation
    import importlib.util
    import os
    
    main_path = os.path.join(os.getcwd(), "main.py")
    print(f"    - Loading main.py from: {main_path}")
    
    spec = importlib.util.spec_from_file_location("main_debug", main_path)
    main_debug = importlib.util.module_from_spec(spec)
    
    print("    - Executing main.py...")
    try:
        spec.loader.exec_module(main_debug)
        print("    ✓ main.py executed successfully")
        
        if hasattr(main_debug, 'app'):
            print(f"    ✓ Found app: {type(main_debug.app)}")
        else:
            print("    ❌ No app found after execution")
            print(f"    Available: {[x for x in dir(main_debug) if not x.startswith('_')]}")
            
    except Exception as exec_error:
        print(f"    ❌ Error during main.py execution: {exec_error}")
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Error in debug test: {e}")
    traceback.print_exc()
