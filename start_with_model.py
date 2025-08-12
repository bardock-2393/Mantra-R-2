#!/usr/bin/env python3
"""
AI Video Detective - Startup with Model Loading
Loads the Qwen2.5-VL-32B model and then starts the Flask application
"""

import os
import sys
import asyncio
import threading
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def load_model_in_background():
    """Load the model in a background thread"""
    try:
        print("🚀 Loading Qwen2.5-VL-32B model in background...")
        
        # Import and load the model
        from services.qwen25vl_32b_service import Qwen25VL32BService
        
        # Create service instance
        service = Qwen25VL32BService()
        
        # Initialize the model
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(service.initialize())
        
        # Update the main AI service
        from services.ai_service import ai_service
        ai_service.model = service.model
        ai_service.tokenizer = service.tokenizer
        ai_service.processor = service.processor
        ai_service.is_initialized = True
        
        print("✅ Qwen2.5-VL-32B model loaded and integrated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_flask_app():
    """Start the Flask application"""
    try:
        print("🌐 Starting Flask application...")
        
        # Import and start the Flask app
        from app import app
        
        # Start the Flask application
        app.run(
            debug=True,
            host='0.0.0.0',
            port=8000,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Flask application failed to start: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to load model and start application"""
    print("🎯 AI Video Detective - Starting with Model Loading")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the project directory.")
        return
    
    # Start model loading in background thread
    print("🔄 Starting model loading in background...")
    model_thread = threading.Thread(target=load_model_in_background, daemon=True)
    model_thread.start()
    
    # Give the model a moment to start loading
    time.sleep(2)
    
    # Start Flask application in main thread
    print("🌐 Starting Flask application...")
    start_flask_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 AI Video Detective stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

