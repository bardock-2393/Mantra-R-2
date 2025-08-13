#!/usr/bin/env python3
"""
Model Loading Script for AI Video Detective
Loads the Qwen2.5-VL-7B model and then starts the Flask application
"""

import os
import sys
import asyncio
import threading
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def load_model_in_background():
    """Load the 7B model in a background thread"""
    try:
        print("🚀 Loading Qwen2.5-VL-7B model in background...")
        
        # Import the model manager
        from services.model_manager import model_manager
        
        # Initialize the 7B model
        asyncio.run(model_manager.initialize_model('qwen25vl_7b'))
        
        print("✅ Qwen2.5-VL-7B model loaded and integrated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load 7B model: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to start the application"""
    print("🚀 Starting AI Video Detective with Qwen2.5-VL-7B...")
    
    # Start model loading in background
    model_thread = threading.Thread(target=load_model_in_background, daemon=True)
    model_thread.start()
    
    # Start Flask app
    try:
        from app import app
        print("🌐 Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

