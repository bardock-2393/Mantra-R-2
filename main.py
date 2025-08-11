#!/usr/bin/env python3
"""
AI Video Detective - Startup Script for Round 2
Simple script to run the application with GPU-powered local AI (7B model only)
"""

import os
import sys
from app import app

def main():
    """Main startup function for Round 2 - 7B Model Only"""
    print("🚀 AI Video Detective Round 2 - GPU-Powered Local AI (7B Model)")
    print("=" * 60)
    
    # Check if required environment variables are set
    from config import Config
    
    # Check GPU availability for Round 2
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🖥️  GPU Detected: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        else:
            print("⚠️  Warning: No CUDA-capable GPU detected")
            print("   GPU is required for Round 2 performance targets")
            print("   Please ensure CUDA drivers are installed")
    except ImportError:
        print("⚠️  Warning: PyTorch not installed")
        print("   Please install PyTorch with CUDA support")
    
    # Check if upload directory exists
    upload_dir = Config.UPLOAD_FOLDER
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"📁 Created upload directory: {upload_dir}")
    
    # Check if session storage directory exists
    session_dir = Config.SESSION_STORAGE_PATH
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        print(f"📁 Created session storage directory: {session_dir}")
    
    print("🤖 Local AI: Qwen2.5-VL-7B ready for video analysis")
    print("🔍 DeepStream: Real-time object detection and tracking")
    print("💾 Vector Search: Fast retrieval and analysis")
    print("🎯 Performance: <1000ms latency, 90fps processing, 120min videos")
    print("🌐 Server: Starting on http://localhost:8000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask application
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=8000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 AI Video Detective Round 2 stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 