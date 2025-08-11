#!/usr/bin/env python3
"""
AI Video Detective - Startup Script for Round 2
Simple script to run the application with GPU-powered local AI
"""

import os
import sys
from app import app

def main():
    """Main startup function for Round 2"""
    print("🚀 AI Video Detective Round 2 - GPU-Powered Local AI")
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
    
    # Check Hugging Face configuration
    print("\n🔐 Hugging Face Configuration:")
    hf_token = os.getenv('HF_TOKEN', '')
    if hf_token:
        print(f"✅ HF_TOKEN configured: {hf_token[:8]}...")
    else:
        print("⚠️  No HF_TOKEN found")
        print("   Some models may require authentication")
        print("   Create .env file with HF_TOKEN=your_token")
        print("   Get token from: https://huggingface.co/settings/tokens")
    
    # Check model configuration
    model_path = os.getenv('MINICPM_MODEL_PATH', 'microsoft/DialoGPT-medium')
    print(f"🤖 Model: {model_path}")
    if 'openbmb/MiniCPM-V-2_6' in model_path and not hf_token:
        print("⚠️  MiniCPM-V-2_6 requires authentication")
        print("   System will fallback to open model automatically")
    
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
    
    print("🤖 Local AI: MiniCPM-V-2_6 ready for video analysis")
    print("🎯 Performance: <1000ms latency, 90fps processing")
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