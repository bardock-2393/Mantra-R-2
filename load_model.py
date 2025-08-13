#!/usr/bin/env python3
"""
Model Loading Script for AI Video Detective
Loads the Qwen2.5-VL-7B model and integrates it with the AI service
"""

import os
import sys
import asyncio
import torch
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check if the environment is ready for model loading"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPU is required for Qwen2.5-VL-7B")
        return False
    
    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
    
    # 7B model requires less memory than 32B
    if gpu_memory < 8:
        print("❌ Error: Insufficient GPU memory for 7B model")
        print("   Required: 8GB+, Recommended: 16GB+")
        return False
    elif gpu_memory < 12:
        print("⚠️ Warning: GPU memory may be insufficient for optimal 7B performance")
        print("   Recommended: 16GB+ for optimal performance")
    
    # Check required packages
    required_packages = [
        'transformers',
        'torch',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Environment check passed")
    return True

async def load_qwen25vl_7b():
    """Load the Qwen2.5-VL-7B model"""
    try:
        print("🚀 Loading Qwen2.5-VL-7B model...")
        
        # Import the model manager
        from services.model_manager import model_manager
        
        # Initialize the 7B model
        success = await model_manager.initialize_model('qwen25vl_7b')
        
        if success:
            print("✅ Qwen2.5-VL-7B model loaded successfully!")
            return model_manager
        else:
            print("❌ Failed to load Qwen2.5-VL-7B model")
            return None
        
    except Exception as e:
        print(f"❌ Failed to load Qwen2.5-VL-7B model: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_ai_service(loaded_service):
    """Update the main AI service to use the loaded model"""
    try:
        print("🔄 Updating AI service...")
        
        # Import the main AI service
        from services.ai_service import ai_service
        
        # Get the current model info
        current_model = loaded_service.get_current_model()
        
        if current_model['initialized'] and current_model['service_instance']:
            # Update the service with the loaded model
            ai_service.model = current_model['service_instance'].model
            ai_service.tokenizer = current_model['service_instance'].tokenizer
            ai_service.processor = current_model['service_instance'].processor
            ai_service.is_initialized = True
            
            print("✅ AI service updated with loaded model")
            return True
        else:
            print("❌ Model not properly initialized")
            return False
        
    except Exception as e:
        print(f"❌ Failed to update AI service: {e}")
        return False

async def main():
    """Main function to load the 7B model"""
    print("🎯 AI Video Detective - Loading Qwen2.5-VL-7B Model")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    # Load the 7B model
    model_manager = await load_qwen25vl_7b()
    if not model_manager:
        print("❌ Model loading failed")
        return False
    
    # Update AI service
    if not update_ai_service(model_manager):
        print("❌ Failed to update AI service")
        return False
    
    print("✅ Qwen2.5-VL-7B model loaded and integrated successfully!")
    print("🚀 Ready to analyze videos!")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Model loading stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

