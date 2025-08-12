#!/usr/bin/env python3
"""
Model Loading Script for AI Video Detective
Loads the Qwen2.5-VL-32B model and integrates it with the AI service
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
        print("❌ CUDA not available. GPU is required for Qwen2.5-VL-32B")
        return False
    
    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
    
    if gpu_memory < 30:
        print("⚠️ Warning: GPU memory may be insufficient for 32B model")
        print("   Recommended: 40GB+ for optimal performance")
    
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

async def load_qwen25vl_32b():
    """Load the Qwen2.5-VL-32B model"""
    try:
        print("🚀 Loading Qwen2.5-VL-32B model...")
        
        # Import the service
        from services.qwen25vl_32b_service import Qwen25VL32BService
        
        # Create service instance
        service = Qwen25VL32BService()
        
        # Initialize the model
        await service.initialize()
        
        print("✅ Qwen2.5-VL-32B model loaded successfully!")
        return service
        
    except Exception as e:
        print(f"❌ Failed to load Qwen2.5-VL-32B model: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_ai_service(loaded_service):
    """Update the main AI service to use the loaded model"""
    try:
        print("🔄 Updating AI service...")
        
        # Import the main AI service
        from services.ai_service import ai_service
        
        # Update the service with the loaded model
        ai_service.model = loaded_service.model
        ai_service.tokenizer = loaded_service.tokenizer
        ai_service.processor = loaded_service.processor
        ai_service.is_initialized = True
        
        print("✅ AI service updated with loaded model")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update AI service: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model(service):
    """Test the loaded model with a simple prompt"""
    try:
        print("🧪 Testing loaded model...")
        
        # Simple test prompt
        test_prompt = "Hello, can you describe what you see in this image?"
        
        # Generate response
        response = await service._generate_text(test_prompt, max_new_tokens=100)
        
        print(f"✅ Model test successful!")
        print(f"📝 Response: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

async def main():
    """Main function to load and integrate the model"""
    print("🎯 AI Video Detective - Model Loading Script")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix issues and try again.")
        return False
    
    # Load the model
    loaded_service = await load_qwen25vl_32b()
    if not loaded_service:
        print("❌ Model loading failed")
        return False
    
    # Update the main AI service
    if not update_ai_service(loaded_service):
        print("❌ Failed to update AI service")
        return False
    
    # Test the model
    if not await test_model(loaded_service):
        print("❌ Model test failed")
        return False
    
    print("\n🎉 Model loading and integration completed successfully!")
    print("🚀 AI Video Detective is now ready for full video analysis!")
    print("\nNext steps:")
    print("1. Restart your Flask application")
    print("2. Upload a video for analysis")
    print("3. Ask questions about the video content")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print("\n✅ Script completed successfully!")
        else:
            print("\n❌ Script failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

