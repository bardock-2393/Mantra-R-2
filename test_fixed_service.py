#!/usr/bin/env python3
"""
Test script for the fixed AI Video Detective services
Tests CUDA availability, model loading, and service integration
"""

import sys
import os
import torch
from PIL import Image
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cuda_availability():
    """Test CUDA availability and basic PyTorch functionality"""
    print("🔍 Testing CUDA availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    print(f"✅ CUDA is available")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name()}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.mm(x, y)
        print("✅ Basic CUDA tensor operations successful")
        return True
    except Exception as e:
        print(f"❌ CUDA tensor operations failed: {e}")
        return False

def test_model_import():
    """Test if the MiniCPM model can be imported and initialized"""
    print("\n🔍 Testing MiniCPM model import...")
    
    try:
        from models.minicpm_v26_model import MiniCPMV26Model
        print("✅ MiniCPM model class imported successfully")
        
        # Test model initialization
        model = MiniCPMV26Model()
        print("✅ MiniCPM model instance created successfully")
        
        return True, model
    except Exception as e:
        print(f"❌ MiniCPM model import failed: {e}")
        return False, None

def test_basic_text_generation():
    """Test basic text generation with the model"""
    print("\n🔍 Testing basic text generation...")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Test with a simple model first
        model_path = 'openbmb/MiniCPM-V-2_6'
        
        print(f"   Loading model from {model_path}...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        )
        model = model.eval().cuda()
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("   Testing text generation...")
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        question = "What color is this image?"
        msgs = [{'role': 'user', 'content': [dummy_image, question]}]
        
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        
        print(f"✅ Text generation successful: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def test_ai_service():
    """Test the AI service integration"""
    print("\n🔍 Testing AI service integration...")
    
    try:
        from services.ai_service_fixed import AIService
        print("✅ AI service imported successfully")
        
        # Initialize the service
        service = AIService()
        print("✅ AI service instance created successfully")
        
        # Test initialization
        service.initialize()
        print("✅ AI service initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ AI service test failed: {e}")
        return False

def test_gpu_service():
    """Test the GPU service"""
    print("\n🔍 Testing GPU service...")
    
    try:
        from services.gpu_service import GPUService
        print("✅ GPU service imported successfully")
        
        # Initialize the service
        gpu_service = GPUService()
        print("✅ GPU service instance created successfully")
        
        # Test initialization
        gpu_service.initialize()
        print("✅ GPU service initialized successfully")
        
        # Test status retrieval
        status = gpu_service.get_status()
        print("✅ GPU status retrieved successfully")
        print(f"   GPU count: {status.get('gpu_count', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU service test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AI Video Detective Service Tests")
    print("=" * 50)
    
    # Test 1: CUDA availability
    cuda_ok = test_cuda_availability()
    if not cuda_ok:
        print("\n❌ CUDA tests failed. Cannot proceed with model tests.")
        return
    
    # Test 2: Model import
    model_ok, model = test_model_import()
    if not model_ok:
        print("\n❌ Model import failed. Cannot proceed with service tests.")
        return
    
    # Test 3: Basic text generation
    generation_ok = test_basic_text_generation()
    
    # Test 4: GPU service
    gpu_ok = test_gpu_service()
    
    # Test 5: AI service (only if GPU service works)
    ai_ok = False
    if gpu_ok:
        ai_ok = test_ai_service()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"   Model Import: {'✅' if model_ok else '❌'}")
    print(f"   Text Generation: {'✅' if generation_ok else '❌'}")
    print(f"   GPU Service: {'✅' if gpu_ok else '❌'}")
    print(f"   AI Service: {'✅' if ai_ok else '❌'}")
    
    if all([cuda_ok, model_ok, generation_ok, gpu_ok, ai_ok]):
        print("\n🎉 All tests passed! The AI Video Detective service is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    # Cleanup
    if model:
        try:
            model.cleanup()
            print("✅ Model cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main() 