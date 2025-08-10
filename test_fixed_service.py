#!/usr/bin/env python3
"""
Test script for the fixed MiniCPM-V-2_6 service
Tests model initialization and basic functionality
"""

import sys
import os
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_minicpm_model():
    """Test the MiniCPM model initialization and basic functionality"""
    try:
        print("🧪 Testing MiniCPM-V-2_6 model...")
        
        # Test 1: Check CUDA availability
        print("\n1. Checking CUDA availability...")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            print("❌ CUDA not available - this will cause issues")
            return False
        
        # Test 2: Import the model
        print("\n2. Importing MiniCPM model...")
        try:
            from models.minicpm_v26_model import minicpm_v26_model
            print("✅ Model imported successfully")
        except Exception as e:
            print(f"❌ Model import failed: {e}")
            return False
        
        # Test 3: Initialize the model
        print("\n3. Initializing model...")
        try:
            minicpm_v26_model.initialize()
            print("✅ Model initialized successfully")
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
        
        # Test 4: Test basic text generation
        print("\n4. Testing text generation...")
        try:
            test_prompt = "Hello, this is a test message. Please respond with a simple greeting."
            response = minicpm_v26_model.generate_text(test_prompt, max_new_tokens=50)
            print(f"✅ Text generation successful: {response[:100]}...")
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return False
        
        # Test 5: Test model status
        print("\n5. Testing model status...")
        try:
            status = minicpm_v26_model.get_model_status()
            print(f"✅ Model status: {status}")
        except Exception as e:
            print(f"❌ Model status failed: {e}")
            return False
        
        print("\n🎉 All tests passed! MiniCPM-V-2_6 is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_service():
    """Test the AI service integration"""
    try:
        print("\n🧪 Testing AI service integration...")
        
        # Import the service
        from services.ai_service_fixed import minicpm_service
        print("✅ AI service imported successfully")
        
        # Test initialization
        print("Initializing AI service...")
        minicpm_service.initialize()
        print("✅ AI service initialized successfully")
        
        # Test status
        status = minicpm_service.get_model_status()
        print(f"✅ AI service status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting MiniCPM-V-2_6 service tests...")
    print("=" * 50)
    
    # Test the model directly
    model_test_passed = test_minicpm_model()
    
    if model_test_passed:
        # Test the AI service
        service_test_passed = test_ai_service()
        
        if service_test_passed:
            print("\n🎉 All tests passed! The service is ready to use.")
        else:
            print("\n❌ AI service tests failed.")
            return 1
    else:
        print("\n❌ Model tests failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 