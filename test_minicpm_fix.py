#!/usr/bin/env python3
"""
Test script for MiniCPM-V 2.6 model fix
"""

import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def test_minicpm_model():
    """Test the fixed MiniCPM-V 2.6 model"""
    try:
        print("🧪 Testing MiniCPM-V 2.6 model fix...")
        
        # Import the model
        from minicpm_v26_model import minicpm_v26_model
        
        print("✅ Model imported successfully")
        
        # Test initialization
        print("🚀 Initializing model...")
        minicpm_v26_model.initialize()
        
        print("✅ Model initialized successfully")
        
        # Test text generation
        print("📝 Testing text generation...")
        test_prompt = "Hello, how are you today?"
        result = minicpm_v26_model.generate_text(test_prompt, max_new_tokens=20)
        
        print(f"✅ Text generation successful: {result}")
        
        # Test model status
        status = minicpm_v26_model.get_model_status()
        print(f"📊 Model status: {status}")
        
        # Cleanup
        minicpm_v26_model.cleanup()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_minicpm_model()
    sys.exit(0 if success else 1) 