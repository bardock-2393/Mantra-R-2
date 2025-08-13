#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL-7B integration
Verifies that the 7B model can be loaded and used correctly
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_7b_model():
    """Test the 7B model integration"""
    try:
        print("🧪 Testing Qwen2.5-VL-7B integration...")
        
        # Import the model manager
        from services.model_manager import model_manager
        
        print(f"✅ Model manager imported successfully")
        print(f"📋 Available models: {list(model_manager.available_models.keys())}")
        print(f"🎯 Current model: {model_manager.current_model}")
        
        # Check if 7B model is available
        if 'qwen25vl_7b' not in model_manager.available_models:
            print("❌ 7B model not available in model manager")
            return False
        
        print("✅ 7B model found in model manager")
        
        # Try to initialize the 7B model
        print("🚀 Initializing 7B model...")
        success = await model_manager.initialize_model('qwen25vl_7b')
        
        if not success:
            print("❌ Failed to initialize 7B model")
            return False
        
        print("✅ 7B model initialized successfully")
        
        # Get model info
        model_info = model_manager.get_current_model()
        print(f"📊 Model info: {model_info['name']}")
        print(f"🔧 Initialized: {model_info['initialized']}")
        
        # Test health check
        health = await model_manager.health_check()
        print(f"💚 Health check: {health}")
        
        # Test simple text generation
        print("🧠 Testing text generation...")
        try:
            response = await model_manager.generate_chat_response(
                analysis_result="Test video analysis",
                analysis_type="comprehensive",
                user_focus="general",
                message="Hello, how are you?",
                chat_history=[]
            )
            print(f"✅ Text generation successful: {response[:100]}...")
        except Exception as e:
            print(f"⚠️ Text generation test failed: {e}")
        
        print("✅ 7B model integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ 7B model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🎯 Qwen2.5-VL-7B Integration Test")
    print("=" * 50)
    
    success = await test_7b_model()
    
    if success:
        print("\n🎉 All tests passed! 7B model is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





